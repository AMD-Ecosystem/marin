# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the June 67B-A2B model on a pre-tokenized flat Levanter cache on one or more Slurm nodes, without Fray or
Iris.

The run counterpart of ``synthetic_benchmark.py``: the same model, mesh and performance options
(``slurm_run_config.py``), so a configuration timed by the benchmark trains at the same speed. On top it reads its
tokens from a flat Levanter cache (for example the FineWeb-Edu cache ``make_fineweb_mini.py`` writes), sizes the
optimizer's warmup and decay to a token budget, saves checkpoints and resumes from the latest one, and can hold out
part of the cache for a loss eval inside the train step (``--eval-every``, ``--eval-microbatches``;
``GrugStepEvalConfig``), which logs ``eval/loss`` and ``eval/bpb``. It logs the same headline, memory and FLOP lines
as the benchmark, then "trained steps A-B: T tokens, final loss L".

Data: the cache is one token stream of documents laid out as [BOS, text, EOS], cut into ``--seq-len`` windows
across document boundaries (Levanter's ``TokenSeqDataset``), block-shuffled with ``--data-seed``. Attention runs
across the packed documents (``block_cross_document_attention=False``), as in the synthetic benchmark:
``gpu_aiter_flash`` takes no segment ids. Every process loads only the rows of its own devices. The run refuses to
start if its budget needs more windows than the cache holds, unless ``--allow-data-repeat``.

Optimizer (``--optimizer``):
- ``muonh``: the June 67B recipe, ``MoeMuonHHeuristic.build_muonh_config`` with the budget and batch: MuonH on the
  matrices, AdamH on the output projection, Adam on embeddings, router and norms (``rmsnorm_to_adam``, as the June
  cooldown run), 1% warmup, linear decay to 5% of the peak (the June runs' floor), no clipping, logit z-loss 1e-4.
- ``adamw``: Levanter's AdamW (the benchmark's optimizer, so its step time is the measured one) with an explicit
  ``--learning-rate``, 1% warmup and linear decay to 10%.

Example, two 8-GPU nodes, one srun process per node, 1B tokens::

    srun --nodes 2 --ntasks-per-node 1 python experiments/june_tpu_67b_a2b/moe/slurm_train.py \\
        --gpus-per-node 8 --size full --expert-axis 8 --replica-axis 2 --microbatches 4 --batch-size 512 \\
        --data-cache-dir /shared/data/fineweb-edu-1000000000 --tokenizer /shared/data/tokenizers/marin-tokenizer \\
        --train-tokens 1000000000 --optimizer muonh --checkpoint-dir /shared/checkpoints [best7 options]
"""

import argparse
import dataclasses
import json
import logging
import math
from datetime import timedelta
from pathlib import Path

import jax
import numpy as np
from fray.cluster import ResourceConfig
from haliax import Axis
from levanter.callbacks import StepInfo
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, NamedLmDataset
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.store.cache import CacheLedger, TreeCache
from levanter.tokenizers import MarinTokenizer
from levanter.tracker import TrackerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from rigging.log_setup import configure_logging

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.slurm_run_config import (
    StepTimes,
    add_run_arguments,
    build_grug_trainer,
    build_model,
    build_trainer,
    global_batch_size,
    initialize_distributed,
    log_model_flops,
    log_run_config,
    report_memory,
    shared_run_id,
    step_eval_sequences,
)
from experiments.june_tpu_67b_a2b.moe.train import GrugRunConfig, GrugStepEvalConfig, _compute_flops, _run_grug_local

logger = logging.getLogger("june_slurm_train")

MARIN_TOKENIZER = "marin-community/marin-tokenizer"
TRAIN_COMPONENT = "train"
CACHE_EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}
# Levanter's default mixture block, in sequences.
MIXTURE_BLOCK_SIZE = 2048
# Documents at the start of the cache whose BOS, EOS and largest token id are checked against the tokenizer and model.
CHECKED_DOCUMENTS = 1024
# The June 67B runs' schedule floor (MoeMuonHHeuristic(min_lr_ratio=0.05) in the cooldown and its source run) and
# logit z-loss weight.
JUNE_MIN_LR_RATIO = 0.05
JUNE_Z_LOSS_WEIGHT = 1e-4
ADAMW_MIN_LR_RATIO = 0.1
DEFAULT_WARMUP = 0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Override the preset vocabulary, e.g. 128256 to run the tiny preset on a Marin-tokenizer cache.",
    )
    data = parser.add_argument_group("data")
    data.add_argument("--data-cache-dir", required=True, help="Flat Levanter token cache (one train split).")
    data.add_argument(
        "--tokenizer",
        default=MARIN_TOKENIZER,
        help="Tokenizer the cache was built with: a Hugging Face id or a local directory with tokenizer.json "
        "(use a staged copy on shared storage when the compute nodes are offline).",
    )
    data.add_argument("--data-seed", type=int, default=0, help="Seed of the data shuffle.")
    data.add_argument(
        "--allow-data-repeat",
        action="store_true",
        help="Start even if the budget needs more windows than the cache holds (Levanter then repeats data).",
    )
    run = parser.add_argument_group("budget and optimizer")
    run.add_argument(
        "--train-tokens",
        type=lambda s: int(float(s)),
        required=True,
        help="Token budget: the run lasts ceil(tokens / (batch x seq_len)) steps, and the LR schedule spans them.",
    )
    run.add_argument(
        "--steps",
        type=int,
        help="Optional cross-check of the step count the budget gives (launchers that always pass --steps).",
    )
    run.add_argument("--optimizer", choices=("muonh", "adamw"), required=True)
    run.add_argument(
        "--learning-rate",
        type=float,
        help="Peak LR: required for adamw; for muonh it replaces the heuristic's MuonH LR (Adam's follows at 3/13).",
    )
    run.add_argument(
        "--warmup",
        type=float,
        default=DEFAULT_WARMUP,
        help="Warmup as a fraction of the steps (<= 1) or a whole number of steps (> 1).",
    )
    run.add_argument(
        "--min-lr-ratio",
        type=float,
        help=f"Final LR / peak (default: {JUNE_MIN_LR_RATIO} for muonh, {ADAMW_MIN_LR_RATIO} for adamw).",
    )
    run.add_argument("--z-loss-weight", type=float, default=JUNE_Z_LOSS_WEIGHT, help="Logit z-loss weight.")
    ckpt = parser.add_argument_group("checkpoints")
    ckpt.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Save checkpoints under <dir>/<run-id> and resume from the latest one there (default: none).",
    )
    ckpt.add_argument(
        "--checkpoint-minutes", type=float, default=60.0, help="Minutes between temporary checkpoints (one kept)."
    )
    ckpt.add_argument("--checkpoint-every", type=int, help="Also keep a permanent checkpoint every this many steps.")
    logs = parser.add_argument_group("logging")
    logs.add_argument("--log-dir", type=Path, default=Path("logs/june-train"))
    logs.add_argument("--run-id", help="Run id under --log-dir and --checkpoint-dir (default: size and timestamp).")
    logs.add_argument("--wandb-project", help="Also log to this W&B project (WANDB_MODE and WANDB_API_KEY apply).")
    logs.add_argument("--wandb-entity", help="W&B entity (default: the API key's).")
    return parser.parse_args(argv)


def budget_steps(train_tokens: int, batch_size: int, seq_len: int) -> int:
    return math.ceil(train_tokens / (batch_size * seq_len))


def mixture_block_size(train_windows: int) -> int:
    """Levanter's mixture permutes each block of this many sequences among the component's indices of the same block
    and wraps indices past the end, so a final partial block would repeat early windows and skip late ones."""
    return min(MIXTURE_BLOCK_SIZE, train_windows)


def usable_windows(train_windows: int) -> int:
    """The windows a run can read once each: the whole mixture blocks of the training split."""
    block = mixture_block_size(train_windows)
    return train_windows // block * block


def real_data_config(
    cache_dir: str, tokenizer: str, vocab_size: int, *, eval_sequences: int, train_windows: int
) -> LmDataConfig:
    """One flat cache as the whole training mixture, with ``eval_sequences`` of its windows held out."""
    component = DatasetComponent(source=None, cache_dir=cache_dir, format=TextLmDatasetFormat(), flat_cache=True)
    return LmDataConfig(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        cache_dir=None,
        components={TRAIN_COMPONENT: component},
        train_weights={TRAIN_COMPONENT: 1.0},
        auto_build_caches=False,
        mixture_block_size=mixture_block_size(train_windows),
        # gpu_aiter_flash takes no segment ids: attend across the packed documents, as the synthetic batches do.
        block_cross_document_attention=False,
        num_validation_sequences={TRAIN_COMPONENT: eval_sequences} if eval_sequences else None,
    )


def training_windows(cache: TreeCache, seq_len: int, eval_sequences: int) -> int:
    """The cache's whole windows of ``seq_len`` tokens, less the held-out ones (TokenSeqDataset drops the tail)."""
    return cache.flat_field_length("input_ids") // seq_len - eval_sequences


def open_cache(cache_dir: str) -> TreeCache:
    return TreeCache.load_from_ledger(cache_dir, CACHE_EXEMPLAR, CacheLedger.load(cache_dir))


def check_cache_tokens(cache: TreeCache, tokenizer: MarinTokenizer, model_vocab_size: int) -> None:
    """Fail fast if the cache and model disagree with the tokenizer: ids beyond the model's vocabulary would index
    the embedding out of range without an error. Checks the tokenizer's size and the cache's first documents."""
    if len(tokenizer) > model_vocab_size:
        raise ValueError(
            f"tokenizer {tokenizer.name_or_path} has {len(tokenizer)} tokens; the model only {model_vocab_size}"
        )
    documents = [row["input_ids"] for row in cache.get_batch_sync(slice(0, min(CHECKED_DOCUMENTS, len(cache))))]
    largest = max(int(doc.max()) for doc in documents)
    if largest >= model_vocab_size:
        raise ValueError(f"the cache holds token id {largest}; the model's vocabulary is {model_vocab_size}")
    for name, token_id, position in (("BOS", tokenizer.bos_token_id, 0), ("EOS", tokenizer.eos_token_id, -1)):
        if token_id is not None and any(int(doc[position]) != token_id for doc in documents):
            raise ValueError(
                f"the cache's documents do not all have the tokenizer's {name} ({token_id}) at "
                f"position {position}: built with another tokenizer?"
            )


def check_data_budget(
    args: argparse.Namespace, train_windows: int, *, steps: int, batch_size: int, seq_len: int
) -> None:
    usable = usable_windows(train_windows)
    needed = steps * batch_size
    manifest_path = Path(args.data_cache_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        logger.info("data manifest: %s", {k: v for k, v in manifest.items() if not k.startswith("document_tokens")})
    logger.info(
        "data: %s, %d training windows of %d tokens (%d readable once each, %.3fB tokens), %d held out; budget %d "
        "tokens = %d steps x %d sequences x %d tokens reads %d windows (%.3f passes)",
        args.data_cache_dir,
        train_windows,
        seq_len,
        usable,
        usable * seq_len / 1e9,
        step_eval_sequences(args, batch_size),
        args.train_tokens,
        steps,
        batch_size,
        seq_len,
        needed,
        needed / usable,
    )
    if needed > usable and not args.allow_data_repeat:
        raise ValueError(
            f"{steps} steps x {batch_size} sequences need {needed} windows but the cache gives {usable} without "
            "repeats: build a larger cache, lower --train-tokens, or pass --allow-data-repeat"
        )


def optimizer_config(args: argparse.Namespace, *, batch_size: int, seq_len: int, hidden_dim: int) -> OptimizerConfig:
    """The schedule's length comes from the trainer's step count, which the budget sets."""
    if args.optimizer == "muonh":
        heuristic = MoeMuonHHeuristic(min_lr_ratio=JUNE_MIN_LR_RATIO)
        config = heuristic.build_muonh_config(batch_size, float(args.train_tokens), hidden_dim, seq_len=seq_len)
        config = dataclasses.replace(config, rmsnorm_to_adam=True, warmup=args.warmup)
        if args.learning_rate is not None:
            config = dataclasses.replace(
                config, learning_rate=args.learning_rate, adam_lr=args.learning_rate / heuristic.adamh_ratio
            )
        if args.min_lr_ratio is not None:
            config = dataclasses.replace(config, min_lr_ratio=args.min_lr_ratio)
        return config
    if args.learning_rate is None:
        raise ValueError("--optimizer adamw needs --learning-rate")
    return AdamConfig(
        learning_rate=args.learning_rate,
        warmup=args.warmup,
        lr_schedule="linear",
        min_lr_ratio=ADAMW_MIN_LR_RATIO if args.min_lr_ratio is None else args.min_lr_ratio,
    )


def step_eval_config(data: LmDataConfig, args: argparse.Namespace, seq_len: int) -> GrugStepEvalConfig | None:
    """The eval inside the train step on the windows ``real_data_config`` held out, with bits per byte."""
    if args.eval_every <= 0:
        return None
    named = data.validation_sets(Axis("position", seq_len))[TRAIN_COMPONENT]
    # The held-out windows as GrugLmExamples, which the NamedLmDataset wrapper converts to Levanter's LmExample.
    assert isinstance(named, NamedLmDataset)
    held_out = named.dataset
    return GrugStepEvalConfig(
        held_out, every=args.eval_every, microbatches_per_eval=args.eval_microbatches, compute_bpb=True
    )


def trackers(args: argparse.Namespace, run_id: str) -> TrackerConfig | tuple[TrackerConfig, ...]:
    # The JSON log is what spurix parses; W&B is optional on top.
    if args.wandb_project is None:
        return JsonLoggerConfig()
    wandb = WandbConfig(project=args.wandb_project, entity=args.wandb_entity, name=run_id, id=run_id, resume="allow")
    return (JsonLoggerConfig(), wandb)


def build_run_config(args: argparse.Namespace) -> GrugRunConfig:
    model, default_mp = build_model(args)
    if args.vocab_size is not None:
        model = dataclasses.replace(model, vocab_size=args.vocab_size)
    batch_size = global_batch_size(args)
    seq_len = model.max_seq_len
    steps = budget_steps(args.train_tokens, batch_size, seq_len)
    if args.steps is not None and args.steps != steps:
        raise ValueError(
            f"--steps {args.steps} differs from the {steps} steps that --train-tokens {args.train_tokens} "
            f"gives at {batch_size} x {seq_len} tokens per step"
        )
    cache = open_cache(args.data_cache_dir)
    eval_sequences = step_eval_sequences(args, batch_size)
    train_windows = training_windows(cache, seq_len, eval_sequences)
    data = real_data_config(
        args.data_cache_dir,
        args.tokenizer,
        model.vocab_size,
        eval_sequences=eval_sequences,
        train_windows=train_windows,
    )
    check_cache_tokens(cache, data.the_tokenizer, model.vocab_size)
    check_data_budget(args, train_windows, steps=steps, batch_size=batch_size, seq_len=seq_len)

    run_id = shared_run_id(args, "june-train")
    checkpointing = args.checkpoint_dir is not None
    if checkpointing:
        checkpointer = CheckpointerConfig(
            base_path=str(args.checkpoint_dir),
            save_interval=timedelta(minutes=args.checkpoint_minutes),
            keep=[{"every": args.checkpoint_every}] if args.checkpoint_every else None,
        )
    else:
        logger.warning("no --checkpoint-dir: this run saves no checkpoints and cannot resume")
        checkpointer = CheckpointerConfig(base_path=str(args.log_dir / run_id / "checkpoints"))
    trainer = build_trainer(
        args,
        batch_size=batch_size,
        num_train_steps=steps,
        run_id=run_id,
        default_mp=default_mp,
        tracker=trackers(args, run_id),
        checkpointer=checkpointer,
        # None resumes from this run id's latest checkpoint under --checkpoint-dir, if there is one.
        load_checkpoint=None if checkpointing else False,
    )
    return GrugRunConfig(
        model=model,
        data=data,
        resources=ResourceConfig(),
        optimizer=optimizer_config(args, batch_size=batch_size, seq_len=seq_len, hidden_dim=model.hidden_dim),
        trainer=build_grug_trainer(
            args,
            trainer,
            save_checkpoints=checkpointing,
            data_seed=args.data_seed,
            z_loss_weight=args.z_loss_weight,
        ),
        eval=None,
        step_eval=step_eval_config(data, args, seq_len),
    )


class LastLoss:
    """A step hook that keeps the latest step's loss, and the first and last steps this process ran."""

    def __init__(self) -> None:
        self.first_step: int | None = None
        self.step: int | None = None
        self.loss: float | None = None

    def __call__(self, info: StepInfo, force: bool = False) -> None:
        if force:
            return
        if self.first_step is None:
            self.first_step = info.step
        self.step, self.loss = info.step, info.loss


def main() -> None:
    args = parse_args()
    # The training loop sets up logging only once it starts; without this, the lines logged before it are dropped.
    configure_logging(logging.INFO)
    initialize_distributed(args)
    config = build_run_config(args)
    model, trainer = config.model, config.trainer.trainer
    batch_size = trainer.train_batch_size
    flops_per_example, flops_summary = _compute_flops(model_config=model)
    log_run_config(args, model, trainer, batch_size, flops_per_example)
    log_model_flops(flops_per_example, flops_summary)
    if jax.process_index() == 0:
        logger.info("optimizer: %s", config.optimizer)
    step_times = StepTimes(config.step_eval, trainer.num_train_steps)
    last = LastLoss()
    _run_grug_local(config, step_hooks=[step_times, last])
    step_times.log_summary(batch_size, model.max_seq_len, flops_per_example)
    if last.step is not None and jax.process_index() == 0:
        tokens = (last.step + 1) * batch_size * model.max_seq_len
        logger.info("trained steps %d-%d: %d tokens, final loss %.4f", last.first_step, last.step, tokens, last.loss)
    report_memory()


if __name__ == "__main__":
    main()
