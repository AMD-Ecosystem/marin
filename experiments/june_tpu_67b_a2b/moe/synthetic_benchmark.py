# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the June Snowball Grug trainer on synthetic tokens on one or more Slurm nodes, without Fray or Iris.

This is the Grug-code counterpart of ``experiments/grug/snowball_synthetic``: the June 67B-A2B recipe's own
model (``experiments/june_tpu_67b_a2b/moe/model.py``) and training loop, on the same shape, batch and portable
kernels (reference or ``xla_flash`` attention, XLA ``ragged_dot``), so the two code paths can be compared on
hardware that has no CUDA or TPU kernels. Every step sees fresh uniform-random tokens with no repeats, so the
loss cannot fall below ln(vocab_size) and a lower value means something leaks the targets. No checkpoint is
written. Like the Snowball harness, it logs the model FLOPs per example up front and ends with the median step
time from step 2 on, tokens/s, model TFLOP/s and MFU, then each device's peak memory. It also logs each step's wall
period (``throughput/wall_step_s``: the time between consecutive steps' hooks, which includes the host work between
steps that ``throughput/duration`` leaves out) and their median from step 2 on.

With ``--eval-every N`` every N-th step (and the last) also runs a forward-only held-out loss inside the train step
(``GrugStepEvalConfig``) on ``--eval-microbatches`` microbatches of separate random sequences (seed 1), so its loss is
about ln(vocab_size). The headline then covers the train-only steps, and a further line gives the eval steps' cost.

Example, 8 GPUs, full 67B shape, expert parallelism over all 8, profile of steps 10-12::

    RAGGED_DOT_IMPL=xla python experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py \\
        --size full --steps 20 --expert-axis 8 --profile-steps 3

Across nodes, launch one process per node with ``srun`` and pass ``--gpus-per-node``. Devices are ordered by
process, so the mesh's inner axes stay within a node: on two 8-GPU nodes, ``--expert-axis 8 --replica-axis 2``
replicates the model per node and all-reduces gradients across nodes, ``--expert-axis 8`` alone shards the
parameters over a cross-node ``data`` axis (FSDP), and ``--expert-axis 16`` spreads the experts over both nodes.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from rigging.log_setup import configure_logging

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

# The spurix launcher replaces this module's report_memory with one that logs through this logger.
logger = logging.getLogger("june_synthetic_benchmark")
# The training examples' seed and the held-out ones': no held-out sequence is a training sequence.
TRAIN_SEED = 0
EVAL_SEED = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/june-synthetic"))
    parser.add_argument("--run-id", help="Run id under --log-dir (default: size and timestamp).")
    return parser.parse_args(argv)


def synthetic_examples(count: int, seq_len: int, vocab_size: int, seed: int) -> list[GrugLmExample]:
    rng = np.random.default_rng(seed)
    loss_weight = np.ones(seq_len, dtype=np.float32)
    loss_weight[-1] = 0
    return [
        GrugLmExample(tokens=rng.integers(0, vocab_size, seq_len, dtype=np.int32), loss_weight=loss_weight)
        for _ in range(count)
    ]


def build_run_config(args: argparse.Namespace) -> GrugRunConfig:
    """The benchmark's run: the command line's model and mesh on fresh random tokens, flat Adam, no checkpoints."""
    model, default_mp = build_model(args)
    batch_size = global_batch_size(args)
    run_id = shared_run_id(args, "june")
    trainer = build_trainer(
        args,
        batch_size=batch_size,
        num_train_steps=args.steps,
        run_id=run_id,
        default_mp=default_mp,
        tracker=JsonLoggerConfig(),
        checkpointer=CheckpointerConfig(base_path=str(args.log_dir / run_id / "checkpoints")),
        load_checkpoint=False,  # never resume a reused --run-id; every benchmark starts from step 0
    )
    # One fresh example per sequence, and a mixture block exactly as long as the dataset so nothing repeats.
    examples = synthetic_examples(args.steps * batch_size, model.max_seq_len, model.vocab_size, TRAIN_SEED)
    data = LmDataConfig(
        components={"synthetic": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
        tokenizer="passthrough",
        vocab_size=model.vocab_size,
        mixture_block_size=len(examples),
    )
    return GrugRunConfig(
        model=model,
        data=data,
        resources=ResourceConfig(),
        optimizer=AdamConfig(learning_rate=args.learning_rate, warmup=0.0),
        trainer=build_grug_trainer(args, trainer, save_checkpoints=False),
        eval=None,
        step_eval=step_eval_config(args, batch_size, model.max_seq_len, model.vocab_size),
    )


def step_eval_config(
    args: argparse.Namespace, batch_size: int, seq_len: int, vocab_size: int
) -> GrugStepEvalConfig | None:
    sequences = step_eval_sequences(args, batch_size)
    if not sequences:
        return None
    held_out = ListAsyncDataset(synthetic_examples(sequences, seq_len, vocab_size, EVAL_SEED))
    return GrugStepEvalConfig(held_out, every=args.eval_every, microbatches_per_eval=args.eval_microbatches)


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
    step_times = StepTimes(config.step_eval, trainer.num_train_steps)
    _run_grug_local(config, step_hooks=[step_times])
    step_times.log_summary(batch_size, model.max_seq_len, flops_per_example)
    report_memory()


if __name__ == "__main__":
    main()
