# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The AMD copy's real-data run (fork overlay): its config against the benchmark's, the token budget, the held-out
split for the step eval, and the data path on four CPU devices."""

import dataclasses
import json

import haliax as hax
import jax
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh
from levanter.schedule import BatchSchedule
from levanter.store.cache import SerialCacheWriter
from levanter.testing.devices import use_test_devices
from levanter.tracker.helpers import hparams_to_dict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from experiments.amd.june_67b_a2b import slurm_train, synthetic_benchmark
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig, build_train_dataset, build_train_loader

SEQ_LEN = 4096
BATCH = 512
BOS, EOS = 1, 2
CACHE_EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}
# best7's options on the AMD copy (expert kernels and loop are fixed in code).
BEST7_ARGV = (
    "--size",
    "full",
    "--expert-axis",
    "8",
    "--replica-axis",
    "2",
    "--batch-size",
    str(BATCH),
    "--microbatches",
    "4",
    "--zero1",
    "--attention",
    "gpu_aiter_flash",
    "--watch-interval",
    "0",
)
# A small batch and window keep these caches to tens of thousands of tokens: 4096 tokens a step.
SMALL_STEP = ("--batch-size", "16", "--seq-len", "256")


def write_tokenizer(path) -> str:
    """A five-token tokenizer with BOS and EOS, saved where Levanter's loader reads local tokenizers."""
    vocab = {"<unk>": 0, "<s>": BOS, "</s>": EOS, "hi": 3, "there": 4}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    path.mkdir()
    tokenizer.save(str(path / "tokenizer.json"))
    config = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    return str(path)


def write_document_cache(path, num_tokens: int, *, seed: int = 0) -> None:
    """A flat cache of documents laid out as the Marin preprocessor writes them: [BOS, text..., EOS]."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(10, 3000, num_tokens // 10)
    lengths = lengths[: np.searchsorted(np.cumsum(lengths + 2), num_tokens) + 1]
    docs = [np.concatenate([[BOS], rng.integers(3, 128_000, n), [EOS]]).astype(np.int32) for n in lengths]
    with SerialCacheWriter(str(path), CACHE_EXEMPLAR) as writer:
        writer.write_batch([{"input_ids": doc} for doc in docs])


def write_stream_cache(path, num_tokens: int, *, seed: int = 0) -> None:
    """A flat cache whose token stream is 0, 1, 2, ...: window i of a run holds exactly i * L ... (i + 1) * L - 1, so
    a batch row names the window it came from. Documents of random lengths, as in a real cache."""
    rng = np.random.default_rng(seed)
    stream = np.arange(num_tokens, dtype=np.int32)
    cuts = np.sort(rng.choice(np.arange(1, num_tokens), size=num_tokens // 1000, replace=False))
    with SerialCacheWriter(str(path), CACHE_EXEMPLAR) as writer:
        writer.write_batch([{"input_ids": doc} for doc in np.split(stream, cuts)])


def train_argv(tmp_path, cache_tokens: int, *extra: str) -> list[str]:
    """best7's command line as the training script takes it, on a fresh cache."""
    write_document_cache(tmp_path / "cache", cache_tokens)
    tokenizer = write_tokenizer(tmp_path / "tokenizer")
    return [*BEST7_ARGV, "--data-cache-dir", str(tmp_path / "cache"), "--tokenizer", tokenizer, *extra]


def test_training_runs_best7_with_the_benchmark_options(tmp_path):
    step_tokens = BATCH * SEQ_LEN
    argv = train_argv(tmp_path, step_tokens + 10_000, "--train-tokens", str(step_tokens), "--optimizer", "muonh")
    train = slurm_train.build_run_config(slurm_train.parse_args(argv))
    benchmark = synthetic_benchmark.build_run_config(
        synthetic_benchmark.parse_args([*BEST7_ARGV, "--steps", "1", "--run-id", train.trainer.trainer.id])
    )

    assert train.model == benchmark.model
    # Only the run's own settings differ: data, budget, optimizer, z-loss, checkpoints and their tracker.
    run_only = dict(save_checkpoints=False, data_seed=None, z_loss_weight=0.0, trainer=None)
    assert dataclasses.replace(train.trainer, **run_only) == dataclasses.replace(benchmark.trainer, **run_only)
    trainer_only = dict(num_train_steps=0, tracker=None, checkpointer=None, load_checkpoint=None, log_dir=None)
    assert dataclasses.replace(train.trainer.trainer, **trainer_only) == dataclasses.replace(
        benchmark.trainer.trainer, **trainer_only
    )
    assert train.trainer.trainer.num_train_steps == 1
    # The cache-backed config logs (log_configuration converts it to a dict first).
    hparams_to_dict(train)


@pytest.mark.parametrize(("optimizer", "extra"), [("muonh", []), ("adamw", ["--learning-rate", "3e-4"])])
def test_lr_schedule_ends_at_the_token_budget(tmp_path, optimizer, extra):
    budget = int(5.5 * 4096)  # not a whole number of steps: the last one is completed
    argv = train_argv(tmp_path, 8 * 4096, *SMALL_STEP, "--train-tokens", str(budget), "--optimizer", optimizer)
    run = slurm_train.build_run_config(slurm_train.parse_args([*argv, "--warmup", "2", *extra]))
    steps = run.trainer.trainer.num_train_steps
    assert steps == 6
    schedule = run.optimizer.lr_scheduler(steps)
    lrs = np.array([float(schedule(step)) for step in range(steps + 2)])
    peak, floor = run.optimizer.learning_rate, run.optimizer.learning_rate * run.optimizer.min_lr_ratio
    assert lrs[0] == 0 and lrs[2] == pytest.approx(peak)
    assert np.all(np.diff(lrs[2 : steps + 1]) < 0)
    assert lrs[steps] == pytest.approx(floor) and lrs[steps - 1] > floor
    assert lrs[steps + 1] == pytest.approx(floor)


def test_budget_beyond_the_cache_is_refused(tmp_path):
    argv = train_argv(tmp_path, 4 * 4096 + 100, *SMALL_STEP, "--train-tokens", str(5 * 4096), "--optimizer", "muonh")
    with pytest.raises(ValueError, match="allow-data-repeat"):
        slurm_train.build_run_config(slurm_train.parse_args(argv))
    slurm_train.build_run_config(slurm_train.parse_args([*argv, "--allow-data-repeat"]))


def test_eval_holds_out_whole_microbatches_for_the_step_eval(tmp_path):
    # best7's 4 microbatches of a 16-sequence batch: --eval-microbatches 2 holds out 8 windows, which June's
    # GrugEvalConfig evaluates as the step eval with bits per byte.
    windows = 6 * 16 + 8
    argv = train_argv(
        tmp_path, windows * 256 + 100, *SMALL_STEP, "--train-tokens", str(6 * 4096), "--optimizer", "muonh"
    )
    run = slurm_train.build_run_config(slurm_train.parse_args([*argv, "--eval-every", "4", "--eval-microbatches", "2"]))
    assert run.eval == GrugEvalConfig(
        eval_batch_size=8, steps_per_eval=4, max_eval_batches=None, eval_ema=False, compute_bpb=True
    )
    assert run.data.num_validation_sequences == {slurm_train.TRAIN_COMPONENT: 8}
    # Without --eval-every nothing is held out, whatever --eval-microbatches says.
    run = slurm_train.build_run_config(slurm_train.parse_args([*argv, "--eval-microbatches", "2"]))
    assert run.eval is None and run.data.num_validation_sequences is None


def test_checkpoint_staging_budget_reaches_the_writer(tmp_path):
    argv = train_argv(tmp_path, 8 * 4096, *SMALL_STEP, "--train-tokens", str(4 * 4096), "--optimizer", "muonh")
    argv += ["--checkpoint-dir", str(tmp_path / "checkpoints")]
    default = slurm_train.build_run_config(slurm_train.parse_args(argv)).trainer.trainer.checkpointer.write
    staged = slurm_train.build_run_config(slurm_train.parse_args([*argv, "--checkpoint-staged-host-gib", "1.5"]))
    write = staged.trainer.trainer.checkpointer.write
    assert write == dataclasses.replace(default, max_staged_host_bytes=3 * 2**29)


@pytest.mark.parametrize(
    ("seq_len", "batch", "steps", "spare", "held_out"),
    [
        # Exactly the budget's windows (plus the held-out ones), so a repeat or an overlap cannot hide.
        (SEQ_LEN, 8, 4, 0, 8),
        # One whole 2048-sequence mixture block plus a partial one: the run may read only the whole block.
        (256, 64, 32, 80, 0),
    ],
)
def test_loader_yields_sharded_whole_windows_without_repeats(tmp_path, seq_len, batch, steps, spare, held_out):
    cache_dir = str(tmp_path / "cache")
    write_stream_cache(tmp_path / "cache", (steps * batch + spare + held_out) * seq_len)
    tokenizer = write_tokenizer(tmp_path / "tokenizer")
    train_windows = slurm_train.training_windows(slurm_train.open_cache(cache_dir), seq_len, held_out)
    assert slurm_train.usable_windows(train_windows) == steps * batch
    data = slurm_train.real_data_config(
        cache_dir, tokenizer, 1 << 24, eval_sequences=held_out, train_windows=train_windows
    )
    schedule = BatchSchedule(batch)

    def windows_from(loader, step: int, count: int) -> list[int]:
        windows = []
        batches = loader.iter_from_step(step)
        for _, example in zip(range(count), batches, strict=False):
            tokens = example.tokens
            assert tokens.shape == (batch, seq_len) and tokens.dtype == np.int32
            assert tokens.sharding.spec == P(("replica_dcn", "data", "expert"), None)
            weights = np.asarray(example.loss_weight)
            assert np.all(weights[:, :-1] == 1) and np.all(weights[:, -1] == 0)
            # Each of the 4 devices holds its own batch / 4 rows, and every row is one whole window of the stream.
            rows = [np.asarray(shard.data) for shard in tokens.addressable_shards]
            assert all(r.shape == (batch // 4, seq_len) for r in rows)
            rows = np.concatenate(rows)
            assert np.array_equal(rows, rows[:, :1] + np.arange(seq_len, dtype=np.int32))
            windows.extend(int(s) // seq_len for s in rows[:, 0])
        # Stop the prefetch thread: one still in a TensorStore read aborts interpreter teardown.
        batches.close()
        return windows

    with use_test_devices(4):
        mesh = compact_grug_mesh(expert_axis_size=2, replica_axis_size=2, model_axis_size=1)
        with set_mesh(mesh):
            dataset = build_train_dataset(data, max_seq_len=seq_len, batch_schedule=schedule, key=jax.random.PRNGKey(0))
            loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh)
            windows = windows_from(loader, 0, steps)
            # A run resumed at step s reads the windows the uninterrupted run reads from step s.
            resumed = windows_from(loader, steps // 2, steps - steps // 2)
    assert len(windows) == steps * batch and len(set(windows)) == len(windows), "a window repeats"
    assert resumed == windows[steps // 2 * batch :]
    if held_out:
        held = data.validation_sets(hax.Axis("position", seq_len))[slurm_train.TRAIN_COMPONENT].as_sync_dataset()
        held = {int(held[i].tokens.array[0]) // seq_len for i in range(len(held))}
        assert len(held) == held_out and not held & set(windows), "eval windows overlap training windows"
