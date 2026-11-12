# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import sys

import numpy as np
import pytest
from levanter.store.cache import SerialCacheWriter
from levanter.testing.cpu_devices import run_on_cpu_devices
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from experiments.june_tpu_67b_a2b.moe import slurm_train
from experiments.june_tpu_67b_a2b.moe.test_synthetic_benchmark import BATCH, BEST7_ARGV, captured_benchmark_run

SEQ_LEN = 4096
BOS, EOS = 1, 2
CACHE_EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}


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
    """best7's command line as the training script takes it, on a fresh cache: the token budget replaces --steps."""
    write_document_cache(tmp_path / "cache", cache_tokens)
    argv = list(BEST7_ARGV)
    del argv[argv.index("--steps") : argv.index("--steps") + 2]
    tokenizer = write_tokenizer(tmp_path / "tokenizer")
    return [*argv, "--data-cache-dir", str(tmp_path / "cache"), "--tokenizer", tokenizer, *extra]


def test_training_runs_best7_with_the_benchmark_options(monkeypatch, tmp_path):
    step_tokens = BATCH * SEQ_LEN
    argv = train_argv(tmp_path, step_tokens + 10_000, "--train-tokens", str(step_tokens), "--optimizer", "muonh")
    benchmark = captured_benchmark_run(monkeypatch)
    train = slurm_train.build_run_config(slurm_train.parse_args(argv))

    assert train.model == benchmark.model
    # Only the run's own settings differ: data, budget, optimizer, z-loss, checkpoints and their tracker.
    run_only = dict(save_checkpoints=False, data_seed=None, z_loss_weight=0.0, trainer=None)
    assert dataclasses.replace(train.trainer, **run_only) == dataclasses.replace(benchmark.trainer, **run_only)
    trainer_only = dict(num_train_steps=0, tracker=None, checkpointer=None, load_checkpoint=None, log_dir=None)
    assert dataclasses.replace(train.trainer.trainer, **trainer_only) == dataclasses.replace(
        benchmark.trainer.trainer, **trainer_only
    )
    assert train.trainer.trainer.num_train_steps == 1


# A small batch and window keep these caches to tens of thousands of tokens: 4096 tokens a step.
SMALL_STEP = ("--batch-size", "16", "--seq-len", "256")


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
    # best7's 4 microbatches of a 16-sequence batch: --eval-microbatches 2 holds out 8 windows, which the eval inside
    # the train step evaluates with bits per byte; Levanter's standalone eval is off.
    windows = 6 * 16 + 8
    argv = train_argv(
        tmp_path, windows * 256 + 100, *SMALL_STEP, "--train-tokens", str(6 * 4096), "--optimizer", "muonh"
    )
    run = slurm_train.build_run_config(slurm_train.parse_args([*argv, "--eval-every", "4", "--eval-microbatches", "2"]))
    assert run.eval is None
    step_eval = run.step_eval
    assert step_eval is not None
    assert (step_eval.every, step_eval.microbatches_per_eval, step_eval.compute_bpb) == (4, 2, True)
    assert len(step_eval.dataset.as_sync_dataset()) == 8
    assert run.data.num_validation_sequences == {slurm_train.TRAIN_COMPONENT: 8}
    # Without --eval-every nothing is held out, whatever --eval-microbatches says.
    run = slurm_train.build_run_config(slurm_train.parse_args([*argv, "--eval-microbatches", "2"]))
    assert run.step_eval is None and run.data.num_validation_sequences is None


DATA_PATH_SCRIPT = """
import os

import haliax as hax
import jax
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh
from levanter.schedule import BatchSchedule

from experiments.june_tpu_67b_a2b.moe.slurm_train import open_cache, real_data_config, training_windows, usable_windows
from experiments.june_tpu_67b_a2b.moe.train import build_train_dataset, build_train_loader

cache_dir, tokenizer = {cache_dir!r}, {tokenizer!r}
seq_len, batch, steps, held_out = {seq_len}, {batch}, {steps}, {held_out}
train_windows = training_windows(open_cache(cache_dir), seq_len, held_out)
assert usable_windows(train_windows) == steps * batch, (train_windows, usable_windows(train_windows))
data = real_data_config(cache_dir, tokenizer, 1 << 24, eval_sequences=held_out, train_windows=train_windows)
mesh = compact_grug_mesh(expert_axis_size=4, replica_axis_size=2, model_axis_size=1)
schedule = BatchSchedule(batch)
with set_mesh(mesh):
    dataset = build_train_dataset(data, max_seq_len=seq_len, batch_schedule=schedule, key=jax.random.PRNGKey(0))
    loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh)
    windows = []
    batches = loader.iter_from_step(0)
    for _, example in zip(range(steps), batches):
        tokens = example.tokens
        assert tokens.shape == (batch, seq_len) and tokens.dtype == np.int32, (tokens.shape, tokens.dtype)
        assert tokens.sharding.spec == P(("replica_dcn", "data", "expert"), None), tokens.sharding.spec
        assert example.attn_mask.is_causal and example.attn_mask.segment_ids is None
        weights = np.asarray(example.loss_weight)
        assert np.all(weights[:, :-1] == 1) and np.all(weights[:, -1] == 0)
        # Each of the 8 devices holds its own batch / 8 rows, and every row is one whole window of the stream.
        rows = [np.asarray(shard.data) for shard in tokens.addressable_shards]
        assert all(r.shape == (batch // 8, seq_len) for r in rows), [r.shape for r in rows]
        rows = np.concatenate(rows)
        assert np.all(rows[:, 0] % seq_len == 0)
        assert np.array_equal(rows, rows[:, :1] + np.arange(seq_len, dtype=np.int32))
        windows.extend(int(s) // seq_len for s in rows[:, 0])
assert len(windows) == steps * batch and len(set(windows)) == len(windows), "a window repeats"
if held_out:
    held = data.validation_sets(hax.Axis("position", seq_len))["train"].as_sync_dataset()
    held = {{int(held[i].tokens.array[0]) // seq_len for i in range(len(held))}}
    assert len(held) == held_out and not held & set(windows), "eval windows overlap training windows"
print("ok", len(windows), flush=True)
# The loader's prefetch thread can still be in a tensorstore read, which aborts interpreter teardown.
os._exit(0)
"""


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    ("seq_len", "batch", "steps", "spare", "held_out"),
    [
        # Exactly the budget's windows (plus the held-out ones), so a repeat or an overlap cannot hide.
        (SEQ_LEN, 16, 4, 0, 16),
        (SEQ_LEN, 16, 4, 0, 0),
        # One whole 2048-sequence mixture block plus a partial one: the run may read only the whole block.
        (256, 64, 32, 80, 0),
    ],
)
def test_loader_yields_sharded_whole_windows_without_repeats(tmp_path, seq_len, batch, steps, spare, held_out):
    write_stream_cache(tmp_path / "cache", (steps * batch + spare + held_out) * seq_len)
    script = DATA_PATH_SCRIPT.format(
        cache_dir=str(tmp_path / "cache"),
        tokenizer=write_tokenizer(tmp_path / "tokenizer"),
        seq_len=seq_len,
        batch=batch,
        steps=steps,
        held_out=held_out,
    )
    run_on_cpu_devices(script, device_count=8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
