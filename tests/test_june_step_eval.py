# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The held-out eval inside the June train step (``GrugStepEvalConfig``), on eight CPU devices.

The step's eval variant, ``train_step(state, batch, eval_batch)``, runs a forward-only pass over the held-out batch
after the microbatch loop, on the compute-dtype weights the step already holds. On the meshes and in the modes of the
ZeRO-1 tests (replicated, ZeRO-1, ZeRO-1 with the gather in the first microbatch; with and without the overlapped last
microbatch, after-scan and in-scan block gradient sums):

- it trains exactly like the plain step: a run with eval steps in it ends bit for bit where the plain run does, with
  the same shardings, and the two variants are the only two compiled programs;
- its eval loss is the loss of the weights the step started from, computed by a separate forward;
- it holds no extra weight memory: its temporaries exceed the plain step's by less than a small fraction of the
  gathered expert stack, and its arguments only by the eval batch.

The training loop dispatches the variant exactly on the eval steps, logs ``eval/loss`` there, and trains as without
the eval, pipelined or not. ``synthetic_benchmark.py`` runs it end to end on the tiny preset and reports the
train-only headline and the eval steps' line.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

from tests.test_june_pipelined_step import _PRELUDE as _PIPELINED_PRELUDE
from tests.test_june_zero1 import _PRELUDE as _ZERO1_PRELUDE

_STEP_EVAL = r"""
# The held-out rows: whole microbatches of BATCH's 32 rows, from another key than the training tokens.
EVAL_TOKENS = np.asarray(jax.random.randint(jax.random.key(7), (32, 8), 0, BASE.vocab_size))
EVAL_WEIGHT = np.ones((32, 8), np.float32)
EVAL_WEIGHT[:, -1] = 0
EVAL_WEIGHT[1::5, 2:5] = 0  # Unequal weight per chunk.


def eval_batch(rows):
    return Batch(EVAL_TOKENS[:rows], EVAL_WEIGHT[:rows], None)


# (zero1, gather_in_first_microbatch, carry_compute_params, microbatches, overlap_last_microbatch,
#  block_grad_accumulation, eval microbatches)
CASES = {
    "replicated_no_carry": (False, False, False, 4, True, "after_scan", 1),
    "replicated_carry": (False, False, True, 4, False, "in_scan", 2),
    "zero1": (True, False, True, 4, False, "after_scan", 1),
    "gather_overlap_in_scan": (True, True, True, 4, True, "in_scan", 1),
    "gather_overlap_after_scan_k2": (True, True, True, 2, True, "after_scan", 2),
    "gather_no_overlap": (True, True, True, 4, False, "after_scan", 1),
}


def case_step(optimizer, case):
    zero1, gather, _, k, overlap, block, _ = CASES[case]
    return june_train._make_train_step(
        optimizer, POLICY, z_loss_weight=0.0, ema_beta=EMA_BETA, microbatches=k, zero1=zero1,
        overlap_last_microbatch=overlap, gather_in_first_microbatch=gather, block_grad_accumulation=block,
    )


def case_state(mesh, optimizer, case, cfg=ZERO1_CFG):
    zero1, _, carry, *_ = CASES[case]
    with jax.set_mesh(mesh):
        return jax.jit(
            lambda key: june_train.initial_state(
                cfg, optimizer=optimizer, mp=POLICY, key=key, ema_beta=EMA_BETA, carry_compute_params=carry,
                zero1=zero1,
            )
        )(jax.random.key(0))


def case_eval_batch(case):
    *_, k, _, _, m = CASES[case]
    return eval_batch(m * 32 // k)


def batch_at(step):
    return Batch(np.roll(TOKENS, step, axis=0), LOSS_WEIGHT, None)


def started_from(state):
    # The compute-dtype model the step starts from: the parameters cast (with ZeRO-1 gathered to the compute copy's
    # sharding), with the pending QB thresholds applied to the router biases.
    if state.compute_params is None:
        compute = POLICY.cast_to_compute(state.params)
    else:
        compute = june_train._resharded_like(POLICY.cast_to_compute(state.params), state.compute_params)
    return june_train._apply_qb_betas(compute, state.pending_qb_betas)


@jax.jit
def reference_mean_loss(state, batch):
    model = started_from(state)
    return model.next_token_loss(
        batch.tokens, batch.loss_weight, mask=batch.attn_mask, reduction="mean", logsumexp_weight=None
    )


@functools.partial(jax.jit, static_argnums=2)
def reference_sums(state, batch, rows_per_chunk):
    return june_train._eval_sums(started_from(state), batch, rows_per_chunk)


def all_specs(state):
    return {
        name: specs(getattr(state, name))
        for name in ("params", "opt_state", "ema_params", "compute_params", "pending_qb_betas")
        if getattr(state, name) is not None
    }
"""

_PRELUDE = _ZERO1_PRELUDE + "\nimport functools\n" + _STEP_EVAL


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=8)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("optimizer", ["adamw", "muonh"])
def test_eval_step_trains_like_the_plain_step(optimizer):
    # [plain, eval, plain, eval, plain] against five plain steps: every step's loss and metrics other than the eval's,
    # and the final parameters, optimizer state, EMA, QB thresholds and compute copy are bitwise equal, with equal
    # shardings after every step. Plain and eval steps take each other's states as they are, and the step is compiled
    # exactly twice.
    _run(
        f"""
        for name, mesh in MESHES.items():
            for case in CASES:
                msg = f"{{name}} {{case}}"
                runs = {{}}
                for with_eval in (False, True):
                    optimizer = OPTIMIZERS[{optimizer!r}]()
                    state = case_state(mesh, optimizer, case)
                    train_step = case_step(optimizer, case)
                    metrics_per_step, specs_per_step = [], []
                    with jax.set_mesh(mesh):
                        for step in range(5):
                            args = (case_eval_batch(case),) if with_eval and step % 2 == 1 else ()
                            state, metrics, _ = train_step(state, batch_at(step), *args)
                            metrics = jax.device_get(metrics)
                            if args:
                                assert set(metrics) - set(metrics_per_step[0]) == {{
                                    june_train.EVAL_CROSS_ENTROPY_SUM_KEY, june_train.EVAL_LOSS_WEIGHT_SUM_KEY
                                }}, msg
                                metrics = {{k: v for k, v in metrics.items() if not k.startswith("eval/")}}
                            metrics_per_step.append(metrics)
                            specs_per_step.append(all_specs(state))
                    if with_eval:
                        assert train_step._cache_size() == 2, (msg, train_step._cache_size())
                    runs[with_eval] = (leaves(state), metrics_per_step, specs_per_step)
                (plain, plain_metrics, plain_specs), (evaled, eval_metrics, eval_specs) = runs[False], runs[True]
                for a, b in zip(evaled, plain, strict=True):
                    np.testing.assert_array_equal(a, b, err_msg=msg)
                for a, b in zip(eval_metrics, plain_metrics, strict=True):
                    for x, y in zip(leaves(a), leaves(b), strict=True):
                        np.testing.assert_array_equal(x, y, err_msg=msg)
                assert eval_specs == plain_specs, msg
        """
    )


@pytest.mark.timeout(1800)
def test_eval_loss_is_the_loss_of_the_weights_the_step_started_from():
    # The eval variant's cross-entropy sum over its loss weight equals a separate forward of the model the step
    # started from (the cast parameters, gathered, with the pending QB thresholds), on the same held-out batch, after
    # a few training steps so that the weights, the QB thresholds and (with the gather) the compute copy's stale
    # expert weights all differ from their initial values. The same chunked forward in its own program gives the
    # same sums to float32 rounding of the sum over chunks; Transformer.next_token_loss over the whole batch at once,
    # which runs the bfloat16 forward on other shapes (and fusions), to bfloat16 rounding. The loss weight is exact.
    _run(
        """
        optimizer = OPTIMIZERS["adamw"]()
        for name, mesh in MESHES.items():
            for case in CASES:
                msg = f"{name} {case}"
                *_, k, _, _, m = CASES[case]
                state = case_state(mesh, optimizer, case)
                train_step = case_step(optimizer, case)
                held_out = case_eval_batch(case)
                with jax.set_mesh(mesh):
                    for step in range(3):
                        state, _, _ = train_step(state, batch_at(step))
                    reference = jax.device_get(reference_sums(state, held_out, 32 // k))
                    mean = float(reference_mean_loss(state, held_out))
                    state, metrics, _ = train_step(state, batch_at(3), held_out)
                cross_entropy = float(metrics[june_train.EVAL_CROSS_ENTROPY_SUM_KEY])
                weight = float(metrics[june_train.EVAL_LOSS_WEIGHT_SUM_KEY])
                assert weight == float(held_out.loss_weight.sum()), (msg, weight)
                np.testing.assert_allclose(
                    cross_entropy, float(reference[june_train.EVAL_CROSS_ENTROPY_SUM_KEY]), rtol=1e-6, err_msg=msg
                )
                np.testing.assert_allclose(cross_entropy / weight, mean, rtol=2e-3, err_msg=msg)
                # Far from the initial loss: the step evaluated the trained weights, not fresh ones.
                assert abs(cross_entropy / weight - np.log(BASE.vocab_size)) > 1e-3, (msg, cross_entropy / weight)
        """
    )


@pytest.mark.timeout(1800)
def test_eval_step_holds_no_extra_weight_memory():
    # Wide experts make the expert weights most of the memory. The eval variant's temporaries exceed the plain step's
    # by less than 1/16 of the expert stack of the compute copy: one chunk's forward activations, no gathered or
    # float32 copy of any weight. Its arguments exceed the plain step's by exactly the eval batch's per-device rows:
    # it arrives sharded over the 8 batch shards, as the train batch does.
    _run(
        """
        cfg = dataclasses.replace(ZERO1_CFG, intermediate_dim=512)
        optimizer = optax.adam(1e-2)
        for name, mesh in MESHES.items():
            for case in ("zero1", "gather_overlap_in_scan", "gather_overlap_after_scan_k2", "replicated_carry"):
                msg = f"{name} {case}"
                state = case_state(mesh, optimizer, case, cfg)
                stack = sum(
                    int(np.prod(x.sharding.shard_shape(x.shape))) * x.dtype.itemsize
                    for path, x in jax.tree_util.tree_leaves_with_path(state.compute_params)
                    if is_expert(path)
                )
                train_step = case_step(optimizer, case)
                held_out = case_eval_batch(case)
                with jax.set_mesh(mesh):
                    plain = train_step.lower(state, BATCH).compile().memory_analysis()
                    evaled = train_step.lower(state, BATCH, held_out).compile().memory_analysis()
                added_arguments = evaled.argument_size_in_bytes - plain.argument_size_in_bytes
                eval_bytes = held_out.tokens.nbytes + held_out.loss_weight.nbytes
                assert added_arguments == eval_bytes // 8, (msg, added_arguments, eval_bytes)
                added = evaled.temp_size_in_bytes - plain.temp_size_in_bytes
                print(msg, "added temp bytes", added, "expert stack", stack, flush=True)
                assert added < stack // 16, (msg, added, stack)
        """
    )


_LOOP = r"""
from experiments.june_tpu_67b_a2b.moe.train import GrugStepEvalConfig

LOOP_STEPS = 7
EVAL_EVERY = 3
LOOP_MICROBATCHES = 2
LOOP_EVAL = Batch(TOKENS[::-1].copy(), LOSS_WEIGHT, None)  # two microbatches of 16 rows


def loop_batches():
    for step in range(LOOP_STEPS):
        yield Batch(np.roll(TOKENS, step, axis=0), LOSS_WEIGHT, None)


def loop_train(*, pipelined, with_eval):
    optimizer = AdamConfig(learning_rate=1e-2, warmup=0.0, weight_decay=0.1).build(LOOP_STEPS)
    train_step = june_train._make_train_step(
        optimizer, POLICY, z_loss_weight=0.0, ema_beta=None, microbatches=LOOP_MICROBATCHES,
        overlap_last_microbatch=True,
    )
    dispatched = []

    def counting_step(state, batch, *eval_args, compute_watch):
        dispatched.append(len(eval_args))
        return train_step(state, batch, *eval_args, compute_watch=compute_watch)

    config = GrugStepEvalConfig(dataset=None, every=EVAL_EVERY, microbatches_per_eval=2)
    step_eval = june_train._LoadedStepEval(
        LOOP_EVAL, functools.partial(config.runs_at, num_train_steps=LOOP_STEPS), byte_count=1000.0
    ) if with_eval else None
    hook_steps = []
    runner = StateCallbackRunner(
        step_getter=lambda s: s.step,
        model_getter=lambda s: s.params,
        eval_model_getter=lambda s: s.params,
        opt_state_getter=lambda s: s.opt_state,
    )
    runner.add_hook(log_step_info(LOOP_STEPS), every=1)
    runner.add_hook(lambda info, force=False: hook_steps.append((info.step, force)), every=1)
    tracker = RecordingTracker()
    with levanter.tracker.current_tracker(tracker), jax.set_mesh(MESH):
        state = jax.jit(
            lambda key: june_train.initial_state(BASE, optimizer=optimizer, mp=POLICY, key=key, ema_beta=None)
        )(jax.random.key(0))
        state = june_train._train_loop(
            counting_step,
            state,
            LoadingTimeTrackerIterator(loop_batches()),
            num_train_steps=LOOP_STEPS,
            state_callbacks=runner,
            checkpointer=None,
            watch_interval=0,
            step_timeout=None,
            diagnose_numerics=False,
            pipelined=pipelined,
            hooks_read_state=lambda step: False,
            step_eval=step_eval,
        )
    logged = {
        step: {key.strip("[]'"): value for key, value in metrics.items() if key.strip("[]'") not in TIMING_KEYS}
        for step, metrics in tracker.steps.items()
    }
    return dict(final=leaves(state), logged=logged, dispatched=dispatched, hooks=hook_steps)
"""


@pytest.mark.timeout(900)
def test_loop_dispatches_the_eval_variant_on_the_eval_steps():
    # Eval steps are the steps s > 0 with s % every == 0 and the last: 3 and 6 of 0..6. Only they pass the held-out
    # batch, and only they log eval/loss (with its weight and bits per byte), at their own step. The run trains
    # exactly as without the eval, and every other logged metric is the same, pipelined or not.
    run_on_cpu_devices(
        _PIPELINED_PRELUDE
        + "\nimport functools\n"
        + _LOOP
        + textwrap.dedent(
            """
            eval_steps = [3, 6]
            plain = loop_train(pipelined=False, with_eval=False)
            assert plain["dispatched"] == [0] * LOOP_STEPS
            for pipelined in (False, True):
                run = loop_train(pipelined=pipelined, with_eval=True)
                assert run["dispatched"] == [int(s in eval_steps) for s in range(LOOP_STEPS)], run["dispatched"]
                for a, b in zip(run["final"], plain["final"], strict=True):
                    np.testing.assert_array_equal(a, b)
                assert run["hooks"] == plain["hooks"]
                assert sorted(run["logged"]) == list(range(LOOP_STEPS))
                for step in range(LOOP_STEPS):
                    evals = {k: v for k, v in run["logged"][step].items() if k.startswith("eval/")}
                    others = {k: v for k, v in run["logged"][step].items() if not k.startswith("eval/")}
                    assert others == plain["logged"][step], step
                    if step not in eval_steps:
                        assert not evals, (step, evals)
                        continue
                    assert set(evals) == {"eval/loss", "eval/loss_weight", "eval/bpb"}, evals
                    assert evals["eval/loss_weight"] == float(LOSS_WEIGHT.sum())
                    assert 0 < evals["eval/loss"] < 2 * np.log(BASE.vocab_size), evals
                    cross_entropy = evals["eval/loss"] * evals["eval/loss_weight"]
                    np.testing.assert_allclose(evals["eval/bpb"], cross_entropy / 1000.0 / np.log(2), rtol=1e-6)
            """
        ),
        device_count=8,
    )


_BENCHMARK = r"""
import logging
import sys
import tempfile

from experiments.june_tpu_67b_a2b.moe import synthetic_benchmark


class Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


records = Records()
# On the harness's own loggers: the trainer's setup reconfigures the root logger's handlers.
for name in ("experiments.june_tpu_67b_a2b.moe.train", "june_slurm_run"):
    logging.getLogger(name).addHandler(records)
with tempfile.TemporaryDirectory() as log_dir:
    sys.argv = [
        "synthetic_benchmark.py", "--size", "tiny", "--steps", "8", "--batch-size", "32", "--microbatches", "2",
        "--expert-axis", "2", "--replica-axis", "2", "--moe-impl", "ring_dedup", "--pooled-transport-capacity-factor",
        "4.0", "--capacity-factor", "8.0", "--carry-compute-params", "--zero1",
        "--gather-in-first-microbatch", "--overlap-last-microbatch", "--block-grad-accumulation", "in_scan",
        "--replicated-grads", "after_scan", "--router-stats", "after_scan", "--pipelined-step-logging",
        "--watch-interval", "0", "--eval-every", "3", "--eval-microbatches", "2", "--log-dir", log_dir,
        "--run-id", "step-eval",
    ]
    synthetic_benchmark.main()
text = "\n".join(records.lines)
evals = [line for line in records.lines if line.startswith("eval at step")]
assert [line.split()[3] for line in evals] == ["3", "6", "7"], evals
for line in evals:
    loss = float(line.split("'eval/loss': ")[1].split(",")[0])
    # Uniform random tokens: about ln(128), the tiny preset's vocabulary.
    assert abs(loss - np.log(128)) < 0.5, line
headline = [line for line in records.lines if line.startswith("median step over steps")]
# Steps 2..7 are timed; 3, 6 and 7 are eval steps: the headline covers the train-only steps 2, 4, 5.
assert len(headline) == 1 and headline[0].startswith("median step over steps 2-5:"), headline
summary = [line for line in records.lines if line.startswith("eval steps:")]
assert len(summary) == 1 and "1 measured (steps 6)" in summary[0] and "first (compiling) step 3" in summary[0], summary
assert "an eval every 3 steps" in summary[0], summary
print(text[-3000:])
"""


@pytest.mark.timeout(900)
def test_synthetic_benchmark_runs_the_eval_end_to_end():
    run_on_cpu_devices("import numpy as np\n" + _BENCHMARK, device_count=8)
