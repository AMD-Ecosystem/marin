# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The held-out eval inside the AMD June train step (fork overlay): the eval variant of the step trains exactly like
the plain step and evaluates the weights the step started from, and the loop dispatches it on the eval steps."""

import concurrent.futures
import dataclasses
import logging
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.testing.devices import use_test_devices
from levanter.trainer import TrainerConfig
from levanter.utils.logging import LoadingTimeTrackerIterator

from experiments.amd.june_67b_a2b import train
from experiments.amd.june_67b_a2b.test_train import (
    ACCUMULATION_CONFIG,
    MIXED_POLICY,
    OPTIMIZER,
    PADDED_BATCH,
    Batch,
    _accumulation_mesh,
)
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig

# Two train microbatches of 4 rows, from a seed the training batch does not use.
_EVAL_TOKENS = np.asarray(jax.random.randint(jax.random.key(7), (8, ACCUMULATION_CONFIG.max_seq_len), 0, 32))
_EVAL_WEIGHT = np.ones(_EVAL_TOKENS.shape, np.float32)
_EVAL_WEIGHT[:, -1] = 0
EVAL_BATCH = Batch(_EVAL_TOKENS, _EVAL_WEIGHT, None)


def test_eval_step_trains_like_the_plain_step_and_evaluates_its_starting_weights():
    # K=2 with ZeRO-1 and a bfloat16 compute copy, as best7 trains: the eval runs on the gathered compute weights.
    with use_test_devices(4), jax.set_mesh(_accumulation_mesh()):
        initial = jax.jit(
            lambda k: train.initial_state(
                ACCUMULATION_CONFIG, optimizer=OPTIMIZER, mp=MIXED_POLICY, key=k, ema_beta=None, zero1=True
            )
        )
        step = train._make_train_step(
            OPTIMIZER, MIXED_POLICY, z_loss_weight=1e-3, ema_beta=None, microbatches=2, zero1=True
        )
        start = jax.tree.map(np.asarray, train._checkpointed(initial(jax.random.key(0))))
        # Both variants compile concurrently (XLA releases the GIL).
        state = initial(jax.random.key(0))
        lowered = (step.lower(state, PADDED_BATCH), step.lower(state, PADDED_BATCH, EVAL_BATCH))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            plain_step, eval_step = pool.map(lambda low: low.compile(), lowered)
        plain_state, plain_metrics, _ = plain_step(initial(jax.random.key(0)), PADDED_BATCH)
        eval_state, eval_metrics, _ = eval_step(initial(jax.random.key(0)), PADDED_BATCH, EVAL_BATCH)

        # The update is the same, bit for bit; only the eval sums are added.
        for name, plain, evaluated in (("state", plain_state, eval_state), ("metrics", plain_metrics, eval_metrics)):
            evaluated = jax.tree.map(np.asarray, evaluated)
            plain = jax.tree.map(np.asarray, plain)
            if name == "metrics":
                evaluated = {k: v for k, v in evaluated.items() if k not in _eval_keys()}
            assert jax.tree.structure(evaluated) == jax.tree.structure(plain), name
            for a, b in zip(jax.tree.leaves(evaluated), jax.tree.leaves(plain), strict=True):
                np.testing.assert_array_equal(a, b, err_msg=name)
        assert not set(_eval_keys()) & set(plain_metrics)

        # The eval loss is the plain forward cross-entropy of the starting weights in the compute dtype.
        @jax.jit
        def forward(params, tokens, weight):
            model = MIXED_POLICY.cast_to_compute(params)
            terms = model.next_token_loss_terms(tokens, weight, reduction="sum", logsumexp_weight=None)
            return terms.cross_entropy_loss

        expected = float(forward(start.params, EVAL_BATCH.tokens, EVAL_BATCH.loss_weight))
    cross_entropy = float(eval_metrics[train.EVAL_CROSS_ENTROPY_SUM_KEY])
    weight = float(eval_metrics[train.EVAL_LOSS_WEIGHT_SUM_KEY])
    assert weight == float(_EVAL_WEIGHT.sum())
    # Chunked into the two microbatches in bfloat16: the sum is reordered, not changed.
    assert cross_entropy == pytest.approx(expected, rel=2e-3)


def _eval_keys() -> tuple[str, str]:
    return (train.EVAL_CROSS_ENTROPY_SUM_KEY, train.EVAL_LOSS_WEIGHT_SUM_KEY)


def test_loop_dispatches_the_eval_variant_on_the_eval_steps(monkeypatch, caplog):
    # Eval every 2 of 5 steps: steps 0, 2 and 4 (the last); the plain and eval variants each compile on first use, so
    # their first steps (1 and 0) have no hang timeout.
    num_steps = 5
    dispatched, timeouts = [], []

    def fake_step(state, batch, *eval_args, compute_watch):
        dispatched.append((int(state.step), bool(eval_args)))
        metrics = {
            "train/loss": jnp.float32(1.0),
            train._LOSS_FINITE_KEY: jnp.bool_(True),
            train._OPTIMIZER_HYPERPARAMS_KEY: {},
            "train/cross_entropy_loss": jnp.float32(1.0),
        }
        if eval_args:
            metrics[train.EVAL_CROSS_ENTROPY_SUM_KEY] = jnp.float32(6.0 * int(state.step))
            metrics[train.EVAL_LOSS_WEIGHT_SUM_KEY] = jnp.float32(3.0)
        return SimpleNamespace(step=state.step + 1), metrics, None

    real_exit_if_hung = train.exit_if_hung

    def recording_exit_if_hung(timeout):
        timeouts.append(timeout)
        return real_exit_if_hung(None)

    logged: dict[int, dict] = {}
    monkeypatch.setattr(train, "exit_if_hung", recording_exit_if_hung)
    monkeypatch.setattr(train.levanter.tracker, "log", lambda metrics, step: logged.setdefault(step, {}).update(metrics))
    runner = StateCallbackRunner(
        step_getter=lambda s: s.step,
        model_getter=lambda s: None,
        eval_model_getter=lambda s: None,
        opt_state_getter=lambda s: None,
    )
    step_eval = train._LoadedStepEval(
        batch="held-out",
        runs_at=lambda step: train.eval_runs_at(step, 2, num_steps),
        prefix="eval",
        byte_count=None,
    )
    with caplog.at_level(logging.INFO, logger=train.logger.name):
        train._train_loop(
            fake_step,
            SimpleNamespace(step=jnp.int32(0)),
            LoadingTimeTrackerIterator(iter(range(num_steps))),
            num_train_steps=num_steps,
            state_callbacks=runner,
            checkpointer=None,
            watch_interval=0,
            step_timeout=60.0,
            diagnose_numerics=False,
            hooks_read_state=lambda step: False,
            step_eval=step_eval,
        )
    assert dispatched == [(0, True), (1, False), (2, True), (3, False), (4, True)]
    evals = {step: metrics["eval/loss"] for step, metrics in logged.items() if "eval/loss" in metrics}
    assert evals == {0: 0.0, 2: 4.0, 4: 8.0}
    assert all("eval/bpb" not in m for m in logged.values())
    assert sum("eval at step 2 (the weights after 2 updates)" in r.getMessage() for r in caplog.records) == 1
    # Steps 0 (eval) and 1 (plain) compile: they and the waits for them have no timeout; later steps have one.
    assert None in timeouts and timeouts[-1] == 60.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (dict(microbatches=1), "microbatches >= 2"),
        (dict(eval_batch_size=6), "whole train microbatches"),
        (dict(max_eval_batches=2), "one batch"),
        (dict(eval_ema=True, ema_beta=0.99), "EMA"),
    ],
)
def test_eval_settings_the_step_cannot_run_are_refused(change, match):
    eval_fields = {k: v for k, v in change.items() if k in ("eval_batch_size", "max_eval_batches", "eval_ema")}
    trainer_fields = {k: v for k, v in change.items() if k in ("microbatches", "ema_beta")}
    config = SimpleNamespace(
        eval=dataclasses.replace(GrugEvalConfig(eval_batch_size=8, steps_per_eval=2, eval_ema=False), **eval_fields),
        trainer=dataclasses.replace(
            train.GrugTrainerConfig(trainer=TrainerConfig(train_batch_size=16), microbatches=4), **trainer_fields
        ),
    )
    with pytest.raises(ValueError, match=match):
        train._check_step_eval(config)
