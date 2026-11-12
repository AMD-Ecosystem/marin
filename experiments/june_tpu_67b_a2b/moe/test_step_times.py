# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``StepTimes`` keeps the eval steps of ``GrugStepEvalConfig`` out of the train-only headline and reports their cost."""

import logging
from types import SimpleNamespace

import pytest
from levanter.callbacks import StepInfo
from levanter.data.dataset import ListAsyncDataset

from experiments.june_tpu_67b_a2b.moe import slurm_run_config
from experiments.june_tpu_67b_a2b.moe.slurm_run_config import StepTimes
from experiments.june_tpu_67b_a2b.moe.train import GrugStepEvalConfig

STEPS = 12
TRAIN_WALL = 3.0
EVAL_WALL = 3.5
COMPILE_WALL = 90.0


def run_hooks(monkeypatch, step_times: StepTimes, walls: dict[int, float]) -> None:
    clock = [0.0]
    monkeypatch.setattr(slurm_run_config.time, "perf_counter", lambda: clock[0])
    for step in range(STEPS):
        clock[0] += walls[step]
        step_times(StepInfo(SimpleNamespace(step=step + 1), loss=1.0, step_duration=walls[step] - 0.1))
    step_times(StepInfo(SimpleNamespace(step=STEPS), loss=1.0, step_duration=0.0), force=True)


@pytest.mark.parametrize("pipelined", [False, True])
def test_eval_steps_are_reported_apart_from_the_train_steps(monkeypatch, caplog, pipelined):
    # Eval every 4 steps: 4 compiles the eval variant, 8 is measured, 11 (the last) is left out. Steps 0 and 1 are
    # the timing warmup. The headline covers train steps 2-9 without 3 and 7, the steps before an eval, whose wall
    # periods are timed with it, and 10, the step before the last. Pipelined, each period holds the next step's time,
    # and the first eval step's flush of the pipeline leaves a near-empty period before it.
    step_eval = GrugStepEvalConfig(ListAsyncDataset([]), every=4)
    walls = {step: TRAIN_WALL for step in range(STEPS)}
    if pipelined:
        walls.update({0: 60.0, 3: 0.0, 4: COMPILE_WALL + TRAIN_WALL, 7: EVAL_WALL, 10: EVAL_WALL, 11: 0.01})
    else:
        walls.update({0: 60.0, 4: COMPILE_WALL, 8: EVAL_WALL, 11: 0.01})
    step_times = StepTimes(step_eval, STEPS)
    run_hooks(monkeypatch, step_times, walls)

    assert step_times.timed_steps == [2, 5, 6, 9]
    assert step_times.wall_steps == [TRAIN_WALL] * 4
    assert step_times.eval_wall_steps == {8: pytest.approx(EVAL_WALL + TRAIN_WALL)}
    assert step_times.first_eval == (4, pytest.approx(COMPILE_WALL + TRAIN_WALL))
    with caplog.at_level(logging.INFO, logger=slurm_run_config.logger.name):
        step_times.log_summary(batch_size=8, seq_len=16, flops_per_example=1e6)
    headlines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("median step over steps 2-9:")]
    assert len(headlines) == 1
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("eval steps:")]
    assert line == (
        "eval steps: 1 measured (steps 8), mean wall step 3.500 s; first (compiling) step 4: 90.000 s; "
        "+0.500 s over the train-only mean wall step 3.000 s; mean wall step with an eval every 4 steps: 3.125 s"
    )


@pytest.mark.parametrize("step_eval", [None, GrugStepEvalConfig(ListAsyncDataset([]), every=100)])
def test_without_eval_steps_every_timed_step_is_a_train_step(monkeypatch, caplog, step_eval):
    # No eval, or none in the window but the last step's: the timed steps are those the summary always timed.
    walls = {step: TRAIN_WALL for step in range(STEPS)}
    step_times = StepTimes(step_eval, STEPS)
    run_hooks(monkeypatch, step_times, walls)
    # With the eval, the last step and the one before it (timed with it) are left out.
    expected = list(range(2, STEPS)) if step_eval is None else list(range(2, STEPS - 2))
    assert step_times.timed_steps == expected
    assert step_times.eval_wall_steps == {}
