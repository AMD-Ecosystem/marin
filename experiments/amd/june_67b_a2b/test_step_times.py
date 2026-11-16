# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``StepTimes`` keeps the eval steps of the in-step eval out of the train-only headline and reports their cost."""

import logging
from types import SimpleNamespace

import pytest
from levanter.callbacks import StepInfo

from experiments.amd.june_67b_a2b import run_config
from experiments.amd.june_67b_a2b.run_config import StepTimes
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig

STEPS = 12
TRAIN_WALL = 3.0
EVAL_WALL = 3.5
COMPILE_WALL = 60.0


def run_hooks(monkeypatch, step_times: StepTimes, walls: dict[int, float]) -> None:
    clock = [0.0]
    monkeypatch.setattr(run_config.time, "perf_counter", lambda: clock[0])
    for step in range(STEPS):
        clock[0] += walls[step]
        step_times(StepInfo(SimpleNamespace(step=step + 1), loss=1.0, step_duration=walls[step] - 0.1))
    step_times(StepInfo(SimpleNamespace(step=STEPS), loss=1.0, step_duration=0.0), force=True)


def test_eval_steps_are_reported_apart_from_the_train_steps(monkeypatch, caplog):
    # Eval every 4 steps: 0 compiles the train step and the eval (it and 1 are the timing warmup), 4 and 8 are
    # measured, 11 (the last) is left out. The headline covers train steps 2-9 without 3 and 7, the steps before an
    # eval, whose wall periods are timed with it, and 10, the step before the last.
    walls = {step: TRAIN_WALL for step in range(STEPS)}
    walls.update({0: COMPILE_WALL, 1: COMPILE_WALL, 4: EVAL_WALL, 8: EVAL_WALL, 11: 0.01})
    step_times = StepTimes(GrugEvalConfig(steps_per_eval=4), STEPS)
    run_hooks(monkeypatch, step_times, walls)

    assert step_times.timed_steps == [2, 5, 6, 9]
    assert step_times.wall_steps == [TRAIN_WALL] * 4
    assert step_times.eval_wall_steps == {
        4: pytest.approx(EVAL_WALL + TRAIN_WALL),
        8: pytest.approx(EVAL_WALL + TRAIN_WALL),
    }
    # The first step has no wall period; its own duration stands in.
    assert step_times.first_eval == (0, pytest.approx(COMPILE_WALL - 0.1))
    with caplog.at_level(logging.INFO, logger=run_config.logger.name):
        step_times.log_summary(batch_size=8, seq_len=16, flops_per_example=1e6)
    headlines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("median step over steps 2-9:")]
    assert len(headlines) == 1
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("eval steps:")]
    # The text spurix's EVAL_RE parses.
    assert line == (
        "eval steps: 2 measured (steps 4,8), mean wall step 3.500 s; first (compiling) step 0: 56.900 s; "
        "+0.500 s over the train-only mean wall step 3.000 s; mean wall step with an eval every 4 steps: 3.125 s"
    )


@pytest.mark.parametrize("eval_cfg", [None, GrugEvalConfig(steps_per_eval=100)])
def test_without_eval_steps_every_timed_step_is_a_train_step(monkeypatch, eval_cfg):
    # No eval, or none in the window but step 0's and the last step's: the timed steps are those the summary always
    # timed, less the last step and the one before it (timed with it).
    walls = {step: TRAIN_WALL for step in range(STEPS)}
    step_times = StepTimes(eval_cfg, STEPS)
    run_hooks(monkeypatch, step_times, walls)
    expected = list(range(2, STEPS)) if eval_cfg is None else list(range(2, STEPS - 2))
    assert step_times.timed_steps == expected
    assert step_times.eval_wall_steps == {}
