# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the June tests' JAX scripts with a fixed CPU device count.

A copy of the ``levanter.testing.cpu_devices`` helper that upstream replaced with in-process multi-device tests
(#10713); the June tests still run their sharded checks as scripts in a fresh interpreter.
"""

import os
import subprocess
import sys
import textwrap


def run_on_cpu_devices(script: str, *, device_count: int) -> None:
    """Run a script in a fresh interpreter so XLA can set the CPU device count."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
