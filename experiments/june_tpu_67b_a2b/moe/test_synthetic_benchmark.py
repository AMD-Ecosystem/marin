# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The synthetic benchmark builds the run that best7 timed.

``BEST7_ARGV`` is the command line spurix ran for best7 on two nodes (radvultr-june-best7-scale, minus the per-job
paths). The expected config is written out from the model heuristic and Levanter's defaults, independently of the
benchmark's code, so a refactor of how the benchmark assembles its run cannot change what every study measures.
"""

import dataclasses
import sys
from pathlib import Path

import jmp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, XprofUploadConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.june_tpu_67b_a2b.moe import synthetic_benchmark
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.train import GrugRunConfig, GrugTrainerConfig

STEPS = 3
BATCH = 512
BEST7_ARGV = (
    "--run-id best7 --log-dir /tmp/best7-logs --compilation-cache-dir /tmp/best7-jaxcache "
    f"--size full --steps {STEPS} --batch-size {BATCH} --microbatches 4 --expert-axis 8 --replica-axis 2 "
    "--moe-impl ring_dedup --pooled-transport-capacity-factor 1.1 --remat-mode save_moe_attn "
    "--rms-norm-impl custom_vjp --replicated-grads after_scan --router-stats after_scan --carry-compute-params "
    "--overlap-last-microbatch --zero1 --gather-in-first-microbatch --pipelined-step-logging "
    "--capacity-drops after_scan --qb-threshold radix_select --no-router-diagnostics --skip-zero-router-z-loss "
    "--moe-gather-sum pallas --qk-norm-rope pallas --attention-schedule cond --fused-norm-impl pallas "
    "--moe-routing-metadata pallas --moe-flydsl-gather-a --moe-scatter-epilogue flydsl "
    "--expert-grad-accumulation flydsl --block-grad-accumulation in_scan --ce-weight-grad in_place "
    "--ce-bwd-weight-blocks weight_dtype --gated-norm-impl pallas --gated-norm-remat save_hidden "
    "--head-gate-remat save --moe-send-gather pallas --moe-token-sum clamped --attention gpu_aiter_flash "
    "--watch-interval 0"
).split()


def captured_benchmark_run(monkeypatch) -> GrugRunConfig:
    """The GrugRunConfig the benchmark's main() hands to the training loop for BEST7_ARGV."""
    runs: list[GrugRunConfig] = []
    monkeypatch.setattr(sys, "argv", ["synthetic_benchmark.py", *BEST7_ARGV])
    monkeypatch.setattr(synthetic_benchmark, "_run_grug_local", lambda config, **kwargs: runs.append(config))
    synthetic_benchmark.main()
    (run,) = runs
    return run


def best7_model():
    base = MoeMuonHHeuristic().build_model_config(2560, seq_len=4096)
    return dataclasses.replace(
        base,
        sliding_window=2048,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
        attention_implementation="gpu_aiter_flash",
        moe_implementation="ring_dedup",
        remat_mode="save_moe_attn",
        remat_prevent_cse=None,
        rms_norm_impl="custom_vjp",
        fused_norm_impl="pallas",
        gated_norm_impl="pallas",
        gated_norm_remat="save_hidden",
        head_gate_remat="save",
        qk_norm_rope="pallas",
        attention_schedule="cond",
        pooled_transport_capacity_factor=1.1,
        num_expert_waves=1,
        moe_token_blocks=1,
        moe_dispatch_barrier=False,
        moe_gather_sum="pallas",
        moe_routing_metadata="pallas",
        moe_flydsl_gather_a=True,
        moe_scatter_epilogue="flydsl",
        moe_send_gather="pallas",
        moe_token_sum="clamped",
        ce_v_block_size=None,
        ce_weight_grad="in_place",
        scan_unroll=1,
        ce_bwd_grad_x="add",
        ce_bwd_weight_blocks="weight_dtype",
        replicated_grad_reduction="after_scan",
        replicated_grad_max_bytes=4 << 20,
        router_stats_reduction="after_scan",
        capacity_drop_reduction="after_scan",
        qb_threshold_impl="radix_select",
        router_diagnostics=False,
        skip_zero_router_z_loss=True,
        qb_balance="global",
    )


def best7_trainer() -> GrugTrainerConfig:
    trainer = TrainerConfig(
        id="best7",
        log_dir=Path("/tmp/best7-logs"),
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=BATCH,
        num_train_steps=STEPS,
        tracker=JsonLoggerConfig(),
        checkpointer=CheckpointerConfig(base_path="/tmp/best7-logs/best7/checkpoints"),
        load_checkpoint=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        profiler=ProfilerConfig(
            enabled=False,
            start_step=10,
            num_steps=0,
            upload=XprofUploadConfig(enabled=False),
            profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
        ),
        jax_compilation_cache_dir="/tmp/best7-jaxcache",
        log_jaxprs=False,
        log_xla_hlo=False,
        watch=WatchConfig(interval=0),
    )
    return GrugTrainerConfig(
        trainer=trainer,
        expert_axis_size=8,
        replica_axis_size=2,
        save_checkpoints=False,
        step_timeout=300.0,
        qb_sync_interval=0,
        carry_compute_params=True,
        microbatches=4,
        zero1=True,
        overlap_last_microbatch=True,
        gather_in_first_microbatch=True,
        expert_grad_accumulation="flydsl",
        block_grad_accumulation="in_scan",
        pipelined_step_logging=True,
    )


def test_best7_command_line_builds_the_timed_run(monkeypatch):
    run = captured_benchmark_run(monkeypatch)

    assert run.model == best7_model()
    assert run.trainer == best7_trainer()
    assert run.optimizer == AdamConfig(learning_rate=3e-4, warmup=0.0)
    assert run.resources == ResourceConfig()
    assert run.eval is None

    # Fresh uniform-random tokens for every sequence of every step, with a mixture block as long as the data.
    (name,) = run.data.components
    examples = run.data.components[name].datasets["train"].data
    assert len(examples) == STEPS * BATCH
    assert run.data.mixture_block_size == STEPS * BATCH
    assert (run.data.tokenizer, run.data.vocab_size) == ("passthrough", 128_256)
    expected = np.random.default_rng(0).integers(0, 128_256, (2, 4096), dtype=np.int32)
    np.testing.assert_array_equal(np.stack([examples[0].tokens, examples[1].tokens]), expected)
    loss_weight = np.ones(4096, dtype=np.float32)
    loss_weight[-1] = 0
    np.testing.assert_array_equal(examples[-1].loss_weight, loss_weight)
