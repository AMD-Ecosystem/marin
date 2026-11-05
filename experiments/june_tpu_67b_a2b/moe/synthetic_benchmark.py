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

Example, 8 GPUs, full 67B shape, expert parallelism over all 8, profile of steps 10-12::

    RAGGED_DOT_IMPL=xla python experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py \\
        --size full --steps 20 --expert-axis 8 --profile-steps 3

Across nodes, launch one process per node with ``srun`` and pass ``--gpus-per-node``. Devices are ordered by
process, so the mesh's inner axes stay within a node: on two 8-GPU nodes, ``--expert-axis 8 --replica-axis 2``
replicates the model per node and all-reduces gradients across nodes, ``--expert-axis 8`` alone shards the
parameters over a cross-node ``data`` axis (FSDP), and ``--expert-axis 16`` spreads the experts over both nodes.
"""

import argparse
import dataclasses
import logging
import statistics
import time
from pathlib import Path
from typing import get_args

import jax
import jmp
import levanter.tracker
import numpy as np
from fray.cluster import ResourceConfig
from levanter.callbacks import StepInfo
from levanter.callbacks._metrics import aggregate_device_flops, compute_instant_throughput
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, XprofUploadConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug._moe.pallas_gather_sum import MoeGatherSum
from levanter.grug.attention import GrugAttentionImplementation
from levanter.grug.grug_moe import QbThresholdImpl
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import multihost_broadcast_sync
from rigging.log_setup import configure_logging

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, QbBalance, RematMode, RmsNormImpl, ScanReduction
from experiments.june_tpu_67b_a2b.moe.train import (
    ATTENTION_FLOPS_PER_EXAMPLE_KEY,
    CAUSAL_ATTENTION_FLOPS_PER_EXAMPLE_KEY,
    GrugRunConfig,
    GrugTrainerConfig,
    _compute_flops,
    _run_grug_local,
)

logger = logging.getLogger("june_synthetic_benchmark")

SNOWBALL_HIDDEN_DIM = 2560
# The production window (half of the 4096-token training context), which Levanter's SnowballConfig pins as well.
# The width heuristic would otherwise derive seq_len // 2 and give the medium preset a shorter window than Snowball's.
SNOWBALL_SLIDING_WINDOW = 2048
POOLED_WAVE_MOE_IMPL = "fixed_pooled_wave_all_to_all"
TRANSPORT_CAPACITY_MOE_IMPLS = ("ring_dedup", POOLED_WAVE_MOE_IMPL)
# Steps 0 and 1 are excluded from the summary, as in the Snowball harness: step 0 compiles.
FIRST_TIMED_STEP = 2


def snowball_model(seq_len: int) -> GrugModelConfig:
    """The June 67B-A2B production model at ``seq_len``, as the cooldown launch builds it, minus its
    long-context YaRN temperature scaling and with the production sliding window at every ``seq_len``."""
    base = MoeMuonHHeuristic().build_model_config(SNOWBALL_HIDDEN_DIM, seq_len=seq_len)
    return dataclasses.replace(
        base,
        sliding_window=SNOWBALL_SLIDING_WINDOW,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
    )


def preset(size: str, seq_len: int | None) -> tuple[GrugModelConfig, str]:
    if size == "full":
        return snowball_model(seq_len or 4096), "p=f32,c=bfloat16"
    if size == "medium":
        return dataclasses.replace(snowball_model(seq_len or 1024), num_layers=4), "p=f32,c=bfloat16"
    tiny = dataclasses.replace(
        snowball_model(seq_len or 32),
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_experts=16,
        num_layers=5,
        num_heads=8,
        num_kv_heads=4,
        head_dim=16,
        sliding_window=4,
    )
    return tiny, "f32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=["tiny", "medium", "full"], required=True)
    parser.add_argument("--layers", type=int, help="Override the preset layer count.")
    parser.add_argument("--seq-len", type=int, help="Override the preset sequence length.")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Global batch of one step in sequences (default: one per device per microbatch).",
    )
    parser.add_argument(
        "--microbatches",
        type=int,
        default=1,
        help="Accumulate each step's gradients over this many microbatches of the batch (1 = whole batch at once).",
    )
    parser.add_argument(
        "--overlap-last-microbatch",
        action="store_true",
        help="With --microbatches > 1, reduce the expert gradients layer by layer in the last microbatch's backward.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mp", help="jmp policy override, e.g. 'p=f32,c=float16' (default: preset).")
    parser.add_argument("--expert-axis", type=int, default=1, help="Expert-parallel mesh axis size (1 = FSDP only).")
    parser.add_argument(
        "--replica-axis",
        type=int,
        default=1,
        help="Replica mesh axis size: full parameter copies that all-reduce their gradients (1 = no replicas).",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        help="Initialize jax.distributed from Slurm with one srun process per node owning this many GPUs "
        "(default: one process, no jax.distributed).",
    )
    parser.add_argument("--attention", choices=get_args(GrugAttentionImplementation), default="reference")
    parser.add_argument("--moe-impl", default="ring", help="MoE dispatch backend (default: ring).")
    parser.add_argument(
        "--capacity-factor", type=float, help="EP receiver capacity factor (default: GrugModelConfig's)."
    )
    parser.add_argument(
        "--pooled-transport-capacity-factor",
        type=float,
        help="Sender capacity factor per destination; required by --moe-impl ring_dedup and "
        "fixed_pooled_wave_all_to_all.",
    )
    parser.add_argument(
        "--num-expert-waves",
        type=int,
        default=1,
        help="Waves for fixed_pooled_wave_all_to_all; must divide the local expert count.",
    )
    parser.add_argument(
        "--moe-token-blocks",
        type=int,
        default=1,
        help="ring_dedup only: split each shard's tokens into this many blocks, each with its own all-to-alls, "
        "so one block's exchange can overlap another's expert matmuls.",
    )
    parser.add_argument(
        "--moe-gather-sum",
        choices=get_args(MoeGatherSum),
        default="xla",
        help="ring_dedup only: run the backward's expert-side dispatch gather-sum and sender-side combine gather as "
        "XLA fusions (xla) or in a Pallas-Triton row-gather kernel (pallas), with the same results bit for bit "
        "(GrugModelConfig.moe_gather_sum).",
    )
    parser.add_argument(
        "--ce-v-block-size",
        type=int,
        help="Vocabulary block of the fused cross-entropy (explicit block sizes, which skip batched_xla's forward "
        "autotune; on MI355X its backward keeps its 8192 block). Default: the backend's choice "
        "(GrugModelConfig.ce_v_block_size).",
    )
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/june-synthetic"))
    parser.add_argument("--run-id", help="Run id under --log-dir (default: size and timestamp).")
    parser.add_argument("--compilation-cache-dir", help="Persistent JAX compilation cache directory.")
    parser.add_argument("--remat-mode", choices=get_args(RematMode), default="recompute_all")
    parser.add_argument(
        "--remat-prevent-cse",
        action="store_true",
        default=None,
        help="Checkpoint the blocks with prevent_cse=True (JAX's default): optimization barriers around the saved "
        "residuals, which keep the backward's slices of them unfused. Without it the stacked blocks are checkpointed "
        "with prevent_cse=False and the unstacked ones with True (GrugModelConfig.remat_prevent_cse).",
    )
    parser.add_argument(
        "--rms-norm-impl",
        choices=get_args(RmsNormImpl),
        default="barrier",
        help="How the RMS norms pin their float32 statistics: an optimization barrier on the variance (barrier), or a "
        "custom VJP that saves the inverse RMS for the backward, letting XLA fuse the reduction with its users "
        "(custom_vjp), or a custom VJP whose backward recomputes the inverse RMS from x (custom_vjp_recompute) "
        "(GrugModelConfig.rms_norm_impl).",
    )
    parser.add_argument(
        "--scan-unroll", type=int, default=1, help="Layers per iteration of the layer scan (lax.scan unroll)."
    )
    parser.add_argument(
        "--replicated-grads",
        choices=get_args(ScanReduction),
        default="per_layer",
        help="Reduce the small replicated block weights' gradients in every layer or once after the layer scan.",
    )
    parser.add_argument(
        "--replicated-grads-max-mib",
        type=float,
        default=4.0,
        help="With --replicated-grads after_scan, defer only weights up to this many MiB per layer.",
    )
    parser.add_argument(
        "--router-stats",
        choices=get_args(ScanReduction),
        default="per_layer",
        help="Reduce routing statistics and QB thresholds in every layer or once after the layer scan.",
    )
    parser.add_argument(
        "--capacity-drops",
        choices=get_args(ScanReduction),
        default="per_layer",
        help="Sum the MoE capacity drop counts across the token shards in every layer or once after the layer scan, "
        "with the routing statistics (needs --router-stats after_scan) (GrugModelConfig.capacity_drop_reduction).",
    )
    parser.add_argument(
        "--qb-threshold",
        choices=get_args(QbThresholdImpl),
        default="top_k",
        help="Select the QB thresholds with lax.top_k (a full sort on GPUs) or an exact radix select in one Pallas "
        "kernel (GrugModelConfig.qb_threshold_impl).",
    )
    parser.add_argument(
        "--no-router-diagnostics",
        action="store_true",
        help="Skip the diagnostic routing statistics (expert counts, probability sums, entropy, load balancing, "
        "z-loss unless weighted) in every layer (GrugModelConfig.router_diagnostics=False).",
    )
    parser.add_argument(
        "--skip-zero-router-z-loss",
        action="store_true",
        help="Leave the router z-loss out of the loss when its coefficient is 0, so its backward does not run "
        "(GrugModelConfig.skip_zero_router_z_loss).",
    )
    parser.add_argument("--qb-balance", choices=get_args(QbBalance), default="global", help="QB load balancing groups.")
    parser.add_argument(
        "--qb-sync-interval",
        type=int,
        default=0,
        help="With --qb-balance per_node, average the nodes' QB thresholds every this many steps (0 never).",
    )
    parser.add_argument(
        "--unstacked-blocks",
        action="store_true",
        help="Run the blocks one by one instead of in one layer scan (no stacked per-layer residuals or weights; "
        "needs per_layer reductions).",
    )
    parser.add_argument(
        "--carry-compute-params",
        action="store_true",
        help="Keep a compute-dtype copy of the parameters in the train state, written by the update.",
    )
    parser.add_argument(
        "--zero1",
        action="store_true",
        help="Shard the expert weights' parameters and optimizer state over the replicas (ZeRO-1); needs "
        "--microbatches >= 2 and --carry-compute-params.",
    )
    parser.add_argument(
        "--gather-in-first-microbatch",
        action="store_true",
        help="With --zero1, all-gather the expert weights layer by layer in the first microbatch's forward, into the "
        "compute copy, instead of at the end of the step.",
    )
    parser.add_argument(
        "--expert-grad-accumulation",
        choices=("xla", "flydsl"),
        default="xla",
        help="With --microbatches > 1: sum the expert weight gradients with an XLA float32 add after each microbatch's "
        "layer scan (xla), or inside jax-flydsl's weight-gradient kernel, layer by layer in place (flydsl; ring_dedup "
        "on an expert axis of 2+, bit-identical) (GrugTrainerConfig.expert_grad_accumulation).",
    )
    parser.add_argument(
        "--watch-interval", type=int, default=10, help="Steps between gradient watch steps (0 disables)."
    )
    parser.add_argument(
        "--pipelined-step-logging",
        action="store_true",
        help="Dispatch step N+1 before fetching, checking and logging step N "
        "(GrugTrainerConfig.pipelined_step_logging).",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=300.0,
        help="Exit with status 1 if a step takes longer than this many seconds (0 disables). "
        "The first step and watch steps have no timeout.",
    )
    return parser.parse_args()


def synthetic_examples(count: int, seq_len: int, vocab_size: int, seed: int = 0) -> list[GrugLmExample]:
    rng = np.random.default_rng(seed)
    loss_weight = np.ones(seq_len, dtype=np.float32)
    loss_weight[-1] = 0
    return [
        GrugLmExample(tokens=rng.integers(0, vocab_size, seq_len, dtype=np.int32), loss_weight=loss_weight)
        for _ in range(count)
    ]


def report_memory() -> None:
    for device in jax.local_devices():
        stats = device.memory_stats()
        if stats is None:
            return
        logger.info(
            "%s: peak %.1f GiB of %.1f GiB", device, stats["peak_bytes_in_use"] / 2**30, stats["bytes_limit"] / 2**30
        )


def log_model_flops(flops_per_example: float, flops_summary: dict[str, float]) -> None:
    """Log the FLOPs per example that MFU charges (3x forward) and their attention-map part, non-causal and causal."""
    if jax.process_index() != 0:
        return
    logger.info(
        "model flops per example: total=%r attention_noncausal=%r attention_causal=%r",
        flops_per_example,
        flops_summary[ATTENTION_FLOPS_PER_EXAMPLE_KEY],
        flops_summary[CAUSAL_ATTENTION_FLOPS_PER_EXAMPLE_KEY],
    )


def log_throughput_summary(
    step_durations: list[float], batch_size: int, seq_len: int, flops_per_example: float, num_steps: int
) -> None:
    median = statistics.median(step_durations)
    throughput = compute_instant_throughput(batch_size, median, seq_len, flops_per_example, aggregate_device_flops())
    logger.info(
        "median step over steps %d-%d: %.3f s  %s tokens/s  %s model TFLOP/s  MFU %s%%",
        FIRST_TIMED_STEP,
        num_steps - 1,
        median,
        f"{throughput.tokens_per_second:,.0f}" if throughput.tokens_per_second is not None else "n/a",
        f"{throughput.model_flops_per_second / 1e12:.1f}" if throughput.model_flops_per_second is not None else "n/a",
        f"{throughput.mfu:.2f}" if throughput.mfu is not None else "n/a",
    )


def main() -> None:
    args = parse_args()
    # The training loop sets up logging only once it starts; without this, the lines logged before it are dropped.
    configure_logging(logging.INFO)
    if args.gpus_per_node is not None:
        # Before any JAX call that starts the backend, including jax.device_count() below. Without explicit device
        # ids, JAX gives each Slurm process the single GPU numbered by its local rank.
        DistributedConfig(local_device_ids=list(range(args.gpus_per_node))).initialize()
    model, default_mp = preset(args.size, args.seq_len)
    model = dataclasses.replace(
        model,
        attention_implementation=args.attention,
        moe_implementation=args.moe_impl,
        remat_mode=args.remat_mode,
        remat_prevent_cse=args.remat_prevent_cse,
        rms_norm_impl=args.rms_norm_impl,
        pooled_transport_capacity_factor=args.pooled_transport_capacity_factor,
        num_expert_waves=args.num_expert_waves,
        moe_token_blocks=args.moe_token_blocks,
        moe_gather_sum=args.moe_gather_sum,
        ce_v_block_size=args.ce_v_block_size,
        scan_unroll=args.scan_unroll,
        replicated_grad_reduction=args.replicated_grads,
        replicated_grad_max_bytes=int(args.replicated_grads_max_mib * (1 << 20)),
        router_stats_reduction=args.router_stats,
        capacity_drop_reduction=args.capacity_drops,
        qb_threshold_impl=args.qb_threshold,
        router_diagnostics=not args.no_router_diagnostics,
        skip_zero_router_z_loss=args.skip_zero_router_z_loss,
        qb_balance=args.qb_balance,
        use_array_stacked_blocks=not args.unstacked_blocks,
    )
    if args.capacity_factor is not None:
        model = dataclasses.replace(model, capacity_factor=args.capacity_factor)
    if args.layers is not None:
        model = dataclasses.replace(model, num_layers=args.layers)
    # The backend checks these only on the first step, after the full model state is built.
    if args.moe_impl in TRANSPORT_CAPACITY_MOE_IMPLS and args.pooled_transport_capacity_factor is None:
        raise ValueError(f"--moe-impl {args.moe_impl} requires --pooled-transport-capacity-factor")
    if args.moe_impl == POOLED_WAVE_MOE_IMPL:
        local_experts = model.num_experts // args.expert_axis
        if local_experts % args.num_expert_waves != 0:
            raise ValueError(
                f"--num-expert-waves {args.num_expert_waves} must divide the {local_experts} local experts "
                f"({model.num_experts} experts over --expert-axis {args.expert_axis})"
            )
    batch_size = jax.device_count() * args.microbatches if args.batch_size is None else args.batch_size
    if batch_size == 1:
        # jnp.roll in the next-token loss slices the (1, seq_len) token grid to (1, 1), and under an explicit mesh JAX
        # drops the sharding of that slice, then rejects concatenating it with the (1, seq_len - 1) remainder.
        raise ValueError("a global batch of 1 fails in the loss under explicit mesh axes; pass --batch-size 2 or more")
    # Every process takes rank 0's id: each one's clock can give a different timestamp, which would split the run's
    # logs and profiles across several ids.
    run_id = multihost_broadcast_sync(args.run_id or f"june-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}")

    trainer = TrainerConfig(
        id=run_id,
        log_dir=args.log_dir,
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        mp=jmp.get_policy(args.mp or default_mp),
        train_batch_size=batch_size,
        num_train_steps=args.steps,
        tracker=JsonLoggerConfig(),
        checkpointer=CheckpointerConfig(base_path=str(args.log_dir / run_id / "checkpoints")),
        load_checkpoint=False,  # never resume a reused --run-id; every benchmark starts from step 0
        distributed=DistributedConfig(initialize_jax_distributed=False),
        profiler=ProfilerConfig(
            enabled=args.profile_steps > 0,
            start_step=args.profile_start,
            num_steps=args.profile_steps,
            upload=XprofUploadConfig(enabled=False),
            # The HLO modules let trace tools attribute kernels to HLO ops (spurix's xplane buckets).
            profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
        ),
        jax_compilation_cache_dir=args.compilation_cache_dir,
        log_jaxprs=False,
        log_xla_hlo=False,
        watch=WatchConfig(interval=args.watch_interval),
    )
    # One fresh example per sequence, and a mixture block exactly as long as the dataset so nothing repeats.
    examples = synthetic_examples(args.steps * batch_size, model.max_seq_len, model.vocab_size)
    data = LmDataConfig(
        components={"synthetic": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
        tokenizer="passthrough",
        vocab_size=model.vocab_size,
        mixture_block_size=len(examples),
    )
    config = GrugRunConfig(
        model=model,
        data=data,
        resources=ResourceConfig(),
        optimizer=AdamConfig(learning_rate=args.learning_rate, warmup=0.0),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            expert_axis_size=args.expert_axis,
            replica_axis_size=args.replica_axis,
            save_checkpoints=False,
            step_timeout=args.step_timeout or None,
            qb_sync_interval=args.qb_sync_interval,
            carry_compute_params=args.carry_compute_params,
            microbatches=args.microbatches,
            zero1=args.zero1,
            overlap_last_microbatch=args.overlap_last_microbatch,
            gather_in_first_microbatch=args.gather_in_first_microbatch,
            expert_grad_accumulation=args.expert_grad_accumulation,
            pipelined_step_logging=args.pipelined_step_logging,
        ),
        eval=None,
    )
    flops_per_example, flops_summary = _compute_flops(model_config=model)
    logger.info(
        "size=%s layers=%d hidden=%d experts=%d topk=%d heads=%d kv=%d batch=%d microbatches=%d "
        "overlap_last_microbatch=%s seq_len=%d "
        "window=%d mp=%s expert_axis=%d replica_axis=%d attention=%s moe=%s capacity=%s pooled_capacity=%s waves=%d "
        "token_blocks=%d gather_sum=%s ce_v_block=%s remat=%s remat_prevent_cse=%s rms_norm_impl=%s "
        "replicated_grads=%s router_stats=%s capacity_drops=%s qb_threshold=%s router_diagnostics=%s "
        "skip_zero_router_z_loss=%s qb_balance=%s qb_sync_interval=%d carry_compute_params=%s "
        "zero1=%s gather_in_first_microbatch=%s expert_grad_accumulation=%s watch_interval=%d pipelined_step_logging=%s "
        "flops_per_example=%.4e",
        args.size,
        model.num_layers,
        model.hidden_dim,
        model.num_experts,
        model.num_experts_per_token,
        model.num_heads,
        model.num_kv_heads,
        batch_size,
        args.microbatches,
        args.overlap_last_microbatch,
        model.max_seq_len,
        model.sliding_window,
        trainer.mp,
        args.expert_axis,
        args.replica_axis,
        args.attention,
        args.moe_impl,
        model.capacity_factor,
        model.pooled_transport_capacity_factor,
        model.num_expert_waves,
        model.moe_token_blocks,
        model.moe_gather_sum,
        model.ce_v_block_size,
        args.remat_mode,
        model.remat_prevent_cse,
        model.rms_norm_impl,
        args.replicated_grads,
        args.router_stats,
        model.capacity_drop_reduction,
        model.qb_threshold_impl,
        model.router_diagnostics,
        model.skip_zero_router_z_loss,
        args.qb_balance,
        args.qb_sync_interval,
        args.carry_compute_params,
        args.zero1,
        args.gather_in_first_microbatch,
        args.expert_grad_accumulation,
        args.watch_interval,
        args.pipelined_step_logging,
        flops_per_example,
    )
    log_model_flops(flops_per_example, flops_summary)
    step_durations: list[float] = []
    wall_steps: list[float] = []
    last_hook_time: float | None = None

    def record_step(info: StepInfo, force: bool = False) -> None:
        nonlocal last_hook_time
        # The loop re-runs every hook with force=True after the last step; that call would duplicate the sample.
        if force:
            return
        now = time.perf_counter()
        if last_hook_time is not None:
            # One loop period: what the devices see, host work between steps included.
            wall_step = now - last_hook_time
            levanter.tracker.log({"throughput/wall_step_s": wall_step}, step=info.step)
            if info.step >= FIRST_TIMED_STEP:
                wall_steps.append(wall_step)
        last_hook_time = now
        if info.step >= FIRST_TIMED_STEP:
            step_durations.append(info.step_duration)

    _run_grug_local(config, step_hooks=[record_step])
    if step_durations:
        log_throughput_summary(step_durations, batch_size, model.max_seq_len, flops_per_example, args.steps)
    if wall_steps:
        logger.info(
            "median wall step over steps %d-%d: %.3f s", FIRST_TIMED_STEP, args.steps - 1, statistics.median(wall_steps)
        )
    report_memory()


if __name__ == "__main__":
    main()
