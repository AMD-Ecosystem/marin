# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command line to ``GrugRunConfig`` for the June 67B-A2B Slurm scripts (no Fray or Iris).

``synthetic_benchmark.py`` (random tokens, throughput) and ``slurm_train.py`` (a training run on a token cache)
take the same model, mesh and performance options from here, so a configuration timed by the benchmark runs
unchanged in training. Each script adds only its own options: the benchmark its step count and flat Adam, the
training script its data, optimizer schedule, checkpoints and eval.
"""

import argparse
import dataclasses
import logging
import statistics
import time
from typing import get_args

import jax
import jmp
import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.callbacks._metrics import aggregate_device_flops, compute_instant_throughput
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, XprofUploadConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig
from levanter.grug._moe.ep_ring_dedup import MoeSendGather, MoeTokenSum
from levanter.grug._moe.pallas_gather_sum import MoeGatherSum
from levanter.grug._moe.pallas_routing_metadata import MoeRoutingMetadata
from levanter.grug._moe.scatter_epilogue import MoeScatterEpilogue
from levanter.grug.attention import GrugAttentionImplementation
from levanter.grug.grug_moe import QbThresholdImpl
from levanter.tracker import TrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import multihost_broadcast_sync

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.model import (
    AttentionSchedule,
    CeBackwardGradX,
    CeBackwardWeightBlocks,
    FusedNormImpl,
    GatedNormImpl,
    GatedNormRemat,
    GrugModelConfig,
    HeadGateRemat,
    QbBalance,
    QkNormRope,
    RematMode,
    RmsNormImpl,
    ScanReduction,
)
from experiments.june_tpu_67b_a2b.moe.train import (
    ATTENTION_FLOPS_PER_EXAMPLE_KEY,
    CAUSAL_ATTENTION_FLOPS_PER_EXAMPLE_KEY,
    GrugStepEvalConfig,
    GrugTrainerConfig,
)

logger = logging.getLogger("june_slurm_run")

SNOWBALL_HIDDEN_DIM = 2560
# The production window (half of the 4096-token training context), which Levanter's SnowballConfig pins as well.
# The width heuristic would otherwise derive seq_len // 2 and give the medium preset a shorter window than Snowball's.
SNOWBALL_SLIDING_WINDOW = 2048
POOLED_WAVE_MOE_IMPL = "fixed_pooled_wave_all_to_all"
TRANSPORT_CAPACITY_MOE_IMPLS = ("ring_dedup", POOLED_WAVE_MOE_IMPL)
# Steps 0 and 1 of a process are excluded from the timing summary, as in the Snowball harness: the first compiles.
TIMING_WARMUP_STEPS = 2


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


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The model, mesh, batch, kernel and logging options both Slurm scripts take."""
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
        "--moe-dispatch-barrier",
        action="store_true",
        help="ring_dedup only: optimization barrier on the expert-ordered dispatch rows, so XLA computes their gather "
        "once instead of also fusing it into the saved-residual update; same results bit for bit "
        "(GrugModelConfig.moe_dispatch_barrier).",
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
        "--moe-routing-metadata",
        choices=get_args(MoeRoutingMetadata),
        default="xla",
        help="ring_dedup only: build the integer routing maps (send slots, expert rows, group sizes, drop counts) "
        "with XLA fusions (xla) or two Pallas-Triton counting-sort kernels per side (pallas), with the same values "
        "(GrugModelConfig.moe_routing_metadata).",
    )
    parser.add_argument(
        "--moe-flydsl-gather-a",
        action="store_true",
        help="ring_dedup with RAGGED_DOT_IMPL=flydsl only: gather the dispatch rows inside the up-projection GEMM "
        "(forward and weight gradient) instead of building the expert-ordered copy; same results bit for bit "
        "(GrugModelConfig.moe_flydsl_gather_a).",
    )
    parser.add_argument(
        "--moe-scatter-epilogue",
        choices=get_args(MoeScatterEpilogue),
        default="off",
        help="ring_dedup only: run the expert-side combine and dispatch backward as separate gathers (off) or as "
        "row-scatter epilogues of jax-flydsl's grouped GEMMs plus a Pallas gather-sum of the other rows (flydsl), "
        "with the same results bit for bit (GrugModelConfig.moe_scatter_epilogue).",
    )
    parser.add_argument(
        "--moe-send-gather",
        choices=get_args(MoeSendGather),
        default="xla",
        help="ring_dedup only: build the forward send buffer with an XLA fusion (xla) or the Pallas row gather "
        "(pallas), with the same results bit for bit (GrugModelConfig.moe_send_gather).",
    )
    parser.add_argument(
        "--moe-token-sum",
        choices=get_args(MoeTokenSum),
        default="take",
        help="ring_dedup only: sum each token's send-slot rows (forward combine, dispatch backward) with jnp.take fill "
        "masks (take) or clamped indices whose validity test stays in the gather's fusion (clamped), with the same "
        "results bit for bit (GrugModelConfig.moe_token_sum).",
    )
    parser.add_argument(
        "--ce-v-block-size",
        type=int,
        help="Vocabulary block of the fused cross-entropy (explicit block sizes, which skip batched_xla's forward "
        "autotune; on MI355X its backward keeps its 8192 block). Default: the backend's choice "
        "(GrugModelConfig.ce_v_block_size).",
    )
    parser.add_argument(
        "--ce-weight-grad",
        choices=("stacked", "in_place"),
        default="stacked",
        help="How batched_xla's cross-entropy backward builds the output projection's gradient: a float32 stack of "
        "vocab blocks cast after the loop, from a padded projection (stacked), or each block written into the [H, V] "
        "result in place, padding only the last block (in_place; same arithmetic, ~1.9 GiB less memory) "
        "(GrugModelConfig.ce_weight_grad).",
    )
    parser.add_argument(
        "--ce-bwd-grad-x",
        choices=get_args(CeBackwardGradX),
        default="add",
        help="batched_xla cross-entropy backward: sum the input gradient over the vocabulary blocks with a separate "
        "add kernel after each bf16 GEMM (add) or inside a float32 GEMM as its C operand, in place (gemm; differs in "
        "the last bits) (GrugModelConfig.ce_bwd_grad_x).",
    )
    parser.add_argument(
        "--ce-bwd-weight-blocks",
        choices=get_args(CeBackwardWeightBlocks),
        default="float32",
        help="batched_xla cross-entropy backward: stack the output projection's per-block weight gradients in float32 "
        "(float32) or in the weight's dtype (weight_dtype; the same values bit for bit) "
        "(GrugModelConfig.ce_bwd_weight_blocks).",
    )
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
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
        "--fused-norm-impl",
        choices=get_args(FusedNormImpl),
        default="off",
        help="Run each RMS norm with its GatedNorm (and the residual add before the MLP norm) as one custom VJP whose "
        "backward does the whole RMS-norm backward in one pass, in JAX ops (xla) or Pallas-Triton kernels (pallas); "
        "off keeps the separate modules (GrugModelConfig.fused_norm_impl).",
    )
    parser.add_argument(
        "--gated-norm-impl",
        choices=get_args(GatedNormImpl),
        default="xla",
        help="With a fused norm, run GatedNorm's up projection, sigmoid and product, and its gate backward, as JAX ops "
        "(xla) or as two Pallas-Triton kernels that write no extra [B, S, D] tensors under remat (pallas) "
        "(GrugModelConfig.gated_norm_impl).",
    )
    parser.add_argument(
        "--gated-norm-remat",
        choices=get_args(GatedNormRemat),
        default="recompute",
        help="Save GatedNorm's rank-128 down projection in the per-block checkpoint so the backward's rerun of the "
        "block skips that GEMM (save_hidden), or rerun it (recompute) (GrugModelConfig.gated_norm_remat).",
    )
    parser.add_argument(
        "--head-gate-remat",
        choices=get_args(HeadGateRemat),
        default="recompute",
        help="Save the attention's head-gate logits in the per-block checkpoint so the backward's rerun of the block "
        "skips that GEMM (save), or rerun it (recompute) (GrugModelConfig.head_gate_remat).",
    )
    parser.add_argument(
        "--qk-norm-rope",
        choices=get_args(QkNormRope),
        default="xla",
        help="How attention normalizes, rotates (half-RoPE) and scales q and k: separate ops (xla), or one function "
        "whose backward recomputes the norm statistic, in JAX ops under a custom VJP (custom_vjp) or in one "
        "Pallas-Triton kernel per direction (pallas) (GrugModelConfig.qk_norm_rope).",
    )
    parser.add_argument(
        "--attention-schedule",
        choices=get_args(AttentionSchedule),
        default="cond",
        help="Stacked blocks: pick each layer's attention (long or sliding window) with a per-layer lax.cond (cond), "
        "a trace-time schedule over groups of layers (static, same model), or use the sliding window in every layer "
        "(all_short: a different model, only to time the conditional) (GrugModelConfig.attention_schedule).",
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
        "--block-grad-accumulation",
        choices=("after_scan", "in_scan"),
        default="after_scan",
        help="With --microbatches > 1: sum the other stacked block weights' gradients (attention, shared expert, "
        "norms, router; the expert weights too without --expert-grad-accumulation flydsl) with a float32 add after "
        "each microbatch's layer scan (after_scan), or inside the layer scan's backward, layer by layer into the "
        "carried float32 sums (in_scan; bit-identical, no stacked per-microbatch gradient) "
        "(GrugTrainerConfig.block_grad_accumulation).",
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
        "The first step, watch steps and the first eval step have no timeout.",
    )
    step_eval = parser.add_argument_group("held-out eval inside the train step (GrugStepEvalConfig)")
    step_eval.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Every this many steps, and on the last step, run the train step's eval variant: a forward-only held-out "
        "loss on the compute-dtype weights the step already holds, no extra weight memory (0: no eval). Needs "
        "--microbatches >= 2.",
    )
    step_eval.add_argument(
        "--eval-microbatches",
        type=int,
        default=1,
        help="The held-out set's size in train microbatches (--batch-size / --microbatches sequences each), all "
        "evaluated at every eval step.",
    )


def initialize_distributed(args: argparse.Namespace) -> None:
    """With --gpus-per-node, start jax.distributed from Slurm. Call before any JAX call that starts the backend."""
    if args.gpus_per_node is not None:
        # Without explicit device ids, JAX gives each Slurm process the single GPU numbered by its local rank.
        DistributedConfig(local_device_ids=list(range(args.gpus_per_node))).initialize()


def build_model(args: argparse.Namespace) -> tuple[GrugModelConfig, str]:
    """The preset model with the command line's options, and the preset's default mixed-precision policy."""
    model, default_mp = preset(args.size, args.seq_len)
    model = dataclasses.replace(
        model,
        attention_implementation=args.attention,
        moe_implementation=args.moe_impl,
        remat_mode=args.remat_mode,
        remat_prevent_cse=args.remat_prevent_cse,
        rms_norm_impl=args.rms_norm_impl,
        fused_norm_impl=args.fused_norm_impl,
        gated_norm_impl=args.gated_norm_impl,
        gated_norm_remat=args.gated_norm_remat,
        head_gate_remat=args.head_gate_remat,
        qk_norm_rope=args.qk_norm_rope,
        attention_schedule=args.attention_schedule,
        pooled_transport_capacity_factor=args.pooled_transport_capacity_factor,
        num_expert_waves=args.num_expert_waves,
        moe_token_blocks=args.moe_token_blocks,
        moe_dispatch_barrier=args.moe_dispatch_barrier,
        moe_gather_sum=args.moe_gather_sum,
        moe_routing_metadata=args.moe_routing_metadata,
        moe_flydsl_gather_a=args.moe_flydsl_gather_a,
        moe_scatter_epilogue=args.moe_scatter_epilogue,
        moe_send_gather=args.moe_send_gather,
        moe_token_sum=args.moe_token_sum,
        ce_v_block_size=args.ce_v_block_size,
        ce_weight_grad=args.ce_weight_grad,
        scan_unroll=args.scan_unroll,
        ce_bwd_grad_x=args.ce_bwd_grad_x,
        ce_bwd_weight_blocks=args.ce_bwd_weight_blocks,
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
    return model, default_mp


def global_batch_size(args: argparse.Namespace) -> int:
    batch_size = jax.device_count() * args.microbatches if args.batch_size is None else args.batch_size
    if batch_size == 1:
        # jnp.roll in the next-token loss slices the (1, seq_len) token grid to (1, 1), and under an explicit mesh JAX
        # drops the sharding of that slice, then rejects concatenating it with the (1, seq_len - 1) remainder.
        raise ValueError("a global batch of 1 fails in the loss under explicit mesh axes; pass --batch-size 2 or more")
    return batch_size


def step_eval_sequences(args: argparse.Namespace, batch_size: int) -> int:
    """The held-out sequences --eval-every and --eval-microbatches ask for: 0 without an eval."""
    if args.eval_every <= 0:
        return 0
    if args.microbatches < 2:
        raise ValueError(f"--eval-every needs --microbatches >= 2, got {args.microbatches}")
    if args.eval_microbatches < 1:
        raise ValueError(f"--eval-microbatches must be at least 1, got {args.eval_microbatches}")
    return args.eval_microbatches * batch_size // args.microbatches


def shared_run_id(args: argparse.Namespace, prefix: str) -> str:
    """--run-id, else ``<prefix>-<size>-<timestamp>``. Every process takes rank 0's id: each one's clock can give a
    different timestamp, which would split the run's logs, profiles and checkpoints across several ids."""
    return multihost_broadcast_sync(args.run_id or f"{prefix}-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}")


def build_trainer(
    args: argparse.Namespace,
    *,
    batch_size: int,
    num_train_steps: int,
    run_id: str,
    default_mp: str,
    tracker: TrackerConfig | tuple[TrackerConfig, ...],
    checkpointer: CheckpointerConfig,
    load_checkpoint: bool | None,
) -> TrainerConfig:
    """The Levanter TrainerConfig: what the scripts share, plus their tracker and checkpoint policy.

    ``load_checkpoint`` is False to always start at step 0, None to resume from this run id's latest checkpoint.
    """
    return TrainerConfig(
        id=run_id,
        log_dir=args.log_dir,
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        mp=jmp.get_policy(args.mp or default_mp),
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        tracker=tracker,
        checkpointer=checkpointer,
        load_checkpoint=load_checkpoint,
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


def build_grug_trainer(
    args: argparse.Namespace,
    trainer: TrainerConfig,
    *,
    save_checkpoints: bool,
    data_seed: int | None = None,
    z_loss_weight: float = 0.0,
) -> GrugTrainerConfig:
    """The Grug trainer: the mesh, microbatching and performance options of the command line."""
    return GrugTrainerConfig(
        trainer=trainer,
        data_seed=data_seed,
        z_loss_weight=z_loss_weight,
        expert_axis_size=args.expert_axis,
        replica_axis_size=args.replica_axis,
        save_checkpoints=save_checkpoints,
        step_timeout=args.step_timeout or None,
        qb_sync_interval=args.qb_sync_interval,
        carry_compute_params=args.carry_compute_params,
        microbatches=args.microbatches,
        zero1=args.zero1,
        overlap_last_microbatch=args.overlap_last_microbatch,
        gather_in_first_microbatch=args.gather_in_first_microbatch,
        expert_grad_accumulation=args.expert_grad_accumulation,
        block_grad_accumulation=args.block_grad_accumulation,
        pipelined_step_logging=args.pipelined_step_logging,
    )


def log_run_config(
    args: argparse.Namespace,
    model: GrugModelConfig,
    trainer: TrainerConfig,
    batch_size: int,
    flops_per_example: float,
) -> None:
    logger.info(
        "size=%s layers=%d hidden=%d experts=%d topk=%d heads=%d kv=%d batch=%d microbatches=%d "
        "overlap_last_microbatch=%s seq_len=%d "
        "window=%d mp=%s expert_axis=%d replica_axis=%d attention=%s moe=%s capacity=%s pooled_capacity=%s waves=%d "
        "token_blocks=%d gather_sum=%s routing_metadata=%s flydsl_gather_a=%s scatter_epilogue=%s send_gather=%s "
        "token_sum=%s ce_v_block=%s "
        "remat=%s remat_prevent_cse=%s rms_norm_impl=%s fused_norm_impl=%s gated_norm_impl=%s gated_norm_remat=%s "
        "head_gate_remat=%s qk_norm_rope=%s attention_schedule=%s "
        "replicated_grads=%s router_stats=%s capacity_drops=%s qb_threshold=%s router_diagnostics=%s "
        "skip_zero_router_z_loss=%s qb_balance=%s qb_sync_interval=%d carry_compute_params=%s dispatch_barrier=%s "
        "zero1=%s gather_in_first_microbatch=%s expert_grad_accumulation=%s block_grad_accumulation=%s "
        "ce_weight_grad=%s watch_interval=%d pipelined_step_logging=%s ce_bwd_grad_x=%s ce_bwd_weight_blocks=%s "
        "eval_every=%d eval_microbatches=%d flops_per_example=%.4e",
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
        model.moe_routing_metadata,
        model.moe_flydsl_gather_a,
        model.moe_scatter_epilogue,
        model.moe_send_gather,
        model.moe_token_sum,
        model.ce_v_block_size,
        args.remat_mode,
        model.remat_prevent_cse,
        model.rms_norm_impl,
        model.fused_norm_impl,
        model.gated_norm_impl,
        model.gated_norm_remat,
        model.head_gate_remat,
        model.qk_norm_rope,
        model.attention_schedule,
        args.replicated_grads,
        args.router_stats,
        model.capacity_drop_reduction,
        model.qb_threshold_impl,
        model.router_diagnostics,
        model.skip_zero_router_z_loss,
        args.qb_balance,
        args.qb_sync_interval,
        args.carry_compute_params,
        model.moe_dispatch_barrier,
        args.zero1,
        args.gather_in_first_microbatch,
        args.expert_grad_accumulation,
        args.block_grad_accumulation,
        model.ce_weight_grad,
        args.watch_interval,
        args.pipelined_step_logging,
        model.ce_bwd_grad_x,
        model.ce_bwd_weight_blocks,
        args.eval_every,
        args.eval_microbatches,
        flops_per_example,
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


def report_memory() -> None:
    for device in jax.local_devices():
        stats = device.memory_stats()
        if stats is None:
            return
        logger.info(
            "%s: peak %.1f GiB of %.1f GiB", device, stats["peak_bytes_in_use"] / 2**30, stats["bytes_limit"] / 2**30
        )


class StepTimes:
    """A step hook that records each step's duration and wall period, skipping the first ``TIMING_WARMUP_STEPS``
    steps this process runs (the first compiles; a resumed run starts at its checkpoint's step).

    The wall period (``throughput/wall_step_s``) is the time between consecutive steps' hooks, which includes the
    host work between steps that ``throughput/duration`` leaves out.

    With ``step_eval``, the steps it runs at (``GrugStepEvalConfig.runs_at``) are kept apart: the headline covers the
    train-only steps, so it stays comparable with runs without an eval, and a separate line gives the eval steps'
    cost. The first eval step this process runs compiles the eval variant and is reported on its own; the run's last
    step is left out of the eval steps' mean (its wall period ends with the loop).
    """

    def __init__(self, step_eval: GrugStepEvalConfig | None, num_train_steps: int) -> None:
        self.step_eval = step_eval
        self.num_train_steps = num_train_steps
        self.step_durations: list[float] = []
        self.wall_steps: list[float] = []
        self.timed_steps: list[int] = []
        self.eval_wall_steps: dict[int, float] = {}
        self.first_eval: tuple[int, float] | None = None
        self._seen = 0
        self._last_hook_time: float | None = None

    def _is_eval(self, step: int) -> bool:
        return self.step_eval is not None and self.step_eval.runs_at(step, self.num_train_steps)

    def __call__(self, info: StepInfo, force: bool = False) -> None:
        # The loop re-runs every hook with force=True after the last step; that call would duplicate the sample.
        if force:
            return
        now = time.perf_counter()
        timed = self._seen >= TIMING_WARMUP_STEPS
        self._seen += 1
        wall_step = None
        if self._last_hook_time is not None:
            # One loop period: what the devices see, host work between steps included.
            wall_step = now - self._last_hook_time
            levanter.tracker.log({"throughput/wall_step_s": wall_step}, step=info.step)
        self._last_hook_time = now
        if self._is_eval(info.step):
            if self.first_eval is None:
                self.first_eval = (info.step, wall_step or info.step_duration)
            elif timed and wall_step is not None and info.step != self.num_train_steps - 1:
                self.eval_wall_steps[info.step] = wall_step
            return
        if not timed:
            return
        if wall_step is not None:
            self.wall_steps.append(wall_step)
        self.step_durations.append(info.step_duration)
        self.timed_steps.append(info.step)

    def log_summary(self, batch_size: int, seq_len: int, flops_per_example: float) -> None:
        """The headline spurix parses: median step, tokens/s, model TFLOP/s and MFU, then the median wall step, over
        the train-only steps; with an eval, then the eval steps' cost."""
        if not self.step_durations:
            return
        first, last = self.timed_steps[0], self.timed_steps[-1]
        median = statistics.median(self.step_durations)
        throughput = compute_instant_throughput(batch_size, median, seq_len, flops_per_example, aggregate_device_flops())
        logger.info(
            "median step over steps %d-%d: %.3f s  %s tokens/s  %s model TFLOP/s  MFU %s%%",
            first,
            last,
            median,
            f"{throughput.tokens_per_second:,.0f}" if throughput.tokens_per_second is not None else "n/a",
            (
                f"{throughput.model_flops_per_second / 1e12:.1f}"
                if throughput.model_flops_per_second is not None
                else "n/a"
            ),
            f"{throughput.mfu:.2f}" if throughput.mfu is not None else "n/a",
        )
        if self.wall_steps:
            logger.info("median wall step over steps %d-%d: %.3f s", first, last, statistics.median(self.wall_steps))
        if self.step_eval is None or self.first_eval is None:
            return
        compile_step, compile_wall = self.first_eval
        if not self.eval_wall_steps or not self.wall_steps:
            logger.info("eval steps: 0 measured; first (compiling) step %d: %.3f s", compile_step, compile_wall)
            return
        train_mean = statistics.fmean(self.wall_steps)
        eval_mean = statistics.fmean(self.eval_wall_steps.values())
        extra = eval_mean - train_mean
        logger.info(
            "eval steps: %d measured (steps %s), mean wall step %.3f s; first (compiling) step %d: %.3f s; "
            "%+.3f s over the train-only mean wall step %.3f s; mean wall step with an eval every %d steps: %.3f s",
            len(self.eval_wall_steps),
            ",".join(map(str, self.eval_wall_steps)),
            eval_mean,
            compile_step,
            compile_wall,
            extra,
            train_mean,
            self.step_eval.every,
            train_mean + extra / self.step_eval.every,
        )
