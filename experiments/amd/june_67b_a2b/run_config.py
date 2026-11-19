# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command line to ``GrugRunConfig`` for the AMD June scripts (no Fray or Iris).

The options cover the model size, the mesh, the batch and its microbatches, ZeRO-1, the attention kernel, profiling
and logging, and (fork overlay) the held-out step eval. The kernels and the loop are fixed in the
model and trainer code, so there are no options for them. A script adds its own options on top: the synthetic
benchmark its step count and flat Adam, ``slurm_train.py`` its data, budget, optimizer and checkpoints.
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
from levanter.grug._moe import flydsl_expert
from levanter.grug.attention import GrugAttentionImplementation
from levanter.tracker import TrackerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import multihost_broadcast_sync

from experiments.amd.june_67b_a2b.model import MOE_IMPLEMENTATION, shared_under_dispatch
from experiments.amd.june_67b_a2b.train import GrugTrainerConfig, eval_runs_at
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.synthetic_benchmark import snowball_model
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig

logger = logging.getLogger("amd_june_run")

# ring_filtered's sender pool per destination shard, in units of the even share: the MI355X selected configuration.
POOLED_TRANSPORT_CAPACITY_FACTOR = 1.1
# Steps 0 and 1 of a process are excluded from the timing summary, as in the Snowball harness: the first compiles.
TIMING_WARMUP_STEPS = 2


def preset(size: str, seq_len: int | None) -> tuple[GrugModelConfig, str]:
    """The model at ``size`` and its mixed-precision policy. ``full`` is June's production shape (``snowball_model``),
    ``medium`` that with 4 layers, ``tiny`` the smallest shape the kernels tile, for CPU smoke runs."""
    if size == "full":
        model, mp = snowball_model(seq_len or 4096), "p=f32,c=bfloat16"
    elif size == "medium":
        model, mp = dataclasses.replace(snowball_model(seq_len or 1024), num_layers=4), "p=f32,c=bfloat16"
    else:
        model = dataclasses.replace(
            snowball_model(seq_len or 32),
            vocab_size=128,
            hidden_dim=128,
            intermediate_dim=64,
            shared_expert_intermediate_dim=64,
            num_experts=16,
            num_layers=5,
            num_heads=2,
            num_kv_heads=1,
            head_dim=64,
            sliding_window=4,
        )
        mp = "f32"
    model = dataclasses.replace(
        model,
        moe_implementation=MOE_IMPLEMENTATION,
        pooled_transport_capacity_factor=POOLED_TRANSPORT_CAPACITY_FACTOR,
    )
    return model, mp


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The model, mesh, batch, attention, profiling and logging options."""
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
        help="Run each step's batch as this many microbatches, reducing the gradients once per step (1 = all at once).",
    )
    parser.add_argument(
        "--zero1",
        action="store_true",
        help="Shard the expert weights' parameters and optimizer state over the replicas (needs --microbatches >= 2).",
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
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
    parser.add_argument("--compilation-cache-dir", help="Persistent JAX compilation cache directory.")
    parser.add_argument(
        "--watch-interval", type=int, default=10, help="Steps between gradient watch steps (0 disables)."
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=300.0,
        help="Exit with status 1 if a step takes longer than this many seconds (0 disables). "
        "The first step, watch steps and the first eval step have no timeout.",
    )
    step_eval = parser.add_argument_group("held-out step eval (GrugEvalConfig)")
    step_eval.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Every this many steps from step 0, and on the last step, run the step eval before the train step: a "
        "forward-only held-out loss on the compute-dtype weights the step starts from (0: no eval). Needs "
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
    model = dataclasses.replace(model, attention_implementation=args.attention)
    if args.layers is not None:
        model = dataclasses.replace(model, num_layers=args.layers)
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


def eval_config(args: argparse.Namespace, batch_size: int, *, compute_bpb: bool) -> GrugEvalConfig | None:
    """June's ``GrugEvalConfig`` for --eval-every and --eval-microbatches, which the AMD trainer runs inside the train
    step on the data's one validation set (``step_eval_sequences`` sequences of it)."""
    sequences = step_eval_sequences(args, batch_size)
    if not sequences:
        return None
    return GrugEvalConfig(
        eval_batch_size=sequences,
        steps_per_eval=args.eval_every,
        max_eval_batches=None,
        eval_current=True,
        eval_ema=False,
        compute_bpb=compute_bpb,
    )


def shared_run_id(args: argparse.Namespace, prefix: str) -> str:
    """--run-id, else ``<prefix>-<size>-<timestamp>``. Every process takes rank 0's id: each one's clock can give a
    different timestamp, which would split the run's logs and profiles across several ids."""
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
    load_checkpoint: bool | None = False,
) -> TrainerConfig:
    """The Levanter TrainerConfig of the run; ``load_checkpoint=None`` resumes from the run id's latest checkpoint
    if there is one, False always starts at step 0."""
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
            # HLO protos add /host:metadata so trace tools can attribute kernels to HLO ops.
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
    """The AMD Grug trainer config with the command line's mesh, microbatches and ZeRO-1."""
    return GrugTrainerConfig(
        trainer=trainer,
        data_seed=data_seed,
        z_loss_weight=z_loss_weight,
        expert_axis_size=args.expert_axis,
        replica_axis_size=args.replica_axis,
        microbatches=args.microbatches,
        zero1=args.zero1,
        save_checkpoints=save_checkpoints,
        step_timeout=args.step_timeout or None,
    )


def log_run_config(
    args: argparse.Namespace, model: GrugModelConfig, trainer: TrainerConfig, batch_size: int, flops_per_example: float
) -> None:
    logger.info(
        "size=%s layers=%d hidden=%d experts=%d topk=%d heads=%d kv=%d batch=%d microbatches=%d zero1=%s "
        "seq_len=%d window=%d mp=%s "
        "expert_axis=%d replica_axis=%d attention=%s moe=%s expert_kernels=%s expert_grad_sums=%s "
        "capacity=%s pooled_capacity=%s shared_under_dispatch=%s "
        "watch_interval=%d eval_every=%d eval_microbatches=%d flops_per_example=%.4e",
        args.size,
        model.num_layers,
        model.hidden_dim,
        model.num_experts,
        model.num_experts_per_token,
        model.num_heads,
        model.num_kv_heads,
        batch_size,
        args.microbatches,
        args.zero1,
        model.max_seq_len,
        model.sliding_window,
        trainer.mp,
        args.expert_axis,
        args.replica_axis,
        model.attention_implementation,
        model.moe_implementation,
        # Fork overlay: "flydsl" with RAGGED_DOT_IMPL=flydsl (gathered A rows, scatter epilogues, in-kernel sums).
        flydsl_expert.expert_gemm() or "ring_filtered",
        # in_kernel when train._in_kernel_expert_sums() holds on a microbatched step; in_scan otherwise.
        (
            "in_kernel"
            if flydsl_expert.expert_gemm() is not None and args.expert_axis > 1 and args.microbatches > 1
            else "in_scan"
        ),
        model.capacity_factor,
        model.pooled_transport_capacity_factor,
        # Fork overlay: MARIN_SHARED_UNDER_DISPATCH.
        shared_under_dispatch(),
        args.watch_interval,
        args.eval_every,
        args.eval_microbatches,
        flops_per_example,
    )


class StepTimes:
    """A step hook that records each step's duration and wall period, skipping the first ``TIMING_WARMUP_STEPS``
    steps this process runs (the first compiles; a resumed run starts at its checkpoint's step).

    The wall period (``throughput/wall_step_s``) is the time between consecutive steps' hooks, which includes the
    host work between steps that ``throughput/duration`` leaves out.

    With the step eval (``eval_cfg``), the steps it runs at (``eval_runs_at``) are kept apart: the
    headline covers the train-only steps, so it stays comparable with runs without an eval, and a separate line gives
    the eval steps' cost. Each eval step is timed together with the step before it, and both are left out of the
    train-only steps: with pipelined logging a step's metrics arrive only once the next step is done, so the eval's
    time lands in the previous step's period, and the first eval step's flush of the pipeline leaves a near-empty one.
    An eval step's wall time is the pair's wall time minus one train-only mean wall step. The first eval step this
    process runs compiles the eval and is reported on its own; the run's last step is left out of the eval
    steps' mean (its wall period ends with the loop).
    """

    def __init__(self, eval_cfg: GrugEvalConfig | None, num_train_steps: int) -> None:
        self.eval_every = None if eval_cfg is None else eval_cfg.steps_per_eval
        self.num_train_steps = num_train_steps
        self.step_durations: list[float] = []
        self.wall_steps: list[float] = []
        self.timed_steps: list[int] = []
        # Wall periods of an eval step and the step before it, summed; the first eval step's apart.
        self.eval_wall_steps: dict[int, float] = {}
        self.first_eval: tuple[int, float] | None = None
        self._before_eval: tuple[bool, float | None] | None = None
        self._seen = 0
        self._last_hook_time: float | None = None

    def _is_eval(self, step: int) -> bool:
        return self.eval_every is not None and eval_runs_at(step, self.eval_every, self.num_train_steps)

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
            before, self._before_eval = self._before_eval, None
            pair = (before[1] or 0.0) if before is not None else 0.0
            pair += wall_step if wall_step is not None else info.step_duration
            if self.first_eval is None:
                self.first_eval = (info.step, pair)
            elif before is not None and before[0] and wall_step is not None and info.step != self.num_train_steps - 1:
                self.eval_wall_steps[info.step] = pair
            return
        if self._is_eval(info.step + 1):
            # Held for the eval step's pair (see the class docstring).
            self._before_eval = (timed, wall_step)
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
        if not self.step_durations or jax.process_index() != 0:
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
        if self.eval_every is None or self.first_eval is None:
            return
        compile_step, compile_pair = self.first_eval
        train_mean = statistics.fmean(self.wall_steps) if self.wall_steps else 0.0
        compile_wall = compile_pair - train_mean
        if not self.eval_wall_steps or not self.wall_steps:
            logger.info("eval steps: 0 measured; first (compiling) step %d: %.3f s", compile_step, compile_wall)
            return
        eval_mean = statistics.fmean(self.eval_wall_steps.values()) - train_mean
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
            self.eval_every,
            train_mean + extra / self.eval_every,
        )
