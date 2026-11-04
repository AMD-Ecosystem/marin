# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient accumulation over microbatches in the June train step, on eight CPU devices with the ring_dedup MoE.

A train step with ``microbatches=K`` gives the loss, gradients and routing counts of the whole batch, on a mesh with
replicas and expert parallelism (replica_dcn=4 x expert=2) and on one that also shards the parameters over ``data``
(replica_dcn=2 x data=2 x expert=2). The capacity factors are high enough that no expert overflows, since capacity
applies per microbatch. This holds with the logit and router z-losses on, with padding that leaves the microbatches
unequal valid tokens, and with a microbatch that carries no loss weight at all. The next step's QB thresholds are the
mean of those of the microbatches. Every weight's gradient is reduced across the replicas once per step, after the
microbatch loop. With ``overlap_last_microbatch`` the step gives the same loss, metrics and gradients, bit for bit, and
the expert weights' gradients are reduced instead one layer at a time in the last microbatch's backward layer scan.
Checkpointing the stacked blocks without ``prevent_cse`` (the default) drops the optimization barriers around the saved
residuals and gives the same step as with it.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import jmp
import optax
import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.june_tpu_67b_a2b.moe import train as june_train

_PRELUDE = r"""
import collections
import dataclasses
import re

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe import train as june_train
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig

BASE = GrugModelConfig(
    vocab_size=32, hidden_dim=16, intermediate_dim=8, shared_expert_intermediate_dim=8, num_experts=4,
    num_experts_per_token=2, num_layers=3, num_heads=2, num_kv_heads=1, head_dim=8, max_seq_len=8,
    sliding_window=4, capacity_factor=8.0, attention_implementation="reference", moe_implementation="ring_dedup",
    pooled_transport_capacity_factor=4.0, use_array_stacked_blocks=True, disable_pko=True,
    replicated_grad_reduction="after_scan", router_stats_reduction="after_scan",
)
MESHES = {
    "replica4_expert2": compact_grug_mesh(expert_axis_size=2, replica_axis_size=4),
    "replica2_data2_expert2": compact_grug_mesh(expert_axis_size=2, replica_axis_size=2),
}
# 32 rows: four microbatches of one row per device.
TOKENS = np.asarray(jax.random.randint(jax.random.key(1), (32, 8), 0, BASE.vocab_size))
LOSS_WEIGHT = np.ones((32, 8), np.float32)
LOSS_WEIGHT[:, -1] = 0
LOSS_WEIGHT[::3, :4] = 0  # Microbatches with unequal loss weight.
Batch = collections.namedtuple("Batch", "tokens loss_weight attn_mask")
BATCH = Batch(TOKENS, LOSS_WEIGHT, None)
# Rows 0, 3, 6, ... end in three padding positions, so the microbatches route unequal numbers of tokens.
SEGMENT_IDS = np.zeros((32, 8), np.int32)
SEGMENT_IDS[::3, -3:] = -1
PADDED_BATCH = Batch(
    TOKENS,
    np.where(SEGMENT_IDS >= 0, LOSS_WEIGHT, 0),
    AttentionMask(is_causal=True, segment_ids=(SEGMENT_IDS, SEGMENT_IDS)),
)
# Rows 0, 4, 8, ... carry no loss weight: microbatch 0 of four has none.
UNWEIGHTED_ROWS_WEIGHT = LOSS_WEIGHT.copy()
UNWEIGHTED_ROWS_WEIGHT[::4] = 0
UNWEIGHTED_MICROBATCH = Batch(TOKENS, UNWEIGHTED_ROWS_WEIGHT, None)


def keep_gradients():
    # Leaves the parameters unchanged and stores the step's gradients as the optimizer state.
    return optax.GradientTransformation(
        init=lambda params: jax.tree.map(jnp.zeros_like, params),
        update=lambda grads, state, params=None: (jax.tree.map(jnp.zeros_like, grads), grads),
    )


def step(
    cfg, mesh, policy, *, microbatches, carry_compute_params=False, batch=BATCH, z_loss_weight=0.0,
    overlap_last_microbatch=False,
):
    optimizer = keep_gradients()
    with jax.set_mesh(mesh):
        state = june_train.initial_state(
            cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None,
            carry_compute_params=carry_compute_params,
        )
        train_step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=z_loss_weight, ema_beta=None, microbatches=microbatches,
            overlap_last_microbatch=overlap_last_microbatch,
        )
        state, metrics, _ = train_step(state, batch)
    return state, jax.device_get(metrics)


def compiled_step_hlo(cfg, mesh, policy, *, microbatches, overlap_last_microbatch=False, combine_all_reduces=True):
    optimizer = optax.adam(1e-2)
    # Without the combiner every reduction keeps its own instruction and op_name.
    options = {} if combine_all_reduces else {"xla_disable_hlo_passes": "cpu-all-reduce-combiner"}
    with jax.set_mesh(mesh):
        state = june_train.initial_state(cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None)
        train_step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=0.0, ema_beta=None, microbatches=microbatches,
            overlap_last_microbatch=overlap_last_microbatch,
        )
        return train_step.lower(state, BATCH).compile(compiler_options=options).as_text()


Reduction = collections.namedtuple("Reduction", "in_loop elements scalar op_name")
_REDUCTIONS = ("all-reduce", "all-reduce-start", "reduce-scatter")


def replica_reductions(hlo, mesh, ops=_REDUCTIONS):
    # The all-reduces and reduce-scatters (or other collectives ``ops``) of a compiled CPU module that cross
    # replica_dcn: whether each runs inside a while loop (the microbatch loop, the layer scans), how many elements it
    # reduces (its operands', so a reduce-scatter counts its output times its group size; an all-gather its output),
    # whether every operand is a scalar, and its source op.
    computations, current = {}, None
    for line in hlo.splitlines():
        header = re.match(r"(ENTRY )?%?([\w.\-]+) .*\{$", line)
        if header and not line.startswith(" "):
            current = header.group(2)
            computations[current] = (bool(header.group(1)), [])
        elif current and line.startswith(" "):
            computations[current][1].append(line.strip())
    # A computation runs in a loop if it is a while body or is called from one.
    in_loop, stack = {}, [(name, False) for name, (entry, _) in computations.items() if entry]
    while stack:
        name, looped = stack.pop()
        if in_loop.get(name, False) >= looped and name in in_loop:
            continue
        in_loop[name] = looped
        for line in computations[name][1]:
            body = re.search(r"\bbody=%?([\w.\-]+)", line)
            for callee in re.findall(r"%([\w.\-]+)", line):
                if callee in computations:
                    stack.append((callee, looped or (body is not None and callee == body.group(1))))
    coords = np.stack(np.unravel_index(np.arange(mesh.devices.size), mesh.devices.shape), axis=1)
    replica = mesh.axis_names.index("replica_dcn")
    out = []
    for name, (_, lines) in computations.items():
        for line in lines:
            m = re.match(r"(?:ROOT )?%?[\w.\-]+ = (.+?) ([\w\-]+)\((.*)$", line)
            if not m or m.group(2) not in ops:
                continue
            shape, _, attrs = m.groups()
            named = re.search(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}", attrs)
            # Iota groups: [groups, size]<=[dims], optionally T(permutation) of the dims.
            iota = re.search(r"replica_groups=\[(\d+),(\d+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?", attrs)
            if named:
                sizes = {axis: int(size) for axis, size in re.findall(r"'(\w+)'=(\d+)", named.group(1))}
                axes = re.findall(r"'(\w+)'", named.group(2))
                crosses = "replica_dcn" in axes
                group_size = int(np.prod([sizes[axis] for axis in axes]))
            else:
                if iota:
                    count, size, dims, permutation = iota.groups()
                    ids = np.arange(mesh.devices.size).reshape([int(d) for d in dims.split(",")])
                    if permutation:
                        ids = ids.transpose([int(d) for d in permutation.split(",")])
                    groups = ids.reshape(int(count), int(size)).tolist()
                else:
                    groups = re.search(r"replica_groups=\{(.*?)\}\}", attrs)
                    assert groups, line
                    groups = [group.split(",") for group in re.findall(r"\{([\d,]+)", groups.group(0))]
                crosses = any(len({coords[int(d)][replica] for d in group}) > 1 for group in groups)
                group_size = len(groups[0])
            if crosses:
                dims = [[int(d) for d in each.split(",") if d] for each in re.findall(r"\[([\d,]*)\]", shape)]
                elements = sum(int(np.prod(each)) for each in dims)
                if m.group(2) == "reduce-scatter":
                    elements *= group_size
                op_name = re.search(r'op_name="([^"]*)"', attrs)
                scalar = all(not each for each in dims)
                out.append(Reduction(in_loop[name], elements, scalar, op_name.group(1) if op_name else ""))
    return out


def leaves(tree):
    return [np.asarray(x, dtype=np.float32) for x in jax.tree_util.tree_leaves(tree)]


def grad_errors(state, reference):
    # Relative L2 error of every gradient leaf against the reference state's.
    return [
        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        for a, b in zip(leaves(state.opt_state), leaves(reference.opt_state), strict=True)
    ]


def assert_matches_whole_batch(cfg, policy, check_grads, *, carry_compute_params=False, **step_args):
    for name, mesh in MESHES.items():
        whole, whole_metrics = step(
            cfg, mesh, policy, microbatches=1, carry_compute_params=carry_compute_params, **step_args
        )
        assert np.sum(whole_metrics["moe/dropped_assignments"]) == 0, name
        for k in (2, 4):
            msg = f"{name} K={k}"
            accumulated, metrics = step(
                cfg, mesh, policy, microbatches=k, carry_compute_params=carry_compute_params, **step_args
            )
            assert metrics.keys() == whole_metrics.keys(), msg
            for key in (
                "train/loss",
                "train/cross_entropy_loss",
                "train/router/aux_loss_weighted",
                "train/router/router_z_loss",
            ):
                np.testing.assert_allclose(metrics[key], whole_metrics[key], rtol=1e-6, err_msg=f"{msg} {key}")
            # Same routing for every token, and no drops: capacity applies per microbatch.
            np.testing.assert_array_equal(
                metrics["train/router/routing_counts_per_layer"],
                whole_metrics["train/router/routing_counts_per_layer"],
                err_msg=msg,
            )
            assert np.sum(metrics["moe/dropped_assignments"]) == 0, msg
            check_grads(accumulated, whole, name, msg)
"""


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=8)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "case",
    [
        "BASE, batch=BATCH",
        # The logit z-loss is weighted by loss weight, the router z-loss averages over valid tokens.
        "dataclasses.replace(BASE, router_z_loss_coef=0.1), batch=PADDED_BATCH, z_loss_weight=1e-2",
        "dataclasses.replace(BASE, router_z_loss_coef=0.1), batch=UNWEIGHTED_MICROBATCH",
    ],
    ids=["unequal_loss_weight", "z_losses_and_padding", "unweighted_microbatch"],
)
def test_microbatches_match_the_whole_batch_in_float32(case):
    # Only the order of the float32 sums changes: a few ulps (2**-23 each); 4e-7 observed.
    _run(
        f"""
        def check_grads(accumulated, whole, name, msg):
            assert max(grad_errors(accumulated, whole)) <= 1e-5, msg

        assert_matches_whole_batch({case}, policy=jmp.get_policy("f32"), check_grads=check_grads)
        """
    )


@pytest.mark.timeout(900)
def test_microbatches_match_the_whole_batch_with_bfloat16_compute():
    # The forward is the same for every token, so the loss matches to float32 rounding. The gradients do not: each
    # microbatch rounds its own bfloat16 weight gradients, and each device's float32 sum of them is rounded to
    # bfloat16 once more before the single cross-device sum after the loop. The whole batch's bfloat16 gradients are
    # themselves off the float32 ones, by up to 2e-2 (the norm weights, whose gradient sums many terms), so require
    # every microbatched gradient leaf to be no further from its float32 counterpart than the whole batch's, with a
    # margin and a floor of one bfloat16 rounding (2**-8).
    _run(
        """
        policy = jmp.get_policy("p=f32,c=bfloat16")
        cfg = dataclasses.replace(BASE, remat_mode="save_moe_attn")
        exact = {name: step(cfg, mesh, jmp.get_policy("f32"), microbatches=1)[0] for name, mesh in MESHES.items()}

        def check_grads(accumulated, whole, name, msg):
            bounds = 1.5 * np.asarray(grad_errors(whole, exact[name])) + 2**-8
            errors = np.asarray(grad_errors(accumulated, exact[name]))
            assert np.all(errors <= bounds), (msg, errors, bounds)

        assert_matches_whole_batch(cfg, policy, check_grads)
        assert_matches_whole_batch(cfg, policy, check_grads, carry_compute_params=True)
        """
    )


@pytest.mark.timeout(900)
def test_qb_thresholds_are_the_mean_over_microbatches():
    _run(
        """
        policy = jmp.get_policy("f32")
        for qb_balance in ("global", "per_node"):
            cfg = dataclasses.replace(BASE, qb_balance=qb_balance)
            mesh = MESHES["replica2_data2_expert2"]
            k = 4
            state, _ = step(cfg, mesh, policy, microbatches=k)
            # Microbatch i holds rows i, i + k, ...: run each through a whole-batch step on its own.
            expected = np.mean(
                [
                    step(cfg, mesh, policy, microbatches=1, batch=Batch(TOKENS[i::k], LOSS_WEIGHT[i::k], None))[
                        0
                    ].pending_qb_betas
                    for i in range(k)
                ],
                axis=0,
            )
            assert jax.typeof(state.pending_qb_betas).sharding.spec == june_train.qb_betas_spec(cfg), qb_balance
            np.testing.assert_allclose(state.pending_qb_betas, expected, rtol=1e-6, atol=1e-7, err_msg=qb_balance)
        """
    )


@pytest.mark.timeout(900)
def test_gradients_are_reduced_across_replicas_once_after_the_microbatch_loop():
    # Backward ops carry "transpose(" in their op name. In the whole-batch step the expert weights' gradients are
    # reduced across the replicas in every layer of the backward layer scan; with microbatches, no backward reduction
    # of an array across the replicas runs in any loop, only the forward's loss and routing statistics. The scalar
    # cotangent of the loss weight is a backward reduction that may stay in the loop (it does on GPU), so it is not
    # counted as a weight gradient. After the loop, the reductions across the replicas cover every weight's per-device
    # gradient exactly once: in total its element count, plus a few scalars, fewer than the smallest weight has.
    _run(
        """
        policy = jmp.get_policy("p=f32,c=bfloat16")
        for name, mesh in MESHES.items():
            whole = replica_reductions(compiled_step_hlo(BASE, mesh, policy, microbatches=1), mesh)
            assert any(r.in_loop and "transpose(" in r.op_name for r in whole), (name, whole)
            with jax.set_mesh(mesh):
                params = june_train.initial_state(
                    BASE, optimizer=optax.adam(1e-2), mp=policy, key=jax.random.key(0), ema_beta=None
                ).params
                marked = jax.eval_shape(june_train._mark_batch_reduced, params)
            sizes = [
                int(np.prod(x.sharding.shard_shape(x.shape)))
                for path, x in jax.tree_util.tree_leaves_with_path(marked)
                if getattr(path[-1], "name", None) != "router_bias"
            ]
            for k in (2, 4):
                msg = f"{name} K={k}"
                reductions = replica_reductions(compiled_step_hlo(BASE, mesh, policy, microbatches=k), mesh)
                in_loop = [r for r in reductions if r.in_loop]
                assert in_loop and not any("transpose(" in r.op_name and not r.scalar for r in in_loop), (msg, in_loop)
                after_loop = sum(r.elements for r in reductions if not r.in_loop)
                assert sum(sizes) <= after_loop < sum(sizes) + min(sizes), (msg, after_loop, sum(sizes), reductions)
        """
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "case",
    [
        'BASE, jmp.get_policy("f32"), batch=PADDED_BATCH, z_loss_weight=1e-2',
        'dataclasses.replace(BASE, router_z_loss_coef=0.1), jmp.get_policy("f32"), batch=UNWEIGHTED_MICROBATCH',
        'dataclasses.replace(BASE, remat_mode="save_moe_attn"), jmp.get_policy("p=f32,c=bfloat16")',
        'dataclasses.replace(BASE, remat_mode="save_moe_attn"), jmp.get_policy("p=f32,c=bfloat16"), '
        "carry_compute_params=True",
    ],
    ids=["f32_z_loss_and_padding", "f32_unweighted_microbatch", "bfloat16", "bfloat16_carry_compute_params"],
)
def test_overlapping_the_last_microbatch_gives_the_same_step(case):
    # The per-layer reduction in the last microbatch's backward adds the same float32 accumulator and gradient, casts
    # the sum the same way and reduces it over the same devices as the single reduction after the loop: on CPU the
    # gradients, the loss, the metrics and the next QB thresholds are bit-identical.
    _run(
        f"""
        def check(cfg, policy, **step_args):
            for name, mesh in MESHES.items():
                for k in (2, 4):
                    msg = f"{{name}} K={{k}}"
                    hoisted, hoisted_metrics = step(cfg, mesh, policy, microbatches=k, **step_args)
                    overlapped, metrics = step(
                        cfg, mesh, policy, microbatches=k, overlap_last_microbatch=True, **step_args
                    )
                    assert metrics.keys() == hoisted_metrics.keys(), msg
                    for key in metrics:
                        np.testing.assert_array_equal(metrics[key], hoisted_metrics[key], err_msg=f"{{msg}} {{key}}")
                    for a, b in zip(
                        jax.tree_util.tree_leaves((overlapped.opt_state, overlapped.pending_qb_betas)),
                        jax.tree_util.tree_leaves((hoisted.opt_state, hoisted.pending_qb_betas)),
                        strict=True,
                    ):
                        assert a.dtype == b.dtype and a.sharding == b.sharding, msg
                        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=msg)

        check({case})
        """
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "cfg", ["BASE", "dataclasses.replace(BASE, replicated_grad_max_bytes=0)"], ids=["after_scan", "per_layer"]
)
def test_overlapping_the_last_microbatch_with_one_expert_shard(cfg):
    # With a size-1 expert axis the expert weights are replicated: small enough, the layer scan itself marks them
    # reduced after the scan (replicated_grad_reduction="after_scan"), and they then take the same path as the other
    # weights; larger ones are reduced per layer. Either way the step is bit-identical to the hoisted one.
    _run(
        f"""
        cfg = {cfg}
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=8)
        for k in (2, 4):
            hoisted, hoisted_metrics = step(cfg, mesh, jmp.get_policy("f32"), microbatches=k)
            overlapped, metrics = step(cfg, mesh, jmp.get_policy("f32"), microbatches=k, overlap_last_microbatch=True)
            for key in metrics:
                np.testing.assert_array_equal(metrics[key], hoisted_metrics[key], err_msg=f"K={{k}} {{key}}")
            overlapped_grads, hoisted_grads = (jax.tree_util.tree_leaves(s.opt_state) for s in (overlapped, hoisted))
            for a, b in zip(overlapped_grads, hoisted_grads, strict=True):
                assert a.dtype == b.dtype and a.sharding == b.sharding, k
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"K={{k}}")
        """
    )


@pytest.mark.timeout(900)
def test_overlapping_the_last_microbatch_reduces_the_expert_gradients_per_layer():
    # With overlap_last_microbatch, the expert weights' gradients are reduced across the replicas inside the backward
    # layer scan of the last microbatch, which runs after the microbatch loop: one layer's slice of each per
    # iteration, so over the scan each expert weight once. No other backward reduction of an array runs in a loop.
    # After it, there is exactly one reduction for every other weight's gradient. Compiled without the all-reduce
    # combiner, which would merge them with each other and with the forward's routing statistics.
    _run(
        """
        policy = jmp.get_policy("p=f32,c=bfloat16")
        layer_scan = "jit(train_step)/transpose(jvp(Transformer))/while/body/"
        for name, mesh in MESHES.items():
            with jax.set_mesh(mesh):
                params = june_train.initial_state(
                    BASE, optimizer=optax.adam(1e-2), mp=policy, key=jax.random.key(0), ema_beta=None
                ).params
                marked = jax.eval_shape(june_train._mark_batch_reduced, params)
                per_layer = june_train._per_layer_reduced(params)
            expert, other = [], []
            for (path, x), is_expert in zip(
                jax.tree_util.tree_leaves_with_path(marked), jax.tree_util.tree_leaves(per_layer), strict=True
            ):
                if getattr(path[-1], "name", None) != "router_bias":
                    (expert if is_expert else other).append(int(np.prod(x.sharding.shard_shape(x.shape))))
            assert len(expert) == 3, (name, expert)
            layer_slices = [size // BASE.num_layers for size in expert]
            for k in (2, 4):
                msg = f"{name} K={k}"
                reductions = replica_reductions(
                    compiled_step_hlo(
                        BASE, mesh, policy, microbatches=k, overlap_last_microbatch=True, combine_all_reduces=False
                    ),
                    mesh,
                )
                backward_in_loop = [r for r in reductions if r.in_loop and "transpose(" in r.op_name and not r.scalar]
                assert backward_in_loop and all(r.op_name.startswith(layer_scan) for r in backward_in_loop), (
                    msg,
                    backward_in_loop,
                )
                assert sum(r.elements for r in backward_in_loop) == sum(layer_slices), (msg, backward_in_loop)
                forward = [r for r in reductions if "jvp(" in r.op_name and "transpose(" not in r.op_name]
                assert forward, (msg, reductions)  # The routing statistics, so the op_name test means something.
                after_loop = [r.elements for r in reductions if not r.in_loop and not r.scalar and r not in forward]
                assert sorted(after_loop) == sorted(other), (msg, reductions)
        """
    )


@pytest.mark.timeout(900)
def test_overlapping_the_last_microbatch_keeps_no_copy_of_the_expert_weights():
    # The last microbatch's layer scan routes each layer's expert weights through their reductions inside the layer's
    # checkpoint, so its backward reads the slices of the weights the scan was given. Routed before the checkpoint, the
    # checkpoint's inputs would be the reductions' outputs, which the scan stacks for its backward: a second copy of
    # the expert weights, one more per expert weight in the forward scan's loop state (6 rather than 3 here). With one
    # data shard, so that marking the weights moves no data; wide experts give the expert weights shapes of their own.
    _run(
        r"""
        policy = jmp.get_policy("p=f32,c=bfloat16")
        cfg = dataclasses.replace(BASE, intermediate_dim=24)
        mesh = MESHES["replica4_expert2"]
        last_forward = 'op_name="jit(train_step)/jvp(Transformer)/while/body/'
        with jax.set_mesh(mesh):
            params = june_train.initial_state(
                cfg, optimizer=optax.adam(1e-2), mp=policy, key=jax.random.key(0), ema_beta=None
            ).params
            compute = jax.eval_shape(policy.cast_to_compute, params)
            per_layer = june_train._per_layer_reduced(params)
        expert_shapes = collections.Counter(
            "bf16[" + ",".join(map(str, x.sharding.shard_shape(x.shape))) + "]"
            for x, is_expert in zip(jax.tree.leaves(compute), jax.tree.leaves(per_layer), strict=True)
            if is_expert
        )
        assert sum(expert_shapes.values()) == 3, expert_shapes
        for k in (2, 4):
            hlo = compiled_step_hlo(cfg, mesh, policy, microbatches=k, overlap_last_microbatch=True)
            computations = dict(re.findall(r"(?m)^%?([\w.\-]+) .*\{$\n((?:[ ].*\n?)*)", hlo))
            held = [
                sum(state.count(shape) for shape in expert_shapes)
                for state, body in re.findall(r"= (\(.*?\)) while\(.*?body=%?([\w.\-]+)", hlo)
                if last_forward in computations.get(body, "")
            ]
            assert held == [3], (k, held)
        """
    )


def test_overlapping_the_last_microbatch_needs_two_microbatches():
    with pytest.raises(ValueError, match="at least 2 microbatches"):
        june_train._make_train_step(
            optax.sgd(1e-2),
            jmp.get_policy("f32"),
            z_loss_weight=0.0,
            ema_beta=None,
            microbatches=1,
            overlap_last_microbatch=True,
        )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("remat_mode", ["recompute_all", "save_moe_attn"])
def test_checkpointing_without_prevent_cse_gives_the_same_step(remat_mode):
    # prevent_cse only adds optimization barriers around the checkpointed blocks' saved residuals in the backward:
    # without them the step computes the same values, up to the float32 rounding of a different fusion.
    _run(
        f"""
        cfg = dataclasses.replace(BASE, remat_mode="{remat_mode}")
        policy = jmp.get_policy("f32")
        for name, mesh in MESHES.items():
            msg = f"{{name}} K=2"
            barriers = {{}}
            for prevent_cse in (True, False, None):
                optimizer = optax.adam(1e-2)
                with jax.set_mesh(mesh):
                    state = june_train.initial_state(
                        dataclasses.replace(cfg, remat_prevent_cse=prevent_cse), optimizer=optimizer, mp=policy,
                        key=jax.random.key(0), ema_beta=None,
                    )
                    train_step = june_train._make_train_step(
                        optimizer, policy, z_loss_weight=0.0, ema_beta=None, microbatches=2
                    )
                    barriers[prevent_cse] = train_step.lower(state, BATCH).as_text().count("optimization_barrier")
            # The default (None) drops them in the stacked blocks' layer scan, and keeps them for unstacked blocks,
            # where nothing else stops XLA from CSE-ing the recomputation with the forward.
            assert barriers[None] == barriers[False] < barriers[True], (msg, barriers)
            assert cfg.use_array_stacked_blocks and cfg.remat_prevent_cse is None, msg
            unstacked = dataclasses.replace(
                cfg, use_array_stacked_blocks=False, replicated_grad_reduction="per_layer",
                router_stats_reduction="per_layer",
            )
            for prevent_cse in (True, None):
                optimizer = optax.adam(1e-2)
                with jax.set_mesh(mesh):
                    state = june_train.initial_state(
                        dataclasses.replace(unstacked, remat_prevent_cse=prevent_cse), optimizer=optimizer, mp=policy,
                        key=jax.random.key(0), ema_beta=None,
                    )
                    train_step = june_train._make_train_step(
                        optimizer, policy, z_loss_weight=0.0, ema_beta=None, microbatches=2
                    )
                    barriers["unstacked", prevent_cse] = train_step.lower(state, BATCH).as_text().count(
                        "optimization_barrier"
                    )
            assert barriers["unstacked", None] == barriers["unstacked", True], (msg, barriers)
            kept, kept_metrics = step(dataclasses.replace(cfg, remat_prevent_cse=True), mesh, policy, microbatches=2)
            dropped, metrics = step(cfg, mesh, policy, microbatches=2)
            assert metrics.keys() == kept_metrics.keys(), msg
            for key in ("train/loss", "train/cross_entropy_loss", "train/router/aux_loss_weighted"):
                np.testing.assert_allclose(metrics[key], kept_metrics[key], rtol=1e-6, err_msg=f"{{msg}} {{key}}")
            np.testing.assert_array_equal(
                metrics["train/router/routing_counts_per_layer"],
                kept_metrics["train/router/routing_counts_per_layer"],
                err_msg=msg,
            )
            assert max(grad_errors(dropped, kept)) <= 1e-5, msg
        """
    )
