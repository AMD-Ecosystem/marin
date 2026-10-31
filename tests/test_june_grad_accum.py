# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient accumulation over microbatches in the June train step, on eight CPU devices with the ring_dedup MoE.

A train step with ``microbatches=K`` gives the loss, gradients and routing counts of the whole batch, on a mesh with
replicas and expert parallelism (replica_dcn=4 x expert=2) and on one that also shards the parameters over ``data``
(replica_dcn=2 x data=2 x expert=2). The capacity factors are high enough that no expert overflows, since capacity
applies per microbatch. This holds with the logit and router z-losses on, with padding that leaves the microbatches
unequal valid tokens, and with a microbatch that carries no loss weight at all. The next step's QB thresholds are the
mean of those of the microbatches. Every weight's gradient is reduced across the replicas once per step, after the
microbatch loop.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

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


def step(cfg, mesh, policy, *, microbatches, carry_compute_params=False, batch=BATCH, z_loss_weight=0.0):
    optimizer = keep_gradients()
    with jax.set_mesh(mesh):
        state = june_train.initial_state(
            cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None,
            carry_compute_params=carry_compute_params,
        )
        train_step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=z_loss_weight, ema_beta=None, microbatches=microbatches
        )
        state, metrics, _ = train_step(state, batch)
    return state, jax.device_get(metrics)


def compiled_step_hlo(cfg, mesh, policy, *, microbatches):
    optimizer = optax.adam(1e-2)
    with jax.set_mesh(mesh):
        state = june_train.initial_state(cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None)
        train_step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=0.0, ema_beta=None, microbatches=microbatches
        )
        return train_step.lower(state, BATCH).compile().as_text()


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
