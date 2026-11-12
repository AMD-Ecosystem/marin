# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""ring_dedup's combine and dispatch backward as GEMM row-scatter epilogues, bit for bit against the unfused path."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.nn.ragged_dot import ragged_dot, ragged_dot_accumulating_grad

from levanter.grug._moe.ep_common import _assignment_sources, _combine_gather_sum, _combine_gather_sum_impl
from levanter.grug._moe.ep_ring import _dispatch_rows, _gather_sum_slots, _select_local_assignments
from levanter.grug._moe.flydsl_gather import gathered_up_projection
from levanter.grug._moe.pallas_gather_sum import gather_sum_rows_into
from levanter.grug._moe.scatter_epilogue import dispatch_up, down_combine, scatter_plan
from levanter.utils.jax_utils import is_rocm_backend


def _bits(x: jax.Array) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def routing(key, received_rows: int, topk: int, local_experts: int, capacity: int, hit_prob: float):
    """A received-row routing as ring_dedup's expert side builds it, from random per-slot local expert ids.

    Each slot of a received row carries a local expert with probability ``hit_prob`` (else -1), so rows have 0..K
    kept slots; ``capacity`` below the demand makes receiver drops. Returns ``(group_sizes, picked, dispatch_valid,
    dispatch_row, position, weights)`` as in ``_ring_dedup_block``.
    """
    k_hit, k_expert, k_weight = jax.random.split(key, 3)
    assignments = received_rows * topk
    hit = jax.random.bernoulli(k_hit, hit_prob, (assignments,))
    local_expert = jnp.where(hit, jax.random.randint(k_expert, (assignments,), 0, local_experts), 0)
    group_sizes, _, picked, dispatch_valid = _select_local_assignments(
        local_expert,
        hit,
        local_experts=local_experts,
        logical_capacity=capacity,
        physical_capacity=capacity,
        selection="rank_scatter",
    )
    picked = jnp.where(dispatch_valid, picked, assignments)
    dispatch_row = jnp.where(dispatch_valid, picked // topk, received_rows)
    position = _assignment_sources(picked, send_size=assignments).reshape(received_rows, topk)
    weights = jnp.where(hit, jax.random.uniform(k_weight, (assignments,)), 0).reshape(received_rows, topk)
    return group_sizes, picked, dispatch_valid, dispatch_row, position, weights.astype(jnp.float32)


def test_scatter_plan_marks_single_slot_rows():
    # Received rows: 0 has slots 0 and 2 kept, 1 has slot 1, 2 has none, 3 has slot 3; capacity 4 dispatch rows.
    capacity, topk = 4, 4
    picked = jnp.asarray([0 * 4 + 0, 0 * 4 + 2, 1 * 4 + 1, 3 * 4 + 3], jnp.int32)
    dispatch_valid = jnp.ones((capacity,), bool)
    position = _assignment_sources(picked, send_size=16).reshape(4, topk)
    dst_row, write = scatter_plan(position, picked, dispatch_valid)
    np.testing.assert_array_equal(dst_row, [-1, -1, 1, 3])
    np.testing.assert_array_equal(write, [True, False, True, False])


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("weighted", [False, True])
def test_gather_sum_rows_into_matches_xla_bitwise(dtype, weighted):
    received_rows, topk, capacity, hidden = 70, 4, 160, 64
    _, picked, dispatch_valid, _, position, weights = routing(jax.random.key(0), received_rows, topk, 6, capacity, 0.5)
    rows = jax.random.normal(jax.random.key(1), (capacity, hidden)).astype(dtype)
    rows = rows.at[::7].set(-0.0)
    _, write = scatter_plan(position, picked, dispatch_valid)
    dst = jnp.full((received_rows, hidden), 3.0, dtype)
    if weighted:
        expected = jax.jit(lambda r: _combine_gather_sum_impl(r, position.reshape(-1), weights, dtype))(rows)
    else:
        expected = jax.jit(_gather_sum_slots)(rows, position)
    actual = jax.jit(
        lambda d, r: gather_sum_rows_into(d, r, position, write, weights if weighted else None, interpret=True)
    )(dst, rows)
    written = np.asarray(write)
    assert written.any() and not written.all()
    np.testing.assert_array_equal(_bits(actual)[written], _bits(expected)[written])
    np.testing.assert_array_equal(_bits(actual)[~written], _bits(dst)[~written])


@pytest.mark.timeout(600)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ring_dedup_scatter_epilogue_matches_off_bitwise(dtype):
    """On 8 expert shards with sender and receiver drops, ring_dedup with the scatter epilogue's data flow (XLA GEMMs,
    the Pallas gather-sum in the interpreter) gives the outputs and gradients of the unfused path, bit for bit, also
    with two token blocks and with the Pallas sender-side gather."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    script = """
        import math

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.grug.grug_moe import moe_mlp

        dtype = jnp.__DTYPE__
        shards, tokens_per_shard, hidden, inter, experts, topk = 8, 24, 32, 16, 32, 4
        tokens = shards * tokens_per_shard
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, shards, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        keys = jax.random.split(jax.random.key(0), 7)
        x = jax.random.normal(keys[0], (tokens, hidden)).astype(dtype)
        selected = jnp.argsort(jax.random.uniform(keys[1], (tokens, experts)), axis=1)[:, :topk].astype(jnp.int32)
        combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, topk)), axis=-1)
        w_up_gate = (jax.random.normal(keys[3], (experts, hidden, 2 * inter)) / math.sqrt(hidden)).astype(dtype)
        w_down = (jax.random.normal(keys[4], (experts, inter, hidden)) / math.sqrt(inter)).astype(dtype)
        cotangent = jax.random.normal(keys[5], (tokens, hidden)).astype(dtype)
        keep_fraction = np.repeat(0.3 + 0.7 * np.arange(shards) / (shards - 1), tokens_per_shard)
        token_valid = np.asarray(jax.random.uniform(keys[6], (tokens,))) < keep_fraction

        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        args = (
            jax.device_put(x, batch),
            jax.device_put(combine_weights, batch),
            jax.device_put(w_up_gate, expert_sharding),
            jax.device_put(w_down, expert_sharding),
        )
        selected_sharded = jax.device_put(selected, batch)
        token_valid_sharded = jax.device_put(jnp.asarray(token_valid), NamedSharding(mesh, P(("data", "expert"))))
        cotangent_sharded = jax.device_put(cotangent, batch)

        def run(scatter_epilogue, gather_sum, token_blocks):
            def loss(x, cw, w_up_gate, w_down):
                out, counts = moe_mlp(
                    x,
                    selected_sharded,
                    cw,
                    w_up_gate,
                    w_down,
                    token_valid=token_valid_sharded,
                    activation=jax.nn.silu,
                    implementation="ring_dedup",
                    mesh=mesh,
                    capacity_factor=0.3,
                    pooled_transport_capacity_factor=0.5,
                    report_capacity_overflow=True,
                    token_blocks=token_blocks,
                    gather_sum=gather_sum,
                    scatter_epilogue=scatter_epilogue,
                )
                return jnp.sum((out * cotangent_sharded).astype(jnp.float32)), (out, counts)

            with jax.set_mesh(mesh):
                grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                (_, (out, counts)), grads = grad_fn(*args)
            return out, counts, grads

        def bits(a):
            a = np.asarray(a)
            return a.view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)

        for token_blocks, gather_sum in ((1, "xla"), (2, "pallas")):
            ref_out, ref_counts, ref_grads = run("off", gather_sum, token_blocks)
            out, counts, grads = run("reference", gather_sum, token_blocks)
            assert int(ref_counts.sender_dropped) > 0 and int(ref_counts.receiver_dropped) > 0
            assert int(counts.receiver_dropped) == int(ref_counts.receiver_dropped)
            np.testing.assert_array_equal(bits(out), bits(ref_out))
            for a, e in zip(grads, ref_grads, strict=True):
                assert np.any(np.asarray(e, np.float32) != 0)
                np.testing.assert_array_equal(bits(a), bits(e))
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script.replace("__DTYPE__", dtype))],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _unfused(gather_sum):
    """The unfused combine and dispatch + up projection of ``_ring_dedup_block`` (jax-flydsl GEMMs)."""

    def combine(h, w2, group_sizes, position, picked, weights):
        y = ragged_dot(h, w2, group_sizes, implementation="flydsl")
        return _combine_gather_sum(y, position.reshape(-1), picked, weights, y.dtype)

    def up(received, dispatch_row, dispatch_valid, w13, group_sizes, position):
        x = _dispatch_rows(received, dispatch_row, dispatch_valid, position, gather_sum)
        return ragged_dot(x, w13, group_sizes, implementation="flydsl")

    return combine, up


@pytest.mark.parametrize("gather_sum", ["xla", "pallas"])
def test_flydsl_epilogues_match_unfused_bitwise_on_rocm(gather_sum):
    if not is_rocm_backend() or importlib.util.find_spec("jax_flydsl") is None:
        pytest.skip("needs a ROCm GPU and jax-flydsl")
    # One expert shard of the June MoE at 1/8 scale: 14988 received rows, 16384 dispatch rows in 32 experts.
    received_rows, topk, experts, capacity, hidden, inter = 14988, 4, 32, 16384, 2560, 1280
    group_sizes, picked, dispatch_valid, dispatch_row, position, weights = routing(
        jax.random.key(3), received_rows, topk, experts, capacity, 0.3
    )
    dst_row, write = scatter_plan(position, picked, dispatch_valid)
    keys = jax.random.split(jax.random.key(4), 6)
    h = jax.random.normal(keys[0], (capacity, inter)).astype(jnp.bfloat16)
    w2 = (jax.random.normal(keys[1], (experts, inter, hidden)) / 32).astype(jnp.bfloat16)
    received = jax.random.normal(keys[2], (received_rows, hidden)).astype(jnp.bfloat16)
    w13 = (jax.random.normal(keys[3], (experts, hidden, 2 * inter)) / 48).astype(jnp.bfloat16)
    combine_ct = jax.random.normal(keys[4], (received_rows, hidden)).astype(jnp.bfloat16)
    up_ct = jax.random.normal(keys[5], (capacity, 2 * inter)).astype(jnp.bfloat16)
    safe_row = jnp.minimum(dispatch_row, received_rows - 1)
    combine, up = _unfused(gather_sum)

    def value_and_grads(f, ct):
        def run(*args):
            out, vjp = jax.vjp(f, *args)
            return (out, *vjp(ct))

        return jax.jit(run)

    def assert_same_bits(actual, expected):
        for a, e in zip(actual, expected, strict=True):
            assert np.any(np.asarray(e, np.float32) != 0)
            np.testing.assert_array_equal(_bits(a), _bits(e))

    # Combine: output, d h, d w2 and the routing-weight gradient.
    assert_same_bits(
        value_and_grads(
            lambda h, w2, w: down_combine(h, w2, group_sizes, position, picked, w, dst_row, write, "flydsl"),
            combine_ct,
        )(h, w2, weights),
        value_and_grads(lambda h, w2, w: combine(h, w2, group_sizes, position, picked, w), combine_ct)(h, w2, weights),
    )
    # Dispatch + up projection: output, d received (the dispatch backward) and d w13.
    assert_same_bits(
        value_and_grads(
            lambda r, w: dispatch_up(r, safe_row, dispatch_valid, w, group_sizes, position, dst_row, write, "flydsl"),
            up_ct,
        )(received, w13),
        value_and_grads(lambda r, w: up(r, safe_row, dispatch_valid, w, group_sizes, position), up_ct)(received, w13),
    )


@pytest.mark.parametrize("mode", ["accum", "final"])
@pytest.mark.parametrize("gather_sum", ["xla", "pallas"])
def test_stacked_expert_path_matches_unfused_bitwise_on_rocm(gather_sum, mode, monkeypatch):
    """The epilogues with the dispatch gather inside the up projection's GEMMs (``gather_a``) and the expert weight
    gradients summed in the wgrad kernel, and ``gathered_up_projection`` with the sums, against the unfused path with
    ``ragged_dot_accumulating_grad``: outputs, data gradients and the sums (or, with "final", the weight gradients)."""
    if not is_rocm_backend() or importlib.util.find_spec("jax_flydsl") is None:
        pytest.skip("needs a ROCm GPU and jax-flydsl")
    monkeypatch.setenv("RAGGED_DOT_IMPL", "flydsl")
    received_rows, topk, experts, capacity, hidden, inter, layers = 14988, 4, 32, 16384, 2560, 1280, 2
    group_sizes, picked, dispatch_valid, dispatch_row, position, weights = routing(
        jax.random.key(3), received_rows, topk, experts, capacity, 0.3
    )
    dst_row, write = scatter_plan(position, picked, dispatch_valid)
    keys = jax.random.split(jax.random.key(5), 9)
    h = jax.random.normal(keys[0], (capacity, inter)).astype(jnp.bfloat16)
    w2 = (jax.random.normal(keys[1], (experts, inter, hidden)) / 32).astype(jnp.bfloat16)
    received = jax.random.normal(keys[2], (received_rows, hidden)).astype(jnp.bfloat16)
    w13 = (jax.random.normal(keys[3], (experts, hidden, 2 * inter)) / 48).astype(jnp.bfloat16)
    combine_ct = jax.random.normal(keys[4], (received_rows, hidden)).astype(jnp.bfloat16)
    up_ct = jax.random.normal(keys[5], (capacity, 2 * inter)).astype(jnp.bfloat16)
    acc13 = tuple(jax.random.normal(k, (layers, experts, hidden, inter), jnp.float32) for k in keys[6:8])
    acc2 = jax.random.normal(keys[8], (layers, experts, inter, hidden), jnp.float32)
    layer = jnp.int32(1)
    safe_row = jnp.minimum(dispatch_row, received_rows - 1)
    row_index = jnp.where(dispatch_valid, safe_row, received_rows)
    sums = {"acc": None, "layer": layer, "mode": mode}

    def unfused_up(r, w, acc):
        x = _dispatch_rows(r, safe_row, dispatch_valid, position, gather_sum)
        return ragged_dot_accumulating_grad(x, w, group_sizes, acc, layer, mode)

    def unfused_combine(h, w2, wts, acc):
        y = ragged_dot_accumulating_grad(h, w2, group_sizes, acc, layer, mode)
        if mode == "final":
            return _combine_gather_sum(y, position.reshape(-1), picked, wts, y.dtype)
        y, acc = y
        return _combine_gather_sum(y, position.reshape(-1), picked, wts, y.dtype), acc

    def fused_up(r, w, acc):
        return dispatch_up(
            r,
            safe_row,
            dispatch_valid,
            w,
            group_sizes,
            position,
            dst_row,
            write,
            "flydsl",
            gather_a=True,
            **{**sums, "acc": acc},
        )

    def gathered_up(r, w, acc):
        return gathered_up_projection(r, row_index, position, w, group_sizes, gather_sum, **{**sums, "acc": acc})

    def fused_combine(h, w2, wts, acc):
        return down_combine(
            h, w2, group_sizes, position, picked, wts, dst_row, write, "flydsl", **{**sums, "acc": acc}
        )

    def value_and_grads(f, ct, *args):
        def run(*args):
            out, vjp = jax.vjp(f, *args)
            return out, vjp(ct)

        return jax.tree.leaves(jax.jit(run)(*args))

    def assert_same_bits(actual, expected):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(_bits(a), _bits(e))

    # "accum": the cotangent of the returned sums is the running sum, which the backward updates in place.
    up_cts = up_ct if mode == "final" else (up_ct, acc13)
    combine_cts = combine_ct if mode == "final" else (combine_ct, acc2)
    expected_up = value_and_grads(unfused_up, up_cts, received, w13, acc13)
    assert any(np.any(np.asarray(e, np.float32) != 0) for e in expected_up)
    assert_same_bits(value_and_grads(fused_up, up_cts, received, w13, acc13), expected_up)
    assert_same_bits(value_and_grads(gathered_up, up_cts, received, w13, acc13), expected_up)
    assert_same_bits(
        value_and_grads(fused_combine, combine_cts, h, w2, weights, acc2),
        value_and_grads(unfused_combine, combine_cts, h, w2, weights, acc2),
    )
