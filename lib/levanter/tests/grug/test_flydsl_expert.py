# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``ring_filtered``'s FlyDSL expert kernels (fork overlay) against its own path, bit for bit.

On CPU the kernels run in their "reference" form: the same data flow (gathered A rows, scatter epilogues plus the
gather-sum of the other rows, in-kernel gradient sums) with XLA standing in for the jax-flydsl GEMMs, so these tests
check the wiring and the arithmetic the kernels must reproduce. The kernels themselves need an AMD Instinct GPU.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe import flydsl_expert
from levanter.grug._moe.ep_common import _assignment_sources, _combine_gather_sum_impl
from levanter.grug._moe.ep_ring import _gather_sum_slots, _select_local_assignments
from levanter.grug._moe.grad_accum import ExpertGradAccumulator
from levanter.grug._moe.pallas_gather_sum import gather_sum_rows_into
from levanter.grug.grug_moe import moe_mlp
from levanter.testing.devices import use_test_devices

SHARDS, TOKENS_PER_SHARD, HIDDEN, INTER, EXPERTS, TOPK, LAYERS = 4, 32, 32, 16, 16, 4, 3


def _bits(x) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def test_scatter_plan_marks_single_slot_rows():
    # Received rows: 0 has slots 0 and 2 kept, 1 has slot 1, 2 has none, 3 has slot 3; four dispatch rows.
    picked = jnp.asarray([0 * 4 + 0, 0 * 4 + 2, 1 * 4 + 1, 3 * 4 + 3], jnp.int32)
    position = _assignment_sources(picked, send_size=16).reshape(4, 4)
    dst_row, write = flydsl_expert.scatter_plan(position, picked, jnp.ones((4,), bool))
    np.testing.assert_array_equal(dst_row, [-1, -1, 1, 3])
    np.testing.assert_array_equal(write, [True, False, True, False])


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("weighted", [False, True])
def test_gather_sum_rows_into_matches_xla_bitwise(dtype, weighted):
    received_rows, topk, capacity, hidden = 70, 4, 160, 64
    keys = jax.random.split(jax.random.key(0), 4)
    hit = jax.random.bernoulli(keys[0], 0.5, (received_rows * topk,))
    local_expert = jnp.where(hit, jax.random.randint(keys[1], (received_rows * topk,), 0, 6), 0)
    _, _, picked, valid = _select_local_assignments(
        local_expert, hit, local_experts=6, logical_capacity=capacity, physical_capacity=capacity
    )
    picked = jnp.where(valid, picked, received_rows * topk)
    position = _assignment_sources(picked, send_size=received_rows * topk).reshape(received_rows, topk)
    weights = jax.random.uniform(keys[2], (received_rows, topk))
    rows = jax.random.normal(keys[3], (capacity, hidden)).astype(dtype).at[::7].set(-0.0)
    _, write = flydsl_expert.scatter_plan(position, picked, valid)
    dst = jnp.full((received_rows, hidden), 3.0, dtype)
    if weighted:
        expected = jax.jit(lambda r: _combine_gather_sum_impl(r, position.reshape(-1), weights, dtype))(rows)
    else:
        expected = jax.jit(_gather_sum_slots)(rows, position)

    actual = jax.jit(lambda d, r: gather_sum_rows_into(d, r, position, write, weights if weighted else None))(
        dst, rows
    )

    written = np.asarray(write)
    assert written.any() and not written.all()
    np.testing.assert_array_equal(_bits(actual)[written], _bits(expected)[written])
    np.testing.assert_array_equal(np.asarray(actual)[~written], np.asarray(dst)[~written])


def _problem(dtype):
    """Tokens with uneven padding and tight capacities (sender and receiver drops) on 4 expert shards."""
    tokens = SHARDS * TOKENS_PER_SHARD
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.normal(keys[0], (tokens, HIDDEN)).astype(dtype)
    selected = jnp.argsort(jax.random.uniform(keys[1], (tokens, EXPERTS)), axis=1)[:, :TOPK].astype(jnp.int32)
    combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, TOPK)), axis=-1).astype(dtype)
    w_up_gate = (jax.random.normal(keys[3], (EXPERTS, HIDDEN, 2 * INTER)) / math.sqrt(HIDDEN)).astype(dtype)
    w_down = (jax.random.normal(keys[4], (EXPERTS, INTER, HIDDEN)) / math.sqrt(INTER)).astype(dtype)
    cotangent = jax.random.normal(keys[5], (tokens, HIDDEN)).astype(dtype)
    keep_fraction = np.repeat(0.3 + 0.7 * np.arange(SHARDS) / (SHARDS - 1), TOKENS_PER_SHARD)
    token_valid = jnp.asarray(np.asarray(jax.random.uniform(keys[6], (tokens,))) < keep_fraction)
    return x, selected, combine_weights, w_up_gate, w_down, cotangent, token_valid


def _sharded(mesh, problem):
    x, selected, combine_weights, w_up_gate, w_down, cotangent, token_valid = problem
    batch = NamedSharding(mesh, P(("data", "expert"), None))
    experts = NamedSharding(mesh, P("expert", None, None))
    return (
        jax.device_put(x, batch),
        jax.device_put(selected, batch),
        jax.device_put(combine_weights, batch),
        jax.device_put(w_up_gate, experts),
        jax.device_put(w_down, experts),
        jax.device_put(cotangent, batch),
        jax.device_put(token_valid, NamedSharding(mesh, P(("data", "expert")))),
    )


def _mesh(devices):
    return Mesh(
        np.asarray(devices).reshape(1, SHARDS, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 3,
    )


def _moe(x, selected, cw, w_up_gate, w_down, token_valid, grad_acc=None):
    return moe_mlp(
        x,
        selected,
        cw,
        w_up_gate,
        w_down,
        token_valid=token_valid,
        implementation="ring_filtered",
        capacity_factor=0.6,
        pooled_transport_capacity_factor=0.6,
        report_capacity_overflow=True,
        grad_acc=grad_acc,
    )


@pytest.mark.timeout(120)
@pytest.mark.multidevice
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ring_filtered_flydsl_expert_kernels_match_its_own_path_bitwise(monkeypatch, dtype):
    """With sender and receiver drops and uneven padding, the gathered-A up projection and the scatter-epilogue
    combine and dispatch backward give ring_filtered's outputs, drop counts and gradients bit for bit."""
    with use_test_devices(SHARDS) as devices:
        mesh = _mesh(devices)
        x, selected, cw, w_up_gate, w_down, cotangent, token_valid = _sharded(mesh, _problem(getattr(jnp, dtype)))

        def run(gemm):
            monkeypatch.setattr(flydsl_expert, "expert_gemm", lambda: gemm)

            def loss(x, cw, w_up_gate, w_down):
                out, counts = _moe(x, selected, cw, w_up_gate, w_down, token_valid)
                return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), (out, counts)

            with jax.set_mesh(mesh):
                (_, (out, counts)), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))(
                    x, cw, w_up_gate, w_down
                )
            return out, counts, grads

        out, counts, grads = run(None)
        kernel_out, kernel_counts, kernel_grads = run("reference")

    assert int(counts.sender_dropped) > 0 and int(counts.receiver_dropped) > 0
    assert (int(kernel_counts.sender_dropped), int(kernel_counts.receiver_dropped)) == (
        int(counts.sender_dropped),
        int(counts.receiver_dropped),
    )
    np.testing.assert_array_equal(_bits(kernel_out), _bits(out))
    for name, got, want in zip(("x", "combine_weights", "w_up_gate", "w_down"), kernel_grads, grads, strict=True):
        np.testing.assert_array_equal(_bits(got), _bits(want), err_msg=name)


@pytest.mark.timeout(120)
@pytest.mark.multidevice
@pytest.mark.parametrize("mode", ["init", "accum", "final"])
def test_ring_filtered_folds_expert_gradients_into_the_accumulator(monkeypatch, mode):
    """The expert weight gradients go into one layer of the float32 sums: "init" writes them there, "accum" adds them
    to the running sums, "final" hands each weight its running sum plus its gradient. Other layers are untouched, and
    the token and routing-weight gradients are those without the accumulator."""
    layer = 1
    with use_test_devices(SHARDS) as devices:
        mesh = _mesh(devices)
        x, selected, cw, w_up_gate, w_down, cotangent, token_valid = _sharded(mesh, _problem(jnp.bfloat16))
        acc_sharding = NamedSharding(mesh, P(None, None, "expert", None, None))
        keys = jax.random.split(jax.random.key(1), 3)
        sums = tuple(
            jax.device_put(jax.random.normal(k, (LAYERS, 1, *shape), jnp.float32), acc_sharding)
            for k, shape in zip(
                keys, ((EXPERTS, HIDDEN, INTER), (EXPERTS, HIDDEN, INTER), (EXPERTS, INTER, HIDDEN)), strict=True
            )
        )

        def loss(x, cw, w_up_gate, w_down):
            out, _ = _moe(x, selected, cw, w_up_gate, w_down, token_valid)
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

        def loss_into(x, cw, w_up_gate, w_down, acc):
            (out, _), acc = _moe(x, selected, cw, w_up_gate, w_down, token_valid, acc.at_layer(jnp.int32(layer)))
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), acc.without_layer()

        def with_sums(*args):
            acc = ExpertGradAccumulator(*sums, mode=mode)
            if mode == "final":
                return (*jax.grad(lambda *a: loss_into(*a, acc)[0], argnums=(0, 1, 2, 3))(*args), None)
            (_, _), vjp = jax.vjp(lambda *a: loss_into(*a), *args, acc)
            *grads, acc_ct = vjp((jnp.ones((), jnp.float32), acc))
            return (*grads, acc_ct)

        monkeypatch.setattr(flydsl_expert, "expert_gemm", lambda: "reference")
        with jax.set_mesh(mesh):
            plain = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(x, cw, w_up_gate, w_down)
            d_x, d_cw, d_w_up_gate, d_w_down, acc_ct = jax.jit(with_sums)(x, cw, w_up_gate, w_down)

    np.testing.assert_array_equal(_bits(d_x), _bits(plain[0]))
    np.testing.assert_array_equal(_bits(d_cw), _bits(plain[1]))
    plain_gate, plain_up = np.split(np.asarray(plain[2]), 2, axis=-1)
    gradients = (plain_gate, plain_up, np.asarray(plain[3]))
    if mode == "final":
        expected = [(np.asarray(s)[layer, 0] + g.astype(np.float32)).astype(g.dtype) for s, g in zip(sums, gradients)]
        got_gate, got_up = np.split(np.asarray(d_w_up_gate), 2, axis=-1)
        for got, want in zip((got_gate, got_up, np.asarray(d_w_down)), expected, strict=True):
            np.testing.assert_array_equal(_bits(got), _bits(want))
        return
    assert not np.any(np.asarray(d_w_up_gate)) and not np.any(np.asarray(d_w_down))
    for s, g, got in zip(sums, gradients, (acc_ct.w_gate, acc_ct.w_up, acc_ct.w_down), strict=True):
        want = np.asarray(s).copy()
        base = 0.0 if mode == "init" else want[layer, 0]
        want[layer, 0] = base + g.astype(np.float32)
        np.testing.assert_array_equal(np.asarray(got), want)
