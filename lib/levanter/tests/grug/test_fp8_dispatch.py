# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``ring_filtered``'s fp8 forward dispatch (fork overlay, lossy prototype).

The fp8 transport only moves rows, so ``unfused`` at ``x`` must be the bfloat16 dispatch at ``x``'s quantized and
dequantized rows, bit for bit, values and gradients (straight-through backward). ``fused`` must equal ``unfused``; on
CPU its fp8 up projection runs in the "reference" form (XLA dequantizes, then the gathered up projection).
"""

import math

import jax
import jax.numpy as jnp
import jax.extend
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe import flydsl_expert
from levanter.grug._moe.common import MOE_REMAT_SAVE_NAMES
from levanter.grug._moe.fp8_dispatch import FP8_MAX, dequantize_rows, quantize_rows
from levanter.grug._moe.grad_accum import ExpertGradAccumulator
from levanter.grug.grug_moe import moe_mlp
from levanter.testing.devices import use_test_devices

SHARDS, TOKENS_PER_SHARD, HIDDEN, INTER, EXPERTS, TOPK, LAYERS = 4, 32, 32, 16, 16, 4, 3


def _bits(x) -> np.ndarray:
    x = np.asarray(x)
    return x.view({1: np.uint8, 2: np.uint16, 4: np.uint32}[x.dtype.itemsize])


def _host_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The quantization with ml_dtypes on the host: s = amax * f32(1 / 448), q = e4m3fn(clip(x / s)) rounded to nearest even."""
    values = x.astype(np.float32)
    amax = np.abs(values).max(axis=1)
    scale = np.where(amax > 0, amax * np.float32(1 / 448), np.float32(1)).astype(np.float32)
    q = np.clip(values / scale[:, None], -448, 448).astype(ml_dtypes.float8_e4m3fn)
    return q.view(np.uint8), scale


def _rows_to_quantize() -> np.ndarray:
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((64, 256)).astype(np.float32)
    rows[1] = 0.0
    rows[2, 7] = 50.0  # an outlier channel
    rows[3] = rng.standard_t(3, 256)  # heavy tails
    rows[4] = np.linspace(-1, 1, 256)
    rows[5, :] = 1e-3
    rows[5, 0] = 448e-3  # amax exactly 448 steps of the other values
    rows[6] *= 1e4
    rows[7] *= 1e-20
    return rows


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_quantize_rows_matches_the_host_reference_bitwise(dtype):
    x = jnp.asarray(_rows_to_quantize()).astype(dtype)
    bits, scale = jax.jit(quantize_rows)(x)
    want_bits, want_scale = _host_quantize(np.asarray(x.astype(jnp.float32)))
    np.testing.assert_array_equal(_bits(scale), _bits(want_scale))
    np.testing.assert_array_equal(np.asarray(bits), want_bits)


def test_dequantized_rows_stay_within_the_fp8_error_bound():
    x = jnp.asarray(_rows_to_quantize()).astype(jnp.bfloat16)
    bits, scale = quantize_rows(x)
    y = np.asarray(dequantize_rows(bits, scale, jnp.bfloat16), np.float32)
    values = np.asarray(x, np.float32)
    s = np.asarray(scale)[:, None]
    # Half an e4m3 step (2^-4 relative for normals, 2^-10 * s below the normals) plus the bfloat16 rounding (2^-9).
    bound = 2.0**-4 * np.abs(values) + 2.0**-10 * s + 2.0**-8 * np.abs(values)
    assert np.all(np.abs(y - values) <= bound)
    assert np.all(np.abs(y) <= np.abs(values).max(axis=1, keepdims=True) * (1 + 2.0**-8))
    np.testing.assert_array_equal(y[1], 0.0)
    # Each row's amax is represented exactly (448 * s).
    np.testing.assert_array_equal(np.abs(y).max(axis=1), np.abs(values).max(axis=1))
    assert FP8_MAX == float(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max)


def _problem():
    """Tokens with uneven padding and tight capacities (sender and receiver drops) on 4 expert shards."""
    tokens = SHARDS * TOKENS_PER_SHARD
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.normal(keys[0], (tokens, HIDDEN)).astype(jnp.bfloat16)
    x = x.at[3, 5].set(40.0)  # an outlier channel
    selected = jnp.argsort(jax.random.uniform(keys[1], (tokens, EXPERTS)), axis=1)[:, :TOPK].astype(jnp.int32)
    combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, TOPK)), axis=-1).astype(jnp.bfloat16)
    w_up_gate = (jax.random.normal(keys[3], (EXPERTS, HIDDEN, 2 * INTER)) / math.sqrt(HIDDEN)).astype(jnp.bfloat16)
    w_down = (jax.random.normal(keys[4], (EXPERTS, INTER, HIDDEN)) / math.sqrt(INTER)).astype(jnp.bfloat16)
    cotangent = jax.random.normal(keys[5], (tokens, HIDDEN)).astype(jnp.bfloat16)
    keep_fraction = np.repeat(0.3 + 0.7 * np.arange(SHARDS) / (SHARDS - 1), TOKENS_PER_SHARD)
    token_valid = jnp.asarray(np.asarray(jax.random.uniform(keys[6], (tokens,))) < keep_fraction)
    return x, selected, combine_weights, w_up_gate, w_down, cotangent, token_valid


def _mesh(devices):
    return Mesh(
        np.asarray(devices).reshape(1, SHARDS, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 3,
    )


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


def _qdq(x):
    """``x``'s rows quantized and dequantized: what the fp8 dispatch delivers."""
    bits, scale = quantize_rows(x)
    return dequantize_rows(bits, scale, x.dtype)


def _run(monkeypatch, mesh, problem, *, gemm, fp8_dispatch, qdq_input=False, remat=False):
    """Output, drop counts and the gradients (x, combine weights, w_up_gate, w_down) of ``<out, cotangent>``."""
    monkeypatch.setattr(flydsl_expert, "expert_gemm", lambda: gemm)
    x, selected, cw, w_up_gate, w_down, cotangent, token_valid = problem
    if qdq_input:
        # The gradients are then the ones at the dequantized rows, which straight-through hands to x.
        with jax.set_mesh(mesh):
            x = jax.jit(_qdq)(x)

    def loss(x, cw, w_up_gate, w_down):
        out, counts = moe_mlp(
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
            fp8_dispatch=fp8_dispatch,
        )
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), (out, counts)

    if remat:
        loss = jax.checkpoint(loss, policy=jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES))
    grad_fn = jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)
    with jax.set_mesh(mesh):
        (_, (out, counts)), grads = jax.jit(grad_fn)(x, cw, w_up_gate, w_down)
        jaxpr = jax.make_jaxpr(grad_fn)(x, cw, w_up_gate, w_down)
    return out, counts, grads, jaxpr


def _assert_same_bits(got, want):
    got_out, got_counts, got_grads, _ = got
    want_out, want_counts, want_grads, _ = want
    assert int(want_counts.sender_dropped) > 0 and int(want_counts.receiver_dropped) > 0
    assert (int(got_counts.sender_dropped), int(got_counts.receiver_dropped)) == (
        int(want_counts.sender_dropped),
        int(want_counts.receiver_dropped),
    )
    np.testing.assert_array_equal(_bits(got_out), _bits(want_out))
    for name, g, w in zip(("x", "combine_weights", "w_up_gate", "w_down"), got_grads, want_grads, strict=True):
        np.testing.assert_array_equal(_bits(g), _bits(w), err_msg=name)


@pytest.mark.timeout(240)
@pytest.mark.multidevice
@pytest.mark.parametrize("gemm", [None, "reference"])
def test_unfused_is_the_bf16_dispatch_of_the_dequantized_rows(monkeypatch, gemm):
    """Values and every gradient equal mode off's at the quantized-and-dequantized input, bit for bit: the forward
    moves exactly those rows and the backward is mode off's (straight-through). The output stays close to mode off's
    at the exact input."""
    with use_test_devices(SHARDS) as devices:
        mesh = _mesh(devices)
        problem = _sharded(mesh, _problem())
        unfused = _run(monkeypatch, mesh, problem, gemm=gemm, fp8_dispatch="unfused")
        off_at_qdq = _run(monkeypatch, mesh, problem, gemm=gemm, fp8_dispatch="off", qdq_input=True)
        off = _run(monkeypatch, mesh, problem, gemm=gemm, fp8_dispatch="off")

    _assert_same_bits(unfused, off_at_qdq)
    got, want = np.asarray(unfused[0], np.float64), np.asarray(off[0], np.float64)
    assert np.linalg.norm(got - want) <= 0.05 * np.linalg.norm(want)


@pytest.mark.timeout(240)
@pytest.mark.multidevice
def test_fused_matches_unfused_bitwise(monkeypatch):
    with use_test_devices(SHARDS) as devices:
        mesh = _mesh(devices)
        problem = _sharded(mesh, _problem())
        fused = _run(monkeypatch, mesh, problem, gemm="reference", fp8_dispatch="fused")
        unfused = _run(monkeypatch, mesh, problem, gemm="reference", fp8_dispatch="unfused")
    _assert_same_bits(fused, unfused)


@pytest.mark.timeout(240)
@pytest.mark.multidevice
@pytest.mark.parametrize("mode", ["init", "accum", "final"])
def test_fused_folds_expert_gradients_like_unfused(monkeypatch, mode):
    """With the expert gradient sums, fused and unfused give the same output, token and routing-weight gradients and
    accumulator cotangents."""
    layer = 1
    with use_test_devices(SHARDS) as devices:
        mesh = _mesh(devices)
        x, selected, cw, w_up_gate, w_down, cotangent, token_valid = _sharded(mesh, _problem())
        acc_sharding = NamedSharding(mesh, P(None, None, "expert", None, None))
        keys = jax.random.split(jax.random.key(1), 3)
        sums = tuple(
            jax.device_put(jax.random.normal(k, (LAYERS, 1, *shape), jnp.float32), acc_sharding)
            for k, shape in zip(
                keys, ((EXPERTS, HIDDEN, INTER), (EXPERTS, HIDDEN, INTER), (EXPERTS, INTER, HIDDEN)), strict=True
            )
        )

        def run(fp8_dispatch):
            monkeypatch.setattr(flydsl_expert, "expert_gemm", lambda: "reference")

            def loss_into(x, cw, w_up_gate, w_down, acc):
                (out, _), acc = moe_mlp(
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
                    grad_acc=acc.at_layer(jnp.int32(layer)),
                    fp8_dispatch=fp8_dispatch,
                )
                return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), acc.without_layer()

            def grads(x, cw, w_up_gate, w_down):
                acc = ExpertGradAccumulator(*sums, mode=mode)
                if mode == "final":
                    return jax.grad(
                        lambda x, cw, w_up_gate, w_down: loss_into(x, cw, w_up_gate, w_down, acc)[0],
                        argnums=(0, 1, 2, 3),
                    )(x, cw, w_up_gate, w_down)
                _, vjp = jax.vjp(loss_into, x, cw, w_up_gate, w_down, acc)
                return vjp((jnp.ones((), jnp.float32), acc))

            with jax.set_mesh(mesh):
                return jax.jit(grads)(x, cw, w_up_gate, w_down)

        fused = run("fused")
        unfused = run("unfused")

    for (path, got), (_, want) in zip(
        jax.tree_util.tree_flatten_with_path(fused)[0], jax.tree_util.tree_flatten_with_path(unfused)[0], strict=True
    ):
        np.testing.assert_array_equal(_bits(got), _bits(want), err_msg=jax.tree_util.keystr(path))


def _all_to_alls(jaxpr) -> list[tuple[str, tuple[int, ...]]]:
    """``(dtype, shape)`` of every ``all_to_all`` operand in ``jaxpr`` and its sub-jaxprs, sorted."""
    found = []

    def walk(j):
        for eqn in j.eqns:
            if eqn.primitive.name == "all_to_all":
                aval = eqn.invars[0].aval
                found.append((str(aval.dtype), tuple(aval.shape)))
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        walk(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        walk(sub)

    walk(jaxpr.jaxpr)
    return sorted(found)


@pytest.mark.timeout(240)
@pytest.mark.multidevice
@pytest.mark.parametrize("fp8_dispatch", ["unfused", "fused"])
def test_fp8_dispatch_keeps_the_backward_collectives_under_remat(monkeypatch, fp8_dispatch):
    """Under the MoE remat policy the gradient runs the same all_to_alls as mode off, nothing recomputed: the
    forward's rows go as uint8 and its metadata carries one more column; the backward's are mode off's."""
    with use_test_devices(SHARDS) as devices:
        mesh = _mesh(devices)
        problem = _sharded(mesh, _problem())
        fp8 = _run(monkeypatch, mesh, problem, gemm="reference", fp8_dispatch=fp8_dispatch, remat=True)
        off = _run(monkeypatch, mesh, problem, gemm="reference", fp8_dispatch="off", remat=True)

    got, want = _all_to_alls(fp8[3]), _all_to_alls(off[3])
    # Mode off: metadata, rows and combine forward, and their three transposes.
    assert len(want) == 6, want
    metadata = next(a for a in want if a[0] == "float32")
    rows = next(a for a in want if a[0] == "bfloat16")
    expected = list(want)
    expected.remove(metadata)
    expected.remove(rows)
    expected += [("float32", (*metadata[1][:-1], metadata[1][-1] + 1)), ("uint8", rows[1])]
    assert got == sorted(expected)


def test_fused_needs_the_flydsl_kernels(monkeypatch):
    monkeypatch.setattr(flydsl_expert, "expert_gemm", lambda: None)
    with use_test_devices(SHARDS) as devices:
        mesh = _mesh(devices)
        x, selected, cw, w_up_gate, w_down, _, token_valid = _sharded(mesh, _problem())
        with jax.set_mesh(mesh), pytest.raises(ValueError, match="FlyDSL"):
            moe_mlp(
                x,
                selected,
                cw,
                w_up_gate,
                w_down,
                token_valid=token_valid,
                implementation="ring_filtered",
                pooled_transport_capacity_factor=1.0,
                fp8_dispatch="fused",
            )
