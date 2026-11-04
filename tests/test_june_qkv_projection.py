# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June attention's fused q/k/v projection (``qkv_projection="fused"`` / ``"fused_with_gate"``) against the
separate one.

The fused projections concatenate ``w_q``, ``w_k``, ``w_v`` (and ``attn_gate``) in the forward and run one einsum, so
each output column is the same dot product over the hidden dim: the outputs and gradients match the separate
projections up to the accumulation order of the GEMM, in float32 to a few ulps and in bfloat16 to its rounding. The
parameters stay separate, as the optimizer sees them. The train step on eight CPU devices is in
test_june_grad_accum.py.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import (
    CausalSelfAttention,
    GrugModelConfig,
    Transformer,
    _mark_reduced,
    fused_projection_weights,
)

_CFG = GrugModelConfig(
    vocab_size=32,
    hidden_dim=64,
    intermediate_dim=8,
    shared_expert_intermediate_dim=8,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=16,
    max_seq_len=16,
    sliding_window=8,
    capacity_factor=8.0,
    attention_implementation="reference",
    use_array_stacked_blocks=True,
    disable_pko=True,
)
_FUSED = ("fused", "fused_with_gate")


def _attention(projection: str, dtype) -> CausalSelfAttention:
    attn = CausalSelfAttention.init(dataclasses.replace(_CFG, qkv_projection=projection), key=jax.random.key(0))
    # attn_gate starts at zero; give it values so its gradient path is exercised.
    gate = jax.random.normal(jax.random.key(5), attn.attn_gate.shape) * 0.1
    attn = dataclasses.replace(attn, attn_gate=gate)
    return jax.tree.map(lambda a: a.astype(dtype) if isinstance(a, jax.Array) else a, attn)


def _value_and_grads(projection: str, dtype, use_pko: bool):
    x = jax.random.normal(jax.random.key(1), (2, _CFG.max_seq_len, _CFG.hidden_dim)).astype(dtype)
    g = jax.random.normal(jax.random.key(2), x.shape).astype(dtype)
    mask = AttentionMask(is_causal=True)

    def f(attn, values):
        return attn(values, mask, use_pko=use_pko)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        attn = _attention(projection, dtype)
        y, vjp = jax.vjp(jax.jit(f), attn, x)
        d_attn, dx = vjp(g)
    grads = {name: getattr(d_attn, name) for name in ("w_q", "w_k", "w_v", "w_o", "attn_gate")}
    return y, dx, grads


def _scaled_error(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-30))


@pytest.mark.parametrize("use_pko", [False, True])
@pytest.mark.parametrize("projection", _FUSED)
def test_fused_projection_matches_separate_in_float32(projection, use_pko):
    y_sep, dx_sep, grads_sep = _value_and_grads("separate", jnp.float32, use_pko)
    y, dx, grads = _value_and_grads(projection, jnp.float32, use_pko)
    # Only the GEMM's float32 accumulation order differs.
    tolerance = 16 * float(jnp.finfo(jnp.float32).eps)
    assert _scaled_error(y, y_sep) <= tolerance
    assert _scaled_error(dx, dx_sep) <= tolerance
    for name, grad in grads.items():
        assert grad.shape == grads_sep[name].shape, name
        assert _scaled_error(grad, grads_sep[name]) <= tolerance, name


@pytest.mark.parametrize("projection", _FUSED)
def test_fused_projection_matches_separate_in_bfloat16(projection):
    # Each path is as close to the float32 result as the other, up to one bfloat16 rounding.
    y_ref, dx_ref, grads_ref = _value_and_grads("separate", jnp.float32, False)
    y_sep, dx_sep, grads_sep = _value_and_grads("separate", jnp.bfloat16, False)
    y, dx, grads = _value_and_grads(projection, jnp.bfloat16, False)
    assert y.dtype == dx.dtype == jnp.bfloat16
    bf16_eps = float(jnp.finfo(jnp.bfloat16).eps)
    for name, new, old, ref in (
        ("y", y, y_sep, y_ref),
        ("dx", dx, dx_sep, dx_ref),
        *((name, grads[name], grads_sep[name], grads_ref[name]) for name in grads),
    ):
        assert _scaled_error(new, ref) <= 1.5 * _scaled_error(old, ref) + bf16_eps, name


@pytest.mark.parametrize("projection", ["separate", *_FUSED])
def test_projection_einsums(projection):
    # One dot over the concatenated weights (with or without the gate) replaces the separate ones, in the forward
    # and, transposed, in the backward: two dots (input and weight gradient) per forward dot, and w_o's three.
    x = jnp.ones((2, _CFG.max_seq_len, _CFG.hidden_dim))
    mask = AttentionMask(is_causal=True)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        attn = _attention(projection, jnp.float32)
        forward = jax.make_jaxpr(lambda a, v: a._project(v))(attn, x)
        backward = jax.make_jaxpr(jax.grad(lambda a, v: jnp.sum(a(v, mask)), argnums=(0, 1)))(attn, x)
    projection_dots = {"separate": 4, "fused": 2, "fused_with_gate": 1}[projection]
    dots = [e for e in forward.jaxpr.eqns if e.primitive.name == "dot_general"]
    assert len(dots) == projection_dots
    if projection != "separate":
        width = sum(getattr(attn, name).shape[1] for name in fused_projection_weights(attn.cfg))
        assert dots[0].outvars[0].aval.shape[-1] == width
    # The reference attention's own two dots (scores, values) and their four transposes are in the backward too.
    assert str(backward).count("dot_general") == 3 * (projection_dots + 1) + 6


@pytest.mark.parametrize("projection", ["separate", *_FUSED])
def test_parameters_stay_separate(projection):
    # The optimizer (MuonH: Newton-Schulz per matrix) sees the same parameters whatever the projection.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        base = Transformer.init(_CFG, key=jax.random.key(0))
        model = Transformer.init(dataclasses.replace(_CFG, qkv_projection=projection), key=jax.random.key(0))

    def parameters(m):
        return [(jax.tree_util.keystr(path), a.shape, a.dtype) for path, a in jax.tree_util.tree_leaves_with_path(m)]

    assert parameters(model) == parameters(base)


@pytest.mark.parametrize(
    ("projection", "max_bytes", "expected"),
    [
        # Per layer in float32: w_q 16 KiB, w_k and w_v 8 KiB each, attn_gate 1 KiB.
        ("separate", 9 << 10, {"w_q": False, "w_k": True, "w_v": True, "attn_gate": True}),
        ("fused", 9 << 10, {"w_q": False, "w_k": False, "w_v": False, "attn_gate": True}),
        ("fused_with_gate", 9 << 10, {"w_q": False, "w_k": False, "w_v": False, "attn_gate": False}),
        ("fused", 32 << 10, {"w_q": True, "w_k": True, "w_v": True, "attn_gate": True}),
        ("fused_with_gate", 33 << 10, {"w_q": True, "w_k": True, "w_v": True, "attn_gate": True}),
        ("fused_with_gate", 32 << 10, {"w_q": False, "w_k": False, "w_v": False, "attn_gate": False}),
    ],
)
def test_fused_weights_are_marked_reduced_together(projection, max_bytes, expected):
    cfg = dataclasses.replace(_CFG, qkv_projection=projection, replicated_grad_reduction="after_scan")
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        stacked = Transformer.init(cfg, key=jax.random.key(0)).stacked_blocks.stacked
        marked = _mark_reduced(stacked, max_bytes)
        reduced = {name: bool(jax.typeof(getattr(marked.attn, name)).sharding.spec.reduced) for name in expected}
    assert reduced == expected
