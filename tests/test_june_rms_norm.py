# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June model's custom-VJP RMS norm (``rms_norm_impl="custom_vjp"``) against the optimization-barrier one.

Both compute the same float32 forward expression, ``x * rsqrt(mean(x^2) + eps)``, but the barrier keeps the reduction
and its users in separate kernels, while without it XLA may fuse them (on CPU at d=2560 into one YNNPACK fusion), which
rounds the reduction and ``rsqrt`` differently: the outputs differ by a few float32 ulps, at most one ulp after the cast
to bfloat16. The custom VJP's backward uses the saved inverse RMS ``r`` and ``r^3`` where AD of the barrier path uses
``rsqrt``'s derivative ``-r / (2 (v + eps))``, so the gradients also agree to float32 rounding. Both paths are compared
with a float64 reference, to which they are equally close.
"""

import collections
import contextlib
import dataclasses
import io

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, RMSNorm, Transformer, rms_norm, rms_normalize

SHAPES = [(4, 16), (2, 8, 128), (2, 3, 4, 128), (3, 7, 2560)]
# Largest error of each path against the float64 reference, in units of the output dtype's epsilon times the
# reference's largest magnitude.
_EPS_MULTIPLE = {jnp.float32: 64, jnp.bfloat16: 2}
# Forward difference between the paths: a few float32 ulps, or one bfloat16 ulp after the cast.
_FORWARD_RTOL = {jnp.float32: 4 * float(jnp.finfo(jnp.float32).eps), jnp.bfloat16: float(jnp.finfo(jnp.bfloat16).eps)}


def _inputs(shape, dtype, seed=0):
    k_x, k_g, k_w = jax.random.split(jax.random.key(seed), 3)
    # Rows of different scales, so eps matters for some of them.
    scale = jnp.exp(jax.random.uniform(k_w, (*shape[:-1], 1), minval=-6.0, maxval=3.0))
    x = (jax.random.normal(k_x, shape) * scale).astype(dtype)
    g = jax.random.normal(k_g, shape, dtype=jnp.float32)
    return x, g


def _reference(x, g, eps, weight=None):
    # float64 value and VJP of x * rsqrt(mean(x^2) + eps) * weight.
    x = np.asarray(x, np.float64)
    g = np.asarray(g, np.float64)
    w = np.ones(x.shape[-1]) if weight is None else np.asarray(weight, np.float64)
    r = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    gw = g * w
    dx = r * gw - x * r**3 * np.mean(gw * x, axis=-1, keepdims=True)
    dw = np.sum((g * x * r).reshape(-1, x.shape[-1]), axis=0)
    return x * r * w, dx, dw


def _value_and_vjp(fn, x, g):
    y, vjp = jax.vjp(fn, x)
    (dx,) = vjp(g.astype(y.dtype))
    return y, dx


def _max_scaled_error(actual, expected):
    actual = np.asarray(actual, np.float64)
    return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-30))


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_custom_vjp_rms_norm_matches_the_barrier_rms_norm(dtype, shape, eps):
    x, g = _inputs(shape, dtype)
    run = jax.jit(_value_and_vjp, static_argnums=0)
    y_barrier, dx_barrier = run(lambda v: rms_norm(v, eps, impl="barrier"), x, g)
    y_custom, dx_custom = run(lambda v: rms_norm(v, eps, impl="custom_vjp"), x, g)
    assert y_custom.dtype == y_barrier.dtype == dtype
    assert dx_custom.dtype == dx_barrier.dtype == dtype
    # Same forward expression; the barrier only changes which kernels compute it, and so the float32 rounding.
    np.testing.assert_allclose(y_custom, y_barrier, rtol=_FORWARD_RTOL[dtype], atol=0)

    tolerance = _EPS_MULTIPLE[dtype] * float(jnp.finfo(dtype).eps)
    y_ref, dx_ref, _ = _reference(x, g, eps)
    for name, dx in (("barrier", dx_barrier), ("custom_vjp", dx_custom)):
        assert _max_scaled_error(dx, dx_ref) <= tolerance, name
    assert _max_scaled_error(y_custom, y_ref) <= tolerance
    # The two gradients differ by float32 rounding, before the cast to the input dtype.
    diff = np.max(np.abs(np.asarray(dx_custom, np.float64) - np.asarray(dx_barrier, np.float64)))
    assert diff <= tolerance * np.max(np.abs(dx_ref))


def test_rms_normalize_is_float32_without_barriers():
    x = jnp.ones((2, 8), jnp.bfloat16)
    assert rms_normalize(x, 1e-6).dtype == jnp.float32
    barriers = {}
    for impl in ("barrier", "custom_vjp"):
        text = str(jax.make_jaxpr(jax.grad(lambda v, i=impl: jnp.sum(rms_norm(v, impl=i).astype(jnp.float32))))(x))
        barriers[impl] = text.count("optimization_barrier")
    assert barriers["barrier"] > 0 and barriers["custom_vjp"] == 0, barriers


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_custom_vjp_rms_norm_layer_matches_the_barrier_layer(dtype, eps):
    # The weighted layer: the weight multiplies the float32 normalized activations outside the custom VJP, so its
    # gradient comes from AD in both paths.
    x, g = _inputs((2, 64, 256), dtype, seed=3)
    weight = jax.random.normal(jax.random.key(4), (256,)) * 0.3 + 1
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):

        def run(impl):
            def f(w, v):
                return RMSNorm(w, eps=eps, impl=impl)(v)

            y, vjp = jax.vjp(f, weight, x)
            dw, dx = vjp(g.astype(y.dtype))
            return y, dx, dw

        y_barrier, dx_barrier, dw_barrier = jax.jit(run, static_argnums=0)("barrier")
        y_custom, dx_custom, dw_custom = jax.jit(run, static_argnums=0)("custom_vjp")
    np.testing.assert_allclose(y_custom, y_barrier, rtol=_FORWARD_RTOL[dtype], atol=0)
    tolerance = _EPS_MULTIPLE[dtype] * float(jnp.finfo(dtype).eps)
    _, dx_ref, dw_ref = _reference(x, np.asarray(g.astype(dtype), np.float64), eps, weight)
    for name, dx in (("barrier", dx_barrier), ("custom_vjp", dx_custom)):
        assert _max_scaled_error(dx, dx_ref) <= tolerance, name
    # The weight gradient comes from AD in both paths; it differs only through the normalized activations' rounding.
    for name, dw in (("barrier", dw_barrier), ("custom_vjp", dw_custom)):
        assert _max_scaled_error(dw, dw_ref) <= 64 * float(jnp.finfo(jnp.float32).eps), name


def test_bf16_custom_vjp_rms_norm_scoring_is_within_one_ulp_of_differentiated_forward():
    # test_june_snowball_model's barrier test, which asserts bitwise equality: the custom VJP does not give that. The
    # scoring forward does not output the inverse RMS, so XLA may fuse the reduction with its users there and not in
    # the differentiated forward (on CPU at this size: one YNNPACK fusion against two kernels).
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        inputs = jax.random.normal(jax.random.PRNGKey(10), (2, 2048, 2560)).astype(jnp.bfloat16)
        weight = (jax.random.normal(jax.random.PRNGKey(11), (2560,)) * 0.3 + 1).astype(jnp.bfloat16)
        norm = RMSNorm(weight, eps=1e-6, impl="custom_vjp")

        def objective(module, values):
            output = module(values)
            return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

        scored = jax.jit(lambda module, values: module(values))(norm, inputs)
        (_, trained), gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(norm, inputs)
        jax.block_until_ready(gradients)
        np.testing.assert_allclose(scored, trained, rtol=float(jnp.finfo(jnp.bfloat16).eps), atol=0)


# The type print_saved_residuals gives a batch dimension of 2 sharded over the batch axes.
_BATCH_SHARDED_ROWS = "2@(replica_dcn,data,expert)"
_SMALL = GrugModelConfig(
    vocab_size=32,
    hidden_dim=16,
    intermediate_dim=8,
    shared_expert_intermediate_dim=8,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=2,
    num_kv_heads=1,
    head_dim=8,
    max_seq_len=8,
    sliding_window=4,
    capacity_factor=8.0,
    attention_implementation="reference",
    use_array_stacked_blocks=True,
    disable_pko=True,
)


def test_rms_norm_impl_reaches_every_norm():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(dataclasses.replace(_SMALL, rms_norm_impl="custom_vjp"), key=jax.random.key(0))
    leaves = jax.tree.leaves(model, is_leaf=lambda node: isinstance(node, RMSNorm))
    norms = [leaf for leaf in leaves if isinstance(leaf, RMSNorm)]
    # embed, final, and the stacked blocks' attention and MLP norms.
    assert len(norms) == 4
    assert {norm.impl for norm in norms} == {"custom_vjp"}


@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("remat_mode", ["recompute_all", "save_moe", "save_moe_attn"])
def test_custom_vjp_rms_norm_saves_no_more_residuals(remat_mode, stacked):
    # The custom VJP's residuals (x and the inverse RMS) are unnamed, so the block checkpoints' policies recompute them
    # in the backward like the barrier path's intermediates: the blocks save the same residuals, whatever the number of
    # layers. Outside the checkpoints, the embedding and final norms each save one float32 activation and one float32
    # statistic fewer than with the barrier: the inverse RMS and the normalized activations (for the weight's
    # gradient), where AD of the barrier path also saves the square's derivative 2x and rsqrt's derivative factor.
    tokens = jax.random.randint(jax.random.key(1), (2, 8), 0, _SMALL.vocab_size)

    def saved(num_layers, impl):
        cfg = dataclasses.replace(
            _SMALL, num_layers=num_layers, remat_mode=remat_mode, use_array_stacked_blocks=stacked, rms_norm_impl=impl
        )
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
            model = Transformer.init(cfg, key=jax.random.key(0))
            params, static = eqx.partition(model, eqx.is_array)

            def loss(p):
                hidden, _ = eqx.combine(p, static)(tokens)
                return jnp.mean(jnp.square(hidden))

            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                jax.ad_checkpoint.print_saved_residuals(loss, params)
        # One line per residual: its type, then where it comes from.
        return collections.Counter(line.split()[0] for line in report.getvalue().splitlines())

    for num_layers in (2, 4):
        barrier, custom = saved(num_layers, "barrier"), saved(num_layers, "custom_vjp")
        assert custom <= barrier, num_layers
        activation, statistic = f"f32[{_BATCH_SHARDED_ROWS},8,16]", f"f32[{_BATCH_SHARDED_ROWS},8,1]"
        assert barrier - custom == collections.Counter({activation: 2, statistic: 2}), num_layers
