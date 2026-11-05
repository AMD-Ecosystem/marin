# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June model's fused residual add + RMS norm + GatedNorm (``fused_norm_impl``) against the separate modules.

The fused path computes the same expressions: the norm in float32 from the rounded residual sum, its output rounded
once, the GatedNorm and its matmuls as ``GatedNorm`` writes them. Its backward adds the same cotangent terms in one
pass. In float32 the loss and every gradient agree with the separate modules to float32 rounding; with bfloat16
activations the norm output may differ by one bfloat16 ulp where the statistic rounds differently, so the loss agrees
to bfloat16 rounding. The kernels themselves are tested in ``lib/levanter/tests/kernels/test_pallas_add_rms_norm.py``.
"""

import dataclasses
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GatedNorm, GrugModelConfig, RMSNorm, Transformer, fused_norm_and_gate
from tests.june_cpu_devices import run_on_cpu_devices

_SMALL = GrugModelConfig(
    vocab_size=200,
    hidden_dim=32,
    intermediate_dim=16,
    shared_expert_intermediate_dim=16,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=2,
    num_kv_heads=1,
    head_dim=16,
    max_seq_len=16,
    sliding_window=8,
    capacity_factor=8.0,
    attention_implementation="reference",
    use_array_stacked_blocks=True,
    disable_pko=True,
    ce_implementation="xla",
    rms_norm_impl="custom_vjp",
)


def _loss_and_grads(cfg: GrugModelConfig, dtype=jnp.float32):
    tokens = jax.random.randint(jax.random.key(1), (4, cfg.max_seq_len), 0, cfg.vocab_size)
    loss_weight = jnp.ones(tokens.shape, jnp.float32)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
        # Perturb every weight (the norm weights start at 1), then cast.
        model = jax.tree.map(
            lambda a: (
                (a * (1 + 0.1 * jnp.sin(jnp.arange(a.size).reshape(a.shape)))).astype(dtype)
                if eqx.is_inexact_array(a)
                else a
            ),
            model,
        )
        params, static = eqx.partition(model, eqx.is_array)

        def loss(p):
            return eqx.combine(p, static).next_token_loss(tokens, loss_weight, logsumexp_weight=1e-3)

        value, grads = jax.jit(jax.value_and_grad(loss))(params)
    return value, jax.tree.leaves(grads)


def _assert_grads_close(grads, base, rtol):
    assert len(grads) == len(base)
    for g, b in zip(grads, base, strict=True):
        scale = max(float(jnp.max(jnp.abs(b.astype(jnp.float32)))), 1e-30)
        assert float(jnp.max(jnp.abs(g.astype(jnp.float32) - b.astype(jnp.float32)))) <= rtol * scale


@pytest.mark.parametrize("impl", ["xla", "pallas"])
@pytest.mark.parametrize(
    "overrides",
    [{}, {"use_array_stacked_blocks": False}],
    ids=["stacked", "unstacked"],
)
def test_fused_norm_matches_separate_modules_float32(impl, overrides):
    cfg = dataclasses.replace(_SMALL, **overrides)
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(dataclasses.replace(cfg, fused_norm_impl=impl))
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-4)


@pytest.mark.parametrize("impl", ["xla", "pallas"])
def test_fused_norm_matches_separate_modules_bfloat16(impl):
    base_loss, base_grads = _loss_and_grads(_SMALL, jnp.bfloat16)
    loss, grads = _loss_and_grads(dataclasses.replace(_SMALL, fused_norm_impl=impl), jnp.bfloat16)
    np.testing.assert_allclose(np.float32(loss), np.float32(base_loss), rtol=2e-2)
    _assert_grads_close(grads, base_grads, rtol=6e-2)


def test_fused_norm_returns_the_residual_sum_and_the_gated_norm():
    k = jax.random.split(jax.random.key(2), 4)
    x = jax.random.normal(k[0], (2, 8, 32)).astype(jnp.bfloat16)
    residual = jax.random.normal(k[1], (2, 8, 32)).astype(jnp.bfloat16)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        norm = RMSNorm.init(32, 1e-5, "custom_vjp")
        gated = jax.tree.map(lambda a: a.astype(jnp.bfloat16), GatedNorm.init(32, 0.5, key=k[2]))
        want = np.asarray(gated(norm(x + residual)), np.float32)
        for impl in ("xla", "pallas"):
            h, out = fused_norm_and_gate(x, residual, norm, gated, impl, return_h=True)
            np.testing.assert_array_equal(np.asarray(h, np.float32), np.asarray(x + residual, np.float32))
            np.testing.assert_allclose(np.asarray(out, np.float32), want, rtol=2**-7, atol=2**-7 * np.abs(want).max())


_SHARDED = """
import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

BASE = GrugModelConfig(
    vocab_size=32, hidden_dim=16, intermediate_dim=8, shared_expert_intermediate_dim=8, num_experts=4,
    num_experts_per_token=2, num_layers=3, num_heads=2, num_kv_heads=1, head_dim=8, max_seq_len=8,
    sliding_window=4, capacity_factor=4.0, attention_implementation="reference", moe_implementation="ring",
    use_array_stacked_blocks=True, disable_pko=True, rms_norm_impl="custom_vjp",
)
TOKENS = jax.random.randint(jax.random.key(1), (8, 8), 0, BASE.vocab_size)
WEIGHT = jnp.ones((8, 8), dtype=jnp.float32)


def run(cfg):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
        model = jax.tree.map(lambda a: a * (1 + 0.1 * jnp.sin(jnp.arange(a.size).reshape(a.shape)))
                             if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jnp.floating) else a, model)
        step = jax.jit(jax.value_and_grad(lambda m: m.next_token_loss(TOKENS, WEIGHT, reduction="mean")))
        loss, grads = step(model)
        hlo = step.lower(model).compile().as_text()
    return loss, jax.tree_util.tree_leaves(grads), hlo


def all_reduces_in_loops(hlo):
    bodies = set(re.findall(r"body=%?([\\w.\\-]+)", hlo))
    count, current = 0, None
    for line in hlo.splitlines():
        header = re.match(r"^(?:ENTRY )?%?([\\w.\\-]+) .*\\{$", line)
        if header and not line.startswith(" "):
            current = header.group(1)
        elif current in bodies and re.search(r"\\ball-reduce(-start)?\\(", line):
            count += 1
    return count


for reduction in ("per_layer", "after_scan"):
    cfg = dataclasses.replace(BASE, replicated_grad_reduction=reduction)
    loss, grads, hlo = run(cfg)
    for impl in ("xla", "pallas"):
        f_loss, f_grads, f_hlo = run(dataclasses.replace(cfg, fused_norm_impl=impl))
        np.testing.assert_allclose(f_loss, loss, rtol=1e-6)
        for got, want in zip(f_grads, grads, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * float(np.max(np.abs(want))))
        # The fused norm reduces its weight gradients where the separate modules do: per layer, or after the scan.
        assert all_reduces_in_loops(f_hlo) == all_reduces_in_loops(hlo), (reduction, impl)
"""


@pytest.mark.timeout(900)
def test_fused_norm_on_four_devices_keeps_gradients_and_their_reductions():
    run_on_cpu_devices(textwrap.dedent(_SHARDED), device_count=4)
