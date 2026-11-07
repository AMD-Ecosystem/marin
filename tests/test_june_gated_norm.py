# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June model's GatedNorm and head-gate options: the Pallas gate kernels (``gated_norm_impl``) against the JAX
ops, and the saved rank-128 down projection (``gated_norm_remat``) and head-gate logits (``head_gate_remat``) against
rerunning them. The kernels themselves are tested in
``lib/levanter/tests/kernels/test_pallas_gated_norm.py``."""

import dataclasses
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jex_core
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

# The Pallas kernels tile the hidden width in 128 columns.
_SMALL = GrugModelConfig(
    vocab_size=200,
    hidden_dim=128,
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
    fused_norm_impl="pallas",
)


def _setup(cfg: GrugModelConfig, dtype):
    tokens = jax.random.randint(jax.random.key(1), (4, cfg.max_seq_len), 0, cfg.vocab_size)
    loss_weight = jnp.ones(tokens.shape, jnp.float32)
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

    return jax.value_and_grad(loss), params


def _loss_and_grads(cfg: GrugModelConfig, dtype=jnp.float32):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        step, params = _setup(cfg, dtype)
        value, grads = jax.jit(step)(params)
    return value, jax.tree.leaves(grads)


def _projections(cfg: GrugModelConfig, width: int) -> int:
    """GEMMs [..., D] x [D, width] -> [..., width] in the differentiated step's jaxpr (GatedNorm's down projection for
    width 128, the head gate for width num_heads), each loop body and conditional branch counted once."""
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        step, params = _setup(cfg, jnp.float32)
        jaxpr = jax.make_jaxpr(step)(params).jaxpr

    def count(jaxpr) -> int:
        total = 0
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                rhs, out = eqn.invars[1].aval, eqn.outvars[0].aval
                total += rhs.shape == (cfg.hidden_dim, width) and out.ndim >= 3 and out.shape[-1] == width
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    if isinstance(sub, jex_core.ClosedJaxpr):
                        total += count(sub.jaxpr)
                    elif isinstance(sub, jex_core.Jaxpr):
                        total += count(sub)
        return total

    return count(jaxpr)


def _assert_grads_close(grads, base, rtol):
    assert len(grads) == len(base)
    for g, b in zip(grads, base, strict=True):
        g, b = np.asarray(g, np.float32), np.asarray(b, np.float32)
        np.testing.assert_allclose(g, b, rtol=rtol, atol=rtol * max(np.abs(b).max(), 1e-30))


@pytest.mark.parametrize("fused", ["xla", "pallas"])
def test_pallas_gate_matches_xla_float32(fused):
    cfg = dataclasses.replace(_SMALL, fused_norm_impl=fused)
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(dataclasses.replace(cfg, gated_norm_impl="pallas"))
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-4)


def test_pallas_gate_in_bfloat16_is_as_close_to_float32_as_xla():
    # bfloat16 rounds the two paths differently; each leaf of the kernels' run must be about as close to a float32
    # run as the JAX ops' run is.
    anchor_loss, anchor = _loss_and_grads(_SMALL)
    base_loss, base_grads = _loss_and_grads(_SMALL, jnp.bfloat16)
    loss, grads = _loss_and_grads(dataclasses.replace(_SMALL, gated_norm_impl="pallas"), jnp.bfloat16)
    for value in (base_loss, loss):
        np.testing.assert_allclose(np.float32(value), np.float32(anchor_loss), rtol=2e-2)

    def rel(x, ref):
        x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
        return np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-30)

    for g, b, ref in zip(grads, base_grads, anchor, strict=True):
        assert rel(g, ref) <= 1.5 * rel(b, ref) + 1e-3


@pytest.mark.parametrize(
    ("fused", "gate", "remat_mode"),
    [
        ("off", "xla", "recompute_all"),
        ("pallas", "xla", "save_moe_attn"),
        ("pallas", "pallas", "save_moe_attn"),
        ("xla", "pallas", "recompute_all"),
    ],
)
def test_save_hidden_skips_the_down_projection_rerun(fused, gate, remat_mode):
    cfg = dataclasses.replace(_SMALL, fused_norm_impl=fused, gated_norm_impl=gate, remat_mode=remat_mode)
    saved = dataclasses.replace(cfg, gated_norm_remat="save_hidden")
    # Each block's two GatedNorms run their down projection in the forward and again in the backward's rerun of the
    # block; with the projection saved, only in the forward.
    assert _projections(cfg, 128) - _projections(saved, 128) == 2
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(saved)
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-5)


@pytest.mark.parametrize("schedule", ["cond", "static"])
def test_saved_head_gate_skips_its_gemm_rerun(schedule):
    cfg = dataclasses.replace(_SMALL, remat_mode="save_moe_attn", attention_schedule=schedule)
    saved = dataclasses.replace(cfg, head_gate_remat="save")
    # The backward's rerun of the block reruns the head-gate GEMM in each attention branch (long and short under the
    # conditional; the one static kind per layer of the two-layer model otherwise); saved, it does not.
    reruns = _projections(cfg, cfg.num_heads) - _projections(saved, cfg.num_heads)
    assert reruns == 2
    base_loss, base_grads = _loss_and_grads(cfg)
    loss, grads = _loss_and_grads(saved)
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-5)


_SHARDED = """
import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

BASE = GrugModelConfig(
    vocab_size=32, hidden_dim=128, intermediate_dim=8, shared_expert_intermediate_dim=8, num_experts=4,
    num_experts_per_token=2, num_layers=3, num_heads=2, num_kv_heads=1, head_dim=8, max_seq_len=8,
    sliding_window=4, capacity_factor=4.0, attention_implementation="reference", moe_implementation="ring",
    use_array_stacked_blocks=True, disable_pko=True, rms_norm_impl="custom_vjp", fused_norm_impl="pallas",
    remat_mode="save_moe_attn",
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
    for overrides in ({"gated_norm_impl": "pallas"}, {"gated_norm_remat": "save_hidden"},
                      {"gated_norm_impl": "pallas", "gated_norm_remat": "save_hidden", "head_gate_remat": "save"}):
        f_loss, f_grads, f_hlo = run(dataclasses.replace(cfg, **overrides))
        np.testing.assert_allclose(f_loss, loss, rtol=1e-6)
        for got, want in zip(f_grads, grads, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * float(np.max(np.abs(want))))
        # The GatedNorm weight gradients are reduced where the JAX ops reduce them: per layer, or after the scan.
        assert all_reduces_in_loops(f_hlo) == all_reduces_in_loops(hlo), (reduction, overrides)
"""


@pytest.mark.timeout(900)
def test_gate_options_on_four_devices_keep_gradients_and_their_reductions():
    run_on_cpu_devices(textwrap.dedent(_SHARDED), device_count=4)
