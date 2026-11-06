# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June model's kernel options: ``ce_v_block_size``, ``moe_gather_sum``, ``moe_routing_metadata``,
``moe_flydsl_gather_a`` and ``moe_scatter_epilogue``.

An explicit fused cross-entropy vocabulary block changes only the summation order of the streamed logsumexp and of
the gradient blocks, so the loss and the gradients match the backend's default block to float32 rounding. The Pallas
gather-sum's numerics are tested in ``lib/levanter/tests/grug/test_pallas_gather_sum.py``, the routing-metadata kernels'
in ``lib/levanter/tests/grug/test_pallas_routing_metadata.py``, the scatter epilogue's in
``lib/levanter/tests/grug/test_scatter_epilogue.py``."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, MoEMLP, Transformer

_SMALL = GrugModelConfig(
    vocab_size=200,
    hidden_dim=16,
    intermediate_dim=8,
    shared_expert_intermediate_dim=8,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=2,
    num_kv_heads=1,
    head_dim=8,
    max_seq_len=16,
    sliding_window=8,
    capacity_factor=8.0,
    attention_implementation="reference",
    use_array_stacked_blocks=True,
    disable_pko=True,
    ce_implementation="xla",
)


def _loss_and_grads(cfg: GrugModelConfig):
    tokens = jax.random.randint(jax.random.key(1), (4, cfg.max_seq_len), 0, cfg.vocab_size)
    loss_weight = jnp.ones(tokens.shape, jnp.float32)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
        params, static = eqx.partition(model, eqx.is_array)

        def loss(p):
            return eqx.combine(p, static).next_token_loss(tokens, loss_weight, logsumexp_weight=1e-3)

        value, grads = jax.jit(jax.value_and_grad(loss))(params)
    return value, jax.tree.leaves(grads)


@pytest.mark.parametrize("v_block_size", [16, 64, 8192])
def test_ce_v_block_size_keeps_loss_and_grads(v_block_size):
    base_loss, base_grads = _loss_and_grads(_SMALL)
    loss, grads = _loss_and_grads(dataclasses.replace(_SMALL, ce_v_block_size=v_block_size))
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    assert len(grads) == len(base_grads)
    for g, base in zip(grads, base_grads, strict=True):
        scale = max(float(jnp.max(jnp.abs(base))), 1e-30)
        assert float(jnp.max(jnp.abs(g - base))) <= 1e-5 * scale
    if v_block_size < _SMALL.vocab_size:
        # Several vocabulary blocks: the block reached the loss, which now sums in a different order.
        assert any(not np.array_equal(g, base) for g, base in zip(grads, base_grads, strict=True))


@pytest.mark.parametrize("gather_sum", ["xla", "pallas"])
def test_moe_gather_sum_reaches_every_expert_mlp(gather_sum):
    cfg = dataclasses.replace(
        _SMALL, moe_implementation="ring_dedup", pooled_transport_capacity_factor=1.0, moe_gather_sum=gather_sum
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
    mlps = [leaf for leaf in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, MoEMLP)) if isinstance(leaf, MoEMLP)]
    assert mlps and {mlp.expert_mlp.gather_sum for mlp in mlps} == {gather_sum}


@pytest.mark.parametrize("routing_metadata", ["xla", "pallas"])
def test_moe_routing_metadata_reaches_every_expert_mlp(routing_metadata):
    cfg = dataclasses.replace(
        _SMALL,
        moe_implementation="ring_dedup",
        pooled_transport_capacity_factor=1.0,
        moe_routing_metadata=routing_metadata,
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
    mlps = [leaf for leaf in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, MoEMLP)) if isinstance(leaf, MoEMLP)]
    assert mlps and {mlp.expert_mlp.routing_metadata for mlp in mlps} == {routing_metadata}


@pytest.mark.parametrize("flydsl_gather_a", [False, True])
def test_moe_flydsl_gather_a_reaches_every_expert_mlp(flydsl_gather_a):
    cfg = dataclasses.replace(
        _SMALL,
        moe_implementation="ring_dedup",
        pooled_transport_capacity_factor=1.0,
        moe_flydsl_gather_a=flydsl_gather_a,
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
    mlps = [leaf for leaf in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, MoEMLP)) if isinstance(leaf, MoEMLP)]
    assert mlps and {mlp.expert_mlp.flydsl_gather_a for mlp in mlps} == {flydsl_gather_a}


@pytest.mark.parametrize("scatter_epilogue", ["off", "flydsl"])
def test_moe_scatter_epilogue_reaches_every_expert_mlp(scatter_epilogue):
    cfg = dataclasses.replace(
        _SMALL,
        moe_implementation="ring_dedup",
        pooled_transport_capacity_factor=1.0,
        moe_scatter_epilogue=scatter_epilogue,
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
    mlps = [leaf for leaf in jax.tree.leaves(model, is_leaf=lambda n: isinstance(n, MoEMLP)) if isinstance(leaf, MoEMLP)]
    assert mlps and {mlp.expert_mlp.scatter_epilogue for mlp in mlps} == {scatter_epilogue}
