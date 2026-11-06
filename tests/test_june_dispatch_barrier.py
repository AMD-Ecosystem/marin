# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``GrugModelConfig.moe_dispatch_barrier``: an optimization barrier on ``ring_dedup``'s expert-ordered dispatch rows.

The barrier only stops XLA from fusing a second copy of the dispatch gather into the update of the saved-residual
stack, so the loss and every gradient are the same bits with and without it, under the remat policy that saves those
rows. The expert-parallel path needs several devices, so each test runs on 8 CPU devices in a fresh interpreter."""

import textwrap

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

_PRELUDE = r"""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

CFG = GrugModelConfig(
    vocab_size=200, hidden_dim=16, intermediate_dim=8, shared_expert_intermediate_dim=8, num_experts=4,
    num_experts_per_token=2, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=8, max_seq_len=16,
    sliding_window=8, capacity_factor=8.0, attention_implementation="reference", use_array_stacked_blocks=True,
    disable_pko=True, ce_implementation="xla", moe_implementation="ring_dedup", pooled_transport_capacity_factor=1.0,
    remat_mode="save_moe",
)
MESH = compact_grug_mesh(expert_axis_size=2, replica_axis_size=4)
TOKENS = np.asarray(jax.random.randint(jax.random.key(1), (8, CFG.max_seq_len), 0, CFG.vocab_size))


def step(cfg):
    with jax.set_mesh(MESH):
        model = Transformer.init(cfg, key=jax.random.key(0))
        params, static = eqx.partition(model, eqx.is_array)
        weight = jnp.ones(TOKENS.shape, jnp.float32)

        def loss(p):
            return eqx.combine(p, static).next_token_loss(jnp.asarray(TOKENS), weight, logsumexp_weight=1e-3)

        return jax.jit(jax.value_and_grad(loss)), params


def loss_and_grads(cfg):
    fn, params = step(cfg)
    with jax.set_mesh(MESH):
        value, grads = fn(params)
    return np.asarray(value), [np.asarray(g) for g in jax.tree.leaves(grads)]


def barriers(cfg):
    fn, params = step(cfg)
    with jax.set_mesh(MESH):
        return fn.lower(params).as_text().count("optimization_barrier")
"""


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=8)


def test_dispatch_barrier_adds_a_barrier_to_the_step():
    _run(
        """
        off = barriers(CFG)
        on = barriers(dataclasses.replace(CFG, moe_dispatch_barrier=True))
        assert on > off, (on, off)
        """
    )


@pytest.mark.parametrize("token_blocks", [1, 2])
def test_dispatch_barrier_keeps_loss_and_grads_bitwise(token_blocks):
    _run(
        f"""
        off = dataclasses.replace(CFG, moe_token_blocks={token_blocks})
        base_loss, base_grads = loss_and_grads(off)
        loss, grads = loss_and_grads(dataclasses.replace(off, moe_dispatch_barrier=True))
        np.testing.assert_array_equal(loss, base_loss)
        assert len(grads) == len(base_grads)
        for g, base in zip(grads, base_grads, strict=True):
            np.testing.assert_array_equal(g, base)
        """
    )
