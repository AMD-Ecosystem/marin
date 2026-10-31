# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient accumulation over microbatches in the June train step, on eight CPU devices with the ring_dedup MoE.

A train step with ``microbatches=K`` gives the loss, gradients and routing counts of the whole batch, on a mesh with
replicas and expert parallelism (replica_dcn=4 x expert=2) and on one that also shards the parameters over ``data``
(replica_dcn=2 x data=2 x expert=2). The capacity factors are high enough that no expert overflows, since capacity
applies per microbatch. The next step's QB thresholds are the mean of those of the microbatches.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

_PRELUDE = """
import collections
import dataclasses

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
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


def keep_gradients():
    # Leaves the parameters unchanged and stores the step's gradients as the optimizer state.
    return optax.GradientTransformation(
        init=lambda params: jax.tree.map(jnp.zeros_like, params),
        update=lambda grads, state, params=None: (jax.tree.map(jnp.zeros_like, grads), grads),
    )


def step(cfg, mesh, policy, *, microbatches, carry_compute_params=False, batch=BATCH):
    optimizer = keep_gradients()
    with jax.set_mesh(mesh):
        state = june_train.initial_state(
            cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None,
            carry_compute_params=carry_compute_params,
        )
        train_step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=0.0, ema_beta=None, microbatches=microbatches
        )
        state, metrics, _ = train_step(state, batch)
    return state, jax.device_get(metrics)


def leaves(tree):
    return [np.asarray(x, dtype=np.float32) for x in jax.tree_util.tree_leaves(tree)]


def grad_errors(state, reference):
    # Relative L2 error of every gradient leaf against the reference state's.
    return [
        np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        for a, b in zip(leaves(state.opt_state), leaves(reference.opt_state), strict=True)
    ]


def assert_matches_whole_batch(cfg, policy, check_grads, *, carry_compute_params=False):
    for name, mesh in MESHES.items():
        whole, whole_metrics = step(cfg, mesh, policy, microbatches=1, carry_compute_params=carry_compute_params)
        assert np.sum(whole_metrics["moe/dropped_assignments"]) == 0, name
        for k in (2, 4):
            msg = f"{name} K={k}"
            accumulated, metrics = step(cfg, mesh, policy, microbatches=k, carry_compute_params=carry_compute_params)
            assert metrics.keys() == whole_metrics.keys(), msg
            np.testing.assert_allclose(metrics["train/loss"], whole_metrics["train/loss"], rtol=1e-6, err_msg=msg)
            np.testing.assert_allclose(
                metrics["train/cross_entropy_loss"], whole_metrics["train/cross_entropy_loss"], rtol=1e-6, err_msg=msg
            )
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
def test_microbatches_match_the_whole_batch_in_float32():
    # Only the order of the float32 sums changes: a few ulps (2**-23 each); 4e-7 observed.
    _run(
        """
        def check_grads(accumulated, whole, name, msg):
            assert max(grad_errors(accumulated, whole)) <= 1e-5, msg

        assert_matches_whole_batch(BASE, jmp.get_policy("f32"), check_grads)
        """
    )


@pytest.mark.timeout(900)
def test_microbatches_match_the_whole_batch_with_bfloat16_compute():
    # The forward is the same for every token, so the loss matches to float32 rounding. The gradients do not: each
    # microbatch rounds its bfloat16 weight gradients and their cross-device sums on its own. The whole batch's
    # bfloat16 gradients are themselves up to 2e-2 off the float32 ones (the norm weights, whose gradient sums many
    # terms), so require the microbatched gradients to be no further from the float32 ones than that, with a margin.
    _run(
        """
        policy = jmp.get_policy("p=f32,c=bfloat16")
        cfg = dataclasses.replace(BASE, remat_mode="save_moe_attn")
        exact = {name: step(cfg, mesh, jmp.get_policy("f32"), microbatches=1)[0] for name, mesh in MESHES.items()}
        bfloat16_error = {}

        def check_grads(accumulated, whole, name, msg):
            bound = bfloat16_error.setdefault(name, max(grad_errors(whole, exact[name])))
            assert max(grad_errors(accumulated, exact[name])) <= 1.5 * bound, (msg, bound)

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
