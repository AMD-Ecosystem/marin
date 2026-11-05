# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Options that cut the June model's in-scan routing statistics, on four CPU devices (data=2 x expert=2), ring_filtered.

- ``capacity_drop_reduction="after_scan"``: the drop counts are summed once after the layer scan, not in every layer.
- ``qb_threshold_impl="radix_select"``: the QB thresholds come from an exact radix select instead of a top_k sort.
- ``router_diagnostics=False`` and ``skip_zero_router_z_loss``: no diagnostic statistics, no z-loss term at 0.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import pytest

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from tests.june_cpu_devices import run_on_cpu_devices

_PRELUDE = """
import collections
import dataclasses
import re

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from levanter.grug.sharding import compact_grug_mesh

from experiments.june_tpu_67b_a2b.moe import train as june_train
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

# Transport and expert capacity well below the demand, so ring_filtered drops on both sides.
BASE = GrugModelConfig(
    vocab_size=32, hidden_dim=16, intermediate_dim=8, shared_expert_intermediate_dim=8, num_experts=4,
    num_experts_per_token=2, num_layers=3, num_heads=2, num_kv_heads=1, head_dim=8, max_seq_len=8,
    sliding_window=4, capacity_factor=0.5, pooled_transport_capacity_factor=0.6, attention_implementation="reference",
    moe_implementation="ring_filtered", use_array_stacked_blocks=True, disable_pko=True,
    router_stats_reduction="after_scan",
)
MESH = compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)
# No expert parallelism: ring_filtered falls back to the local path, whose drops are per data shard.
DATA_MESH = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
TOKENS = jax.random.randint(jax.random.key(1), (8, 8), 0, BASE.vocab_size)
WEIGHT = jnp.ones((8, 8), dtype=jnp.float32)


def run(cfg, mesh=MESH):
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.key(0))

        def loss_fn(m):
            return m.next_token_loss(TOKENS, WEIGHT, reduction="mean", return_router_metrics=True)

        step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        (loss, metrics), grads = step(model)
        hlo = step.lower(model).compile().as_text()
    return loss, metrics, grads, hlo


def train_metrics(cfg, microbatches):
    batch = collections.namedtuple("Batch", "tokens loss_weight attn_mask")(TOKENS, WEIGHT, None)
    policy = jmp.get_policy("f32")
    optimizer = optax.adam(1e-3)
    with jax.set_mesh(MESH):
        state = june_train.initial_state(cfg, optimizer=optimizer, mp=policy, key=jax.random.key(0), ema_beta=None)
        step = june_train._make_train_step(
            optimizer, policy, z_loss_weight=0.0, ema_beta=None, microbatches=microbatches
        )
        state, metrics, _ = step(state, batch)
    return state, metrics


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


def leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def assert_same(got, want):
    for g, w in zip(leaves(got), leaves(want), strict=True):
        np.testing.assert_array_equal(g, w)
"""


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=4)


@pytest.mark.timeout(600)
def test_capacity_drops_after_scan_match_per_layer_and_leave_the_scan():
    _run(
        """
        loss, metrics, grads, hlo = run(BASE)
        a_loss, a_metrics, a_grads, a_hlo = run(dataclasses.replace(BASE, capacity_drop_reduction="after_scan"))
        drops = np.asarray(metrics["moe/dropped_assignments"])
        assert drops.shape == (BASE.num_layers,) and (drops > 0).all(), drops
        np.testing.assert_array_equal(a_loss, loss)
        assert a_metrics.keys() == metrics.keys()
        assert_same(a_metrics, metrics)
        assert_same(a_grads, grads)
        assert all_reduces_in_loops(a_hlo) < all_reduces_in_loops(hlo)

        # The local path (no expert axis; it has no capacity limit, so it drops nothing) reports per-shard counts too.
        loss, metrics, grads, _ = run(BASE, DATA_MESH)
        a_loss, a_metrics, a_grads, _ = run(dataclasses.replace(BASE, capacity_drop_reduction="after_scan"), DATA_MESH)
        np.testing.assert_array_equal(a_loss, loss)
        assert_same(a_metrics, metrics)
        assert_same(a_grads, grads)

        # The same with microbatches, whose drop counts add up.
        state, m = train_metrics(BASE, microbatches=2)
        a_state, a_m = train_metrics(dataclasses.replace(BASE, capacity_drop_reduction="after_scan"), microbatches=2)
        assert_same(a_m, m)
        assert_same(a_state.params, state.params)
        """
    )


def test_capacity_drops_after_scan_need_router_stats_after_scan():
    with pytest.raises(ValueError, match="capacity_drop_reduction"):
        GrugModelConfig(vocab_size=32, use_array_stacked_blocks=True, capacity_drop_reduction="after_scan")


@pytest.mark.timeout(600)
def test_radix_select_qb_thresholds_match_top_k():
    _run(
        """
        for reduction in ("per_layer", "after_scan"):
            cfg = dataclasses.replace(BASE, router_stats_reduction=reduction)
            loss, metrics, grads, _ = run(cfg)
            r_loss, r_metrics, r_grads, _ = run(dataclasses.replace(cfg, qb_threshold_impl="radix_select"))
            np.testing.assert_array_equal(r_loss, loss)
            assert_same(r_metrics, metrics)
            assert_same(r_grads, grads)
        """
    )


@pytest.mark.timeout(600)
def test_without_router_diagnostics_training_is_unchanged():
    _run(
        """
        for reduction in ("per_layer", "after_scan"):
            cfg = dataclasses.replace(BASE, router_stats_reduction=reduction)
            loss, metrics, grads, _ = run(cfg)
            off = dataclasses.replace(cfg, router_diagnostics=False)
            o_loss, o_metrics, o_grads, _ = run(off)
            np.testing.assert_array_equal(o_loss, loss)
            assert_same(o_grads, grads)
            kept = {"qb_beta_per_layer", "moe/dropped_assignments", "moe/skipped_padding_assignments",
                    "train/cross_entropy_loss", "train/router/aux_loss_weighted"}
            assert set(o_metrics) == kept, sorted(set(o_metrics) ^ kept)
            for name in kept:
                np.testing.assert_array_equal(np.asarray(o_metrics[name]), np.asarray(metrics[name]))

            # A weighted z-loss is still computed, as before.
            weighted = dataclasses.replace(cfg, router_z_loss_coef=1e-2)
            w_loss, w_metrics, w_grads, _ = run(weighted)
            ow_loss, ow_metrics, ow_grads, _ = run(dataclasses.replace(weighted, router_diagnostics=False))
            np.testing.assert_allclose(ow_loss, w_loss, rtol=1e-6)
            z_loss = "train/router/router_z_loss"
            np.testing.assert_allclose(np.asarray(ow_metrics[z_loss]), np.asarray(w_metrics[z_loss]), rtol=1e-6)
            for got, want in zip(leaves(ow_grads), leaves(w_grads), strict=True):
                np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)

        # Microbatches combine the statistics that remain.
        state, m = train_metrics(BASE, microbatches=2)
        o_state, o_m = train_metrics(dataclasses.replace(BASE, router_diagnostics=False), microbatches=2)
        np.testing.assert_array_equal(np.asarray(o_m["train/loss"]), np.asarray(m["train/loss"]))
        np.testing.assert_array_equal(np.asarray(o_m["qb_beta_per_layer"]), np.asarray(m["qb_beta_per_layer"]))
        assert "train/router/routing_counts_per_layer" not in o_m
        assert_same(o_state.params, state.params)
        """
    )


@pytest.mark.timeout(600)
def test_skipping_a_zero_router_z_loss_drops_its_backward():
    _run(
        """
        loss, metrics, grads, hlo = run(BASE)
        s_loss, s_metrics, s_grads, s_hlo = run(dataclasses.replace(BASE, skip_zero_router_z_loss=True))
        np.testing.assert_array_equal(s_loss, loss)
        assert_same(s_metrics, metrics)
        assert_same(s_grads, grads)
        exps = lambda text: len(re.findall(r"\\bexponential\\(", text))
        assert exps(s_hlo) < exps(hlo), (exps(s_hlo), exps(hlo))
        """
    )
