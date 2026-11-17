# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-shard routing statistics reduced once after the layers match QBRoutedMoE's per-layer statistics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard

from levanter.grug.grug_moe import (
    MoEExpertMlp,
    QBRoutedMoE,
    moe_mlp,
    moe_router_z_loss_local,
    moe_routing_stats_local,
    qb_beta_topk_local,
    reduce_moe_routing_stats,
)
from levanter.testing.devices import use_test_devices

_AXES = ("data", "expert")
_TOKENS, _HIDDEN, _EXPERTS, _TOPK, _LAYERS = 64, 8, 4, 2, 2
_Z_LOSS_COEF = 0.1


def _per_shard_layer(x, valid, router, expert_mlp, mesh, *, z_loss_only):
    # QBRoutedMoE's routing, with every statistic left per shard for one reduction after the layers.
    logits = jnp.einsum("td,de->te", x, reshard(router, P(None, None))).astype(jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    topk_logits, selected = jax.lax.top_k(logits, _TOPK + 1)
    alpha, selected = topk_logits[:, -1:], selected[:, :-1]
    weights = jax.lax.optimization_barrier(jax.nn.sigmoid(jnp.take_along_axis(logits, selected, axis=-1)))
    weights *= 2.5 / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-9)
    tokens = P(_AXES, None)
    if z_loss_only:
        stats = moe_router_z_loss_local(reshard(logits, tokens), valid, mesh, batch_axes=_AXES)
    else:
        stats = moe_routing_stats_local(
            reshard(selected, tokens),
            reshard(probs, tokens),
            reshard(logits, tokens),
            valid,
            mesh,
            batch_axes=_AXES,
            num_experts=_EXPERTS,
        )
    stats["qb_beta_local"], stats["qb_beta_weight_local"] = qb_beta_topk_local(
        reshard(logits - alpha, tokens),
        valid,
        mesh,
        batch_axes=_AXES,
        num_experts_per_token=_TOPK,
        num_experts=_EXPERTS,
    )
    out, counts = expert_mlp(
        x,
        selected,
        weights.astype(x.dtype),
        token_valid=valid,
        mesh=mesh,
        report_capacity_overflow=True,
        local_capacity_drops=True,
    )
    stats["routing_sender_drops_local"] = counts.sender_dropped
    stats["routing_receiver_drops_local"] = counts.receiver_dropped
    return out, stats


def _run(mesh, x, valid, routers, expert_mlps, *, per_shard, z_loss_only=False):
    def loss(x, routers, expert_mlps):
        layer_stats = []
        for router, expert_mlp in zip(routers, expert_mlps, strict=True):
            if per_shard:
                out, stats = _per_shard_layer(x, valid, router, expert_mlp, mesh, z_loss_only=z_loss_only)
            else:
                out, stats = QBRoutedMoE(num_experts_per_token=_TOPK, batch_axes=_AXES)(
                    x,
                    valid,
                    router=router,
                    router_bias=jnp.zeros((_EXPERTS,), jnp.float32),
                    expert_mlp=expert_mlp,
                    mesh=mesh,
                )
            x = x + out
            layer_stats.append(stats)
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *layer_stats)
        if per_shard:
            stacked = reduce_moe_routing_stats(stacked, num_experts=_EXPERTS, num_experts_per_token=_TOPK)
        return jnp.sum(x**2) + _Z_LOSS_COEF * jnp.sum(stacked["router_z_loss"]), stacked

    (value, stats), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(x, routers, expert_mlps)
    return value, jax.tree.map(np.asarray, stats), [np.asarray(g) for g in jax.tree.leaves(grads)]


@pytest.mark.multidevice
@pytest.mark.parametrize(
    ("mesh_shape", "z_loss_only"),
    [
        pytest.param((2, 2), False, id="expert_parallel"),
        pytest.param((2, 2), True, id="expert_parallel_z_loss_only"),
        pytest.param((4, 1), False, id="local"),
    ],
)
def test_per_shard_stats_reduced_after_the_layers_match_per_layer_stats(mesh_shape, z_loss_only):
    with use_test_devices(4) as devices:
        mesh = jax.make_mesh(mesh_shape, _AXES, axis_types=(AxisType.Explicit,) * 2, devices=devices)
        rng = np.random.default_rng(0)
        with jax.set_mesh(mesh):
            tokens = NamedSharding(mesh, P(_AXES))
            x = jax.device_put(rng.standard_normal((_TOKENS, _HIDDEN)).astype(np.float32), tokens)
            valid = jax.device_put(rng.random(_TOKENS) < 0.8, tokens)
            routers = [
                jax.device_put(rng.standard_normal((_HIDDEN, _EXPERTS)).astype(np.float32), NamedSharding(mesh, P()))
                for _ in range(_LAYERS)
            ]
            # Capacities small enough that both the senders and the receivers drop assignments.
            expert_mlps = [
                MoEExpertMlp.init(
                    num_experts=_EXPERTS,
                    hidden_dim=_HIDDEN,
                    intermediate_dim=4,
                    initializer_std=0.25,
                    key=jax.random.key(layer),
                    implementation="ring_filtered",
                    capacity_factor=0.4,
                    pooled_transport_capacity_factor=0.6,
                )
                for layer in range(_LAYERS)
            ]
            loss, stats, grads = _run(mesh, x, valid, routers, expert_mlps, per_shard=False)
            p_loss, p_stats, p_grads = _run(
                mesh, x, valid, routers, expert_mlps, per_shard=True, z_loss_only=z_loss_only
            )

    np.testing.assert_allclose(p_loss, loss, rtol=1e-6)
    # The statistics do not feed the layers, and a sum's gradient does not depend on its order: exact.
    for got, want in zip(p_grads, grads, strict=True):
        np.testing.assert_array_equal(got, want)
    for name in ("routing_sender_drops", "routing_receiver_drops", "capacity_overflow"):
        np.testing.assert_array_equal(p_stats[name], stats[name])
    if mesh_shape[1] > 1:
        assert stats["routing_sender_drops"].min() > 0 and stats["routing_receiver_drops"].min() > 0, stats
    # Sums over the shards taken in a different order than the per-layer psums.
    for name in ("qb_beta", "router_z_loss"):
        np.testing.assert_allclose(p_stats[name], stats[name], rtol=1e-6, atol=1e-7)
    if z_loss_only:
        assert "routing_counts" not in p_stats
    else:
        np.testing.assert_array_equal(p_stats["routing_counts"], stats["routing_counts"])
        for name in ("routing_entropy", "load_balancing_loss"):
            np.testing.assert_allclose(p_stats[name], stats[name], rtol=1e-6)


@pytest.mark.multidevice
@pytest.mark.parametrize(
    ("implementation", "report_capacity_overflow", "match"),
    [
        pytest.param("ring_filtered", False, "needs report_capacity_overflow", id="without_overflow_report"),
        pytest.param("ring", True, "needs ring_filtered", id="other_expert_parallel_implementation"),
    ],
)
def test_local_capacity_drops_rejects_unsupported_calls(implementation, report_capacity_overflow, match):
    with use_test_devices(4) as devices:
        mesh = jax.make_mesh((2, 2), _AXES, axis_types=(AxisType.Explicit,) * 2, devices=devices)
        with jax.set_mesh(mesh):
            tokens = NamedSharding(mesh, P(_AXES))
            x = jax.device_put(np.zeros((_TOKENS, _HIDDEN), np.float32), tokens)
            selected = jax.device_put(np.zeros((_TOKENS, _TOPK), np.int32), tokens)
            weights = jax.device_put(np.ones((_TOKENS, _TOPK), np.float32), tokens)
            with pytest.raises(ValueError, match=match):
                moe_mlp(
                    x,
                    selected,
                    weights,
                    jnp.zeros((_EXPERTS, _HIDDEN, 8), jnp.float32),
                    jnp.zeros((_EXPERTS, 4, _HIDDEN), jnp.float32),
                    mesh=mesh,
                    implementation=implementation,
                    pooled_transport_capacity_factor=1.0,
                    report_capacity_overflow=report_capacity_overflow,
                    local_capacity_drops=True,
                )


def test_reducing_routing_counts_without_valid_tokens_raises():
    stacked = {
        "routing_counts_local": jnp.ones((_LAYERS, 2, _EXPERTS), jnp.int32),
        "router_prob_sum_local": jnp.ones((_LAYERS, 2, _EXPERTS), jnp.float32),
        "qb_beta": jnp.zeros((_LAYERS, _EXPERTS), jnp.float32),
    }
    with pytest.raises(ValueError, match="needs valid_tokens_local"):
        reduce_moe_routing_stats(stacked, num_experts=_EXPERTS, num_experts_per_token=_TOPK)


def _all_reduce_types(hlo_text: str) -> list[str]:
    # The result types of each all-reduce, one string per collective (a tuple when XLA combined several).
    return [
        line.split(" = ", 1)[1].split(" all-reduce(")[0] for line in hlo_text.splitlines() if " all-reduce(" in line
    ]


@pytest.mark.multidevice
def test_float32_drop_partials_reduce_exactly_with_the_float32_statistics():
    with use_test_devices(4) as devices:
        mesh = jax.make_mesh((2, 2), _AXES, axis_types=(AxisType.Explicit,) * 2, devices=devices)
        rng = np.random.default_rng(0)
        # Up to 2**22 per shard: the four shards' sums reach close to 2**24, where float32 integers stop being exact.
        drops = {
            name: rng.integers(2**22 - 64, 2**22, size=(_LAYERS, 4), dtype=np.int32)
            for name in ("routing_sender_drops_local", "routing_receiver_drops_local")
        }
        stats = {
            "router_z_sq_sum_local": rng.random((_LAYERS, 4), dtype=np.float32),
            "valid_tokens_local": np.full((_LAYERS, 4), 16.0, np.float32),
        }

        def reduce(stacked):
            out = reduce_moe_routing_stats(
                {**stacked, "qb_beta": jnp.zeros((_LAYERS, _EXPERTS))},
                num_experts=_EXPERTS,
                num_experts_per_token=_TOPK,
            )
            return {
                name: out[name] for name in ("routing_sender_drops", "routing_receiver_drops", "capacity_overflow")
            }

        results, all_reduces = {}, {}
        with jax.set_mesh(mesh):
            shards = NamedSharding(mesh, P(None, _AXES))
            for dtype in (np.int32, np.float32):
                stacked = jax.device_put({**stats, **{k: v.astype(dtype) for k, v in drops.items()}}, shards)
                compiled = jax.jit(reduce).lower(stacked).compile()
                results[dtype] = jax.tree.map(np.asarray, compiled(stacked))
                all_reduces[dtype] = _all_reduce_types(compiled.as_text())

    sender = drops["routing_sender_drops_local"].sum(axis=1, dtype=np.int64)
    receiver = drops["routing_receiver_drops_local"].sum(axis=1, dtype=np.int64)
    for dtype in (np.int32, np.float32):
        assert results[dtype]["routing_sender_drops"].dtype == np.int32
        np.testing.assert_array_equal(results[dtype]["routing_sender_drops"], sender)
        np.testing.assert_array_equal(results[dtype]["routing_receiver_drops"], receiver)
        np.testing.assert_array_equal(results[dtype]["capacity_overflow"], (sender + receiver).astype(np.float32))
    # int32 partials take an all-reduce of their own; float32 ones none outside the float32 statistics'.
    assert any("s32" in types for types in all_reduces[np.int32]), all_reduces[np.int32]
    assert all_reduces[np.float32] and not any("s32" in types for types in all_reduces[np.float32]), all_reduces
