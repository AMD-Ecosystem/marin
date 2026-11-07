# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``ring_dedup``'s sender-side options (``send_gather``, ``token_sum``) against the defaults, bit for bit."""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.ep_ring import _gather_sum_slots
from levanter.grug._moe.ep_ring_dedup import _gather_sum_slots_clamped


def _bits(x: jax.Array) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_gather_sum_slots_clamped_matches_take_bitwise(dtype):
    k_rows, k_slots = jax.random.split(jax.random.key(0))
    num_rows, tokens, slots_per_token, hidden = 90, 40, 8, 48
    rows = jax.random.normal(k_rows, (num_rows, hidden), jnp.float32)
    # Signed zeros: a sum that starts from zero or skips a term would round them differently.
    rows = rows.at[::5].set(-0.0).astype(dtype)
    # The sender's slots: in [0, num_rows), or the sentinel num_rows for a destination the token does not reach.
    slots = jax.random.randint(k_slots, (tokens, slots_per_token), 0, 2 * num_rows)
    slots = jnp.where(slots < num_rows, slots, num_rows).astype(jnp.int32)
    expected = jax.jit(_gather_sum_slots)(rows, slots)
    actual = jax.jit(_gather_sum_slots_clamped)(rows, slots)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ring_dedup_token_side_options_match_defaults_bitwise(dtype):
    """On 8 expert shards with sender and receiver drops, ring_dedup with the Pallas send gather (in the interpreter)
    and the clamped token sums gives the outputs and gradients of the defaults, bit for bit, also with two token
    blocks and with the Pallas backward gathers."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    script = """
        import math

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

        from levanter.grug.grug_moe import moe_mlp

        dtype = jnp.__DTYPE__
        shards, tokens_per_shard, hidden, inter, experts, topk = 8, 24, 32, 16, 32, 4
        tokens = shards * tokens_per_shard
        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, shards, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        keys = jax.random.split(jax.random.key(0), 7)
        x = jax.random.normal(keys[0], (tokens, hidden)).astype(dtype)
        selected = jnp.argsort(jax.random.uniform(keys[1], (tokens, experts)), axis=1)[:, :topk].astype(jnp.int32)
        combine_weights = jax.nn.softmax(jax.random.normal(keys[2], (tokens, topk)), axis=-1)
        w_up_gate = (jax.random.normal(keys[3], (experts, hidden, 2 * inter)) / math.sqrt(hidden)).astype(dtype)
        w_down = (jax.random.normal(keys[4], (experts, inter, hidden)) / math.sqrt(inter)).astype(dtype)
        cotangent = jax.random.normal(keys[5], (tokens, hidden)).astype(dtype)
        keep_fraction = np.repeat(0.3 + 0.7 * np.arange(shards) / (shards - 1), tokens_per_shard)
        token_valid = np.asarray(jax.random.uniform(keys[6], (tokens,))) < keep_fraction

        batch = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
        args = (
            jax.device_put(x, batch),
            jax.device_put(combine_weights, batch),
            jax.device_put(w_up_gate, expert_sharding),
            jax.device_put(w_down, expert_sharding),
        )
        selected_sharded = jax.device_put(selected, batch)
        token_valid_sharded = jax.device_put(jnp.asarray(token_valid), NamedSharding(mesh, P(("data", "expert"))))
        cotangent_sharded = jax.device_put(cotangent, batch)

        def run(send_gather, token_sum, gather_sum, token_blocks):
            def loss(x, cw, w_up_gate, w_down):
                out, counts = moe_mlp(
                    x,
                    selected_sharded,
                    cw,
                    w_up_gate,
                    w_down,
                    token_valid=token_valid_sharded,
                    activation=jax.nn.silu,
                    implementation="ring_dedup",
                    mesh=mesh,
                    capacity_factor=0.3,
                    pooled_transport_capacity_factor=0.5,
                    report_capacity_overflow=True,
                    token_blocks=token_blocks,
                    gather_sum=gather_sum,
                    send_gather=send_gather,
                    token_sum=token_sum,
                )
                return jnp.sum((out * cotangent_sharded).astype(jnp.float32)), (out, counts)

            with jax.set_mesh(mesh):
                grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                jaxpr = str(jax.make_jaxpr(grad_fn)(*args))
                (_, (out, counts)), grads = grad_fn(*args)
            # The Pallas send gather runs once per token block.
            assert jaxpr.count("grug_gather_rows") >= (token_blocks if send_gather == "pallas" else 0), send_gather
            return out, counts, grads

        def bits(a):
            a = np.asarray(a)
            return a.view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)

        for gather_sum in ("xla", "pallas"):
            for token_blocks in (1, 2):
                ref_out, ref_counts, ref_grads = run("xla", "take", gather_sum, token_blocks)
                assert int(ref_counts.sender_dropped) > 0 and int(ref_counts.receiver_dropped) > 0
                for send_gather, token_sum in (("pallas", "take"), ("xla", "clamped"), ("pallas", "clamped")):
                    out, counts, grads = run(send_gather, token_sum, gather_sum, token_blocks)
                    assert int(counts.sender_dropped) == int(ref_counts.sender_dropped)
                    assert int(counts.receiver_dropped) == int(ref_counts.receiver_dropped)
                    np.testing.assert_array_equal(bits(out), bits(ref_out))
                    for a, e in zip(grads, ref_grads, strict=True):
                        assert np.any(np.asarray(e, np.float32) != 0)
                        np.testing.assert_array_equal(bits(a), bits(e))
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script.replace("__DTYPE__", dtype))],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
