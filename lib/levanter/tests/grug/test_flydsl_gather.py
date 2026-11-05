# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``ring_filtered`` with the dispatch gather inside the up-projection GEMM against the explicit gather, bit for bit."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.ep_ring import _dispatch_rows
from levanter.grug._moe.flydsl_gather import gathered_up_projection
from levanter.utils.jax_utils import is_rocm_backend


def _bits(x: jax.Array) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ring_filtered_flydsl_gather_a_matches_explicit_gather_bitwise(dtype):
    """On 8 expert shards with sender and receiver drops, ring_filtered with flydsl_gather_a (its CPU path) gives the
    outputs and gradients of the explicit dispatch gather, bit for bit, with either backward gather-sum and with one
    and two token blocks."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    env.pop("RAGGED_DOT_IMPL", None)
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

        def run(flydsl_gather_a, gather_sum, token_blocks):
            def loss(x, cw, w_up_gate, w_down):
                out, counts = moe_mlp(
                    x,
                    selected_sharded,
                    cw,
                    w_up_gate,
                    w_down,
                    token_valid=token_valid_sharded,
                    activation=jax.nn.silu,
                    implementation="ring_filtered",
                    mesh=mesh,
                    capacity_factor=0.3,
                    pooled_transport_capacity_factor=0.5,
                    report_capacity_overflow=True,
                    token_blocks=token_blocks,
                    gather_sum=gather_sum,
                    flydsl_gather_a=flydsl_gather_a,
                )
                return jnp.sum((out * cotangent_sharded).astype(jnp.float32)), (out, counts)

            with jax.set_mesh(mesh):
                grad_fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True))
                gathered = str(jax.make_jaxpr(loss)(*args)).count("name=_gathered_up_projection")
                (_, (out, counts)), grads = grad_fn(*args)
            return out, counts, grads, gathered

        def bits(a):
            a = np.asarray(a)
            return a.view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)

        for gather_sum in ("xla", "pallas"):
            for token_blocks in (1, 2):
                ref_out, ref_counts, ref_grads, ref_gathered = run(False, gather_sum, token_blocks)
                out, counts, grads, gathered = run(True, gather_sum, token_blocks)
                assert int(ref_counts.sender_dropped) > 0 and int(ref_counts.receiver_dropped) > 0
                assert int(counts.receiver_dropped) == int(ref_counts.receiver_dropped)
                # One gathered up projection per token block, and none without the option.
                assert (ref_gathered, gathered) == (0, token_blocks), (ref_gathered, gathered)
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


@pytest.mark.parametrize("gather_sum", ["xla", "pallas"])
def test_gathered_up_projection_matches_explicit_gather_on_flydsl(gather_sum, monkeypatch):
    if not is_rocm_backend() or importlib.util.find_spec("jax_flydsl") is None:
        pytest.skip("needs a ROCm GPU and jax-flydsl")
    monkeypatch.setenv("RAGGED_DOT_IMPL", "flydsl")
    # One 1/8-scale expert shard: 1900 received rows, 2048 dispatch rows (some padding), 4 experts, hidden 256.
    received, rows, experts, hidden, width, topk = 1900, 2048, 4, 256, 512, 4
    keys = jax.random.split(jax.random.key(3), 4)
    source = jax.random.normal(keys[0], (received, hidden), jnp.float32).astype(jnp.bfloat16)
    w13 = (jax.random.normal(keys[1], (experts, hidden, width), jnp.float32) / 16).astype(jnp.bfloat16)
    d_out = jax.random.normal(keys[2], (rows, width), jnp.float32).astype(jnp.bfloat16)
    group_sizes = jnp.asarray([700, 0, 513, 790], jnp.int32)
    live = int(group_sizes.sum())
    # Ascending received rows per group, as ring_filtered picks them; the padding rows hold the sentinel.
    rng = np.random.default_rng(0)
    row_index = np.full(rows, received, np.int32)
    start = 0
    for size in np.asarray(group_sizes):
        row_index[start : start + size] = np.sort(rng.choice(received, size, replace=False))
        start += size
    # The inverse map: each received row's dispatch rows, out of range for none.
    slots = np.full((received, topk), rows, np.int32)
    fill = np.zeros(received, np.int32)
    for p in range(live):
        r = row_index[p]
        slots[r, fill[r]] = p
        fill[r] += 1
    row_index, slots = jnp.asarray(row_index), jnp.asarray(slots)

    def explicit(source, w13):
        valid = row_index < received
        x = _dispatch_rows(source, jnp.minimum(row_index, received - 1), valid, slots, gather_sum)
        return ragged_dot(x, w13, group_sizes)

    def gathered(source, w13):
        return gathered_up_projection(source, row_index, slots, w13, group_sizes, gather_sum)

    ref, ref_vjp = jax.vjp(jax.jit(explicit), source, w13)
    out, vjp = jax.vjp(jax.jit(gathered), source, w13)
    np.testing.assert_array_equal(_bits(out), _bits(ref))
    for a, e in zip(vjp(d_out), ref_vjp(d_out), strict=True):
        assert np.any(np.asarray(e, np.float32) != 0)
        np.testing.assert_array_equal(_bits(a), _bits(e))
