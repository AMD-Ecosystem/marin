# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert weight gradients summed inside the weight-gradient kernel (``expert_grad_accumulation="flydsl"``).

With it, gradient accumulation threads the float32 sums of the stacked expert weights through every microbatch's
layer scan, and the expert matmuls' backward folds each layer's weight gradient into them (``ExpertGradAccumulator``):
written in the first microbatch with ``gather_in_first_microbatch``, added in the loop, and with
``overlap_last_microbatch`` emitted as the bf16 sum the last microbatch reduces per layer. jax-flydsl's kernels need a
GPU, so here jnp stand-ins with the kernels' contracts (float32 products rounded to bf16, then the same float32
arithmetic) replace them in both arms: the step must give the same loss and gradients, bit for bit, as summing the
bf16 gradients with XLA, on meshes with replicas, ``data`` sharding and expert parallelism, with ZeRO-1, the
overlapped last microbatch and the first microbatch's gathers.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import pytest

from tests.june_cpu_devices import run_on_cpu_devices
from tests.test_june_grad_accum import _PRELUDE as _GRAD_ACCUM_PRELUDE

_PRELUDE = (
    _GRAD_ACCUM_PRELUDE
    + r"""
import os
import types

import importlib

rd = importlib.import_module("haliax.nn.ragged_dot")

os.environ["RAGGED_DOT_IMPL"] = "flydsl"


def _groups(offsets, rows, groups):
    # One-hot group of each row; rows past the last offset belong to none.
    gid = jnp.searchsorted(offsets[1:], jnp.arange(rows), side="right")
    return jax.nn.one_hot(gid, groups, dtype=jnp.float32)


def _f32(x):
    return x.astype(jnp.float32)


def _wgrad_f32(a, b, offsets):
    groups = offsets.shape[0] - 1
    return jnp.einsum("mk,mn,mg->gkn", _f32(a), _f32(b), _groups(offsets, a.shape[0], groups))


def _wgrad_accum(a, b, offsets, acc, layer, *, mode="accum", round_bf16=True):
    dw = _wgrad_f32(a, b, offsets)
    if round_bf16:
        dw = _f32(dw.astype(jnp.bfloat16))
    parts = tuple(acc) if isinstance(acc, tuple) else (acc,)
    pieces = jnp.split(dw, len(parts), axis=-1)
    if mode == "final":
        return (jnp.concatenate([p[layer] for p in parts], axis=-1) + dw).astype(jnp.bfloat16)
    new = tuple(p.at[layer].set(d) if mode == "init" else p.at[layer].add(d) for p, d in zip(parts, pieces))
    return new if isinstance(acc, tuple) else new[0]


# The jax-flydsl ops with their contracts (float32 accumulation, bf16 output), in jnp.
FAKE_OPS = types.SimpleNamespace(
    grouped_gemm_bf16_nn=lambda a, b, o: jnp.einsum(
        "mk,gkn,mg->mn", _f32(a), _f32(b), _groups(o, a.shape[0], b.shape[0])
    ).astype(jnp.bfloat16),
    grouped_gemm_bf16_nt=lambda a, b, o: jnp.einsum(
        "mk,gnk,mg->mn", _f32(a), _f32(b), _groups(o, a.shape[0], b.shape[0])
    ).astype(jnp.bfloat16),
    grouped_gemm_bf16_wgrad=lambda a, b, o: _wgrad_f32(a, b, o).astype(jnp.bfloat16),
    grouped_gemm_bf16_wgrad_accum=_wgrad_accum,
)
rd._flydsl_grouped_gemm_ops = lambda: FAKE_OPS
rd._check_flydsl_operands = lambda *args: None
rd._zero_rows_past_groups_in_place = rd._zero_rows_past_groups

POLICY = jmp.get_policy("p=f32,c=bfloat16")
GA_CFG = dataclasses.replace(BASE, num_experts=8, remat_mode="save_moe_attn")


def accumulated_grads(mesh, *, accumulation, zero1=False, overlap=False, gather=False, microbatches=4, cfg=GA_CFG):
    optimizer = keep_gradients()
    with jax.set_mesh(mesh):
        state = june_train.initial_state(
            cfg, optimizer=optimizer, mp=POLICY, key=jax.random.key(0), ema_beta=None,
            carry_compute_params=zero1, zero1=zero1,
        )
        train_step = june_train._make_train_step(
            optimizer, POLICY, z_loss_weight=0.0, ema_beta=None, microbatches=microbatches, zero1=zero1,
            overlap_last_microbatch=overlap, gather_in_first_microbatch=gather,
            expert_grad_accumulation=accumulation,
        )
        state, metrics, _ = train_step(state, BATCH)
    return jax.device_get(metrics["train/loss"]), jax.device_get(state.opt_state)


def assert_same_step(**kwargs):
    for name, mesh in MESHES.items():
        loss, grads = accumulated_grads(mesh, accumulation="xla", **kwargs)
        k_loss, k_grads = accumulated_grads(mesh, accumulation="flydsl", **kwargs)
        assert np.array_equal(loss, k_loss), (name, loss, k_loss)
        for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(grads), jax.tree.leaves(k_grads), strict=True):
            a, b = np.asarray(a), np.asarray(b)
            assert a.shape == b.shape and a.dtype == b.dtype, (name, jax.tree_util.keystr(path))
            assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (name, jax.tree_util.keystr(path))
        expert = [x for p, x in jax.tree_util.tree_leaves_with_path(k_grads) if "expert_mlp" in jax.tree_util.keystr(p)]
        assert len(expert) == 3 and all(np.abs(np.asarray(x)).max() > 0 for x in expert), name
"""
)


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=8)


@pytest.mark.timeout(900)
def test_in_kernel_sums_match_xla_in_the_microbatch_loop():
    _run("assert_same_step()")


@pytest.mark.timeout(900)
def test_in_kernel_sums_match_xla_with_the_overlapped_last_microbatch():
    _run("assert_same_step(overlap=True)")


@pytest.mark.timeout(900)
@pytest.mark.parametrize("overlap", [False, True], ids=["loop", "overlap"])
def test_in_kernel_sums_match_xla_with_zero1_and_first_microbatch_gathers(overlap):
    _run(f"assert_same_step(zero1=True, gather=True, overlap={overlap})")


@pytest.mark.timeout(900)
def test_in_kernel_sums_with_two_microbatches_and_the_overlapped_last():
    # The loop then runs one microbatch; with the gathers it runs none, the first and last running outside it.
    _run(
        """
        assert_same_step(microbatches=2, overlap=True)
        assert_same_step(microbatches=2, zero1=True, gather=True, overlap=True)
        """
    )


def test_in_kernel_sums_need_microbatches():
    _run(
        """
        try:
            june_train._make_train_step(
                keep_gradients(), POLICY, z_loss_weight=0.0, ema_beta=None, microbatches=1,
                expert_grad_accumulation="flydsl",
            )
        except ValueError as e:
            assert "microbatches" in str(e)
        else:
            raise AssertionError("expected a ValueError")
        """
    )


def test_in_kernel_sums_need_ring_filtered():
    _run(
        """
        try:
            ring = dataclasses.replace(GA_CFG, moe_implementation="ring")
            accumulated_grads(MESHES["replica4_expert2"], accumulation="flydsl", cfg=ring)
        except NotImplementedError as e:
            assert "ring_filtered" in str(e)
        else:
            raise AssertionError("expected a NotImplementedError")
        """
    )


@pytest.mark.timeout(900)
def test_in_kernel_sums_with_token_blocks():
    # Two token blocks fold two gradients into each layer of the sums: the backward's first one ("init") writes it and
    # the second adds to it. XLA sums the blocks' bf16 gradients in bf16 first, so only the rounding differs.
    _run(
        """
        cfg = dataclasses.replace(GA_CFG, moe_token_blocks=2)
        mesh = MESHES["replica4_expert2"]
        for kwargs in ({}, {"zero1": True, "gather": True}):
            loss, grads = accumulated_grads(mesh, accumulation="xla", cfg=cfg, **kwargs)
            k_loss, k_grads = accumulated_grads(mesh, accumulation="flydsl", cfg=cfg, **kwargs)
            assert np.array_equal(loss, k_loss), (kwargs, loss, k_loss)
            for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(grads), jax.tree.leaves(k_grads), strict=True):
                a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
                scale = max(float(np.abs(a).max()), 1e-30)
                assert float(np.abs(a - b).max()) <= 2e-2 * scale, (kwargs, jax.tree_util.keystr(path))
        try:
            accumulated_grads(mesh, accumulation="flydsl", cfg=cfg, zero1=True, gather=True, overlap=True)
        except NotImplementedError as e:
            assert "token_blocks" in str(e)
        else:
            raise AssertionError("expected a NotImplementedError")
        """
    )
