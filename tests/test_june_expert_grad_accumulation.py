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
    # Padded rows as in _padded below: M is the contraction here.
    pad = -a.shape[0] % 512
    a, b = jnp.pad(a, ((0, pad), (0, 0))), jnp.pad(b, ((0, pad), (0, 0)))
    return jnp.einsum("mk,mn,mg->gkn", _f32(a), _f32(b), _groups(offsets, a.shape[0], groups))


def _gather(a, row_index):
    # The kernels' row index: row r of A is a[row_index[r]], zeros outside [0, R).
    if row_index is None:
        return a
    valid = (row_index >= 0) & (row_index < a.shape[0])
    return jnp.where(valid[:, None], a[jnp.clip(row_index, 0, a.shape[0] - 1)], jnp.zeros((), a.dtype))


def _wgrad_accum(a, b, offsets, acc, layer, *, mode="accum", round_bf16=True, a_row_index=None):
    dw = _wgrad_f32(_gather(a, a_row_index), b, offsets)
    if round_bf16:
        dw = _f32(dw.astype(jnp.bfloat16))
    parts = tuple(acc) if isinstance(acc, tuple) else (acc,)
    pieces = jnp.split(dw, len(parts), axis=-1)
    if mode == "final":
        return (jnp.concatenate([p[layer] for p in parts], axis=-1) + dw).astype(jnp.bfloat16)
    new = tuple(p.at[layer].set(d) if mode == "init" else p.at[layer].add(d) for p, d in zip(parts, pieces))
    return new if isinstance(acc, tuple) else new[0]


def _padded(f):
    # ragged_dot pads its rows to a multiple of 512 and the epilogue paths call the kernels unpadded: XLA's CPU dot may
    # sum in another order for another row count, which the kernels do not, so every stand-in runs on padded rows.
    def op(a, *rest, **kwargs):
        rows = a.shape[0]
        pad = -rows % 512
        return f(jnp.pad(a, ((0, pad), (0, 0))), *rest, **kwargs)[:rows]

    return op


@_padded
def _nn_rows(a, b, o):
    return jnp.einsum("mk,gkn,mg->mn", _f32(a), _f32(b), _groups(o, a.shape[0], b.shape[0])).astype(jnp.bfloat16)


def _nn(a, b, o, row_index=None):
    return _nn_rows(_gather(a, row_index), b, o)


@_padded
def _nt(a, b, o):
    return jnp.einsum("mk,gnk,mg->mn", _f32(a), _f32(b), _groups(o, a.shape[0], b.shape[0])).astype(jnp.bfloat16)


def _scatter(gemm):
    # The row-scatter epilogue: dst[dst_row[r]] = bf16(f32(C[r]) * row_scale[r] + 0.0) for the computed rows.
    def op(a, b, o, dst_row, *, row_scale=None, dst=None, dst_rows=None, store_c="all"):
        c = gemm(a, b, o)
        rows = _f32(c) if row_scale is None else _f32(c) * row_scale[:, None]
        rows = (rows + jnp.float32(0.0)).astype(jnp.bfloat16)
        if dst is None:
            dst = jnp.zeros((dst_rows, c.shape[1]), jnp.bfloat16)
        live = (dst_row >= 0) & (jnp.arange(c.shape[0]) < o[-1])
        out = dst.at[jnp.where(live, dst_row, dst.shape[0])].set(rows, mode="drop", unique_indices=True)
        return (None if store_c == "none" else c), out

    return op


# The jax-flydsl ops with their contracts (float32 accumulation, bf16 output), in jnp.
FAKE_OPS = types.SimpleNamespace(
    grouped_gemm_bf16_nn=_nn,
    grouped_gemm_bf16_nt=_nt,
    grouped_gemm_bf16_wgrad=lambda a, b, o, a_row_index=None: _wgrad_f32(_gather(a, a_row_index), b, o).astype(
        jnp.bfloat16
    ),
    grouped_gemm_bf16_wgrad_accum=_wgrad_accum,
    grouped_gemm_bf16_nn_scatter=_scatter(_nn),
    grouped_gemm_bf16_nt_scatter=_scatter(_nt),
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


@pytest.mark.timeout(900)
def test_in_kernel_sums_with_the_static_attention_schedule():
    # The trace-time attention schedule runs its own layer loop (a scanned group of four layers, then two), which
    # carries the sums like the per-layer cond's scan; with the overlapped last microbatch it reads them per layer.
    _run(
        """
        cfg = dataclasses.replace(GA_CFG, num_layers=6, attention_schedule="static")
        assert_same_step(cfg=cfg)
        assert_same_step(cfg=cfg, zero1=True, gather=True, overlap=True)
        """
    )


@pytest.mark.timeout(900)
def test_stacked_options_give_the_all_off_step():
    # The stack1 options together: the fused residual add + RMS norm + GatedNorm, the fused q/k norm + RoPE and the
    # routing-metadata kernels in Pallas (the interpreter off GPU), and the in-kernel expert gradient sums (the jnp
    # stand-ins above), in the loop and with ZeRO-1, the first microbatch's gathers and the overlapped last
    # microbatch. Against the fused norm alone, the others change only float32 rounding of the q/k norm's statistic
    # (gradients within 1%). Against all four off, the gradients differ by what the fused norm alone changes with bf16
    # compute (its backward rounds once where the separate modules round at each op): up to ~9% in norm for the router's
    # in this toy model, whose bf16 gradients are ~6-20% from float32 ones either way. head_dim 64 as in the q/k
    # kernel's step test.
    # The stack2 options on top (the dispatch gather inside the up-projection GEMM, the combine and dispatch backward as
    # GEMM row-scatter epilogues, the Pallas gather-sums; the jnp stand-ins above for the gathered-A and scatter
    # kernels) are exact rewrites of the stack1 step, including the expert gradient sums in every accumulator mode
    # ("init" in the first microbatch, "accum" in the loop, "final" in the overlapped last one): the loss matches bit
    # for bit; the gradients match to rounding, since the scatter epilogue takes the routing-weight gradient from the
    # saved expert outputs and the unfused path from the down projection's backward (_down_project_combine).
    _run(
        """
        off = dataclasses.replace(GA_CFG, head_dim=64)
        fused = dataclasses.replace(off, fused_norm_impl="pallas")
        on = dataclasses.replace(fused, qk_norm_rope="pallas", moe_routing_metadata="pallas")
        stack2 = dataclasses.replace(
            on, moe_gather_sum="pallas", moe_flydsl_gather_a=True, moe_scatter_epilogue="flydsl"
        )

        def assert_close(grads, reference, rtol, label):
            for (path, a), b in zip(
                jax.tree_util.tree_leaves_with_path(reference), jax.tree.leaves(grads), strict=True
            ):
                a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
                error = float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(a)), 1e-30)
                assert error <= rtol, (label, jax.tree_util.keystr(path), error)

        cases = [("replica4_expert2", {}), ("replica2_data2_expert2", {"zero1": True, "gather": True, "overlap": True})]
        for name, kwargs in cases:
            mesh = MESHES[name]
            loss, grads = accumulated_grads(mesh, accumulation="xla", cfg=off, **kwargs)
            _, fused_grads = accumulated_grads(mesh, accumulation="xla", cfg=fused, **kwargs)
            on_loss, on_grads = accumulated_grads(mesh, accumulation="flydsl", cfg=on, **kwargs)
            np.testing.assert_allclose(on_loss, loss, rtol=2e-2, err_msg=name)
            assert_close(on_grads, fused_grads, 2e-2, (name, "fused norm only"))
            assert_close(on_grads, grads, 0.15, (name, "all off"))
            s2_loss, s2_grads = accumulated_grads(mesh, accumulation="flydsl", cfg=stack2, **kwargs)
            assert np.array_equal(s2_loss, on_loss), (name, s2_loss, on_loss)
            assert_close(s2_grads, on_grads, 2e-2, (name, "stack2"))
        """
    )
