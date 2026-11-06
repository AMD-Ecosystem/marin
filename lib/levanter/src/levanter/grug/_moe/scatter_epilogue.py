# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``ring_dedup``'s combine and dispatch backward as row-scatter epilogues of the expert GEMMs.

Each received row ``q`` of an expert shard sums its at most ``K`` dispatch rows: the forward combine
``out[q] = sum_j w[q, j] * Y[position[q, j]]`` after the down projection, and the dispatch backward
``dReceived[q] = sum_j dX[position[q, j]]`` after the up projection's data gradient. On the June shapes about 81% of
the received rows that carry an assignment have exactly one kept slot. For those the sum is a single scaled row, so
the GEMM that produces the row writes it to its received row directly (jax-flydsl's ``grouped_gemm_bf16_nn_scatter``
for the down projection, ``grouped_gemm_bf16_nt_scatter`` for the up projection's data gradient); the GEMM output
does not have to be read back for them. The other received rows (two or more kept slots, or none) are written
afterwards, in place, by the Pallas gather-sum of :mod:`pallas_gather_sum`, which reads only their dispatch rows.

Both give the XLA formulations' bits: a single-slot row's sum is ``f32(bf16 row) * weight`` plus the zero fill of the
other slots, which the epilogue reproduces with its ``+ 0.0``, and the gather-sum adds the same terms in the same
order. ``"reference"`` runs the same data flow with XLA standing in for the jax-flydsl GEMMs, for CPU tests.

The down projection still stores its output ``Y``, which the backward needs for the routing-weight gradient (it is
saved under the same checkpoint name as without the epilogue). The up projection's data gradient stores only the
rows the gather-sum reads. Gradients are those of the unfused path: the backward runs the same kernels.
"""

import functools
import importlib
from typing import Any, Literal

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import GradAccumulation
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_DISPATCH_OUTPUT
from levanter.grug._moe.ep_common import _combine_gather_sum_bwd, _combine_gather_sum_impl
from levanter.grug._moe.ep_ring import _gather_sum_slots, _take_valid_rows
from levanter.grug._moe.flydsl_gather import _gathered_rows, weight_grad
from levanter.grug._moe.pallas_gather_sum import gather_sum_rows_into

# haliax.nn re-exports the function under the module's name, so the module (with the flydsl helpers) is imported by path.
_ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")

MoeScatterEpilogue = Literal["off", "flydsl", "reference"]
"""How ``ring_dedup`` runs the combine and the dispatch backward: as separate gathers ("off"), as row-scatter
epilogues of jax-flydsl's grouped GEMMs plus a gather-sum of the other rows ("flydsl"), or the same data flow with XLA
GEMMs ("reference", for tests off GPU)."""


def scatter_plan(
    position: Int[Array, "R K"], picked: Int[Array, " C"], dispatch_valid: Bool[Array, " C"]
) -> tuple[Int[Array, " C"], Bool[Array, " R"]]:
    """Which dispatch rows the GEMM epilogue scatters, and which received rows the gather-sum writes.

    ``position[q, j]`` is the dispatch row of received row ``q``'s ``j``-th assignment, or ``C`` (the dispatch row
    count) if it was not kept; ``picked[r]`` is dispatch row ``r``'s flat assignment ``q * K + j``.

    Returns ``dst_row`` (dispatch row ``r``'s received row when that row has exactly one kept slot, else -1; distinct
    over its non-negative entries, since every received row owns its dispatch rows) and ``write`` (received rows with
    zero or several kept slots).
    """
    num_rows, topk = position.shape
    capacity = picked.shape[0]
    count = jnp.sum(position < capacity, axis=1, dtype=jnp.int32)
    received_row = picked // topk
    single = (count == 1)[jnp.minimum(received_row, num_rows - 1)]
    dst_row = jnp.where(dispatch_valid & single, received_row, -1).astype(jnp.int32)
    return dst_row, count != 1


def _flydsl_ops():
    return _ragged_dot_module._flydsl_grouped_gemm_ops()


def _offsets(group_sizes: Int[Array, " G"], rows: int) -> Int[Array, " G1"]:
    # As haliax's flydsl ragged_dot: clamped to the row count.
    return jnp.minimum(jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True), rows)


def _scatter_reference(dst, rows, dst_row, scale):
    """The epilogue's write in XLA: ``dst[dst_row[r]] = bf16(f32(rows[r]) * scale[r] + 0.0)``."""
    values = rows.astype(jnp.float32)
    if scale is not None:
        values = values * scale[:, None]
    values = (values + jnp.float32(0.0)).astype(dst.dtype)
    target = jnp.where(dst_row >= 0, dst_row, dst.shape[0])
    return dst.at[target].set(values, mode="drop", unique_indices=True)


def _up(gemm, x, w13, group_sizes):
    implementation = "flydsl" if gemm == "flydsl" else "auto"
    return _ragged_dot_module.ragged_dot(x, w13, group_sizes, implementation=implementation)


def _ragged_dot_vjp(gemm, lhs, rhs, group_sizes, dout):
    """``(dlhs, drhs)`` of ``ragged_dot(lhs, rhs, group_sizes)``, through the kernels the unfused path uses."""
    if gemm == "flydsl":
        dlhs, drhs, _ = _ragged_dot_module._ragged_dot_flydsl_bwd((lhs, rhs, group_sizes), dout)
        return dlhs, drhs
    _, vjp = jax.vjp(lambda a, b: _ragged_dot_module.ragged_dot(a, b, group_sizes), lhs, rhs)
    return vjp(dout)


def _down_scatter(gemm, h, w2, group_sizes, dst_row, row_scale, num_rows):
    """``Y = ragged_dot(h, w2)`` (rows past the groups zeroed, as ragged_dot does) and a fresh ``[num_rows, N]``
    buffer holding Y's single-slot rows scaled into their received rows (the other rows undefined)."""
    if gemm == "flydsl":
        ops = _flydsl_ops()
        _ragged_dot_module._check_flydsl_operands(h, w2, group_sizes)
        y, dst = ops.grouped_gemm_bf16_nn_scatter(
            h, w2, _offsets(group_sizes, h.shape[0]), dst_row, row_scale=row_scale, dst_rows=num_rows
        )
        return _ragged_dot_module._zero_rows_past_groups_in_place(y, group_sizes), dst
    y = _ragged_dot_module.ragged_dot(h, w2, group_sizes)
    dst = jnp.zeros((num_rows, y.shape[1]), y.dtype)
    return y, _scatter_reference(dst, y, dst_row, row_scale)


def _dgrad_scatter(gemm, dy, w13, group_sizes, dst_row, num_rows):
    """``dX = dy @ w13[g].T`` for the rows the gather-sum reads (``dst_row < 0``; the others undefined) and a fresh
    ``[num_rows, K]`` buffer holding dX's single-slot rows in their received rows (the other rows undefined)."""
    if gemm == "flydsl":
        ops = _flydsl_ops()
        return ops.grouped_gemm_bf16_nt_scatter(
            dy, w13, _offsets(group_sizes, dy.shape[0]), dst_row, dst_rows=num_rows, store_c="unscattered"
        )
    dx, _ = _ragged_dot_vjp(gemm, jnp.zeros((dy.shape[0], w13.shape[1]), dy.dtype), w13, group_sizes, dy)
    dst = jnp.zeros((num_rows, dx.shape[1]), dx.dtype)
    return dx, _scatter_reference(dst, dx, dst_row, None)


def _gather_sum(dst, rows, position, write, weights, gemm):
    if gemm == "flydsl":
        return gather_sum_rows_into(dst, rows, position, write, weights)
    # The XLA formulations the kernel reproduces, for every row, then selected.
    if weights is None:
        full = _gather_sum_slots(rows, position)
    else:
        full = _combine_gather_sum_impl(rows, position.reshape(-1), weights, rows.dtype)
    return jnp.where(write[:, None], full, dst)


def _ops_or_none(gemm):
    """The jax-flydsl module for ``gemm="flydsl"``, None (XLA) for "reference"."""
    return _flydsl_ops() if gemm == "flydsl" else None


def _data_grad(gemm, dout, rhs, group_sizes):
    """``dout @ rhs[g].T``, the data gradient of ``ragged_dot(., rhs, group_sizes)``, through the unfused path's kernel."""
    if gemm == "flydsl":
        return _ragged_dot_module._flydsl_grouped_matmul(dout, rhs, group_sizes, _ragged_dot_module._DLHS_DIM_NUMS)
    lhs = jnp.zeros((dout.shape[0], rhs.shape[1]), dout.dtype)
    _, vjp = jax.vjp(lambda a: _ragged_dot_module.ragged_dot(a, rhs, group_sizes), lhs)
    return vjp(dout)[0]


def _accumulating_bwd(gemm, mode, lhs, dout, rhs, group_sizes, acc, layer, row_index=None):
    """``(d_lhs, d_rhs, d_acc)`` of a projection whose weight gradient goes into ``acc`` with ``mode`` (see
    ``ragged_dot_accumulating_grad``); ``row_index`` gathers ``lhs``'s rows for the weight gradient."""
    d_w = weight_grad(_ops_or_none(gemm), lhs, dout, rhs, group_sizes, acc, layer, mode, row_index=row_index)
    if mode == "final":
        return d_w, None
    return None, d_w


# Combine (forward of the down projection): out[q] = sum_j weights[q, j] * Y[position[q, j]].
def down_combine(
    h: Float[Array, "C I"],
    w2: Float[Array, "E I H"],
    group_sizes: Int[Array, " E"],
    position: Int[Array, "R K"],
    picked: Int[Array, " C"],
    weights: Float[Array, "R K"],
    dst_row: Int[Array, " C"],
    write: Bool[Array, " R"],
    gemm: MoeScatterEpilogue,
    *,
    acc: Any = None,
    layer: jax.Array | None = None,
    mode: GradAccumulation | None = None,
) -> Any:
    """``_combine_gather_sum(ragged_dot(h, w2, group_sizes), position, picked, weights)`` with the single-slot rows
    written by the down projection's epilogue. ``dst_row`` and ``write`` come from :func:`scatter_plan`.

    With ``mode`` the down projection's weight gradient goes into layer ``layer`` of ``acc`` (the in-kernel expert
    gradient sums), as ``ragged_dot_accumulating_grad`` does: "init"/"accum" return ``(out, acc)``, "final" ``out``."""
    if mode is None:
        acc = layer = None
    return _down_combine(h, w2, group_sizes, position, picked, weights, dst_row, write, acc, layer, gemm, mode)


@functools.partial(jax.custom_vjp, nondiff_argnums=(10, 11))
def _down_combine(h, w2, group_sizes, position, picked, weights, dst_row, write, acc, layer, gemm, mode):
    return _down_combine_fwd(h, w2, group_sizes, position, picked, weights, dst_row, write, acc, layer, gemm, mode)[0]


def _down_combine_fwd(h, w2, group_sizes, position, picked, weights, dst_row, write, acc, layer, gemm, mode):
    num_rows, topk = position.shape
    row_scale = weights.astype(jnp.float32).reshape(-1)[jnp.minimum(picked, num_rows * topk - 1)]
    y, out = _down_scatter(gemm, h, w2, group_sizes, dst_row, row_scale, num_rows)
    # Saved like the unfused path's down projection output: the backward needs Y for the routing-weight gradient.
    y = tree_checkpoint_name(y, _CHECKPOINT_DISPATCH_OUTPUT)
    out = _gather_sum(out, y, position, write, weights, gemm)
    residuals = (h, w2, group_sizes, y, position, picked, weights, layer, acc if mode == "final" else None)
    return (out if mode in (None, "final") else (out, acc)), residuals


def _down_combine_bwd(gemm, mode, residuals, cotangent):
    h, w2, group_sizes, y, position, picked, weights, layer, acc = residuals
    if mode in ("init", "accum"):
        cotangent, acc = cotangent
    d_y, _, _, d_weights = _combine_gather_sum_bwd(None, (y, position.reshape(-1), picked, weights), cotangent)
    if mode is None:
        d_h, d_w2 = _ragged_dot_vjp(gemm, h, w2, group_sizes, d_y)
        d_acc = None
    else:
        d_h = _data_grad(gemm, d_y, w2, group_sizes)
        d_w2, d_acc = _accumulating_bwd(gemm, mode, h, d_y, w2, group_sizes, acc, layer)
    return d_h, d_w2, None, None, None, d_weights, None, None, d_acc, None


_down_combine.defvjp(_down_combine_fwd, _down_combine_bwd)


# Dispatch and up projection: Y13 = ragged_dot(received[dispatch_row] (0 where invalid), w13). Its backward sums each
# received row's dispatch-row gradients: dReceived[q] = sum_j dX[position[q, j]].
def dispatch_up(
    received: Float[Array, "R H"],
    dispatch_row: Int[Array, " C"],
    dispatch_valid: Bool[Array, " C"],
    w13: Float[Array, "E H I2"],
    group_sizes: Int[Array, " E"],
    position: Int[Array, "R K"],
    dst_row: Int[Array, " C"],
    write: Bool[Array, " R"],
    gemm: MoeScatterEpilogue,
    *,
    gather_a: bool = False,
    acc: Any = None,
    layer: jax.Array | None = None,
    mode: GradAccumulation | None = None,
) -> Any:
    """``ragged_dot(_dispatch_rows(received, dispatch_row, dispatch_valid, position), w13, group_sizes)`` whose
    backward writes the single-slot received rows from the data-gradient GEMM's epilogue. ``dispatch_row`` must lie
    in ``[0, R)``; ``dst_row`` and ``write`` come from :func:`scatter_plan`.

    ``gather_a`` reads the dispatch rows of ``received`` inside the forward and weight-gradient GEMMs (jax-flydsl's
    row index, ``flydsl_gather``) instead of building the gathered copy, which is then not saved: the caller saves
    ``received``. With ``mode`` the weight gradient goes into layer ``layer`` of ``acc``, as in :func:`down_combine`.
    Values and gradients are the same bit for bit either way."""
    if mode is None:
        acc = layer = None
    return _dispatch_up(
        received,
        dispatch_row,
        dispatch_valid,
        w13,
        group_sizes,
        position,
        dst_row,
        write,
        acc,
        layer,
        gemm,
        gather_a,
        mode,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(10, 11, 12))
def _dispatch_up(
    received,
    dispatch_row,
    dispatch_valid,
    w13,
    group_sizes,
    position,
    dst_row,
    write,
    acc,
    layer,
    gemm,
    gather_a,
    mode,
):
    return _dispatch_up_fwd(
        received,
        dispatch_row,
        dispatch_valid,
        w13,
        group_sizes,
        position,
        dst_row,
        write,
        acc,
        layer,
        gemm,
        gather_a,
        mode,
    )[0]


def _up_gathered(gemm, received, row_index, w13, group_sizes):
    if gemm == "flydsl":
        ops = _flydsl_ops()
        _ragged_dot_module._check_flydsl_operands(received, w13, group_sizes)
        out = ops.grouped_gemm_bf16_nn(received, w13, _offsets(group_sizes, row_index.shape[0]), row_index=row_index)
        return _ragged_dot_module._zero_rows_past_groups_in_place(out, group_sizes)
    return _up(gemm, _gathered_rows(received, row_index), w13, group_sizes)


def _dispatch_up_fwd(
    received,
    dispatch_row,
    dispatch_valid,
    w13,
    group_sizes,
    position,
    dst_row,
    write,
    acc,
    layer,
    gemm,
    gather_a,
    mode,
):
    if gather_a:
        # The A rows by index, the sentinel R (zeros) where invalid; the residual is ``received`` itself.
        row_index = jnp.where(dispatch_valid, dispatch_row, received.shape[0]).astype(jnp.int32)
        x = received
        out = _up_gathered(gemm, received, row_index, w13, group_sizes)
    else:
        row_index = None
        x = tree_checkpoint_name(_take_valid_rows(received, dispatch_row, dispatch_valid), _CHECKPOINT_DISPATCH_INPUT)
        out = _up(gemm, x, w13, group_sizes)
    residuals = (x, row_index, w13, group_sizes, position, dst_row, write, layer, acc if mode == "final" else None)
    return (out if mode in (None, "final") else (out, acc)), residuals


def _dispatch_up_bwd(gemm, gather_a, mode, residuals, cotangent):
    x, row_index, w13, group_sizes, position, dst_row, write, layer, acc = residuals
    if mode in ("init", "accum"):
        cotangent, acc = cotangent
    dx, d_received = _dgrad_scatter(gemm, cotangent, w13, group_sizes, dst_row, write.shape[0])
    d_received = _gather_sum(d_received, dx, position, write, None, gemm)
    if mode is None and row_index is None:
        _, d_w13 = _ragged_dot_vjp(gemm, x, w13, group_sizes, cotangent)
        d_acc = None
    elif mode is None:
        d_w13 = weight_grad(_ops_or_none(gemm), x, cotangent, w13, group_sizes, row_index=row_index)
        d_acc = None
    else:
        d_w13, d_acc = _accumulating_bwd(gemm, mode, x, cotangent, w13, group_sizes, acc, layer, row_index)
    return d_received, None, None, d_w13, None, None, None, None, d_acc, None


_dispatch_up.defvjp(_dispatch_up_fwd, _dispatch_up_bwd)
