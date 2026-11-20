# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``ring_filtered``'s expert matmuls as jax-flydsl GEMMs that gather, scatter and accumulate in their epilogues.

On the expert side ``ring_filtered`` gathers the received rows into expert order (``x_dispatch``), runs the up
projection on them, runs the down projection, and sums each received row's at most ``K`` dispatch rows back
(``out[q] = sum_j w[q, j] * Y[position[q, j]]``); the up projection's backward sums the dispatch rows' data gradient
the same way (``dReceived[q] = sum_j dX[position[q, j]]``). With jax-flydsl:

- the up projection's GEMMs read their A rows from the received buffer through a row index (``gather_a``), so
  ``x_dispatch`` is never built and the received rows are what the remat policy saves;
- a received row with exactly one kept slot (about 81% of those that carry an assignment at the June shapes) is a
  single scaled row, so the GEMM that produces it writes it to its received row from its epilogue (``nn_scatter`` for
  the down projection, ``nt_scatter`` for the up projection's data gradient), and the Pallas gather-sum of
  ``pallas_gather_sum.gather_sum_rows_into`` then writes only the other received rows, in place;
- with an ``ExpertGradAccumulator`` the weight-gradient GEMMs fold each layer's gradient into float32 sums in place
  (``grouped_gemm_bf16_wgrad_accum``), instead of returning a bf16 gradient for a separate add.

Values and gradients equal ``ring_filtered``'s own path bit for bit: a single-slot row's sum is ``f32(bf16 row) *
weight`` plus the other slots' zero terms, which the epilogue reproduces with its ``+ 0.0``, and the gather-sum adds
the same terms in the same order. The down projection keeps ``ring_filtered``'s backward
(``_down_project_combine``): the routing-weight gradient comes from ``<hidden, dhidden> / w``, so the down
projection's output is neither saved nor recomputed.

``expert_gemm()`` selects the kernels: "flydsl" when ``ragged_dot`` runs jax-flydsl (``RAGGED_DOT_IMPL=flydsl``), None
otherwise, when ``ring_filtered`` runs its own path. "reference" runs the same data flow with XLA standing in for the
jax-flydsl GEMMs, for tests off GPU. Fork overlay: jax-flydsl is a private AMD package.
"""

import functools
import importlib
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.ep_common import _combine_gather_sum_impl
from levanter.grug._moe.ep_ring import _gather_sum_slots, _take_valid_rows
from levanter.grug._moe.fp8_dispatch import FP8_DTYPE, SendBack, dequantize_rows
from levanter.grug._moe.grad_accum import GradAccumulation, accumulate_reference
from levanter.grug._moe.pallas_gather_sum import gather_sum_rows_into

# haliax.nn re-exports the function under the module's name, so the module (with the flydsl helpers) is imported by
# path, and its helpers are looked up at call time.
_rd = importlib.import_module("haliax.nn.ragged_dot")

ExpertGemm: TypeAlias = Literal["flydsl", "reference"]

# jax-flydsl's gathered nn GEMM with an fp8 A operand (the fp8 A loader of the fp8 dispatch, fork overlay):
# ``grouped_gemm_fp8_rows_nn(a_q, a_scale, b, group_offsets, *, row_index)`` with ``a_q`` [R, K] float8_e4m3fn (or its
# uint8 bits), ``a_scale`` [R] float32, ``b`` [G, K, N] bf16 and ``row_index`` [total_M] ints. Output row ``r`` is
# ``bf16(a_scale[i] * sum_k f32(a_q[i, k]) * b[g, k, :])`` with ``i = row_index[r]`` (zeros for an index outside
# [0, R)): the kernel converts the fp8 rows exactly and applies the row scale to the f32 accumulator in its epilogue.
# That is ``grouped_gemm_bf16_nn(dequantized a, b, group_offsets, row_index=row_index)`` without the bf16 rounding of
# ``q * scale``: within about one bf16 ulp of the unfused path, not bit for bit (bitwise only for power-of-two scales).
FP8_UP_KERNEL = "grouped_gemm_fp8_rows_nn"


def expert_gemm() -> ExpertGemm | None:
    """The expert kernels ``ring_filtered`` runs: "flydsl" when ``ragged_dot`` runs jax-flydsl, else None."""
    return "flydsl" if _rd.flydsl_selected() else None


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


def _ops(gemm: ExpertGemm):
    """The jax-flydsl module for "flydsl", None (XLA) for "reference"."""
    return _rd._flydsl_grouped_gemm_ops() if gemm == "flydsl" else None


def _gathered_rows(source: Float[Array, "R H"], row_index: Int[Array, "P"]) -> Float[Array, "P H"]:
    """``source[row_index]``, zeros for an index past the source."""
    valid = row_index < source.shape[0]
    return _take_valid_rows(source, jnp.minimum(row_index, source.shape[0] - 1), valid)


def _ragged_dot_vjp(gemm: ExpertGemm, lhs, rhs, group_sizes, dout):
    """``(dlhs, drhs)`` of ``ragged_dot(lhs, rhs, group_sizes)``, through the kernels the unfused path uses."""
    if gemm == "flydsl":
        dlhs, drhs, _ = _rd.ragged_dot_flydsl_bwd((lhs, rhs, group_sizes), dout)
        return dlhs, drhs
    _, vjp = jax.vjp(lambda a, b: _rd.ragged_dot(a, b, group_sizes), lhs, rhs)
    return vjp(dout)


def _data_grad(gemm: ExpertGemm, dout, rhs, group_sizes):
    """``dout @ rhs[g].T``, the data gradient of ``ragged_dot(., rhs, group_sizes)``."""
    if gemm == "flydsl":
        return _rd._flydsl_grouped_matmul(dout, rhs, group_sizes, _rd._DLHS_DIM_NUMS)
    lhs = jnp.zeros((dout.shape[0], rhs.shape[1]), dout.dtype)
    _, vjp = jax.vjp(lambda a: _rd.ragged_dot(a, rhs, group_sizes), lhs)
    return vjp(dout)[0]


def weight_grad(
    gemm: ExpertGemm,
    lhs: jax.Array,
    dout: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    acc: Any = None,
    layer: jax.Array | None = None,
    mode: GradAccumulation | None = None,
    row_index: jax.Array | None = None,
) -> Any:
    """The weight gradient of ``ragged_dot(A, rhs, group_sizes)`` for the output cotangent ``dout``, ``A`` being
    ``lhs`` or, with ``row_index``, ``lhs``'s rows gathered through it (zeros past ``lhs``).

    Without ``mode`` it returns the gradient; with one it folds it into layer ``layer`` of ``acc`` as
    ``grouped_gemm_bf16_wgrad_accum`` does: the updated ``acc`` for "init"/"accum", the running sum for "final".
    """
    if gemm == "reference":
        a = lhs if row_index is None else _gathered_rows(lhs, row_index)
        _, vjp = jax.vjp(lambda w: _rd.ragged_dot(a, w, group_sizes), rhs)
        (dw,) = vjp(dout)
        if mode is None:
            return dw
        return accumulate_reference(dw, acc, layer, mode)
    ops = _ops(gemm)
    offsets = _rd.flydsl_offsets(group_sizes, dout.shape[0])
    gather = {} if row_index is None else {"a_row_index": row_index}
    if mode is None:
        return ops.grouped_gemm_bf16_wgrad(lhs, dout, offsets, **gather)
    return ops.grouped_gemm_bf16_wgrad_accum(lhs, dout, offsets, acc, layer, mode=mode, **gather)


def _accumulated_weight_grad(gemm, mode, lhs, dout, rhs, group_sizes, acc, layer, row_index=None):
    """``(d_rhs, d_acc)`` of a projection whose weight gradient goes into ``acc`` with ``mode``: with "final" the
    weight's cotangent is the running sum and ``acc`` gets none, otherwise the weight's is zero."""
    result = weight_grad(gemm, lhs, dout, rhs, group_sizes, acc, layer, mode, row_index=row_index)
    if mode == "final":
        return result, None
    return None, result


def _scatter_reference(dst, rows, dst_row, scale):
    """The epilogue's write in XLA: ``dst[dst_row[r]] = round(f32(rows[r]) * scale[r] + 0.0)``."""
    values = rows.astype(jnp.float32)
    if scale is not None:
        values = values * scale[:, None]
    values = (values + jnp.float32(0.0)).astype(dst.dtype)
    target = jnp.where(dst_row >= 0, dst_row, dst.shape[0])
    return dst.at[target].set(values, mode="drop", unique_indices=True)


def _gather_sum_into(gemm: ExpertGemm, dst, rows, position, write, weights):
    """``dst`` with each received row where ``write`` replaced by its gather-sum of ``rows``."""
    if gemm == "flydsl":
        return gather_sum_rows_into(dst, rows, position, write, weights)
    # The XLA formulations the kernel reproduces, for every row, then selected.
    if weights is None:
        full = _gather_sum_slots(rows, position)
    else:
        full = _combine_gather_sum_impl(rows, position.reshape(-1), weights, rows.dtype)
    return jnp.where(write[:, None], full, dst)


# Down projection and combine: out[q] = sum_j weights[q, j] * Y[position[q, j]], Y = ragged_dot(h, w2, group_sizes).
def down_project_combine(
    hidden: Float[Array, "C I"],
    w2: Float[Array, "E I H"],
    group_sizes: Int[Array, " E"],
    weights: Float[Array, "R K"],
    position: Int[Array, "R K"],
    picked: Int[Array, " C"],
    dst_row: Int[Array, " C"],
    write: Bool[Array, " R"],
    gemm: ExpertGemm,
    *,
    acc: Any = None,
    layer: jax.Array | None = None,
    mode: GradAccumulation | None = None,
) -> Any:
    """``ep_ring_filtered._down_project_combine`` with the single-slot rows written by the down projection's epilogue.
    ``dst_row`` and ``write`` come from :func:`scatter_plan`; ``picked`` is the dispatch rows' flat assignments (the
    sentinel ``R * K`` for padding).

    With ``mode`` the down projection's weight gradient goes into layer ``layer`` of ``acc``: "init"/"accum" return
    ``(out, acc)`` and the backward updates the cotangent of the returned ``acc``; "final" returns ``out`` and gives
    ``w2`` the running sum as its cotangent."""
    if mode is None:
        acc = layer = None
    return _down_combine(hidden, w2, group_sizes, weights, position, picked, dst_row, write, acc, layer, gemm, mode)


@functools.partial(jax.custom_vjp, nondiff_argnums=(10, 11))
def _down_combine(hidden, w2, group_sizes, weights, position, picked, dst_row, write, acc, layer, gemm, mode):
    return _down_combine_fwd(
        hidden, w2, group_sizes, weights, position, picked, dst_row, write, acc, layer, gemm, mode
    )[0]


def _down_scatter(gemm: ExpertGemm, hidden, w2, group_sizes, dst_row, row_scale, num_rows):
    """``Y = ragged_dot(hidden, w2)`` and a fresh ``[num_rows, H]`` buffer holding Y's single-slot rows scaled into
    their received rows (the other rows undefined). Y's rows past the groups are not zeroed: nothing reads them."""
    if gemm == "flydsl":
        _rd._check_flydsl_operands(hidden, w2, group_sizes)
        offsets = _rd.flydsl_offsets(group_sizes, hidden.shape[0])
        return _ops(gemm).grouped_gemm_bf16_nn_scatter(
            hidden, w2, offsets, dst_row, row_scale=row_scale, dst_rows=num_rows, store_c="unscattered"
        )
    y = _rd.ragged_dot(hidden, w2, group_sizes)
    return y, _scatter_reference(jnp.zeros((num_rows, y.shape[1]), y.dtype), y, dst_row, row_scale)


def _down_combine_fwd(hidden, w2, group_sizes, weights, position, picked, dst_row, write, acc, layer, gemm, mode):
    num_rows, topk = position.shape
    row_scale = weights.astype(jnp.float32).reshape(-1)[jnp.minimum(picked, num_rows * topk - 1)]
    y, out = _down_scatter(gemm, hidden, w2, group_sizes, dst_row, row_scale, num_rows)
    out = _gather_sum_into(gemm, out, y, position, write, weights)
    residuals = (hidden, w2, group_sizes, weights, position, picked, layer, acc if mode == "final" else None)
    return (out if mode in (None, "final") else (out, acc)), residuals


def _down_combine_bwd(gemm, mode, residuals, cotangent):
    hidden, w2, group_sizes, weights, position, picked, layer, acc = residuals
    if mode in ("init", "accum"):
        cotangent, acc = cotangent
    # ep_ring_filtered._down_project_combine_bwd, with the GEMMs on ``gemm``.
    num_rows, num_slots = weights.shape
    flat_valid = picked < num_rows * num_slots
    safe_flat = jnp.minimum(picked, num_rows * num_slots - 1)
    row_weight = jnp.where(flat_valid, weights.reshape(-1)[safe_flat].astype(jnp.float32), 0.0)
    d_y = (cotangent[safe_flat // num_slots].astype(jnp.float32) * row_weight[:, None]).astype(hidden.dtype)
    if mode is None:
        d_hidden, d_w2 = _ragged_dot_vjp(gemm, hidden, w2, group_sizes, d_y)
        d_acc = None
    else:
        d_hidden = _data_grad(gemm, d_y, w2, group_sizes)
        d_w2, d_acc = _accumulated_weight_grad(gemm, mode, hidden, d_y, w2, group_sizes, acc, layer)

    weighted_score = jnp.sum(hidden.astype(jnp.float32) * d_hidden.astype(jnp.float32), axis=-1)
    nonzero = row_weight != 0
    row_score = jnp.where(nonzero, weighted_score / jnp.where(nonzero, row_weight, 1.0), 0.0)
    d_weights = jnp.take(row_score, position, mode="fill", fill_value=0).astype(weights.dtype)
    return d_hidden, d_w2, None, d_weights, None, None, None, None, d_acc, None


_down_combine.defvjp(_down_combine_fwd, _down_combine_bwd)


# Dispatch and up projection: ragged_dot(received[row_index] (zeros past it), w13). Its backward sums each received
# row's dispatch-row gradients: dReceived[q] = sum_j dX[position[q, j]].
def dispatch_up(
    received: Float[Array, "R H"],
    row_index: Int[Array, " C"],
    w13: Float[Array, "E H I2"],
    group_sizes: Int[Array, " E"],
    position: Int[Array, "R K"],
    dst_row: Int[Array, " C"],
    write: Bool[Array, " R"],
    gemm: ExpertGemm,
    *,
    acc: Any = None,
    layer: jax.Array | None = None,
    mode: GradAccumulation | None = None,
) -> Any:
    """``ragged_dot(_dispatch_rows(received, ...), w13, group_sizes)`` with the dispatch gather inside the forward and
    weight-gradient GEMMs (their A rows read through ``row_index``, the sentinel ``R`` reading zeros), and the backward's
    single-slot received rows written by the data-gradient GEMM's epilogue. The gathered copy is never built, so the
    caller saves ``received``. ``dst_row`` and ``write`` come from :func:`scatter_plan`. With ``mode`` the weight
    gradient goes into layer ``layer`` of ``acc``, as in :func:`down_project_combine`."""
    if mode is None:
        acc = layer = None
    return _dispatch_up(received, row_index, w13, group_sizes, position, dst_row, write, acc, layer, gemm, mode)


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10))
def _dispatch_up(received, row_index, w13, group_sizes, position, dst_row, write, acc, layer, gemm, mode):
    return _dispatch_up_fwd(received, row_index, w13, group_sizes, position, dst_row, write, acc, layer, gemm, mode)[0]


def _up_gathered(gemm: ExpertGemm, received, row_index, w13, group_sizes):
    if gemm == "flydsl":
        _rd._check_flydsl_operands(received, w13, group_sizes)
        offsets = _rd.flydsl_offsets(group_sizes, row_index.shape[0])
        out = _ops(gemm).grouped_gemm_bf16_nn(received, w13, offsets, row_index=row_index)
        return _rd._zero_rows_past_groups_in_place(out, group_sizes)
    return _rd.ragged_dot(_gathered_rows(received, row_index), w13, group_sizes)


def _dgrad_scatter(gemm: ExpertGemm, dy, w13, group_sizes, dst_row, num_rows):
    """``dX = dy @ w13[g].T`` for the rows the gather-sum reads (``dst_row < 0``; the others undefined) and a fresh
    ``[num_rows, H]`` buffer holding dX's single-slot rows in their received rows (the other rows undefined)."""
    if gemm == "flydsl":
        offsets = _rd.flydsl_offsets(group_sizes, dy.shape[0])
        return _ops(gemm).grouped_gemm_bf16_nt_scatter(
            dy, w13, offsets, dst_row, dst_rows=num_rows, store_c="unscattered"
        )
    dx = _data_grad(gemm, dy, w13, group_sizes)
    return dx, _scatter_reference(jnp.zeros((num_rows, dx.shape[1]), dx.dtype), dx, dst_row, None)


def _dispatch_up_fwd(received, row_index, w13, group_sizes, position, dst_row, write, acc, layer, gemm, mode):
    out = _up_gathered(gemm, received, row_index, w13, group_sizes)
    residuals = (
        received,
        row_index,
        w13,
        group_sizes,
        position,
        dst_row,
        write,
        layer,
        acc if mode == "final" else None,
    )
    return (out if mode in (None, "final") else (out, acc)), residuals


def _dispatch_up_bwd(gemm, mode, residuals, cotangent):
    received, row_index, w13, group_sizes, position, dst_row, write, layer, acc = residuals
    if mode in ("init", "accum"):
        cotangent, acc = cotangent
    dx, d_received = _dgrad_scatter(gemm, cotangent, w13, group_sizes, dst_row, write.shape[0])
    d_received = _gather_sum_into(gemm, d_received, dx, position, write, None)
    if mode is None:
        d_w13 = weight_grad(gemm, received, cotangent, w13, group_sizes, row_index=row_index)
        d_acc = None
    else:
        d_w13, d_acc = _accumulated_weight_grad(
            gemm, mode, received, cotangent, w13, group_sizes, acc, layer, row_index
        )
    return d_received, None, d_w13, None, None, None, None, d_acc, None


_dispatch_up.defvjp(_dispatch_up_fwd, _dispatch_up_bwd)


# fp8 dispatch, fused: the up projection reads the received rows as float8_e4m3fn bits and float32 row scales.
def dispatch_up_fp8(
    x_source: Float[Array, "T H"],
    received_bits: Any,
    received_scale: Float[Array, " R"],
    row_index: Int[Array, " C"],
    w13: Float[Array, "E H I2"],
    group_sizes: Int[Array, " E"],
    position: Int[Array, "R K"],
    dst_row: Int[Array, " C"],
    write: Bool[Array, " R"],
    gemm: ExpertGemm,
    send_back: SendBack,
    send_back_args: tuple[jax.Array, ...],
    *,
    acc: Any = None,
    layer: jax.Array | None = None,
    mode: GradAccumulation | None = None,
) -> Any:
    """:func:`dispatch_up` on the received rows ``dequantize_rows(received_bits, received_scale, x_source.dtype)``,
    which the forward never builds: the jax-flydsl fp8 kernel (``FP8_UP_KERNEL``) converts the fp8 rows exactly as it
    loads them and applies the row scales to its f32 accumulator, so the forward is within about one bf16 ulp of the
    unfused path (it skips the bf16 rounding of ``q * s``), not bitwise equal to it.

    The backward is :func:`dispatch_up`'s at the dequantized rows (rebuilt from the saved bits and scales), and the
    received rows' cotangent goes on to ``x_source``, the sender's rows, through ``send_back(d_received,
    *send_back_args)`` (straight-through: the transpose of the bfloat16 transport). ``x_source`` itself is not read.
    The bits and the scales get no gradient. ``acc``/``layer``/``mode`` as in :func:`dispatch_up`."""
    if mode is None:
        acc = layer = None
    return _dispatch_up_fp8(
        x_source,
        received_bits,
        received_scale,
        row_index,
        w13,
        group_sizes,
        position,
        dst_row,
        write,
        send_back_args,
        acc,
        layer,
        gemm,
        mode,
        send_back,
    )


def _up_gathered_fp8(gemm: ExpertGemm, bits, scale, row_index, w13, group_sizes, dtype):
    if gemm == "reference":
        return _up_gathered(gemm, dequantize_rows(bits, scale, dtype), row_index, w13, group_sizes)
    if dtype != jnp.bfloat16:
        raise ValueError(f"the fp8 up projection needs bfloat16 tokens, got {dtype}")
    _rd._check_flydsl_operands(jax.ShapeDtypeStruct(bits.shape, jnp.bfloat16), w13, group_sizes)
    kernel = getattr(_ops(gemm), FP8_UP_KERNEL, None)
    if kernel is None:
        raise NotImplementedError(
            f"fp8_dispatch='fused' needs jax-flydsl's {FP8_UP_KERNEL} (the fp8 A loader), which this jax-flydsl lacks"
        )
    offsets = _rd.flydsl_offsets(group_sizes, row_index.shape[0])
    out = kernel(jax.lax.bitcast_convert_type(bits, FP8_DTYPE), scale, w13, offsets, row_index=row_index)
    return _rd._zero_rows_past_groups_in_place(out, group_sizes)


@functools.partial(jax.custom_vjp, nondiff_argnums=(12, 13, 14))
def _dispatch_up_fp8(
    x_source, bits, scale, row_index, w13, group_sizes, position, dst_row, write, sb_args, acc, layer, gemm, mode, sb
):
    return _dispatch_up_fp8_fwd(
        x_source,
        bits,
        scale,
        row_index,
        w13,
        group_sizes,
        position,
        dst_row,
        write,
        sb_args,
        acc,
        layer,
        gemm,
        mode,
        sb,
    )[0]


def _dispatch_up_fp8_fwd(
    x_source, bits, scale, row_index, w13, group_sizes, position, dst_row, write, sb_args, acc, layer, gemm, mode, sb
):
    out = _up_gathered_fp8(gemm, bits, scale, row_index, w13, group_sizes, x_source.dtype)
    # The zero-size array carries the dtype of the dequantized rows and of x_source's cotangent.
    residuals = (
        bits,
        scale,
        row_index,
        w13,
        group_sizes,
        position,
        dst_row,
        write,
        layer,
        acc if mode == "final" else None,
        sb_args,
        jnp.zeros((0,), x_source.dtype),
    )
    return (out if mode in (None, "final") else (out, acc)), residuals


def _dispatch_up_fp8_bwd(gemm, mode, send_back, residuals, cotangent):
    bits, scale, row_index, w13, group_sizes, position, dst_row, write, layer, acc, sb_args, dtype_carrier = residuals
    received = dequantize_rows(bits, scale, dtype_carrier.dtype)
    d_received, _, d_w13, _, _, _, _, d_acc, _ = _dispatch_up_bwd(
        gemm, mode, (received, row_index, w13, group_sizes, position, dst_row, write, layer, acc), cotangent
    )
    d_source = send_back(d_received, *sb_args)
    return d_source, None, None, None, d_w13, None, None, None, None, None, d_acc, None


_dispatch_up_fp8.defvjp(_dispatch_up_fp8_fwd, _dispatch_up_fp8_bwd)
