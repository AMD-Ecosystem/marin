# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""``ring_dedup``'s expert up projection with the dispatch gather done by the jax-flydsl GEMM's A loads.

``ring_dedup`` gathers the received rows into expert order (``x_dispatch``, a ``[P, H]`` copy) and runs the up
projection on it; the up projection's weight gradient reads that copy again. jax-flydsl's grouped NN and wgrad kernels
can read the rows straight from the received buffer through a row index instead, with the same bits, so
:func:`gathered_up_projection` never materializes ``x_dispatch``. Its backward is the one ``ring_dedup`` already runs:
the data gradient through the grouped NT kernel, then the dispatch gather-sum of that gradient back onto the received
rows (XLA or Pallas, as ``gather_sum`` picks).

The kernels run with ``ragged_dot``'s flydsl implementation (``RAGGED_DOT_IMPL=flydsl``). Off GPU the same custom VJP
runs with an explicit gather and ``ragged_dot``, so CPU tests cover the wiring; on a GPU without flydsl it raises.
"""

import importlib
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from haliax.nn.ragged_dot import GradAccumulation
from jaxtyping import Array, Float, Int

from levanter.grug._moe.ep_ring import _gather_sum_slots, _take_valid_rows, _use_pallas_gather
from levanter.grug._moe.grad_accum import accumulate_reference
from levanter.grug._moe.pallas_gather_sum import MoeGatherSum, gather_sum_rows

# haliax.nn re-exports the function under the module's name, so the module (with the flydsl helpers) is imported by
# path, and its helpers are looked up at call time.
_rd = importlib.import_module("haliax.nn.ragged_dot")


def _flydsl_ops():
    """The jax-flydsl grouped GEMM module when ``ragged_dot`` runs flydsl, None off GPU; raises otherwise."""
    if _rd._preferred_implementations("auto")[0] == "flydsl":
        return _rd._flydsl_grouped_gemm_ops()
    if jax.default_backend() == "cpu":
        return None
    raise ValueError(
        "moe_flydsl_gather_a needs ragged_dot's flydsl implementation (RAGGED_DOT_IMPL=flydsl), got "
        f"{_rd._preferred_implementations('auto')}"
    )


def _offsets(group_sizes: Int[Array, "E"], rows: int) -> Int[Array, "E1"]:
    # As haliax's flydsl ragged_dot: clamped, so groups that overrun the rows are truncated.
    return jnp.minimum(jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True), rows)


def _gathered_rows(source: Float[Array, "R H"], row_index: Int[Array, "P"]) -> Float[Array, "P H"]:
    """``x_dispatch`` as ``ring_dedup`` builds it: ``source[row_index]``, zeros for an index past the source."""
    valid = row_index < source.shape[0]
    return _take_valid_rows(source, jnp.minimum(row_index, source.shape[0] - 1), valid)


def weight_grad(
    ops: Any,
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

    ``ops`` is the jax-flydsl module (the kernels gather A themselves) or None (XLA on the explicit gather). Without
    ``mode`` it returns the bf16 gradient; with one it folds it into layer ``layer`` of ``acc`` as
    ``grouped_gemm_bf16_wgrad_accum`` does: the updated ``acc`` for "init"/"accum", the bf16 running sum for "final".
    """
    if ops is None:
        a = lhs if row_index is None else _gathered_rows(lhs, row_index)
        _, vjp = jax.vjp(lambda w: _rd.ragged_dot(a, w, group_sizes), rhs)
        (dw,) = vjp(dout)
        if mode is None:
            return dw
        result = accumulate_reference(dw, acc, layer, mode)
        return result.astype(rhs.dtype) if mode == "final" else result
    offsets = _offsets(group_sizes, dout.shape[0])
    gather = {} if row_index is None else {"a_row_index": row_index}
    if mode is None:
        return ops.grouped_gemm_bf16_wgrad(lhs, dout, offsets, **gather)
    return ops.grouped_gemm_bf16_wgrad_accum(lhs, dout, offsets, acc, layer, mode=mode, **gather)


def gathered_up_projection(
    source: Float[Array, "R H"],
    row_index: Int[Array, "P"],
    slots: Int[Array, "R K"],
    w13: Float[Array, "E H I2"],
    group_sizes: Int[Array, "E"],
    gather_sum: MoeGatherSum,
    *,
    acc: Any = None,
    layer: jax.Array | None = None,
    mode: GradAccumulation | None = None,
) -> Any:
    """``ragged_dot(source[row_index], w13, group_sizes)``, gathering inside the GEMM.

    ``row_index[p]`` is the received row of dispatch row ``p``; an index ``>= R`` (the sentinel ``R``) reads zeros.
    ``slots`` is its inverse, ``ring_dedup``'s ``position``: received row ``r``'s at most ``K`` dispatch rows, with
    out-of-range slots for none. Values and gradients equal the explicit gather followed by ``ragged_dot`` bit for bit.

    With ``mode`` the weight gradient goes into layer ``layer`` of the float32 accumulator ``acc`` instead (the
    gathered-A ``grouped_gemm_bf16_wgrad_accum``), as ``ragged_dot_accumulating_grad`` does: "init"/"accum" return
    ``(out, acc)`` and the backward updates the cotangent of the returned ``acc``; "final" returns ``out`` and gives
    ``w13`` the bf16 running sum as its cotangent.
    """
    if mode is None:
        acc = layer = None
    return _gathered_up_projection(source, row_index, slots, w13, group_sizes, acc, layer, gather_sum, mode)


@partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def _gathered_up_projection(source, row_index, slots, w13, group_sizes, acc, layer, gather_sum, mode):
    return _gathered_up_projection_fwd(source, row_index, slots, w13, group_sizes, acc, layer, gather_sum, mode)[0]


def _forward(source, row_index, w13, group_sizes):
    ops = _flydsl_ops()
    if ops is None:
        return _rd.ragged_dot(_gathered_rows(source, row_index), w13, group_sizes)
    _rd._check_flydsl_operands(source, w13, group_sizes)
    rows = row_index.shape[0]
    out = ops.grouped_gemm_bf16_nn(source, w13, _offsets(group_sizes, rows), row_index=row_index)
    return _rd._zero_rows_past_groups_in_place(out, group_sizes)


def _gathered_up_projection_fwd(source, row_index, slots, w13, group_sizes, acc, layer, gather_sum, mode):
    out = _forward(source, row_index, w13, group_sizes)
    residuals = (source, row_index, slots, w13, group_sizes, layer, acc if mode == "final" else None)
    return (out if mode in (None, "final") else (out, acc)), residuals


def _gathered_up_projection_bwd(gather_sum, mode, residuals, cotangent):
    source, row_index, slots, w13, group_sizes, layer, acc = residuals
    d_out = cotangent
    if mode in ("init", "accum"):
        d_out, acc = cotangent
    ops = _flydsl_ops()
    if ops is None:
        _, vjp = jax.vjp(lambda x: _rd.ragged_dot(x, w13, group_sizes), _gathered_rows(source, row_index))
        (d_rows,) = vjp(d_out)
    else:
        # haliax's flydsl ragged_dot backward, with the wgrad's A rows gathered from the source.
        offsets = _offsets(group_sizes, row_index.shape[0])
        d_rows = _rd._zero_rows_past_groups_in_place(ops.grouped_gemm_bf16_nt(d_out, w13, offsets), group_sizes)
    d_w = weight_grad(ops, source, d_out, w13, group_sizes, acc, layer, mode, row_index=row_index)
    # The dispatch backward of ``ep_ring._dispatch_rows``.
    d_source = (
        gather_sum_rows(d_rows, slots)
        if _use_pallas_gather(gather_sum, d_rows.shape[-1])
        else _gather_sum_slots(d_rows, slots)
    )
    if mode in ("init", "accum"):
        # The weight's gradient went into the accumulator.
        return d_source, None, None, None, None, d_w, None
    return d_source, None, None, d_w, None, None, None


_gathered_up_projection.defvjp(_gathered_up_projection_fwd, _gathered_up_projection_bwd)
