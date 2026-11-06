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

from functools import partial

import jax
import jax.numpy as jnp
from haliax.nn.ragged_dot import (
    _check_flydsl_operands,
    _flydsl_grouped_gemm_ops,
    _preferred_implementations,
    _zero_rows_past_groups_in_place,
    ragged_dot,
)
from jaxtyping import Array, Float, Int

from levanter.grug._moe.ep_ring import _gather_sum_slots, _take_valid_rows
from levanter.grug._moe.pallas_gather_sum import MoeGatherSum, gather_sum_rows


def _flydsl_ops():
    """The jax-flydsl grouped GEMM module when ``ragged_dot`` runs flydsl, None off GPU; raises otherwise."""
    if _preferred_implementations("auto")[0] == "flydsl":
        return _flydsl_grouped_gemm_ops()
    if jax.default_backend() == "cpu":
        return None
    raise ValueError(
        "moe_flydsl_gather_a needs ragged_dot's flydsl implementation (RAGGED_DOT_IMPL=flydsl), got "
        f"{_preferred_implementations('auto')}"
    )


def _offsets(group_sizes: Int[Array, "E"], rows: int) -> Int[Array, "E1"]:
    # As haliax's flydsl ragged_dot: clamped, so groups that overrun the rows are truncated.
    return jnp.minimum(jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True), rows)


def _gathered_rows(source: Float[Array, "R H"], row_index: Int[Array, "P"]) -> Float[Array, "P H"]:
    """``x_dispatch`` as ``ring_dedup`` builds it: ``source[row_index]``, zeros for an index past the source."""
    valid = row_index < source.shape[0]
    return _take_valid_rows(source, jnp.minimum(row_index, source.shape[0] - 1), valid)


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def gathered_up_projection(
    source: Float[Array, "R H"],
    row_index: Int[Array, "P"],
    slots: Int[Array, "R K"],
    w13: Float[Array, "E H I2"],
    group_sizes: Int[Array, "E"],
    gather_sum: MoeGatherSum,
) -> Float[Array, "P I2"]:
    """``ragged_dot(source[row_index], w13, group_sizes)``, gathering inside the GEMM.

    ``row_index[p]`` is the received row of dispatch row ``p``; an index ``>= R`` (the sentinel ``R``) reads zeros.
    ``slots`` is its inverse, ``ring_dedup``'s ``position``: received row ``r``'s at most ``K`` dispatch rows, with
    out-of-range slots for none. Values and gradients equal the explicit gather followed by ``ragged_dot`` bit for bit.
    """
    return _forward(source, row_index, w13, group_sizes)


def _forward(source, row_index, w13, group_sizes):
    ops = _flydsl_ops()
    if ops is None:
        return ragged_dot(_gathered_rows(source, row_index), w13, group_sizes)
    _check_flydsl_operands(source, w13, group_sizes)
    rows = row_index.shape[0]
    out = ops.grouped_gemm_bf16_nn(source, w13, _offsets(group_sizes, rows), row_index=row_index)
    return _zero_rows_past_groups_in_place(out, group_sizes)


def _gathered_up_projection_fwd(source, row_index, slots, w13, group_sizes, gather_sum):
    return _forward(source, row_index, w13, group_sizes), (source, row_index, slots, w13, group_sizes)


def _gathered_up_projection_bwd(gather_sum, residuals, d_out):
    source, row_index, slots, w13, group_sizes = residuals
    ops = _flydsl_ops()
    if ops is None:
        _, vjp = jax.vjp(lambda x, w: ragged_dot(x, w, group_sizes), _gathered_rows(source, row_index), w13)
        d_rows, d_w13 = vjp(d_out)
    else:
        offsets = _offsets(group_sizes, row_index.shape[0])
        # haliax's flydsl ragged_dot backward, with the wgrad's A rows gathered from the source.
        d_rows = _zero_rows_past_groups_in_place(ops.grouped_gemm_bf16_nt(d_out, w13, offsets), group_sizes)
        d_w13 = ops.grouped_gemm_bf16_wgrad(source, d_out, offsets, a_row_index=row_index)
    # The dispatch backward of ``ep_ring._dispatch_rows``.
    d_source = gather_sum_rows(d_rows, slots) if gather_sum == "pallas" else _gather_sum_slots(d_rows, slots)
    return d_source, None, None, d_w13, None


gathered_up_projection.defvjp(_gathered_up_projection_fwd, _gathered_up_projection_bwd)
