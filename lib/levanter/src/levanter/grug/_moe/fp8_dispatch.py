# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""fp8 forward dispatch for ``ring_filtered``: the dispatch ``all_to_all`` moves the token rows as float8_e4m3fn with
one float32 scale per row, and the backward stays bfloat16 (straight-through). Fork overlay, lossy prototype.

Modes (``Fp8Dispatch``):

- ``off``: the bfloat16 dispatch, unchanged.
- ``unfused``: the sender quantizes its token rows (:func:`quantize_rows`), gathers and sends their bits; the expert
  side dequantizes the received rows (:func:`dequantize_rows`) and runs the bfloat16 expert path on them.
- ``fused``: the same transport, but the expert up projection reads the fp8 rows and their scales itself (the
  jax-flydsl fp8 kernel, ``flydsl_expert.dispatch_up_fp8``); the remat policy saves the fp8 rows, not bfloat16 ones.

``unfused`` multiplies the bfloat16 rows ``round(f32(q) * s)``. The fused kernel converts ``q`` exactly and multiplies
its f32 accumulator row by ``s`` before the bfloat16 store, so its forward skips the bfloat16 rounding of ``q * s``: it
is within about one bfloat16 ulp of ``unfused``, not bitwise equal (bitwise for power-of-two scales). On CPU the fused
mode's reference path is the unfused math. The backward of both modes is the same bfloat16 straight-through one.

Quantization, per row: ``s = amax * f32(1 / 448)`` in float32 (1 for an all-zero row) and ``q = e4m3fn(clip(x / s, -448,
448))``, rounded to nearest even. The explicit clip makes the result independent of the conversion's saturation mode.
The scale is exact, not a power of two: on rows with outlier channels or heavy tails a power-of-two scale doubles the
worst-case error.

The backward is straight-through: the rows' cotangent goes back through the bfloat16 transport's transpose (the same
``all_to_all`` and sender gather-sum as the bfloat16 dispatch), as if the rows had been sent exactly. The scales get no
gradient.
"""

import functools
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, UInt8

Fp8Dispatch: TypeAlias = Literal["off", "unfused", "fused"]
FP8_DISPATCH_MODES: tuple[Fp8Dispatch, ...] = ("off", "unfused", "fused")

FP8_DTYPE = jnp.float8_e4m3fn
# The largest finite float8_e4m3fn.
FP8_MAX = 448.0
# The scale multiplies by this float32 constant: XLA rewrites a division by a constant into that multiplication anyway,
# so spelling it out keeps the scale's bits independent of the compiler.
_INV_FP8_MAX = np.float32(1.0 / FP8_MAX)


def check_fp8_dispatch(mode: str) -> Fp8Dispatch:
    """``mode`` as an ``Fp8Dispatch``; raises for anything else."""
    if mode not in FP8_DISPATCH_MODES:
        raise ValueError(f"fp8_dispatch must be one of {FP8_DISPATCH_MODES}, got {mode!r}")
    return cast(Fp8Dispatch, mode)


def quantize_rows(x: Float[Array, "T H"]) -> tuple[UInt8[Array, "T H"], Float[Array, " T"]]:
    """The float8_e4m3fn bits (as uint8) and float32 scale of each row of ``x``.

    ``s = amax * f32(1 / 448)`` (1 for an all-zero row), ``q = e4m3fn(clip(x / s, -448, 448))`` rounded to nearest even. The bits
    travel as uint8, which every collective moves."""
    values = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(values), axis=1)
    scale = jnp.where(amax > 0, amax * _INV_FP8_MAX, jnp.float32(1.0))
    q = jnp.clip(values / scale[:, None], -FP8_MAX, FP8_MAX).astype(FP8_DTYPE)
    return jax.lax.bitcast_convert_type(q, jnp.uint8), scale


def dequantize_rows(bits: UInt8[Array, "R H"], scale: Float[Array, " R"], dtype: Any) -> Float[Array, "R H"]:
    """``round_dtype(f32(q) * s)`` for the float8_e4m3fn bits ``bits`` and float32 row scales ``scale``."""
    q = jax.lax.bitcast_convert_type(bits, FP8_DTYPE).astype(jnp.float32)
    return (q * scale.astype(jnp.float32)[:, None]).astype(dtype)


SendBack: TypeAlias = Callable[..., jax.Array]
"""``send_back(d_received, *send_back_args)``: the sender rows' cotangent for the received rows' cotangent, i.e. the
transpose of the bfloat16 transport the fp8 rows stand in for."""


def straight_through_rows(
    x_source: Float[Array, "T H"],
    bits: UInt8[Array, "R H"],
    scale: Float[Array, " R"],
    send_back: SendBack,
    send_back_args: tuple[jax.Array, ...],
) -> Float[Array, "R H"]:
    """The received rows ``dequantize_rows(bits, scale, x_source.dtype)``, differentiable as the transport of
    ``x_source``: their cotangent goes to ``x_source`` through ``send_back``. ``x_source`` itself is not read."""
    return _straight_through_rows(x_source, bits, scale, send_back_args, send_back)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _straight_through_rows(x_source, bits, scale, send_back_args, send_back):
    return dequantize_rows(bits, scale, x_source.dtype)


def _straight_through_rows_fwd(x_source, bits, scale, send_back_args, send_back):
    return dequantize_rows(bits, scale, x_source.dtype), send_back_args


def _straight_through_rows_bwd(send_back, send_back_args, cotangent):
    return send_back(cotangent, *send_back_args), None, None, None


_straight_through_rows.defvjp(_straight_through_rows_fwd, _straight_through_rows_bwd)
