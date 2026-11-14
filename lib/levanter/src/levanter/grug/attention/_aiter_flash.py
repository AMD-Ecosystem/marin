# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Grug's ``gpu_aiter_flash`` attention: ROCm/jax-aiter's ``flash_attn_func`` (AITER MHA through XLA FFI).

jax-aiter wraps ``aiter::mha_fwd`` / ``aiter::mha_bwd`` in a ``custom_vjp`` over ``custom_partitioning``, and AITER
picks the kernel per call. Going by jax-aiter's dispatch code (not yet confirmed in a profile), on gfx950 with bf16
and head_dim 128:

- Full causal layers take the asm v3 forward and the asm v3 backward.
- Sliding-window layers take CK both ways. The v3 forward has no window, and jax-aiter routes every windowed
  backward to CK even though a v3 window backward exists on gfx950.

How a Grug call maps to ``flash_attn_func(q, k, v, ...)``:

- q, k and v stay BSHD with the model's ``Hkv`` K/V heads; AITER does grouped-query attention itself, for any
  ratio that divides (20/5 and 40/8 alike).
- ``causal`` is the mask's flag; for self-attention AITER's bottom-right alignment equals Grug's top-left one.
- A causal sliding window ``W`` is ``window_size=(W - 1, 0)`` (the inclusive key range ``[q - left, q + right]``).
  Unwindowed calls pass ``(-1, -1)``, never ``(-1, 0)``: jax-aiter treats any ``right >= 0`` as a window and would
  send the backward to CK.
- ``deterministic=False``. jax-aiter defaults to True, which on gfx950 silently selects the CK backward because the
  v3 backward is nondeterministic only.
- ``softmax_scale = 1 / sqrt(head_dim)``, as in ``reference_attention``; dropout 0, no bias.

Anything AITER cannot run raises at trace time: segment ids (packed documents need ``flash_attn_varlen``), dense
masks, non-causal windows, dtypes other than bf16 and fp16, and head_dim above 256 or not a multiple of 8 (jax-aiter
would pad the latter to the next multiple of 8 on every call).

Under a mesh the kernel runs per device inside ``shard_map``, like ``gpu_pallas_triton_flash``: batch and heads may
be sharded, sequence and head_dim may not. Inside the all-manual ``shard_map`` jax-aiter's ``custom_partitioning``
lowers straight to the per-device FFI call. Keep aiter#5894 in mind: dK came back zero when several processes shared
a GPU.
"""

import importlib
from collections.abc import Callable

import jax
from jaxtyping import Array, Float

from levanter.grug.attention import _gpu_flash_common as common
from levanter.grug.attention._core import AttentionMask

NAME = "gpu_aiter_flash"
MAX_HEAD_DIM = 256
HEAD_DIM_MULTIPLE = 8
NO_WINDOW = (-1, -1)


def _flash_attn_func() -> Callable:
    """``jax_aiter.mha.flash_attn_func``, imported on first use."""
    try:
        return importlib.import_module("jax_aiter.mha").flash_attn_func
    except ImportError as e:
        # jax_aiter.mha also raises ModuleNotFoundError when the wheel lacks the MHA libraries
        # (``jax-aiter-fetch-mha`` adds them); keep its message.
        raise ImportError(f"{NAME} needs jax-aiter's flash attention (jax_aiter.mha.flash_attn_func): {e}") from e


def aiter_flash_attention(
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
    mask: AttentionMask | jax.Array | None,
) -> Float[Array, "B S Hq D"]:
    """Self-attention through jax-aiter's ``flash_attn_func``, per device under a mesh."""
    flash_attn_func = _flash_attn_func()
    call = common.flash_call(NAME, q, k, v, mask)
    head_dim = q.shape[-1]
    if head_dim > MAX_HEAD_DIM or head_dim % HEAD_DIM_MULTIPLE:
        raise ValueError(
            f"{NAME} needs head_dim <= {MAX_HEAD_DIM} and a multiple of {HEAD_DIM_MULTIPLE}, got {head_dim}"
        )

    def local(q, k, v):
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=call.scale,
            causal=call.causal,
            window_size=call.window or NO_WINDOW,
            deterministic=False,
        )
        # flash_attn_func returns a tuple, (out,) unless return_lse or return_attn_probs adds more.
        return out[0]

    return common.run_sharded(NAME, local, q, k, v)
