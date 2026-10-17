# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Grug's ``gpu_te_flash`` attention: TransformerEngine's JAX fused attention, ``transformer_engine.jax.attention``.

Written against ROCm TransformerEngine 2.17 (``transformer_engine==2.17.0+rocm10.0.0``, the spurix refstack
``amd-jax-0.11.0-te``), whose fused attention runs AITER/CK kernels on AMD Instinct GPUs (``NVTE_CK_USES_FWD_V3`` and
``NVTE_CK_USES_BWD_V3`` let it use AITER's asm v3 kernels) and AOTriton where CK does not cover a shape. NVIDIA TE
2.x (cuDNN) takes the same call.

How a Grug call maps to ``fused_attn``:

- q, k and v stay BSHD (``QKVLayout.BSHD_BSHD_BSHD``) with the model's ``Hkv`` K/V heads; TE does grouped-query
  attention itself.
- A causal mask is ``AttnMaskType.CAUSAL_MASK`` (top-left aligned, as in ``reference_attention``); no mask is
  ``NO_MASK``. Every sequence is full length, so the sequence descriptor carries ``S`` for each row.
- A causal sliding window ``W`` is ``window_size=(W - 1, 0)``: TE attends to the inclusive key range
  ``[q - left, q + right]``, while Grug's ``W`` keeps the last ``W`` keys including the query. A window covering
  the whole sequence is dropped.
- ``softmax_type`` is ``VANILLA_SOFTMAX``. ROCm TE and NVIDIA TE 2.10 and later require it; older NVIDIA releases
  do not take it, so it is passed only when ``fused_attn`` has the parameter.
- The scale is ``1 / sqrt(head_dim)``, as in ``reference_attention``; no bias, dropout 0, ``is_training=True``.
- Determinism is TE's: ``NVTE_ALLOW_NONDETERMINISTIC_ALGO`` (TE's default is 1, and the refstack sets 1) selects the
  nondeterministic backward. This module does not set it.

Anything TE cannot run raises at trace time instead of falling back: segment ids (packed documents need TE's THD
layout and a static ``max_segments_per_seq``), dense masks, non-causal windows, dtypes other than bf16 and fp16, and
any shape for which ``is_fused_attn_kernel_available`` finds no fused kernel (on ROCm, neither CK, which stops below
head_dim 512, nor AOTriton). ``NVTE_LOG_CK_CONFIG=1`` prints CK's reason.

Under a mesh the kernel runs per device inside ``shard_map``, like ``gpu_pallas_triton_flash``: batch and heads may
be sharded, sequence and head_dim may not. TE asserts that a global ``MeshResource`` is set whenever a mesh is
active; Grug sets none, so the call goes through ``levanter.layers.te_fused_attention.call_with_empty_mesh_resource``,
which traces TE's forward and backward under an empty one.

Like every aiter-backed kernel on gfx950, keep aiter#5894 in mind: dK came back zero when several processes shared
a GPU.
"""

import functools
import importlib
import inspect
from types import ModuleType

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from levanter.grug.attention import _gpu_flash_common as common
from levanter.grug.attention._core import AttentionMask
from levanter.layers.te_fused_attention import call_with_empty_mesh_resource, softmax_type_kwargs

NAME = "gpu_te_flash"


def _te_attention_module() -> ModuleType:
    """``transformer_engine.jax.attention``, imported on first use."""
    try:
        return importlib.import_module("transformer_engine.jax.attention")
    except ImportError as e:
        raise ImportError(
            f"{NAME} needs TransformerEngine's JAX extension (transformer_engine.jax), for example ROCm TE 2.17 from "
            f"the spurix refstack amd-jax-0.11.0-te; importing it failed: {e}"
        ) from e


def _mask_type(attention: ModuleType, call: common.FlashCall):
    return attention.AttnMaskType.CAUSAL_MASK if call.causal else attention.AttnMaskType.NO_MASK


def _check_kernel_available(attention: ModuleType, q: jax.Array, k: jax.Array, call: common.FlashCall) -> None:
    """Raise unless TE has a fused kernel for this per-device shape."""
    batch, seq_len, num_q_heads, head_dim = q.shape
    candidates = dict(
        is_training=True,
        batch_size=batch,
        q_dtype=q.dtype,
        kv_dtype=k.dtype,
        qkv_layout=attention.QKVLayout.BSHD_BSHD_BSHD,
        attn_bias_type=attention.AttnBiasType.NO_BIAS,
        attn_mask_type=_mask_type(attention, call),
        dropout_probability=0.0,
        q_num_heads=num_q_heads,
        kv_num_heads=k.shape[2],
        q_max_seqlen=seq_len,
        kv_max_seqlen=seq_len,
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        window_size=call.window,
        **softmax_type_kwargs(attention),
    )
    # The checker's parameters differ across TE releases (softmax_type from 2.10, batch_size later).
    params = inspect.signature(attention.is_fused_attn_kernel_available).parameters
    if not attention.is_fused_attn_kernel_available(**{n: x for n, x in candidates.items() if n in params}):
        raise NotImplementedError(
            f"{NAME}: TransformerEngine has no fused attention kernel for q={q.shape} {q.dtype}, Hkv={k.shape[2]}, "
            f"causal={call.causal}, window={call.window} (NVTE_LOG_CK_CONFIG=1 prints CK's reason)"
        )


def _fused_attn(attention: ModuleType, call: common.FlashCall, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    batch, seq_len = q.shape[:2]
    seqlens = jnp.full((batch,), seq_len, dtype=jnp.int32)
    return attention.fused_attn(
        (q, k, v),
        None,  # bias
        attention.SequenceDescriptor.from_seqlens((seqlens, seqlens)),
        None,  # dropout seed
        attn_bias_type=attention.AttnBiasType.NO_BIAS,
        attn_mask_type=_mask_type(attention, call),
        qkv_layout=attention.QKVLayout.BSHD_BSHD_BSHD,
        scaling_factor=call.scale,
        dropout_probability=0.0,
        is_training=True,
        window_size=call.window,
        **softmax_type_kwargs(attention),
    )


def te_flash_attention(
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
    mask: AttentionMask | jax.Array | None,
) -> Float[Array, "B S Hq D"]:
    """Self-attention through TransformerEngine's fused attention, per device under a mesh."""
    attention = _te_attention_module()
    call = common.flash_call(NAME, q, k, v, mask)

    def local(q, k, v):
        _check_kernel_available(attention, q, k, call)
        return call_with_empty_mesh_resource(functools.partial(_fused_attn, attention, call), q, k, v)

    return common.run_sharded(NAME, local, q, k, v)
