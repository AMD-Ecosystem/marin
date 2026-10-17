# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the Grug backends that call an external fused flash-attention kernel.

``gpu_te_flash`` (TransformerEngine) and ``gpu_aiter_flash`` (jax-aiter) both take BSHD q/k/v with grouped-query K/V
heads, a causal flag and a FlashAttention-style ``(left, right)`` window, and both run one kernel per device. This
module turns a Grug ``AttentionMask`` into those static arguments, rejects what the kernels cannot express, and runs
the per-device call under ``shard_map`` the way ``gpu_pallas_triton_flash`` does.
"""

import dataclasses
import math
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from jaxtyping import Array, Float

from levanter.grug.attention._core import AttentionMask
from levanter.sharding import partition_spec_of, partitioned_dims

KERNEL_DTYPES = (jnp.bfloat16, jnp.float16)


@dataclasses.dataclass(frozen=True)
class FlashCall:
    """Static kernel arguments for one attention call.

    ``window`` is ``None`` for unwindowed attention, else the inclusive key range ``[q - left, q + right]`` as
    ``(left, right)``, the convention TransformerEngine and FlashAttention/aiter share.
    """

    causal: bool
    window: tuple[int, int] | None
    scale: float


def flash_call(
    name: str,
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
    mask: AttentionMask | jax.Array | None,
) -> FlashCall:
    """Validate a self-attention call and map its mask to kernel arguments; raise for anything the kernels lack."""
    if q.ndim != 4 or k.shape != v.shape or k.ndim != 4:
        raise ValueError(f"{name} needs BSHD q/k/v with equal k and v shapes, got q={q.shape} k={k.shape} v={v.shape}")
    batch, seq_len, num_q_heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[1] != seq_len or k.shape[3] != head_dim:
        raise ValueError(f"{name} needs self-attention shapes, got q={q.shape} k={k.shape} v={v.shape}")
    num_kv_heads = k.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"{name} needs Hq divisible by Hkv, got Hq={num_q_heads} Hkv={num_kv_heads}")
    if not q.dtype == k.dtype == v.dtype or q.dtype not in KERNEL_DTYPES:
        raise ValueError(f"{name} needs bf16 or fp16 q/k/v of one dtype, got q={q.dtype} k={k.dtype} v={v.dtype}")

    if mask is None:
        return FlashCall(causal=False, window=None, scale=1.0 / math.sqrt(head_dim))
    if not isinstance(mask, AttentionMask):
        raise NotImplementedError(f"{name} needs an AttentionMask or None, not a dense mask")
    if mask.segment_ids is not None:
        raise NotImplementedError(
            f"{name} does not support segment ids (packed documents); "
            "use gpu_pallas_triton_flash, xla_flash or reference"
        )
    if mask.fa4_bounds is not None:
        raise NotImplementedError(f"{name} does not read FA4/CuTe precomputed bounds")

    window = None
    if mask.sliding_window is not None:
        if mask.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {mask.sliding_window}")
        if not mask.is_causal:
            raise NotImplementedError(f"{name} supports sliding windows on causal masks only")
        # Grug's window W keeps the last W keys including the query itself, keys [q - (W - 1), q]. A window that
        # covers the whole sequence is plain causal attention, and passing it would steer aiter's backward to CK.
        if mask.sliding_window < seq_len:
            window = (mask.sliding_window - 1, 0)
    return FlashCall(causal=mask.is_causal, window=window, scale=1.0 / math.sqrt(head_dim))


def run_sharded(
    name: str,
    local: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    q: Float[Array, "B S Hq D"],
    k: Float[Array, "B S Hkv D"],
    v: Float[Array, "B S Hkv D"],
) -> Float[Array, "B S Hq D"]:
    """Run ``local(q, k, v)`` once per device under ``shard_map`` when q is sharded, else directly.

    Batch and heads may be sharded (heads in contiguous blocks, so each device keeps whole GQA groups); sequence and
    head_dim may not. Under the all-manual ``shard_map`` the kernels' ``custom_partitioning`` rules lower straight to
    the per-device call. Unsharded inputs (no mesh, or Auto axes whose placement a tracer cannot see) call the kernel
    directly and leave partitioning to those rules.
    """
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return local(q, k, v)
    q_dims = partitioned_dims(q, mesh)
    if q_dims[1] or q_dims[3]:
        raise ValueError(f"{name} needs unsharded sequence and head_dim, got {partition_spec_of(q)}")
    for label, x in (("k", k), ("v", v)):
        if partitioned_dims(x, mesh) != q_dims:
            raise ValueError(
                f"{name} needs {label} sharded like q, got q={partition_spec_of(q)} {label}={partition_spec_of(x)}"
            )
    if not any(q_dims):
        return local(q, k, v)
    q_spec = tuple(partition_spec_of(q))
    q_spec = P(*q_spec, *((None,) * (q.ndim - len(q_spec))))

    @shard_map(mesh=mesh, out_specs=q_spec, check_vma=False)
    def _local(q_local, k_local, v_local):
        return local(q_local, k_local, v_local)

    return _local(q, k, v)
