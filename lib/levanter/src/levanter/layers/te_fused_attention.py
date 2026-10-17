# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""What Levanter's and Grug's TransformerEngine attention paths need beyond a plain ``fused_attn`` call.

- ``fused_attn`` takes a ``softmax_type`` on ROCm TE and on NVIDIA TE 2.10 and later, which require it; older NVIDIA
  releases have neither the parameter nor ``AttnSoftmaxType``.
- TE reads a global ``MeshResource`` to find a context-parallel axis while tracing ``fused_attn``'s forward and
  again its backward, and asserts when none is set: NVIDIA TE 2.7 to 2.11 always, NVIDIA TE 2.12 and later and ROCm
  TE 2.17 only when a mesh is active, which the Levanter trainer always makes. Levanter sets none, and needs none: it
  uses neither TE's context parallelism nor its logical axes, and TE's partitioning rules for fused attention follow
  the input shardings.
"""

import functools
import importlib
import inspect
from collections.abc import Callable
from types import ModuleType

import jax


def softmax_type_kwargs(te_attention: ModuleType) -> dict:
    """``{"softmax_type": VANILLA_SOFTMAX}`` when ``te_attention.fused_attn`` takes a softmax type, else ``{}``."""
    if "softmax_type" not in inspect.signature(te_attention.fused_attn).parameters:
        return {}
    return {"softmax_type": te_attention.AttnSoftmaxType.VANILLA_SOFTMAX}


def _empty_mesh_resource():
    sharding = importlib.import_module("transformer_engine.jax.sharding")
    return sharding.global_shard_guard(sharding.MeshResource())


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def call_with_empty_mesh_resource(fn: Callable, *args):
    """``fn(*args)`` with an empty TE ``MeshResource`` set while its forward and its backward are traced.

    ``fn`` calls TE's fused attention and must close over no traced values. The backward applies the forward's own
    ``jax.vjp``, so TE's saved softmax statistics are reused and nothing is recomputed.
    """
    with _empty_mesh_resource():
        return fn(*args)


def _call_fwd(fn: Callable, *args):
    with _empty_mesh_resource():
        return jax.vjp(fn, *args)


def _call_bwd(fn: Callable, pullback, d_out):
    del fn
    with _empty_mesh_resource():
        return pullback(d_out)


call_with_empty_mesh_resource.defvjp(_call_fwd, _call_bwd)
