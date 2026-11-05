# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert weight gradients accumulated inside the weight-gradient kernel (``ragged_dot_accumulating_grad``).

Gradient accumulation over microbatches keeps a float32 sum of every weight's gradient. For the stacked expert weights
the layer scan's backward would otherwise write each layer's bf16 gradient into a stacked cotangent, and the sum would
be updated after the scan by a separate float32 add over the whole stack. ``ExpertGradAccumulator`` carries the sums
into the MoE instead: each layer's expert matmuls fold their weight gradient into that layer of the sums in the backward.
"""

import equinox as eqx
import jax
from haliax.nn.ragged_dot import GradAccumulation


class ExpertGradAccumulator(eqx.Module):
    """Float32 sums of the stacked expert weight gradients, threaded through a layer scan and its MoE layers.

    Each sum is ``[L, S, E, ...]``: L layers, then one row per batch shard holding that device's partial sum (the
    ``unreduced`` gradient as an ordinary array, so it can cross a ``shard_map``), then the weight's own expert-sharded
    shape. ``w_gate`` and ``w_up`` are ``[L, S, E, D, I]`` and ``w_down`` is ``[L, S, E, I, D]``.

    ``layer`` is the scan's layer index, set per layer. ``mode`` (``GradAccumulation``): with "init" and "accum" the MoE
    returns the sums and its backward sets or adds each weight gradient into the cotangent of those returned sums, which
    becomes theirs; differentiate with ``jax.vjp`` and seed that cotangent with the running sums. With "final" the sums
    are only read: each expert weight's own cotangent is the bf16 sum including this gradient.
    """

    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    layer: jax.Array | None = None
    mode: GradAccumulation = eqx.field(static=True, default="accum")

    def at_layer(self, layer: jax.Array) -> "ExpertGradAccumulator":
        return ExpertGradAccumulator(self.w_gate, self.w_up, self.w_down, layer, self.mode)

    def without_layer(self) -> "ExpertGradAccumulator":
        return ExpertGradAccumulator(self.w_gate, self.w_up, self.w_down, None, self.mode)


def local_view(x: jax.Array) -> jax.Array:
    """One device's block of a sum inside a ``shard_map``, ``[L, 1, E_local, ...]``, as the kernel's ``[L, E_local, ...]``."""
    return x.reshape(x.shape[0], *x.shape[2:])


def from_local_view(x: jax.Array) -> jax.Array:
    return x.reshape(x.shape[0], 1, *x.shape[1:])
