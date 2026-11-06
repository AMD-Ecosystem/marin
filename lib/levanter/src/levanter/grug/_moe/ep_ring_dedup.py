# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-deduplicated expert-parallel Grug MoE backend.

Each shard sends every local token at most once to each expert shard that owns one of its
selected experts, through a fixed-capacity ``all_to_all``. The expert shard runs ring's grouped
``ragged_dot`` path on the received rows, sums each token's local expert outputs, and returns one
row per received token through a second ``all_to_all``. The sender then sums the returned rows.

Compared with ``ring``, which all-gathers every token to every shard and reduce-scatters a
full-size output, this moves only the rows a destination works on: with top-4 of 256 experts over
8 shards, about 42% of ring's bytes at a sender capacity factor of 1.0.

Every hidden-width data movement is a gather whose transpose is another gather, so neither the
forward nor the backward uses a hidden-width scatter-add.

With ``token_blocks > 1`` each shard splits its tokens into that many contiguous blocks and runs the
whole exchange once per block, so XLA can overlap one block's all-to-alls with another block's expert
matmuls. Capacities are per block, sized from the block's tokens.
"""

import functools
import math
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot, ragged_dot_accumulating_grad
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_ROUTING,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.ep_common import _assignment_sources, _combine_gather_sum
from levanter.grug._moe.ep_ring import _combine_rows, _dispatch_rows, _select_local_assignments
from levanter.grug._moe.grad_accum import ExpertGradAccumulator, from_local_view, local_view
from levanter.grug._moe.pallas_gather_sum import MoeGatherSum
from levanter.grug._moe.pallas_routing_metadata import MoeRoutingMetadata, expert_rows, send_slots


def destination_hit_fraction(*, num_experts: int, local_experts: int, topk: int) -> float:
    """Probability that a token's ``topk`` distinct uniform experts include one on a given shard."""
    return 1.0 - math.comb(num_experts - local_experts, topk) / math.comb(num_experts, topk)


def _moe_mlp_ep_ring_dedup_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    token_valid_local: Bool[Array, "Tlocal"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    transport_capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
    token_blocks: int,
    reduce_drops: bool = True,
    gather_sum: MoeGatherSum = "xla",
    routing_metadata: MoeRoutingMetadata = "xla",
    grad_acc: ExpertGradAccumulator | None = None,
) -> (
    tuple[Float[Array, "Tlocal H"], CapacityDrops]
    | tuple[Float[Array, "Tlocal H"], CapacityDrops, ExpertGradAccumulator]
):
    """Deduplicated all-to-all dispatch, ring's grouped expert MLP, expert-side combine.

    The shard's tokens are split into ``token_blocks`` contiguous blocks, each dispatched, run and
    combined on its own, with the capacities below computed from the block's tokens.

    ``transport_capacity_factor`` limits each (sender, destination) pair to a multiple of the
    distinct tokens the sender would send that destination under uniform routing,
    ``destination_hit_fraction`` times its valid tokens; the send buffer holds the limit for
    ``Tlocal`` valid tokens. A token past the limit loses all its assignments to the destination
    (sender drops). The limit is an expectation for uniform routing, with no worst-case
    guarantee: when routing concentrates on a destination, its tokens past the limit drop even if
    the destination has expert capacity to spare, so a factor that drops nothing on one workload
    can drop on another. ``capacity_factor`` keeps ring's meaning: each expert shard
    runs at most ``capacity_factor * Tlocal * K`` assignments, allocated in local-expert order
    (receiver drops).

    The drop counts are summed over ``token_sharding_axes`` (one small all-reduce per call), or, with
    ``reduce_drops=False``, returned as this shard's own counts, shape [1], for the caller to reduce later.

    ``gather_sum`` picks how the backward's expert-side dispatch gather-sum and sender-side combine gather run: XLA
    fusions ("xla") or the Pallas-Triton kernel of ``pallas_gather_sum`` ("pallas"), with the same results bit for bit.
    The token-side gathers stay on XLA, which fuses them with the residual stream.

    ``routing_metadata`` picks how the integer routing maps (send slots, expert rows, group sizes, drop counts) are
    built from the top-k expert ids: XLA fusions ("xla") or the Pallas-Triton counting sorts of
    ``pallas_routing_metadata`` ("pallas"), with the same values.

    With ``grad_acc`` (this shard's block of an ``ExpertGradAccumulator``) the expert matmuls fold their weight
    gradients into it (``ragged_dot_accumulating_grad``), one token block after another, and the accumulator comes
    back as a third output. With several token blocks the float32 sums take each block's bf16 gradient in turn, where
    XLA first adds the blocks' gradients in bf16, so the rounding differs; mode "final" needs one block.
    """
    tokens = x_local.shape[0]
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")
    if transport_capacity_factor <= 0:
        raise ValueError("transport_capacity_factor must be positive")
    if token_blocks < 1 or tokens % token_blocks != 0:
        raise ValueError(f"token_blocks={token_blocks} must be positive and divide the local token count {tokens}")

    block = functools.partial(
        _ring_dedup_block,
        moe_w13_local=moe_w13_local,
        moe_w2_local=moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        transport_capacity_factor=transport_capacity_factor,
        gather_sum=gather_sum,
        routing_metadata=routing_metadata,
    )
    if token_blocks == 1:
        out_local, sender_dropped, receiver_dropped, grad_acc = block(
            x_local, selected_experts_local, combine_weights_local, token_valid_local, grad_acc=grad_acc
        )
    else:
        block_tokens = tokens // token_blocks
        acc_mode = None if grad_acc is None else grad_acc.mode
        if acc_mode == "final":
            # "final" reads the sums the earlier microbatches left, not the other blocks' gradients, so every block
            # would add them again.
            raise NotImplementedError("grad_acc mode 'final' (the overlapped last microbatch) needs token_blocks == 1")
        outputs, sender_drops, receiver_drops = [], [], []
        for i in range(token_blocks):
            rows = slice(i * block_tokens, (i + 1) * block_tokens)
            if acc_mode == "init":
                # The backward runs the blocks last to first: the last block's gradient writes the layer of the sums
                # and the others add to it.
                grad_acc = grad_acc.with_mode("init" if i == token_blocks - 1 else "accum")
            with jax.named_scope(f"token_block_{i}"):
                out, sender, receiver, grad_acc = block(
                    x_local[rows],
                    selected_experts_local[rows],
                    combine_weights_local[rows],
                    token_valid_local[rows],
                    grad_acc=grad_acc,
                )
            outputs.append(out)
            sender_drops.append(sender)
            receiver_drops.append(receiver)
        if acc_mode is not None:
            grad_acc = grad_acc.with_mode(acc_mode)
        out_local = jnp.concatenate(outputs, axis=0)
        sender_dropped = sum(sender_drops)
        receiver_dropped = sum(receiver_drops)

    if not reduce_drops:
        drops = CapacityDrops(
            sender_dropped=jnp.reshape(sender_dropped, (1,)), receiver_dropped=jnp.reshape(receiver_dropped, (1,))
        )
    else:
        with jax.named_scope("combine"):
            reduced = jax.lax.psum(jnp.stack([sender_dropped, receiver_dropped]), token_sharding_axes)
        drops = CapacityDrops(sender_dropped=reduced[0], receiver_dropped=reduced[1])
    if grad_acc is None:
        return out_local, drops
    return out_local, drops, grad_acc


def _ring_dedup_block(
    x_local: Float[Array, "T H"],
    selected_experts_local: Int[Array, "T K"],
    combine_weights_local: Float[Array, "T K"],
    token_valid_local: Bool[Array, "T"],
    *,
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    transport_capacity_factor: float,
    gather_sum: MoeGatherSum,
    routing_metadata: MoeRoutingMetadata,
    grad_acc: ExpertGradAccumulator | None = None,
) -> tuple[Float[Array, "T H"], Int[Array, ""], Int[Array, ""], ExpertGradAccumulator | None]:
    """One block of tokens through the exchange: its output, its local sender and receiver drops, and ``grad_acc``
    with this block's expert weight gradients folded in (None without one)."""
    tokens, topk = selected_experts_local.shape
    hidden_dim = x_local.shape[1]
    local_experts = moe_w13_local.shape[0]
    ep_size = num_experts // local_experts

    hit_fraction = destination_hit_fraction(num_experts=num_experts, local_experts=local_experts, topk=topk)
    pair_capacity = min(tokens, max(1, math.ceil(transport_capacity_factor * hit_fraction * tokens)))
    send_rows = ep_size * pair_capacity
    received_assignments = send_rows * topk
    expert_capacity = min(received_assignments, max(local_experts, math.ceil(capacity_factor * tokens * topk)))

    with jax.named_scope("dispatch"):
        slot, safe_source, slot_valid, metadata, sender_dropped = _send_routing(
            selected_experts_local,
            token_valid_local,
            combine_weights_local,
            local_experts=local_experts,
            destinations=ep_size,
            pair_capacity=pair_capacity,
            pair_capacity_factor=transport_capacity_factor * hit_fraction,
            routing_metadata=routing_metadata,
        )
        received_metadata = jax.lax.all_to_all(metadata, "expert", split_axis=0, concat_axis=0, tiled=True)
        received_metadata = tree_checkpoint_name(received_metadata, _CHECKPOINT_DISPATCH_INPUT)

        send_x = _dispatch_rows(x_local, safe_source, slot_valid, slot, "xla")
        received_x = jax.lax.all_to_all(
            send_x.reshape(ep_size, pair_capacity, hidden_dim), "expert", split_axis=0, concat_axis=0, tiled=True
        ).reshape(send_rows, hidden_dim)

    with jax.named_scope("expert_dispatch"):
        received_weights, group_sizes, receiver_dropped, picked, dispatch_valid, dispatch_row, position = (
            _expert_routing(
                received_metadata,
                local_experts=local_experts,
                capacity_factor=capacity_factor,
                expert_capacity=expert_capacity,
                routing_metadata=routing_metadata,
            )
        )
        x_dispatch = _dispatch_rows(
            received_x, jnp.minimum(dispatch_row, send_rows - 1), dispatch_valid, position, gather_sum
        )
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        moe_dim = moe_w2_local.shape[1]
        if grad_acc is None:
            w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        else:
            w13_out, grad_acc = _accumulating_ragged_dot(x_dispatch, moe_w13_local, group_sizes, grad_acc, "w13")
        w13_out = tree_checkpoint_name(w13_out, _CHECKPOINT_EXPERT_HIDDEN)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        hidden = activation_fn(gate) * up
        if grad_acc is None:
            out_dispatch = ragged_dot(hidden, moe_w2_local, group_sizes)
        else:
            out_dispatch, grad_acc = _accumulating_ragged_dot(hidden, moe_w2_local, group_sizes, grad_acc, "w2")
        out_dispatch = tree_checkpoint_name(out_dispatch, _CHECKPOINT_DISPATCH_OUTPUT)

    with jax.named_scope("combine"):
        expert_combined = _combine_gather_sum(
            out_dispatch,
            position.reshape(-1),
            picked,
            received_weights,
            out_dispatch.dtype,
        )
        returned = jax.lax.all_to_all(
            expert_combined.reshape(ep_size, pair_capacity, hidden_dim),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(send_rows, hidden_dim)
        out_local = _combine_rows(returned, safe_source, slot_valid, slot, gather_sum).astype(x_local.dtype)
    return out_local, sender_dropped, receiver_dropped, grad_acc


def _send_routing(
    selected_experts_local: Int[Array, "T K"],
    token_valid_local: Bool[Array, "T"],
    combine_weights_local: Float[Array, "T K"],
    *,
    local_experts: int,
    destinations: int,
    pair_capacity: int,
    pair_capacity_factor: float,
    routing_metadata: MoeRoutingMetadata,
) -> tuple[Int[Array, "T D"], Int[Array, "R"], Bool[Array, "R"], Float[Array, "D C1 K2"], Int[Array, ""]]:
    """The sender's routing maps and the metadata it sends.

    Returns each token's send row per destination (``slot``), each send row's token (clamped) and validity, the f32
    [destinations, C + 1, 2K] metadata buffer for the all-to-all (combine weights, local expert ids or -1, and a
    trailer row carrying this shard's valid demand) and the sender drops.
    """
    tokens, topk = selected_experts_local.shape
    send_rows = destinations * pair_capacity
    selected = selected_experts_local.astype(jnp.int32)
    destination = selected // local_experts
    valid_tokens = jnp.sum(token_valid_local, dtype=jnp.int32)
    logical_pair_capacity = _scaled_capacity(valid_tokens, capacity_factor=pair_capacity_factor, maximum=pair_capacity)
    assignment_valid = jnp.broadcast_to(token_valid_local[:, None], selected.shape)
    valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
    if routing_metadata == "pallas":
        send_maps = send_slots(
            selected,
            token_valid_local,
            logical_pair_capacity,
            local_experts=local_experts,
            destinations=destinations,
            pair_capacity=pair_capacity,
        )
        slot, source = send_maps.slot, send_maps.source
        sender_dropped = valid_assignments - send_maps.kept_assignments
    else:
        shard_ids = jnp.arange(destinations, dtype=jnp.int32)
        hit = jnp.any(destination[:, :, None] == shard_ids, axis=1) & token_valid_local[:, None]
        rank = jnp.cumsum(hit, axis=0, dtype=jnp.int32) - 1
        keep = hit & (rank < logical_pair_capacity)
        # slot[t, d]: the token's row in the flat [ep, C] send buffer, or the `send_rows` sentinel.
        slot = jnp.where(keep, shard_ids * pair_capacity + rank, send_rows)
        token_ids = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32)[:, None], slot.shape)
        source = jnp.full((send_rows,), tokens, jnp.int32).at[slot.reshape(-1)].set(token_ids.reshape(-1), mode="drop")
        sent_assignments = jnp.sum(jnp.take_along_axis(keep, destination, axis=1) & assignment_valid, dtype=jnp.int32)
        sender_dropped = valid_assignments - sent_assignments

    slot_valid = source < tokens
    safe_source = jnp.minimum(source, tokens - 1)
    slot, safe_source, slot_valid = tree_checkpoint_name((slot, safe_source, slot_valid), _CHECKPOINT_ROUTING)
    # Per send row: the token's combine weights and local expert ids on that destination. Scattering each
    # assignment to its send row, rather than gathering per send row, makes the weights' gradient a gather
    # instead of a scatter-add.
    # From the saved `slot`, so the backward (the weights' gradient gathers by it) need not rerun the routing.
    assignment_slot = jnp.take_along_axis(slot, destination, axis=1)
    route = jnp.arange(topk)
    slot_weights = (
        jnp.zeros((send_rows, topk), jnp.float32)
        .at[assignment_slot, route]
        .set(combine_weights_local.astype(jnp.float32), mode="drop", unique_indices=True)
    )
    slot_local_ids = (
        jnp.full((send_rows, topk), -1, jnp.int32)
        .at[assignment_slot, route]
        .set(selected % local_experts, mode="drop", unique_indices=True)
    )
    metadata = jnp.concatenate([slot_weights, slot_local_ids.astype(jnp.float32)], axis=1)
    metadata = metadata.reshape(destinations, pair_capacity, 2 * topk)
    # One trailer row per destination carries this shard's valid demand, which the receiver
    # sums into ring's logical capacity without a separate psum.
    trailer = (
        jnp.zeros((destinations, 1, 2 * topk), jnp.float32).at[:, 0, 0].set(valid_assignments.astype(jnp.float32))
    )
    metadata = jnp.concatenate([metadata, trailer], axis=1)
    return slot, safe_source, slot_valid, metadata, sender_dropped


def _expert_routing(
    received_metadata: Float[Array, "D C1 K2"],
    *,
    local_experts: int,
    capacity_factor: float,
    expert_capacity: int,
    routing_metadata: MoeRoutingMetadata,
) -> tuple[
    Float[Array, "R K"],
    Int[Array, "Elocal"],
    Int[Array, ""],
    Int[Array, "P"],
    Bool[Array, "P"],
    Int[Array, "P"],
    Int[Array, "R K"],
]:
    """The receiver's routing maps from the received metadata.

    Returns the received combine weights, the expert group sizes, the receiver drops, the received assignment in each
    expert row (``picked``, the assignment count for padding rows), which expert rows are valid, each expert row's
    received row (``dispatch_row``, the received row count for padding rows) and each received assignment's expert
    row (``position``, ``expert_capacity`` for none).
    """
    ep_size, pair_capacity = received_metadata.shape[0], received_metadata.shape[1] - 1
    topk = received_metadata.shape[2] // 2
    send_rows = ep_size * pair_capacity
    received_assignments = send_rows * topk
    group_valid_assignments = jnp.sum(received_metadata[:, pair_capacity, 0]).astype(jnp.int32)
    received_weights = received_metadata[:, :pair_capacity, :topk].reshape(send_rows, topk)
    logical_capacity = _scaled_capacity(
        group_valid_assignments,
        capacity_factor=capacity_factor,
        divisor=ep_size,
        minimum=local_experts,
        maximum=expert_capacity,
    )
    if routing_metadata == "pallas":
        rows = expert_rows(
            received_metadata, logical_capacity, local_experts=local_experts, physical_capacity=expert_capacity
        )
        group_sizes, receiver_dropped, picked, position = rows
        dispatch_valid = jnp.arange(expert_capacity, dtype=jnp.int32) < jnp.sum(group_sizes)
    else:
        received_ids = jax.lax.stop_gradient(received_metadata[:, :pair_capacity, topk:])
        received_ids = jnp.round(received_ids).astype(jnp.int32).reshape(received_assignments)
        assignment_mask = received_ids >= 0
        local_expert = jnp.where(assignment_mask, received_ids, 0)
        group_sizes, receiver_dropped, picked, dispatch_valid = _select_local_assignments(
            local_expert,
            assignment_mask,
            local_experts=local_experts,
            logical_capacity=logical_capacity,
            physical_capacity=expert_capacity,
        )
        picked = jnp.where(dispatch_valid, picked, received_assignments)
        position = _assignment_sources(picked, send_size=received_assignments)
    dispatch_row = jnp.where(dispatch_valid, picked // topk, send_rows)
    # position[r, j]: the dispatch row of received row r's j-th assignment, or the sentinel.
    position = position.reshape(send_rows, topk)
    group_sizes, picked, dispatch_valid, dispatch_row, position = tree_checkpoint_name(
        (group_sizes, picked, dispatch_valid, dispatch_row, position), _CHECKPOINT_ROUTING
    )
    return received_weights, group_sizes, receiver_dropped, picked, dispatch_valid, dispatch_row, position

def _accumulating_ragged_dot(
    lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, grad_acc: ExpertGradAccumulator, which: str
) -> tuple[jax.Array, ExpertGradAccumulator]:
    """``ragged_dot`` of the up (``which="w13"``, ``rhs`` = w_gate | w_up) or down (``"w2"``) projection with its weight
    gradient folded into this shard's block of ``grad_acc``."""
    if which == "w13":
        acc = (local_view(grad_acc.w_gate), local_view(grad_acc.w_up))
    else:
        acc = local_view(grad_acc.w_down)
    result = ragged_dot_accumulating_grad(lhs, rhs, group_sizes, acc, grad_acc.layer, grad_acc.mode)
    if grad_acc.mode == "final":
        return cast(jax.Array, result), grad_acc
    out, acc = cast(tuple[jax.Array, Any], result)
    if which == "w13":
        w_gate, w_up = acc
        return out, ExpertGradAccumulator(
            from_local_view(w_gate), from_local_view(w_up), grad_acc.w_down, grad_acc.layer, grad_acc.mode
        )
    return out, ExpertGradAccumulator(
        grad_acc.w_gate, grad_acc.w_up, from_local_view(acc), grad_acc.layer, grad_acc.mode
    )
