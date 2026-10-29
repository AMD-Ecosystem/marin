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

With ``token_block_groups`` (two blocks) the overlap is requested rather than left to XLA's scheduler:
the stages are emitted in pipeline order and tagged with XLA scheduling groups, block 1's dispatch
all-to-alls with block 0's expert MLP, then block 0's combine all-to-all with block 1's expert MLP;
the backward pass gets its own pair of groups. XLA keeps the annotations on the expert MLP's kernels
only with ``--xla_gpu_experimental_keep_scheduling_annotations_on_compute``.
"""

import dataclasses
import itertools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jax.experimental.xla_metadata import xla_metadata_call2
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

# XLA frontend attribute that puts ops in a latency-hiding-scheduler group: scheduled together, the
# group's async collectives overlapping its compute.
SCHEDULING_GROUP_ATTR = "_scheduling_group_id"
# Group ids must be unique per traced layer: one id in two unrolled layers makes one group with
# unrelated ops between its members, which XLA's annotation legalization rejects.
_scheduling_group_ids = itertools.count()


def destination_hit_fraction(*, num_experts: int, local_experts: int, topk: int) -> float:
    """Probability that a token's ``topk`` distinct uniform experts include one on a given shard."""
    return 1.0 - math.comb(num_experts - local_experts, topk) / math.comb(num_experts, topk)


@dataclasses.dataclass(frozen=True)
class _BlockSizes:
    """Static sizes of one block's exchange."""

    tokens: int
    topk: int
    hidden_dim: int
    local_experts: int
    ep_size: int
    hit_fraction: float
    transport_capacity_factor: float
    capacity_factor: float
    pair_capacity: int
    expert_capacity: int

    @property
    def send_rows(self) -> int:
        return self.ep_size * self.pair_capacity

    @property
    def received_assignments(self) -> int:
        return self.send_rows * self.topk


def _block_sizes(
    *,
    tokens: int,
    topk: int,
    hidden_dim: int,
    num_experts: int,
    local_experts: int,
    capacity_factor: float,
    transport_capacity_factor: float,
) -> _BlockSizes:
    hit_fraction = destination_hit_fraction(num_experts=num_experts, local_experts=local_experts, topk=topk)
    pair_capacity = min(tokens, max(1, math.ceil(transport_capacity_factor * hit_fraction * tokens)))
    ep_size = num_experts // local_experts
    received_assignments = ep_size * pair_capacity * topk
    expert_capacity = min(received_assignments, max(local_experts, math.ceil(capacity_factor * tokens * topk)))
    return _BlockSizes(
        tokens=tokens,
        topk=topk,
        hidden_dim=hidden_dim,
        local_experts=local_experts,
        ep_size=ep_size,
        hit_fraction=hit_fraction,
        transport_capacity_factor=transport_capacity_factor,
        capacity_factor=capacity_factor,
        pair_capacity=pair_capacity,
        expert_capacity=expert_capacity,
    )


class _Sent(NamedTuple):
    """A block's send buffers, and what its sender needs to sum the returned rows."""

    metadata: Float[Array, "ep C1 K2"]
    send_x: Float[Array, "S H"]
    safe_source: Int[Array, " S"]
    slot_valid: Bool[Array, " S"]
    slot: Int[Array, "T ep"]
    sender_dropped: Int[Array, ""]


class _Dispatched(NamedTuple):
    """A block's rows and metadata as the expert shard received them."""

    metadata: Float[Array, "ep C1 K2"]
    x: Float[Array, "S H"]


_ExpertFn = Callable[[_Dispatched], tuple[Float[Array, "S H"], Int[Array, ""]]]
_BlockResult = tuple[Float[Array, "T H"], Int[Array, ""], Int[Array, ""]]


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
    token_block_groups: bool,
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
    """Deduplicated all-to-all dispatch, ring's grouped expert MLP, expert-side combine.

    The shard's tokens are split into ``token_blocks`` contiguous blocks, each dispatched, run and
    combined on its own, with the capacities below computed from the block's tokens.
    ``token_block_groups`` (requires two blocks) pipelines them through XLA scheduling groups.

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
    """
    tokens, topk = selected_experts_local.shape
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")
    if transport_capacity_factor <= 0:
        raise ValueError("transport_capacity_factor must be positive")
    if token_blocks < 1 or tokens % token_blocks != 0:
        raise ValueError(f"token_blocks={token_blocks} must be positive and divide the local token count {tokens}")
    if token_block_groups and token_blocks != 2:
        raise ValueError(f"token_block_groups requires token_blocks=2, got {token_blocks}")

    sizes = _block_sizes(
        tokens=tokens // token_blocks,
        topk=topk,
        hidden_dim=x_local.shape[1],
        num_experts=num_experts,
        local_experts=local_experts,
        capacity_factor=capacity_factor,
        transport_capacity_factor=transport_capacity_factor,
    )
    blocks = [
        (x_local[rows], selected_experts_local[rows], combine_weights_local[rows], token_valid_local[rows])
        for rows in (slice(i * sizes.tokens, (i + 1) * sizes.tokens) for i in range(token_blocks))
    ]

    def expert(dispatched: _Dispatched) -> tuple[Float[Array, "S H"], Int[Array, ""]]:
        return _expert_mlp(dispatched, moe_w13_local, moe_w2_local, activation_fn, sizes)

    if token_block_groups:
        results = _pipelined_blocks(blocks, expert, sizes)
    elif token_blocks == 1:
        results = [_block(*blocks[0], expert=expert, sizes=sizes)]
    else:
        results = []
        for i, block in enumerate(blocks):
            with jax.named_scope(f"token_block_{i}"):
                results.append(_block(*block, expert=expert, sizes=sizes))

    if token_blocks == 1:
        out_local = results[0][0]
    else:
        out_local = jnp.concatenate([out for out, _, _ in results], axis=0)
    sender_dropped = sum(sender for _, sender, _ in results)
    receiver_dropped = sum(receiver for _, _, receiver in results)
    with jax.named_scope("combine"):
        drops = jax.lax.psum(jnp.stack([sender_dropped, receiver_dropped]), token_sharding_axes)
    return out_local, CapacityDrops(sender_dropped=drops[0], receiver_dropped=drops[1])


def _block(
    x_local: Float[Array, "T H"],
    selected_experts_local: Int[Array, "T K"],
    combine_weights_local: Float[Array, "T K"],
    token_valid_local: Bool[Array, "T"],
    *,
    expert: _ExpertFn,
    sizes: _BlockSizes,
) -> _BlockResult:
    """One block of tokens through the exchange: its output and its local sender and receiver drops."""
    sent = _send(x_local, selected_experts_local, combine_weights_local, token_valid_local, sizes)
    expert_combined, receiver_dropped = expert(_dispatch(sent, sizes))
    out_local = _receive(_return_rows(expert_combined, sizes), sent, x_local.dtype)
    return out_local, sent.sender_dropped, receiver_dropped


def _pipelined_blocks(
    blocks: list[tuple[Array, Array, Array, Array]],
    expert: _ExpertFn,
    sizes: _BlockSizes,
) -> list[_BlockResult]:
    """Two blocks in pipeline order, each overlapping pair of stages in an XLA scheduling group.

    Forward: b0 dispatch, [b1 dispatch | b0 expert MLP], [b0 combine | b1 expert MLP], b1 combine.
    The backward pass runs the stages in reverse, in fresh groups: [b0 combine^T | b1 expert MLP^T],
    then [b1 dispatch^T | b0 expert MLP^T].
    """
    (x0, *routing0), (x1, *routing1) = blocks
    group_a, group_a_bwd, group_b, group_b_bwd = (next(_scheduling_group_ids) for _ in range(4))

    def grouped(fn, group: int, backward_group: int):
        return xla_metadata_call2(
            fn, {SCHEDULING_GROUP_ATTR: group}, ad_metadata={SCHEDULING_GROUP_ATTR: backward_group}
        )

    def stage_a(sent1: _Sent, dispatched0: _Dispatched):
        with jax.named_scope("token_block_1"):
            dispatched1 = _dispatch(sent1, sizes)
        with jax.named_scope("token_block_0"):
            return dispatched1, expert(dispatched0)

    def stage_b(expert_combined0: Float[Array, "S H"], dispatched1: _Dispatched):
        with jax.named_scope("token_block_0"):
            returned0 = _return_rows(expert_combined0, sizes)
        with jax.named_scope("token_block_1"):
            return returned0, expert(dispatched1)

    with jax.named_scope("token_block_0"):
        sent0 = _send(x0, *routing0, sizes)
        dispatched0 = _dispatch(sent0, sizes)
    with jax.named_scope("token_block_1"):
        sent1 = _send(x1, *routing1, sizes)
    dispatched1, (expert_combined0, receiver_dropped0) = grouped(stage_a, group_a, group_a_bwd)(sent1, dispatched0)
    returned0, (expert_combined1, receiver_dropped1) = grouped(stage_b, group_b, group_b_bwd)(
        expert_combined0, dispatched1
    )
    with jax.named_scope("token_block_0"):
        out0 = _receive(returned0, sent0, x0.dtype)
    with jax.named_scope("token_block_1"):
        out1 = _receive(_return_rows(expert_combined1, sizes), sent1, x1.dtype)
    return [(out0, sent0.sender_dropped, receiver_dropped0), (out1, sent1.sender_dropped, receiver_dropped1)]


def _send(
    x_local: Float[Array, "T H"],
    selected_experts_local: Int[Array, "T K"],
    combine_weights_local: Float[Array, "T K"],
    token_valid_local: Bool[Array, "T"],
    sizes: _BlockSizes,
) -> _Sent:
    """Routes a block's tokens to send rows and builds the send buffer and its metadata."""
    tokens, topk, local_experts, ep_size = sizes.tokens, sizes.topk, sizes.local_experts, sizes.ep_size
    pair_capacity, send_rows = sizes.pair_capacity, sizes.send_rows
    with jax.named_scope("dispatch"):
        selected = selected_experts_local.astype(jnp.int32)
        destination = selected // local_experts
        shard_ids = jnp.arange(ep_size, dtype=jnp.int32)
        hit = jnp.any(destination[:, :, None] == shard_ids, axis=1) & token_valid_local[:, None]
        rank = jnp.cumsum(hit, axis=0, dtype=jnp.int32) - 1
        valid_tokens = jnp.sum(token_valid_local, dtype=jnp.int32)
        logical_pair_capacity = _scaled_capacity(
            valid_tokens,
            capacity_factor=sizes.transport_capacity_factor * sizes.hit_fraction,
            maximum=pair_capacity,
        )
        keep = hit & (rank < logical_pair_capacity)
        # slot[t, d]: the token's row in the flat [ep, C] send buffer, or the `send_rows` sentinel.
        slot = jnp.where(keep, shard_ids * pair_capacity + rank, send_rows)
        token_ids = jnp.broadcast_to(jnp.arange(tokens, dtype=jnp.int32)[:, None], slot.shape)
        source = jnp.full((send_rows,), tokens, jnp.int32).at[slot.reshape(-1)].set(token_ids.reshape(-1), mode="drop")

        assignment_valid = jnp.broadcast_to(token_valid_local[:, None], selected.shape)
        valid_assignments = jnp.sum(assignment_valid, dtype=jnp.int32)
        sent_assignments = jnp.sum(jnp.take_along_axis(keep, destination, axis=1) & assignment_valid, dtype=jnp.int32)
        sender_dropped = valid_assignments - sent_assignments

        slot_valid = source < tokens
        safe_source = jnp.minimum(source, tokens - 1)
        slot, safe_source, slot_valid = tree_checkpoint_name((slot, safe_source, slot_valid), _CHECKPOINT_ROUTING)
        # Per send row: the token's combine weights and local expert ids on that destination. Scattering each
        # assignment to its send row, rather than gathering per send row, makes the weights' gradient a gather
        # instead of a scatter-add.
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
        metadata = metadata.reshape(ep_size, pair_capacity, 2 * topk)
        # One trailer row per destination carries this shard's valid demand, which the receiver
        # sums into ring's logical capacity without a separate psum.
        trailer = jnp.zeros((ep_size, 1, 2 * topk), jnp.float32).at[:, 0, 0].set(valid_assignments.astype(jnp.float32))
        metadata = jnp.concatenate([metadata, trailer], axis=1)

        send_x = _dispatch_rows(x_local, safe_source, slot_valid, slot)
    return _Sent(metadata, send_x, safe_source, slot_valid, slot, sender_dropped)


def _dispatch(sent: _Sent, sizes: _BlockSizes) -> _Dispatched:
    """The dispatch all-to-alls: each destination receives its rows and their metadata."""
    with jax.named_scope("dispatch"):
        received_metadata = jax.lax.all_to_all(sent.metadata, "expert", split_axis=0, concat_axis=0, tiled=True)
        received_metadata = tree_checkpoint_name(received_metadata, _CHECKPOINT_DISPATCH_INPUT)
        received_x = jax.lax.all_to_all(
            sent.send_x.reshape(sizes.ep_size, sizes.pair_capacity, sizes.hidden_dim),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(sizes.send_rows, sizes.hidden_dim)
    return _Dispatched(received_metadata, received_x)


def _expert_mlp(
    dispatched: _Dispatched,
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    activation_fn: Callable[[jax.Array], jax.Array],
    sizes: _BlockSizes,
) -> tuple[Float[Array, "S H"], Int[Array, ""]]:
    """The expert shard's side: group the received assignments, run the MLP, sum each row's outputs."""
    topk, local_experts, ep_size = sizes.topk, sizes.local_experts, sizes.ep_size
    pair_capacity, send_rows, received_assignments = sizes.pair_capacity, sizes.send_rows, sizes.received_assignments
    received_metadata = dispatched.metadata
    with jax.named_scope("expert_dispatch"):
        group_valid_assignments = jnp.sum(received_metadata[:, pair_capacity, 0]).astype(jnp.int32)
        received_weights = received_metadata[:, :pair_capacity, :topk].reshape(send_rows, topk)
        received_ids = jax.lax.stop_gradient(received_metadata[:, :pair_capacity, topk:])
        received_ids = jnp.round(received_ids).astype(jnp.int32).reshape(received_assignments)
        assignment_mask = received_ids >= 0
        local_expert = jnp.where(assignment_mask, received_ids, 0)
        logical_capacity = _scaled_capacity(
            group_valid_assignments,
            capacity_factor=sizes.capacity_factor,
            divisor=ep_size,
            minimum=local_experts,
            maximum=sizes.expert_capacity,
        )
        group_sizes, receiver_dropped, picked, dispatch_valid = _select_local_assignments(
            local_expert,
            assignment_mask,
            local_experts=local_experts,
            logical_capacity=logical_capacity,
            physical_capacity=sizes.expert_capacity,
        )
        picked = jnp.where(dispatch_valid, picked, received_assignments)
        dispatch_row = jnp.where(dispatch_valid, picked // topk, send_rows)
        # position[r, j]: the dispatch row of received row r's j-th assignment, or the sentinel.
        position = _assignment_sources(picked, send_size=received_assignments).reshape(send_rows, topk)
        group_sizes, picked, dispatch_valid, dispatch_row, position = tree_checkpoint_name(
            (group_sizes, picked, dispatch_valid, dispatch_row, position), _CHECKPOINT_ROUTING
        )

        x_dispatch = _dispatch_rows(dispatched.x, jnp.minimum(dispatch_row, send_rows - 1), dispatch_valid, position)
        x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13_local, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = tree_checkpoint_name(
            ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("combine"):
        expert_combined = _combine_gather_sum(
            out_dispatch,
            position.reshape(-1),
            picked,
            received_weights,
            out_dispatch.dtype,
        )
    return expert_combined, receiver_dropped


def _return_rows(expert_combined: Float[Array, "S H"], sizes: _BlockSizes) -> Float[Array, "S H"]:
    """The combine all-to-all: each sender gets back one summed row per row it sent."""
    with jax.named_scope("combine"):
        return jax.lax.all_to_all(
            expert_combined.reshape(sizes.ep_size, sizes.pair_capacity, sizes.hidden_dim),
            "expert",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(sizes.send_rows, sizes.hidden_dim)


def _receive(returned: Float[Array, "S H"], sent: _Sent, dtype: jnp.dtype) -> Float[Array, "T H"]:
    """Sums each token's returned rows."""
    with jax.named_scope("combine"):
        return _combine_rows(returned, sent.safe_source, sent.slot_valid, sent.slot).astype(dtype)
