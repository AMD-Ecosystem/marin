# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Destination-filtered expert-parallel Grug MoE backend.

Each shard sends every local token only to the expert shards that own one of its selected experts,
once per shard, through a fixed-capacity ``all_to_all``. The expert shard runs ring's grouped
``ragged_dot`` path on the received rows, sums each token's local expert outputs, and returns one
row per received token through a second ``all_to_all``. The sender then sums the returned rows.

Compared with ``ring``, which all-gathers every token to every shard and reduce-scatters a
full-size output, this moves only the rows a destination works on: with top-4 of 256 experts over
8 shards, about 42% of ring's bytes at a sender capacity factor of 1.0.

Every hidden-width data movement is a gather whose transpose is another gather, so neither the
forward nor the backward uses a hidden-width scatter-add. The routing-weight gradient comes from
the down projection's backward (``_down_project_combine``), so the backward never needs the expert
outputs: remat neither saves them nor recomputes the down projection.

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
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_ROUTING,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.ep_common import _assignment_sources, _combine_gather_sum_impl
from levanter.grug._moe.ep_ring import _combine_rows, _dispatch_rows, _select_local_assignments
from levanter.grug._moe.flydsl_gather import gathered_up_projection
from levanter.grug._moe.grad_accum import ExpertGradAccumulator, from_local_view, local_view
from levanter.grug._moe.pallas_gather_sum import MoeGatherSum
from levanter.grug._moe.pallas_routing_metadata import MoeRoutingMetadata, expert_rows, send_slots
from levanter.grug._moe.scatter_epilogue import MoeScatterEpilogue, dispatch_up, down_combine, scatter_plan


def destination_hit_fraction(*, num_experts: int, local_experts: int, topk: int) -> float:
    """Probability that a token's ``topk`` distinct uniform experts include one on a given shard."""
    return 1.0 - math.comb(num_experts - local_experts, topk) / math.comb(num_experts, topk)


@jax.custom_vjp
def _slot_weights(
    combine_weights: Float[Array, "T K"],
    safe_source: Int[Array, "S"],
    on_destination: Bool[Array, "S K"],
    assignment_slot: Int[Array, "T K"],
) -> Float[Array, "S K"]:
    """``where(on_destination, combine_weights[safe_source], 0)`` in float32.

    ``assignment_slot[t, k]`` is the send slot that carries token ``t``'s assignment ``k`` (or an
    out-of-range sentinel). Each kept assignment has exactly one such slot, so the backward is a
    gather through that map instead of autodiff's scatter-add over ``safe_source``, which is slow
    in bfloat16 on AMD GPUs.
    """
    return jnp.where(on_destination, combine_weights[safe_source].astype(jnp.float32), 0.0)


def _slot_weights_fwd(combine_weights, safe_source, on_destination, assignment_slot):
    out = _slot_weights(combine_weights, safe_source, on_destination, assignment_slot)
    # The zero-size array carries the input dtype the cotangent must have.
    return out, (assignment_slot, jnp.zeros((0,), combine_weights.dtype))


def _slot_weights_bwd(residual, cotangent):
    assignment_slot, dtype_carrier = residual
    topk = assignment_slot.shape[1]
    flat = assignment_slot * topk + jnp.arange(topk, dtype=jnp.int32)
    d_weights = jnp.take(cotangent.reshape(-1), flat, mode="fill", fill_value=0)
    return d_weights.astype(dtype_carrier.dtype), None, None, None


_slot_weights.defvjp(_slot_weights_fwd, _slot_weights_bwd)


@jax.custom_vjp
def _down_project_combine(
    hidden: Float[Array, "M I"],
    moe_w2: Float[Array, "Elocal I H"],
    group_sizes: Int[Array, "Elocal"],
    weights: Float[Array, "N J"],
    inv: Int[Array, "N J"],
    flat_idx: Int[Array, "M"],
) -> Float[Array, "N H"]:
    """``out[n] = sum_j weights[n, j] * y[inv[n, j]]`` with ``y = ragged_dot(hidden, moe_w2, group_sizes)``.

    ``flat_idx[m]`` is the flat ``n * J + j`` entry that reads row ``m`` (or a sentinel), so the
    backward for ``y`` is a gather.

    The backward takes the routing-weight gradient from the down projection's own backward
    instead of from ``y``, so ``y`` is never saved or recomputed. For ``m = inv[n, j]``,
    ``dy[m] = weights[n, j] * cotangent[n]`` and ``dhidden = dy @ moe_w2[e].T``, hence
    ``<hidden[m], dhidden[m]> = weights[n, j] * <y[m], cotangent[n]> = weights[n, j] * dweights[n, j]``.
    The gradient is that row sum, in float32, divided by the weight; a zero weight gets zero.
    """
    y = ragged_dot(hidden, moe_w2, group_sizes)
    return _combine_gather_sum_impl(y, inv.reshape(-1), weights, y.dtype)


def _down_project_combine_fwd(hidden, moe_w2, group_sizes, weights, inv, flat_idx):
    y = ragged_dot(hidden, moe_w2, group_sizes)
    out = _combine_gather_sum_impl(y, inv.reshape(-1), weights, y.dtype)
    return out, (hidden, moe_w2, group_sizes, weights, inv, flat_idx)


def _expert_row_cotangent(cotangent, weights, flat_idx, dtype):
    """The cotangent of the down projection's rows ``y`` and each row's routing weight (zero for unread rows)."""
    num_rows, num_slots = weights.shape
    flat_valid = flat_idx < num_rows * num_slots
    safe_flat = jnp.minimum(flat_idx, num_rows * num_slots - 1)
    row_weight = jnp.where(flat_valid, weights.reshape(-1)[safe_flat].astype(jnp.float32), 0.0)
    d_y = (cotangent[safe_flat // num_slots].astype(jnp.float32) * row_weight[:, None]).astype(dtype)
    return d_y, row_weight


def _routing_weight_grad(hidden, d_hidden, row_weight, weights, inv):
    """``<hidden, dhidden> / w`` per row, in float32, gathered back to ``weights``' layout; a zero weight gets zero."""
    weighted_score = jnp.sum(hidden.astype(jnp.float32) * d_hidden.astype(jnp.float32), axis=-1)
    nonzero = row_weight != 0
    row_score = jnp.where(nonzero, weighted_score / jnp.where(nonzero, row_weight, 1.0), 0.0)
    return jnp.take(row_score, inv, mode="fill", fill_value=0).astype(weights.dtype)


def _down_project_combine_bwd(residual, cotangent):
    hidden, moe_w2, group_sizes, weights, inv, flat_idx = residual
    d_y, row_weight = _expert_row_cotangent(cotangent, weights, flat_idx, hidden.dtype)

    _, down_vjp = jax.vjp(lambda h, w: ragged_dot(h, w, group_sizes), hidden, moe_w2)
    d_hidden, d_w2 = down_vjp(d_y)

    d_weights = _routing_weight_grad(hidden, d_hidden, row_weight, weights, inv)
    return d_hidden, d_w2, None, d_weights, None, None


_down_project_combine.defvjp(_down_project_combine_fwd, _down_project_combine_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def _down_project_combine_accumulating(
    hidden: Float[Array, "M I"],
    moe_w2: Float[Array, "Elocal I H"],
    group_sizes: Int[Array, "Elocal"],
    weights: Float[Array, "N J"],
    inv: Int[Array, "N J"],
    flat_idx: Int[Array, "M"],
    acc: Float[Array, "L Elocal I H"],
    layer: Int[Array, ""],
    mode: str,
):
    """``_down_project_combine`` with the down projection as ``ragged_dot_accumulating_grad(..., acc, layer, mode)``:
    the weight gradient goes into the accumulator as there, and the routing-weight gradient comes from the down
    projection's backward as in ``_down_project_combine``, so the two paths match bit for bit. Returns the combined
    rows, and with modes "init"/"accum" also the accumulator."""
    return _down_project_combine_accumulating_fwd(
        hidden, moe_w2, group_sizes, weights, inv, flat_idx, acc, layer, mode
    )[0]


def _down_project_combine_accumulating_fwd(hidden, moe_w2, group_sizes, weights, inv, flat_idx, acc, layer, mode):
    out = ragged_dot_accumulating_grad(hidden, moe_w2, group_sizes, acc, layer, mode)
    y = out if mode == "final" else out[0]
    combined = _combine_gather_sum_impl(y, inv.reshape(-1), weights, y.dtype)
    # "final" reads the accumulator in the backward; "init"/"accum" take it from the cotangent.
    kept_acc = acc if mode == "final" else None
    result = combined if mode == "final" else (combined, out[1])
    return result, (hidden, moe_w2, group_sizes, weights, inv, flat_idx, kept_acc, layer)


def _down_project_combine_accumulating_bwd(mode, residual, cotangent):
    hidden, moe_w2, group_sizes, weights, inv, flat_idx, acc, layer = residual
    if mode == "final":
        out_cotangent = cotangent
    else:
        # The accumulator's cotangent carries the running sum; it also stands in for the primal, which the
        # accumulating backward does not read.
        out_cotangent, acc = cotangent
    d_y, row_weight = _expert_row_cotangent(out_cotangent, weights, flat_idx, hidden.dtype)

    _, down_vjp = jax.vjp(
        lambda h, w, a: ragged_dot_accumulating_grad(h, w, group_sizes, a, layer, mode), hidden, moe_w2, acc
    )
    d_hidden, d_w2, d_acc = down_vjp(d_y if mode == "final" else (d_y, acc))

    d_weights = _routing_weight_grad(hidden, d_hidden, row_weight, weights, inv)
    # "final" only reads the accumulator, which gets no cotangent, as in ragged_dot_accumulating_grad.
    return d_hidden, d_w2, None, d_weights, None, None, None if mode == "final" else d_acc, None


_down_project_combine_accumulating.defvjp(
    _down_project_combine_accumulating_fwd, _down_project_combine_accumulating_bwd
)


def _moe_mlp_ep_ring_filtered_local(
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
    dispatch_barrier: bool = False,
    gather_sum: MoeGatherSum = "xla",
    routing_metadata: MoeRoutingMetadata = "xla",
    flydsl_gather_a: bool = False,
    scatter_epilogue: MoeScatterEpilogue = "off",
    grad_acc: ExpertGradAccumulator | None = None,
) -> (
    tuple[Float[Array, "Tlocal H"], CapacityDrops]
    | tuple[Float[Array, "Tlocal H"], CapacityDrops, ExpertGradAccumulator]
):
    """Filtered all-to-all dispatch, ring's grouped expert MLP, expert-side combine.

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

    ``dispatch_barrier`` puts an optimization barrier on the expert-ordered dispatch rows ``x_dispatch``: the remat
    policy saves them, and without it XLA fuses a second copy of their gather into the update of the saved stack. With
    ``flydsl_gather_a`` or a ``scatter_epilogue`` those rows are never built and it does nothing.

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

    ``flydsl_gather_a`` runs the up projection on the received rows through ``flydsl_gather.gathered_up_projection``:
    the jax-flydsl GEMM gathers its A rows itself, so the expert-ordered copy ``x_dispatch`` is never built and the
    received rows are what the remat policy saves. Same values and gradients bit for bit; needs ``ragged_dot``'s flydsl
    implementation on GPU.

    ``scatter_epilogue`` other than "off" runs the expert-side combine and dispatch backward as row-scatter epilogues
    of the expert GEMMs plus a gather-sum of the remaining rows (see ``scatter_epilogue``), with the same results bit
    for bit. It replaces the expert-side dispatch gather-sum, so ``gather_sum`` then only affects the sender side. With ``flydsl_gather_a`` the up projection inside it gathers its
    A rows as well, and with ``grad_acc`` its weight gradients go into the accumulator.
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
        _ring_filtered_block,
        moe_w13_local=moe_w13_local,
        moe_w2_local=moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        transport_capacity_factor=transport_capacity_factor,
        dispatch_barrier=dispatch_barrier,
        gather_sum=gather_sum,
        routing_metadata=routing_metadata,
        flydsl_gather_a=flydsl_gather_a,
        scatter_epilogue=scatter_epilogue,
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


def _ring_filtered_block(
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
    dispatch_barrier: bool,
    gather_sum: MoeGatherSum,
    routing_metadata: MoeRoutingMetadata,
    flydsl_gather_a: bool,
    scatter_epilogue: MoeScatterEpilogue,
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
        if scatter_epilogue != "off":
            scatter_rows, gather_rows = scatter_plan(position, picked, dispatch_valid)
            scatter_rows, gather_rows = tree_checkpoint_name((scatter_rows, gather_rows), _CHECKPOINT_ROUTING)
        if flydsl_gather_a:
            # The up projection reads the received rows through dispatch_row (sentinel send_rows reads zeros), so
            # they, not an expert-ordered copy, are the saved dispatch input.
            received_x = tree_checkpoint_name(received_x, _CHECKPOINT_DISPATCH_INPUT)
        elif scatter_epilogue == "off":
            # (dispatch_up saves its own dispatch gather under the same name.)
            x_dispatch = _dispatch_rows(
                received_x, jnp.minimum(dispatch_row, send_rows - 1), dispatch_valid, position, gather_sum
            )
            if dispatch_barrier:
                x_dispatch = jax.lax.optimization_barrier(x_dispatch)
            x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    moe_dim = moe_w2_local.shape[1]
    with jax.named_scope("moe_up_down"):
        if scatter_epilogue != "off":
            w13_out = dispatch_up(
                received_x,
                jnp.minimum(dispatch_row, send_rows - 1),
                dispatch_valid,
                moe_w13_local,
                group_sizes,
                position,
                scatter_rows,
                gather_rows,
                scatter_epilogue,
                gather_a=flydsl_gather_a,
                **_acc_args(grad_acc, "w13"),
            )
        elif flydsl_gather_a:
            w13_out = gathered_up_projection(
                received_x,
                dispatch_row,
                position,
                moe_w13_local,
                group_sizes,
                gather_sum,
                **_acc_args(grad_acc, "w13"),
            )
        elif grad_acc is None:
            w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        else:
            w13_out = ragged_dot_accumulating_grad(
                x_dispatch, moe_w13_local, group_sizes, **_acc_args(grad_acc, "w13")
            )
        w13_out, grad_acc = _acc_result(w13_out, grad_acc, "w13")
        w13_out = tree_checkpoint_name(w13_out, _CHECKPOINT_EXPERT_HIDDEN)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        hidden = activation_fn(gate) * up
        if scatter_epilogue == "off":
            if grad_acc is None:
                expert_combined = _down_project_combine(
                    hidden, moe_w2_local, group_sizes, received_weights, position, picked
                )
            else:
                acc_args = _acc_args(grad_acc, "w2")
                expert_combined = _down_project_combine_accumulating(
                    hidden,
                    moe_w2_local,
                    group_sizes,
                    received_weights,
                    position,
                    picked,
                    acc_args["acc"],
                    acc_args["layer"],
                    acc_args["mode"],
                )
                expert_combined, grad_acc = _acc_result(expert_combined, grad_acc, "w2")

    if scatter_epilogue != "off":
        with jax.named_scope("combine"):
            # The down projection's output is saved inside down_combine under _CHECKPOINT_DISPATCH_OUTPUT.
            expert_combined = down_combine(
                hidden,
                moe_w2_local,
                group_sizes,
                position,
                picked,
                received_weights,
                scatter_rows,
                gather_rows,
                scatter_epilogue,
                **_acc_args(grad_acc, "w2"),
            )
            expert_combined, grad_acc = _acc_result(expert_combined, grad_acc, "w2")

    with jax.named_scope("combine"):
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
    # Per send row: the token's combine weights and local expert ids on that destination.
    slot_destination = jnp.arange(send_rows, dtype=jnp.int32) // pair_capacity
    slot_selected = selected[safe_source]
    on_destination = slot_valid[:, None] & (slot_selected // local_experts == slot_destination[:, None])
    slot_local_ids = jnp.where(on_destination, slot_selected - slot_destination[:, None] * local_experts, -1)
    # assignment_slot[t, k]: the send slot carrying assignment (t, k), or the `send_rows` sentinel. From the saved
    # `slot`, so the weights' gradient (a gather by it) need not rerun the routing.
    assignment_slot = jnp.take_along_axis(slot, destination, axis=1)
    slot_weights = _slot_weights(combine_weights_local, safe_source, on_destination, assignment_slot)
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


def _acc_args(grad_acc: ExpertGradAccumulator | None, which: str) -> dict[str, Any]:
    """The ``acc``/``layer``/``mode`` keywords of an expert matmul that folds the weight gradient of the up
    (``which="w13"``, w_gate | w_up as two column halves) or down (``"w2"``) projection into this shard's block of
    ``grad_acc`` (``ragged_dot_accumulating_grad``, ``gathered_up_projection``, ``dispatch_up``, ``down_combine``)."""
    if grad_acc is None:
        return {}
    if which == "w13":
        acc: Any = (local_view(grad_acc.w_gate), local_view(grad_acc.w_up))
    else:
        acc = local_view(grad_acc.w_down)
    return {"acc": acc, "layer": grad_acc.layer, "mode": grad_acc.mode}


def _acc_result(
    result: Any, grad_acc: ExpertGradAccumulator | None, which: str
) -> tuple[jax.Array, ExpertGradAccumulator | None]:
    """``(output, grad_acc)`` from what a matmul called with ``_acc_args(grad_acc, which)`` returned: the output, or
    with modes "init"/"accum" the output and the updated accumulator block."""
    if grad_acc is None or grad_acc.mode == "final":
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
