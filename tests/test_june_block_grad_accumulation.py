# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Block weight gradients summed inside the layer scan's backward (``block_grad_accumulation="in_scan"``) and the
cross-entropy backward writing the output projection's gradient in place (``ce_weight_grad="in_place"``).

With "in_scan", gradient accumulation threads the float32 sums of the stacked block weights through every microbatch's
layer scan, and each layer's backward adds its weight gradients into its row of them; with "after_scan" the scan
returns the bf16 gradients stacked over the layers and a float32 add after it updates the sums. The arithmetic is the
same, so the step must give the same loss and gradients bit for bit: in the loop, with the overlapped last microbatch,
with ZeRO-1 and the first microbatch's gathers, with the expert weights summed by XLA (in the scan too) or by the
weight-gradient kernel (jnp stand-ins, as in ``test_june_expert_grad_accumulation``), and with the static attention
schedule's layer loop.

Each case runs in a subprocess so XLA can configure the CPU device count before JAX initializes.
"""

import textwrap

import pytest

from tests.june_cpu_devices import run_on_cpu_devices
from tests.test_june_expert_grad_accumulation import _PRELUDE as _EXPERT_PRELUDE

_PRELUDE = (
    _EXPERT_PRELUDE
    + r"""


def block_grads(
    mesh, *, block, accumulation="xla", zero1=False, overlap=False, gather=False, microbatches=4, cfg=GA_CFG
):
    optimizer = keep_gradients()
    with jax.set_mesh(mesh):
        state = june_train.initial_state(
            cfg, optimizer=optimizer, mp=POLICY, key=jax.random.key(0), ema_beta=None,
            carry_compute_params=zero1, zero1=zero1,
        )
        train_step = june_train._make_train_step(
            optimizer, POLICY, z_loss_weight=0.0, ema_beta=None, microbatches=microbatches, zero1=zero1,
            overlap_last_microbatch=overlap, gather_in_first_microbatch=gather,
            expert_grad_accumulation=accumulation, block_grad_accumulation=block,
        )
        state, metrics, _ = train_step(state, BATCH)
    return jax.device_get(metrics["train/loss"]), jax.device_get(state.opt_state)


def assert_bitwise(name, loss, grads, k_loss, k_grads):
    assert np.array_equal(loss, k_loss), (name, loss, k_loss)
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(grads), jax.tree.leaves(k_grads), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, (name, jax.tree_util.keystr(path))
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (name, jax.tree_util.keystr(path))


def assert_same_block_step(**kwargs):
    for name, mesh in MESHES.items():
        loss, grads = block_grads(mesh, block="after_scan", **kwargs)
        k_loss, k_grads = block_grads(mesh, block="in_scan", **kwargs)
        assert_bitwise(name, loss, grads, k_loss, k_grads)
        stacked = [x for p, x in jax.tree_util.tree_leaves_with_path(k_grads) if "stacked" in jax.tree_util.keystr(p)]
        nonzero = [x for x in stacked if np.abs(np.asarray(x, np.float32)).max() > 0]
        # Every stacked weight but router_bias (no gradient) gets one.
        assert len(nonzero) == len(stacked) - 1, (name, len(nonzero), len(stacked))
"""
)


def _run(body: str) -> None:
    run_on_cpu_devices(_PRELUDE + textwrap.dedent(body), device_count=8)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("accumulation", ["xla", "flydsl"])
def test_in_scan_sums_match_after_scan_in_the_microbatch_loop(accumulation):
    _run(f"assert_same_block_step(accumulation={accumulation!r})")


@pytest.mark.timeout(900)
@pytest.mark.parametrize("accumulation", ["xla", "flydsl"])
def test_in_scan_sums_match_after_scan_with_the_overlapped_last_microbatch(accumulation):
    # The last microbatch reduces the expert weights per layer from offsets (the in-scan sums with "xla") and carries
    # the other sums through its scan.
    _run(f"assert_same_block_step(accumulation={accumulation!r}, overlap=True)")


@pytest.mark.timeout(900)
@pytest.mark.parametrize("accumulation", ["xla", "flydsl"])
def test_in_scan_sums_match_after_scan_with_zero1_and_first_microbatch_gathers(accumulation):
    # The first microbatch starts the sums after its scan; the loop and the last microbatch then sum in the scan.
    _run(f"assert_same_block_step(accumulation={accumulation!r}, zero1=True, gather=True, overlap=True)")


@pytest.mark.timeout(900)
def test_in_scan_sums_with_two_microbatches():
    # The loop then runs one microbatch; with the gathers it runs none, the first and last running outside it.
    _run(
        """
        assert_same_block_step(microbatches=2)
        assert_same_block_step(microbatches=2, overlap=True)
        assert_same_block_step(microbatches=2, accumulation="flydsl", zero1=True, gather=True, overlap=True)
        """
    )


@pytest.mark.timeout(900)
def test_in_scan_sums_with_the_static_attention_schedule():
    # The trace-time schedule slices each layer from the whole stacks inside its checkpoint; the sums go in with them.
    _run(
        """
        cfg = dataclasses.replace(GA_CFG, num_layers=6, attention_schedule="static")
        assert_same_block_step(cfg=cfg)
        assert_same_block_step(cfg=cfg, accumulation="flydsl", zero1=True, gather=True, overlap=True)
        """
    )


def test_in_scan_sums_need_microbatches_and_the_stacked_layout():
    _run(
        """
        try:
            june_train._make_train_step(
                keep_gradients(), POLICY, z_loss_weight=0.0, ema_beta=None, microbatches=1,
                block_grad_accumulation="in_scan",
            )
        except ValueError as e:
            assert "microbatches" in str(e)
        else:
            raise AssertionError("expected a ValueError")
        try:
            unstacked = dataclasses.replace(
                GA_CFG, use_array_stacked_blocks=False, replicated_grad_reduction="per_layer",
                router_stats_reduction="per_layer",
            )
            block_grads(MESHES["replica4_expert2"], block="in_scan", cfg=unstacked)
        except ValueError as e:
            assert "stacked" in str(e)
        else:
            raise AssertionError("expected a ValueError")
        """
    )


@pytest.mark.timeout(900)
def test_in_place_ce_weight_grad_gives_the_stacked_step():
    # batched_xla refuses non-GPU backends; with that check skipped its streaming backward runs on CPU. A vocabulary of
    # 40 in blocks of 16: two whole blocks and a partial one (padded with "stacked", narrower with "in_place").
    _run(
        """
        from levanter.kernels.pallas.fused_cross_entropy_loss import batched_xla

        batched_xla._validate_input_contract = lambda *args, **kwargs: None
        stacked = dataclasses.replace(GA_CFG, vocab_size=40, ce_implementation="batched_xla", ce_v_block_size=16)
        in_place = dataclasses.replace(stacked, ce_implementation=None, ce_weight_grad="in_place")
        for name, mesh in MESHES.items():
            for kwargs in ({}, {"zero1": True, "gather": True, "overlap": True}):
                loss, grads = block_grads(mesh, block="after_scan", cfg=stacked, **kwargs)
                k_loss, k_grads = block_grads(mesh, block="in_scan", cfg=in_place, **kwargs)
                assert_bitwise((name, kwargs), loss, grads, k_loss, k_grads)
        try:
            dataclasses.replace(GA_CFG, ce_implementation="xla", ce_weight_grad="in_place")
        except ValueError as e:
            assert "ce_implementation" in str(e)
        else:
            raise AssertionError("expected a ValueError")
        """
    )
