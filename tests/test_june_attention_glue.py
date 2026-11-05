# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June attention's glue options: ``qk_norm_rope`` (fused q/k head norm, half-RoPE and scale) and
``attention_schedule`` (per-layer ``lax.cond``, trace-time schedule, or every layer sliding-window).

The fused q/k functions' numerics are tested in ``lib/levanter/tests/kernels/test_pallas_qk_norm_rope.py``; here the
model with each of them trains like the separate ops. The static schedule runs the same layers with the same weights,
so it gives the same loss and gradients; the HLO has no conditional left. "all_short" is a different model, equal to
the others only when the window covers the whole sequence.

The train-step case runs in a subprocess so XLA can configure eight CPU devices before JAX initializes.
"""

import dataclasses
import re
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from tests.test_june_zero1 import _PRELUDE as _ZERO1_PRELUDE

_SMALL = GrugModelConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=16,
    shared_expert_intermediate_dim=16,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=6,
    num_heads=4,
    num_kv_heads=2,
    head_dim=64,
    max_seq_len=16,
    sliding_window=8,
    capacity_factor=8.0,
    attention_implementation="reference",
    use_array_stacked_blocks=True,
    disable_pko=True,
    ce_implementation="xla",
    rms_norm_impl="custom_vjp",
)


def _loss_and_grads(cfg: GrugModelConfig, *, position_ids=None):
    # A random projection of the final hidden states (next_token_loss takes no positions).
    tokens = jax.random.randint(jax.random.key(1), (4, cfg.max_seq_len), 0, cfg.vocab_size)
    target = jax.random.normal(jax.random.key(3), (4, cfg.max_seq_len, cfg.hidden_dim))
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.key(0))
        params, static = eqx.partition(model, eqx.is_array)

        def loss(p):
            hidden, _ = eqx.combine(p, static)(tokens, position_ids=position_ids)
            return jnp.mean(hidden * target)

        fn = jax.jit(jax.value_and_grad(loss))
        hlo = fn.lower(params).compile().as_text()
        value, grads = fn(params)
    return float(value), [np.asarray(g, np.float32) for g in jax.tree.leaves(grads)], hlo


def _assert_grads_close(grads, reference, rtol):
    for g, r in zip(grads, reference, strict=True):
        assert np.linalg.norm(g - r) <= rtol * max(np.linalg.norm(r), 1e-30), (np.linalg.norm(g - r), np.linalg.norm(r))


@pytest.mark.parametrize("impl", ["custom_vjp", "pallas"])
@pytest.mark.parametrize(
    "variant",
    [
        {},
        # The long layers' own options and per-example positions.
        {"use_array_stacked_blocks": False, "disable_long_rope": True, "qk_mult_long_scale": 1.5, "positions": True},
    ],
)
def test_fused_qk_norm_rope_trains_like_the_separate_ops(impl, variant):
    # float32: the fused functions differ from the separate ops only in how the norm's statistic is summed, a few
    # float32 ulps, which the loss and gradients carry through at that level.
    variant = dict(variant)
    positions = variant.pop("positions", False)
    cfg = dataclasses.replace(_SMALL, **variant)
    position_ids = None
    if positions:
        position_ids = jax.random.randint(jax.random.key(2), (4, cfg.max_seq_len), 0, 100)
    base_loss, base_grads, _ = _loss_and_grads(cfg, position_ids=position_ids)
    loss, grads, _ = _loss_and_grads(dataclasses.replace(cfg, qk_norm_rope=impl), position_ids=position_ids)
    np.testing.assert_allclose(loss, base_loss, rtol=1e-5)
    _assert_grads_close(grads, base_grads, rtol=1e-4)


def _conditionals(hlo: str) -> int:
    return len(re.findall(r" conditional\(", hlo))


@pytest.mark.parametrize("num_layers", [6, 8, 3])
def test_static_attention_schedule_matches_the_per_layer_cond(num_layers):
    # 6 layers: one scanned group of four and two after it (short, long); 8: two groups and none after; 3: no group.
    # The same layers in the same order: equal loss and gradients up to XLA fusing the unrolled group differently.
    cfg = dataclasses.replace(_SMALL, num_layers=num_layers)
    base_loss, base_grads, base_hlo = _loss_and_grads(cfg)
    loss, grads, hlo = _loss_and_grads(dataclasses.replace(cfg, attention_schedule="static"))
    assert _conditionals(base_hlo) > 0
    assert _conditionals(hlo) == 0
    np.testing.assert_allclose(loss, base_loss, rtol=1e-6)
    _assert_grads_close(grads, base_grads, rtol=1e-5)


def test_all_short_schedule_differs_unless_the_window_covers_the_sequence():
    base_loss, _, _ = _loss_and_grads(_SMALL)
    loss, _, hlo = _loss_and_grads(dataclasses.replace(_SMALL, attention_schedule="all_short"))
    assert _conditionals(hlo) == 0
    assert abs(loss - base_loss) > 1e-6
    whole = dataclasses.replace(_SMALL, sliding_window=2 * _SMALL.max_seq_len)
    base_loss, base_grads, _ = _loss_and_grads(whole)
    for schedule in ("static", "all_short"):
        loss, grads, _ = _loss_and_grads(dataclasses.replace(whole, attention_schedule=schedule))
        np.testing.assert_allclose(loss, base_loss, rtol=1e-6, err_msg=schedule)
        _assert_grads_close(grads, base_grads, rtol=1e-5)


@pytest.mark.timeout(900)
def test_static_schedule_matches_the_cond_in_the_zero1_train_step():
    # The production step: ZeRO-1, the expert weights gathered layer by layer in the first microbatch (into carried
    # buffers indexed by layer), the overlapped last microbatch, after-scan reductions of the replicated gradients
    # (their per-layer offsets and sinks are scanned stacks too), the fused q/k norm, bf16 compute, four
    # microbatches. Six layers: a scanned group and two after it. One step matches the cond's to float32 rounding.
    body = """
        cfg = dataclasses.replace(ZERO1_CFG, num_layers=6, head_dim=64, qk_norm_rope="pallas")
        mesh = MESHES["replica4_expert2"]
        optimizer = OPTIMIZERS["adamw"]()
        runs = {}
        for schedule in ("cond", "static"):
            c = dataclasses.replace(cfg, attention_schedule=schedule)
            state = zero1_state(mesh, optimizer, zero1=True, cfg=c)
            train_step = zero1_step(optimizer, zero1=True, overlap=True, gather=True)
            with jax.set_mesh(mesh):
                hlo = train_step.lower(state, BATCH).compile().as_text()
                state, metrics, _ = train_step(state, BATCH)
            runs[schedule] = (float(metrics["train/loss"]), jax.tree.map(np.asarray, state.params), hlo)
        assert " conditional(" in runs["cond"][2]
        assert " conditional(" not in runs["static"][2]
        np.testing.assert_allclose(runs["static"][0], runs["cond"][0], rtol=1e-6)
        errors = relative_errors(runs["static"][1], runs["cond"][1])
        assert max(errors.values()) <= 1e-5, errors
    """
    run_on_cpu_devices(_ZERO1_PRELUDE + textwrap.dedent(body), device_count=8)
