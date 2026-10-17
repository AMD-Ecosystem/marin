# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""``gpu_te_flash`` and ``gpu_aiter_flash`` against fakes of TransformerEngine and jax-aiter (see ``conftest.py``).

The fakes compute attention from each library's documented window and mask semantics, so a wrong window, scale or
head layout changes the output. They record what Grug passes, the translation into each library's API. No real
kernel runs here.
"""

import functools
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, NamedSharding, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import AttentionMask, attention, reference_attention

SEQ_LEN = 64
HEAD_DIM = 16
WINDOW = 24
# bf16 outputs of the same f32 math, as in the Pallas-Triton flash tests.
TOLERANCE = 1e-2

# name -> (query heads, KV heads, mask, (left, right) window the kernel should receive, or None)
CASES = {
    "causal": (4, 4, AttentionMask.causal(), None),
    "causal_gqa_ratio_5": (10, 2, AttentionMask.causal(), None),
    "window_gqa": (8, 2, AttentionMask.causal(sliding_window=WINDOW), (WINDOW - 1, 0)),
    "window_covering_sequence": (4, 2, AttentionMask.causal(sliding_window=SEQ_LEN), None),
}


def _qkv(num_q_heads, num_kv_heads, *, batch=2):
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (batch, SEQ_LEN, num_q_heads, HEAD_DIM), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (batch, SEQ_LEN, num_kv_heads, HEAD_DIM), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (batch, SEQ_LEN, num_kv_heads, HEAD_DIM), jnp.float32).astype(jnp.bfloat16)
    return q, k, v


def _reference(q, k, v, mask):
    with jax.default_matmul_precision("highest"):
        return reference_attention(*(x.astype(jnp.float32) for x in (q, k, v)), mask, logits_dtype=jnp.float32)


@pytest.mark.parametrize("case", list(CASES))
def test_gpu_te_flash_passes_bshd_gqa_and_window_to_fused_attn_and_matches_reference(fake_transformer_engine, case):
    num_q_heads, num_kv_heads, mask, window = CASES[case]
    record = fake_transformer_engine()
    q, k, v = _qkv(num_q_heads, num_kv_heads)

    out = attention(q, k, v, mask, implementation="gpu_te_flash")

    np.testing.assert_allclose(np.asarray(out, np.float32), _reference(q, k, v, mask), atol=TOLERANCE, rtol=TOLERANCE)
    [call] = record.calls
    te = sys.modules["transformer_engine.jax.attention"]
    assert call["qkv_layout"] == te.QKVLayout.BSHD_BSHD_BSHD
    assert call["q_shape"] == q.shape and call["kv_shape"] == k.shape
    assert call["attn_mask_type"] == te.AttnMaskType.CAUSAL_MASK
    assert call["window_size"] == window
    assert call["softmax_type"] == te.AttnSoftmaxType.VANILLA_SOFTMAX
    assert call["scaling_factor"] == pytest.approx(1 / math.sqrt(HEAD_DIM))
    assert (call["bias"], call["seed"], call["dropout_probability"]) == (None, None, 0.0)
    for seqlens in call["seqlens"]:
        np.testing.assert_array_equal(seqlens, [SEQ_LEN, SEQ_LEN])
    [check] = record.checks
    assert (check["attn_mask_type"], check["window_size"]) == (te.AttnMaskType.CAUSAL_MASK, window)
    assert (check["q_num_heads"], check["kv_num_heads"]) == (num_q_heads, num_kv_heads)


@pytest.mark.parametrize("case", list(CASES))
def test_gpu_aiter_flash_passes_window_and_nondeterministic_backward_and_matches_reference(fake_jax_aiter, case):
    num_q_heads, num_kv_heads, mask, window = CASES[case]
    q, k, v = _qkv(num_q_heads, num_kv_heads)

    out = attention(q, k, v, mask, implementation="gpu_aiter_flash")

    np.testing.assert_allclose(np.asarray(out, np.float32), _reference(q, k, v, mask), atol=TOLERANCE, rtol=TOLERANCE)
    [call] = fake_jax_aiter.calls
    assert call["q_shape"] == q.shape and call["kv_shape"] == k.shape
    assert call["causal"] is True
    # (-1, 0) would also mean "causal" to AITER but sends jax-aiter's backward to CK.
    assert call["window_size"] == (window or (-1, -1))
    assert call["deterministic"] is False
    assert call["dropout_p"] == 0.0
    assert call["softmax_scale"] == pytest.approx(1 / math.sqrt(HEAD_DIM))


@pytest.fixture
def fake_backend(request, fake_transformer_engine, fake_jax_aiter):
    """The fake library behind ``request.param``, installed."""
    if request.param == "gpu_te_flash":
        fake_transformer_engine()
    return request.param


@pytest.mark.parametrize("fake_backend", ["gpu_te_flash", "gpu_aiter_flash"], indirect=True)
def test_gpu_flash_backend_gradients_match_reference(fake_backend):
    mask = AttentionMask.causal(sliding_window=WINDOW)
    q, k, v = _qkv(8, 2)
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32) * cotangent)

    def flash(q, k, v):
        return attention(q, k, v, mask, implementation=fake_backend)

    # The TE fake asserts, like TE, that a MeshResource is set while its backward is traced.
    actual = jax.jit(jax.grad(lambda *a: loss(flash, *a), argnums=(0, 1, 2)))(q, k, v)
    expected = jax.grad(lambda *a: loss(lambda *x: _reference(*x, mask), *a), argnums=(0, 1, 2))(q, k, v)
    for name, want, got in zip("qkv", expected, actual, strict=True):
        want = np.asarray(want, np.float32)
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(
            np.asarray(got, np.float32) / scale, want / scale, atol=TOLERANCE, rtol=TOLERANCE, err_msg=f"d{name}"
        )


@pytest.mark.parametrize("fake_backend", ["gpu_te_flash", "gpu_aiter_flash"], indirect=True)
def test_gpu_flash_backend_gradients_match_reference_under_scan_remat_and_cond(fake_backend):
    """The June block's control flow: a scan over rematerialized layers, each picking its mask with ``lax.cond``."""
    q, k, v = _qkv(8, 2)
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)
    use_full_mask = jnp.array([False, True])

    def loss(implementation, q, k, v):
        def attend(mask):
            return lambda x: attention(x, k, v, mask, implementation=implementation)

        @jax.checkpoint
        def layer(x, use_full):
            full, windowed = attend(AttentionMask.causal()), attend(AttentionMask.causal(sliding_window=WINDOW))
            return jax.lax.cond(use_full, full, windowed, x)

        out, _ = jax.lax.scan(lambda x, use_full: (layer(x, use_full), None), q, use_full_mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent)

    # The TE fake fails unless a MeshResource is set wherever these transforms trace fused_attn forward or backward.
    actual = jax.jit(jax.grad(functools.partial(loss, fake_backend), argnums=(0, 1, 2)))(q, k, v)
    with jax.default_matmul_precision("highest"):
        expected = jax.jit(jax.grad(functools.partial(loss, "reference"), argnums=(0, 1, 2)))(q, k, v)
    for name, want, got in zip("qkv", expected, actual, strict=True):
        want = np.asarray(want, np.float32)
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(
            np.asarray(got, np.float32) / scale, want / scale, atol=TOLERANCE, rtol=TOLERANCE, err_msg=f"d{name}"
        )


@pytest.mark.parametrize("implementation", ["gpu_te_flash", "gpu_aiter_flash"])
def test_gpu_flash_backend_runs_the_kernel_on_local_batch_and_heads_under_a_mesh(
    implementation, fake_transformer_engine, fake_jax_aiter
):
    te_record = fake_transformer_engine()
    record = te_record if implementation == "gpu_te_flash" else fake_jax_aiter
    mesh = AbstractMesh(axis_sizes=(2, 2), axis_names=("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    sharding = NamedSharding(mesh, P("data", None, "model", None))
    q = jax.ShapeDtypeStruct((4, SEQ_LEN, 8, HEAD_DIM), jnp.bfloat16, sharding=sharding)
    kv = jax.ShapeDtypeStruct((4, SEQ_LEN, 2, HEAD_DIM), jnp.bfloat16, sharding=sharding)
    mask = AttentionMask.causal(sliding_window=WINDOW)

    def attend(q, k, v):
        return attention(q, k, v, mask, implementation=implementation)

    def loss(q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32))

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(attend, q, kv, kv)
        grads = jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv)

    assert output.sharding == sharding
    assert [g.sharding for g in grads] == [sharding] * 3
    assert {(c["q_shape"], c["kv_shape"]) for c in record.calls} == {
        ((2, SEQ_LEN, 4, HEAD_DIM), (2, SEQ_LEN, 1, HEAD_DIM))
    }


def test_gpu_te_flash_raises_when_te_has_no_kernel(fake_transformer_engine):
    fake_transformer_engine(kernel_available=False)
    q, k, v = _qkv(4, 2)

    with pytest.raises(NotImplementedError, match="no fused attention kernel"):
        attention(q, k, v, AttentionMask.causal(), implementation="gpu_te_flash")


@pytest.mark.parametrize("fake_backend", ["gpu_te_flash", "gpu_aiter_flash"], indirect=True)
def test_gpu_flash_backend_rejects_segment_ids_instead_of_ignoring_them(fake_backend):
    q, k, v = _qkv(4, 2)
    segment_ids = jnp.repeat(jnp.arange(4, dtype=jnp.int32), SEQ_LEN // 4)
    mask = AttentionMask.causal().with_segment_ids(segment_ids)

    with pytest.raises(NotImplementedError, match="segment ids"):
        attention(q, k, v, mask, implementation=fake_backend)


@pytest.mark.parametrize(
    "implementation, package, modules",
    [
        ("gpu_te_flash", "transformer_engine", ["transformer_engine.jax", "transformer_engine.jax.attention"]),
        ("gpu_aiter_flash", "jax_aiter", ["jax_aiter.mha"]),
    ],
)
def test_gpu_flash_backend_names_its_missing_package(monkeypatch, implementation, package, modules):
    for name in [package, *modules]:
        monkeypatch.setitem(sys.modules, name, None)
    q, k, v = _qkv(4, 2)

    with pytest.raises(ImportError, match=f"{implementation} needs .*{package}"):
        attention(q, k, v, AttentionMask.causal(), implementation=implementation)
