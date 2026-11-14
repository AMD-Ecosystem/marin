# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""``gpu_aiter_flash`` against a fake of jax-aiter's ``flash_attn_func``.

The fake computes attention from FlashAttention/aiter's documented window semantics, so a wrong window, scale or head
layout changes the output, and it records what Grug passes. No real kernel runs here.
"""

import math
import sys
import types

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


def _library_attention(q, k, v, *, scale, causal, window):
    """Attention as FlashAttention/aiter define it, in f32 on BSHD inputs with GQA K/V heads: query ``i`` sees keys
    ``[i - left, i + right]`` for ``window=(left, right)``, ``-1`` meaning unbounded, and causal caps keys at ``i``."""
    group = q.shape[2] // k.shape[2]
    q32, k32, v32 = (x.astype(jnp.float32) for x in (q, k, v))
    k32, v32 = (jnp.repeat(x, group, axis=2) for x in (k32, v32))
    query_pos = jnp.arange(q.shape[1])[:, None]
    key_pos = jnp.arange(k.shape[1])[None, :]
    left, right = window
    allowed = jnp.ones((q.shape[1], k.shape[1]), dtype=bool)
    if left >= 0:
        allowed &= key_pos >= query_pos - left
    if right >= 0:
        allowed &= key_pos <= query_pos + right
    if causal:
        allowed &= key_pos <= query_pos
    logits = jnp.einsum("bqhd,bkhd->bhqk", q32, k32, precision=jax.lax.Precision.HIGHEST) * scale
    probs = jax.nn.softmax(jnp.where(allowed, logits, -jnp.inf), axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, v32, precision=jax.lax.Precision.HIGHEST).astype(q.dtype)


@pytest.fixture
def fake_jax_aiter(monkeypatch):
    """A fake ``jax_aiter.mha`` whose ``flash_attn_func`` records its arguments and returns a tuple, as jax-aiter's."""
    record = types.SimpleNamespace(calls=[])

    def flash_attn_func(
        q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), deterministic=True, **_
    ):
        record.calls.append(
            dict(
                q_shape=q.shape,
                kv_shape=k.shape,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        return (_library_attention(q, k, v, scale=softmax_scale, causal=causal, window=window_size),)

    mha = types.ModuleType("jax_aiter.mha")
    mha.flash_attn_func = flash_attn_func
    package = types.ModuleType("jax_aiter")
    package.mha = mha
    monkeypatch.setitem(sys.modules, "jax_aiter", package)
    monkeypatch.setitem(sys.modules, "jax_aiter.mha", mha)
    return record


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


def test_gpu_aiter_flash_gradients_match_reference(fake_jax_aiter):
    mask = AttentionMask.causal(sliding_window=WINDOW)
    q, k, v = _qkv(8, 2)
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32) * cotangent)

    def flash(q, k, v):
        return attention(q, k, v, mask, implementation="gpu_aiter_flash")

    actual = jax.jit(jax.grad(lambda *a: loss(flash, *a), argnums=(0, 1, 2)))(q, k, v)
    expected = jax.grad(lambda *a: loss(lambda *x: _reference(*x, mask), *a), argnums=(0, 1, 2))(q, k, v)
    for name, want, got in zip("qkv", expected, actual, strict=True):
        want = np.asarray(want, np.float32)
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(
            np.asarray(got, np.float32) / scale, want / scale, atol=TOLERANCE, rtol=TOLERANCE, err_msg=f"d{name}"
        )


def test_gpu_aiter_flash_runs_the_kernel_on_local_batch_and_heads_under_a_mesh(fake_jax_aiter):
    mesh = AbstractMesh(axis_sizes=(2, 2), axis_names=("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    sharding = NamedSharding(mesh, P("data", None, "model", None))
    q = jax.ShapeDtypeStruct((4, SEQ_LEN, 8, HEAD_DIM), jnp.bfloat16, sharding=sharding)
    kv = jax.ShapeDtypeStruct((4, SEQ_LEN, 2, HEAD_DIM), jnp.bfloat16, sharding=sharding)
    mask = AttentionMask.causal(sliding_window=WINDOW)

    def attend(q, k, v):
        return attention(q, k, v, mask, implementation="gpu_aiter_flash")

    def loss(q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32))

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(attend, q, kv, kv)
        grads = jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv)

    assert output.sharding == sharding
    assert [g.sharding for g in grads] == [sharding] * 3
    assert {(c["q_shape"], c["kv_shape"]) for c in fake_jax_aiter.calls} == {
        ((2, SEQ_LEN, 4, HEAD_DIM), (2, SEQ_LEN, 1, HEAD_DIM))
    }


def test_gpu_aiter_flash_rejects_segment_ids_instead_of_ignoring_them(fake_jax_aiter):
    q, k, v = _qkv(4, 2)
    segment_ids = jnp.repeat(jnp.arange(4, dtype=jnp.int32), SEQ_LEN // 4)
    mask = AttentionMask.causal().with_segment_ids(segment_ids)

    with pytest.raises(NotImplementedError, match="segment ids"):
        attention(q, k, v, mask, implementation="gpu_aiter_flash")


def test_gpu_aiter_flash_names_its_missing_package(monkeypatch):
    for name in ("jax_aiter", "jax_aiter.mha"):
        monkeypatch.setitem(sys.modules, name, None)
    q, k, v = _qkv(4, 2)

    with pytest.raises(ImportError, match="gpu_aiter_flash needs .*jax_aiter"):
        attention(q, k, v, AttentionMask.causal(), implementation="gpu_aiter_flash")
