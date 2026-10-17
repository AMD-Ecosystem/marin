# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import enum
import functools
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer as HfBaseTokenizer

from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import HfMarinTokenizer, load_tokenizer


@pytest.fixture(scope="session")
def local_gpt2_tokenizer(tmp_path_factory):
    """Load a GPT2 MarinTokenizer from a local directory to avoid network downloads."""
    source_dir = Path(__file__).parent
    output_dir = tmp_path_factory.mktemp("gpt2_tokenizer")
    tmpdir = stage_gpt2_tokenizer(source_dir, output_dir)
    return load_tokenizer(str(tmpdir))


@pytest.fixture(scope="session")
def local_gpt2_tokenizer_path(tmp_path_factory) -> str:
    """Path to a local GPT-2 tokenizer directory.

    Lets HF roundtrip tests build an ``HFCheckpointConverter`` with a local
    tokenizer (and no ``reference_checkpoint``) so they never reach the Hub.
    """
    source_dir = Path(__file__).parent
    output_dir = tmp_path_factory.mktemp("gpt2_tokenizer")
    return str(stage_gpt2_tokenizer(source_dir, output_dir))


@pytest.fixture(scope="session")
def local_gpt2_marin_tokenizer(tmp_path_factory) -> HfMarinTokenizer:
    """Load a GPT2 MarinTokenizer from a local JSON file to avoid network downloads."""

    test_dir = Path(__file__).parent
    tmpdir = tmp_path_factory.mktemp("gpt2_marin_tok")
    shutil.copy(test_dir / "gpt2_tokenizer.json", tmpdir / "tokenizer.json")
    shutil.copy(test_dir / "gpt2_tokenizer_config.json", tmpdir / "tokenizer_config.json")

    tok = HfBaseTokenizer.from_file(str(tmpdir / "tokenizer.json"))

    # GPT-2 uses <|endoftext|> as both BOS and EOS
    vocab = tok.get_vocab()
    eos_token = "<|endoftext|>"
    eos_id = vocab.get(eos_token)

    all_special_ids = [eos_id] if eos_id is not None else []

    return HfMarinTokenizer(
        _tokenizer=tok,
        _name_or_path=str(tmpdir),
        _bos_id=None,
        _eos_id=eos_id,
        _pad_id=None,
        _bos_token=None,
        _eos_token=eos_token,
        _chat_template=None,
        _vocab_size=tok.get_vocab_size(),
        _all_special_ids=all_special_ids,
        _id_to_token={v: k for k, v in vocab.items()},
        _vocab=vocab,
    )


@pytest.fixture(autouse=True)
def _configure_marin_prefix():
    """Set MARIN_PREFIX to a temp directory for tests that rely on it."""
    if "MARIN_PREFIX" in os.environ:
        yield
        return

    with tempfile.TemporaryDirectory(prefix="marin_prefix") as temp_dir:
        os.environ["MARIN_PREFIX"] = temp_dir
        yield
        del os.environ["MARIN_PREFIX"]


def _library_attention(q, k, v, *, scale, causal, window):
    """Attention as TransformerEngine and FlashAttention/aiter define it, in f32 on BSHD inputs with GQA K/V heads.

    Query ``i`` sees keys ``[i - left, i + right]`` for ``window=(left, right)``, ``-1`` meaning unbounded, and a
    causal mask also caps keys at ``i``. Written from the libraries' docs, not from Grug's or Levanter's mask code.
    """
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
def fake_transformer_engine(monkeypatch):
    """Factory installing a fake ``transformer_engine.jax`` (attention and sharding) in ``sys.modules``.

    ``fused_attn`` follows TE 2.17's signature (``softmax_type`` required) or, with ``takes_softmax_type=False``, the
    pre-2.10 one, records each call, and computes ``_library_attention``. Like TE 2.9 and later, it asserts that a
    ``MeshResource`` is set while its forward and its backward are traced. Returns the record of calls.
    """

    def install(*, takes_softmax_type: bool = True, kernel_available: bool = True):
        record = types.SimpleNamespace(calls=[], checks=[])
        sharding = types.ModuleType("transformer_engine.jax.sharding")
        sharding.active_resource = None

        @dataclasses.dataclass(frozen=True)
        class MeshResource:
            cp_resource: str | None = None

        @contextlib.contextmanager
        def global_shard_guard(resource):
            previous, sharding.active_resource = sharding.active_resource, resource
            try:
                yield
            finally:
                sharding.active_resource = previous

        sharding.MeshResource = MeshResource
        sharding.global_shard_guard = global_shard_guard

        def require_mesh_resource():
            assert sharding.active_resource is not None, "Global mesh resource is not set."

        @functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
        def kernel(q, k, v, scale, causal, window):
            require_mesh_resource()
            return _library_attention(q, k, v, scale=scale, causal=causal, window=window)

        def kernel_fwd(q, k, v, scale, causal, window):
            return kernel(q, k, v, scale, causal, window), (q, k, v)

        def kernel_bwd(scale, causal, window, residuals, d_out):
            require_mesh_resource()
            attend = functools.partial(_library_attention, scale=scale, causal=causal, window=window)
            return jax.vjp(attend, *residuals)[1](d_out)

        kernel.defvjp(kernel_fwd, kernel_bwd)

        attention = types.ModuleType("transformer_engine.jax.attention")
        attention.AttnBiasType = enum.Enum("AttnBiasType", ["NO_BIAS", "PRE_SCALE_BIAS", "POST_SCALE_BIAS"])
        attention.AttnMaskType = enum.Enum("AttnMaskType", ["NO_MASK", "PADDING_MASK", "CAUSAL_MASK"])
        attention.QKVLayout = enum.Enum("QKVLayout", ["BS3HD", "BSHD_BS2HD", "BSHD_BSHD_BSHD"])

        @dataclasses.dataclass
        class SequenceDescriptor:
            seqlens: tuple

            @classmethod
            def from_seqlens(cls, seqlens):
                return cls(seqlens)

        attention.SequenceDescriptor = SequenceDescriptor

        def run(
            qkv,
            bias,
            sequence_descriptor,
            seed,
            *,
            attn_bias_type,
            attn_mask_type,
            qkv_layout,
            softmax_type,
            scaling_factor,
            dropout_probability,
            is_training,
            window_size,
        ):
            q, k, v = qkv
            record.calls.append(
                dict(
                    q_shape=q.shape,
                    kv_shape=k.shape,
                    bias=bias,
                    seqlens=sequence_descriptor.seqlens,
                    seed=seed,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    qkv_layout=qkv_layout,
                    softmax_type=softmax_type,
                    scaling_factor=scaling_factor,
                    dropout_probability=dropout_probability,
                    is_training=is_training,
                    window_size=window_size,
                )
            )
            causal = attn_mask_type == attention.AttnMaskType.CAUSAL_MASK
            return kernel(q, k, v, scaling_factor, causal, window_size or (-1, -1))

        if takes_softmax_type:
            attention.AttnSoftmaxType = enum.Enum("AttnSoftmaxType", ["VANILLA_SOFTMAX", "OFF_BY_ONE_SOFTMAX"])

            def fused_attn(
                qkv,
                bias,
                sequence_descriptor,
                seed,
                attn_bias_type,
                attn_mask_type,
                qkv_layout,
                softmax_type,
                scaling_factor,
                dropout_probability,
                is_training,
                max_segments_per_seq=1,
                window_size=None,
            ):
                return run(
                    qkv,
                    bias,
                    sequence_descriptor,
                    seed,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    qkv_layout=qkv_layout,
                    softmax_type=softmax_type,
                    scaling_factor=scaling_factor,
                    dropout_probability=dropout_probability,
                    is_training=is_training,
                    window_size=window_size,
                )

            def is_fused_attn_kernel_available(
                is_training,
                q_dtype,
                kv_dtype,
                qkv_layout,
                attn_bias_type,
                attn_mask_type,
                softmax_type,
                dropout_probability,
                q_num_heads,
                kv_num_heads,
                q_max_seqlen,
                kv_max_seqlen,
                head_dim_qk,
                head_dim_v,
                window_size=None,
            ):
                record.checks.append(
                    dict(
                        attn_mask_type=attn_mask_type,
                        window_size=window_size,
                        softmax_type=softmax_type,
                        q_num_heads=q_num_heads,
                        kv_num_heads=kv_num_heads,
                    )
                )
                return kernel_available

        else:

            def fused_attn(
                qkv,
                bias,
                sequence_descriptor,
                seed,
                attn_bias_type,
                attn_mask_type,
                qkv_layout,
                scaling_factor,
                dropout_probability,
                is_training,
                max_segments_per_seq=1,
                window_size=None,
            ):
                return run(
                    qkv,
                    bias,
                    sequence_descriptor,
                    seed,
                    attn_bias_type=attn_bias_type,
                    attn_mask_type=attn_mask_type,
                    qkv_layout=qkv_layout,
                    softmax_type=None,
                    scaling_factor=scaling_factor,
                    dropout_probability=dropout_probability,
                    is_training=is_training,
                    window_size=window_size,
                )

            def is_fused_attn_kernel_available(
                is_training,
                q_dtype,
                kv_dtype,
                qkv_layout,
                attn_bias_type,
                attn_mask_type,
                dropout_probability,
                q_num_heads,
                kv_num_heads,
                q_max_seqlen,
                kv_max_seqlen,
                head_dim_qk,
                head_dim_v,
                window_size=None,
            ):
                record.checks.append(
                    dict(
                        attn_mask_type=attn_mask_type,
                        window_size=window_size,
                        softmax_type=None,
                        q_num_heads=q_num_heads,
                        kv_num_heads=kv_num_heads,
                    )
                )
                return kernel_available

        attention.fused_attn = fused_attn
        attention.is_fused_attn_kernel_available = is_fused_attn_kernel_available

        te_jax = types.ModuleType("transformer_engine.jax")
        te_jax.attention = attention
        te_jax.sharding = sharding
        package = types.ModuleType("transformer_engine")
        package.jax = te_jax
        for name, module in (
            ("transformer_engine", package),
            ("transformer_engine.jax", te_jax),
            ("transformer_engine.jax.attention", attention),
            ("transformer_engine.jax.sharding", sharding),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return record

    return install

