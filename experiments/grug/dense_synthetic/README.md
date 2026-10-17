# grug-dense-synthetic

Single-node harness that trains a dense transformer on uniform-random tokens through Levanter's `Trainer` and reports step time, tokens/s, MFU, and peak device memory. It is the dense counterpart of [`snowball_synthetic`](../snowball_synthetic/README.md): the same trainer setup and summary lines, so dense GEMM and attention throughput can be measured, and compared with MaxText, on the same footing as the MoE.

[`model.py`](./model.py) reuses the June 67B Grug block machinery (`experiments/june_tpu_67b_a2b/moe/model.py`): the layers stacked into one `jax.lax.scan`, a per-block remat policy, the Grug attention dispatch (`--attention`), FSDP over `data` and tensor parallelism over `model`, and June's `DenseMLP` (SwiGLU) as the MLP. Attention is grouped-query with full RoPE and, where the model has it, a per-head RMSNorm on q and k (QK-norm). Embeddings are untied.

| Preset | Model | Layers | Hidden | MLP | Q / KV heads | head_dim | QK-norm | Vocab | Seq |
|---|---|---|---|---|---|---|---|---|---|
| `tiny` | CPU check | 4 | 64 | 128 | 4 / 2 | 16 | yes | 128 | 32 |
| `8b` | Llama-3 8B | 32 | 4096 | 14336 | 32 / 8 | 128 | no | 128256 | 4096 |
| `32b` | Marin 32B (Qwen3 layout) | 64 | 5120 | 27648 | 40 / 8 | 128 | yes | 128256 | 4096 |

`--layers` gives shallower proxies of the same width, for example `--size 32b --layers 16` to time per-layer kernels and FSDP traffic; extrapolate those to the full depth instead of quoting their MFU. `--remat-mode` is `recompute_all` (rerun each block in backward; the default), `save_dots` (keep the matmul outputs) or `none`. `--model-axis` must divide the KV-head count. Mixed precision defaults to `p=f32,c=bfloat16`.

`--attention` takes any Grug attention implementation. On AMD GPUs the fused choices are:

| `--attention` | Kernel | Needs | `8b` (32 / 8) | `32b` (40 / 8) |
|---|---|---|---|---|
| `gpu_pallas_triton_flash` | Grug's Pallas-Triton flash | nothing extra | yes | no: needs a power-of-two GQA ratio |
| `gpu_aiter_flash` | jax-aiter `flash_attn_func` (AITER asm v3 / CK) | `jax_aiter` with its MHA libraries | yes | yes |
| `gpu_te_flash` | TransformerEngine `fused_attn` (AITER / CK) | ROCm TE's JAX extension, refstack `amd-jax-0.11.0-te` | yes | yes |
| `xla_flash` | Levanter's pure-JAX flash attention | nothing extra | yes | yes |

For `32b`, use `gpu_aiter_flash` or `gpu_te_flash`; `xla_flash` is the fallback when neither package is installed. Both take the 40 query and 8 KV heads as they are, run per device under the mesh, and use a nondeterministic backward (`gpu_aiter_flash` passes `deterministic=False`; `gpu_te_flash` follows `NVTE_ALLOW_NONDETERMINISTIC_ALGO`, which the refstack sets to 1). Neither supports segment ids, which this harness does not use. The module docstrings of `levanter/grug/attention/_aiter_flash.py` and `_te_flash.py` describe the argument mapping.

FLOPs per example are Levanter's count for its Llama and Qwen3 models, so MFU matches theirs. The harness also logs a `model flops per example: total=... attention_noncausal=... attention_causal=...` line: the attention-map part of that total as Levanter counts it (non-causal) and with causal masking counted exactly, for a causal MFU.

The default batch is one sequence per device; a global batch of 1 fails in the next-token loss under explicit mesh axes, so single-device runs need `--batch-size 2`. Every step sees fresh random tokens, so the loss cannot fall below ln(vocab_size) and a lower value means something leaks targets. No checkpoint is written.

```bash
python experiments/grug/dense_synthetic/train.py \
    --size 32b --layers 16 --attention xla_flash --steps 20 --profile-steps 3 --compilation-cache-dir "$WORK/jax-cache"
```

The profile lands under `<log-dir>/<run-id>/profiler/`; summarize it with `lib/marin/tools/profile_summary.py summarize --xplane-file <trace>.xplane.pb`.
