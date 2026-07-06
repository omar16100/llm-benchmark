# Local 1M-Context LLM Findings for M3 Ultra Mac Studio

Date: 2026-07-04

## Hardware Baseline

Test machine:

- Mac Studio
- Apple M3 Ultra
- 32-core CPU
- 80-core GPU
- 512 GB unified memory
- 819 GB/s unified memory bandwidth

This is enough memory to load many very large quantized models, but the target is stricter: **100 generated tokens/sec while using a native 1M-token context**.

## Bottom Line

I did not find a model I would confidently claim can run locally on this machine at both:

1. **100 tokens/sec decode speed**, and
2. **native 1M-token context**, not merely RoPE/YaRN-extended context.

The best practical candidate is **NVIDIA Nemotron 3 Nano 30B-A3B** because it combines 1M context support with a small active-parameter count and a hybrid Mamba/attention/MoE architecture. It is the only candidate that looks plausibly close to the speed target, but it still needs a local benchmark.

> **Update 2026-07-06:** The native-1M half of the target is now demonstrated. `Kimi-Linear-48B-A3B-Instruct` (MLX 8-bit) completed a full needle-in-haystack run at a genuine **1,048,692-token** prompt with **8/8** recall and a truncation guard proving full ingestion, on this machine. It works because its long-context mechanism is **linear attention (KDA) + a few full-attention MLA layers**, both implemented in mlx-lm, rather than the sparse-attention paths that remain absent on Apple Silicon. It clears the native-1M bar but **not** the 100 tok/s bar: decode is 5.1 tok/s at 1M and prefill takes ~5.0 h (58 tok/s), peak 219.9 GB. Full method, ladder, and raw data: [06072026_kimi_linear_1m_verification.md](06072026_kimi_linear_1m_verification.md).

## Local Benchmark Attempt

Attempt date: 2026-07-04

Runner selected for the first pass: `llama.cpp llama-server` version `9730 (e475fa2b5)`, because the MLX runtime in this shell crashed while enumerating the Metal device before model load. The attempted llama.cpp configuration was:

- `-c 1000000`
- `-np 1`
- `-ngl all`
- `-fa on`
- `--no-webui`

No measured speed or recall numbers were produced for the three target models in this run. The model weights were not already present under the local Hugging Face, llama.cpp, LM Studio, Ollama, or `~/models` caches, and shell network/DNS access could not resolve `huggingface.co`.

| Model | Target local artifact | Local weights found | Load/download result | 100-token decode | 1M-context result |
|---|---|---:|---|---:|---:|
| **NVIDIA Nemotron 3 Nano 30B-A3B** | `unsloth/Nemotron-3-Nano-30B-A3B-GGUF:UD-Q4_K_XL` | No | Failed before load: Hugging Face repo commit could not be resolved | Not measured | Not measured |
| **Qwen2.5-7B-Instruct-1M** | `bartowski/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M` | No | Failed before load: Hugging Face repo commit could not be resolved | Not measured | Not measured |
| **Qwen2.5-14B-Instruct-1M** | `bartowski/Qwen2.5-14B-Instruct-1M-GGUF:Q4_K_M` | No | Failed before load: Hugging Face repo commit could not be resolved | Not measured | Not measured |

Key failure evidence:

- `curl -I --max-time 15 https://huggingface.co` returned `Could not resolve host: huggingface.co`.
- `llama-server -hf ...` returned `HTTPLIB failed: Could not establish connection` and `failed to download model from Hugging Face` for all three target repos.
- The recorded attempt rows are in `results/1m_candidates/2026-07-04_attempts.csv`.

### Nemotron Local Smoke Result

After `Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf` was downloaded locally, a short 100-token decode smoke test was run from:

`/Users/macmini/models/1m-candidates/nemotron-3-nano-30b-a3b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf`

The intended Metal run failed before inference:

- `ggml_metal_init: error: failed to create command queue`
- `ggml_backend_metal_device_init_backend: error: failed to allocate context`

This happened with both `-c 1000000` and a 4K control run, so it appears to be a shell/Metal-access issue rather than a 1M-context allocation issue.

CPU-only fallback result with `-dev none -ngl 0 -c 1000000 -n 100`:

| Model | Mode | Context allocation | Filled context | Prompt tok/s | Decode tok/s | Verdict |
|---|---|---:|---:|---:|---:|---|
| **NVIDIA Nemotron 3 Nano 30B-A3B UD-Q4_K_XL** | CPU-only fallback | 1,000,000 | Short prompt only | 44.3 | 46.6 | Below 100 tok/s; not representative of Metal/GPU throughput |

The recorded smoke row is in `results/1m_candidates/2026-07-04_nemotron_smoke.csv`.

### Nemotron MLX 8-bit Setup Attempt

The MLX 8-bit artifact was downloaded to:

`/Users/macmini/models/1m-candidates/nemotron-3-nano-30b-a3b-mlx-8bit`

Local artifact status:

- Disk size: 31 GiB
- Free disk after download: 124 GiB
- Files present: `config.json`, tokenizer files, and seven `model-0000x-of-00007.safetensors` shards

The attempted MLX smoke command was:

```bash
/Users/macmini/projects/llm-benchmark/.venv/bin/mlx_lm.generate \
  --model /Users/macmini/models/1m-candidates/nemotron-3-nano-30b-a3b-mlx-8bit \
  --prompt 'Write a concise explanation of why long-context inference can be slower than short-context inference. Keep the answer factual.' \
  --max-tokens 100
```

Result: no tokens/sec measured. MLX failed before model load while enumerating the Metal device:

`NSRangeException: index 0 beyond bounds for empty array`

The recorded setup row is in `results/1m_candidates/2026-07-04_nemotron_mlx8_smoke.csv`.

### Nemotron MLX 8-bit Token Test

On 2026-07-05, `mlx_lm.benchmark` successfully reached the Apple M3 Ultra GPU for one run:

- `default_device: Device(gpu, 0)`
- `metal_available: True`
- device: `Apple M3 Ultra`
- model artifact: `/Users/macmini/models/1m-candidates/nemotron-3-nano-30b-a3b-mlx-8bit`

Short-context profile, after warmup:

| Prompt tokens | Generated tokens | Trials | Avg prompt tok/s | Avg generation tok/s | Peak memory |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 100 | 3 | 2,514.455 | 109.964 | 37.062 GB |

100K-context profile, after warmup:

| Prompt tokens | Generated tokens | Trials | Prompt tok/s | Generation tok/s | Peak memory | Total time |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 100 | 1 | 1,611.916 | 85.684 | 37.665 GB | 63.257 s |

True 1M filled-context profile:

| Prompt tokens | Generated tokens | Result |
|---:|---:|---|
| 1,000,000 | 100 | No throughput line after roughly 35 minutes; interrupted manually |

Interpretation: **Nemotron MLX 8-bit clears 100 generated tok/s at 4K context**, but drops to **85.684 generated tok/s after a 100K-token prompt**. This run therefore does **not** meet the 100 tok/s target at 100K filled context, and it did **not** prove 100 tok/s after a filled 1M-token prompt. The 1M prefill/generation path was too slow to produce a completed result in the observed window.

The recorded token-test rows are in:

- `results/1m_candidates/2026-07-05_nemotron_mlx8_token_test.csv`
- `results/1m_candidates/nemotron_mlx8_token_test_20260705_172459.csv`

## Candidate Models

| Model | Architecture | Params | Official Context | Rough Weight Memory | Local Verdict |
|---|---|---:|---:|---:|---|
| **NVIDIA Nemotron 3 Nano 30B-A3B** | Hybrid Mamba-2 + Attention + MoE | 30B total, ~3.5B active | 1M | BF16 ~60 GB, FP8/Q8 ~30 GB, Q4 ~15-22 GB | Best shot for speed + 1M. Needs local benchmark. |
| **Qwen2.5-7B-Instruct-1M** | Dense Transformer | 7.61B total, 6.53B non-embedding | 1,010,000 tokens | BF16 ~15 GB, Q8 ~8 GB, Q4 ~4-6 GB | Fits easily, but official full-1M deployment expects 120 GB VRAM and custom vLLM. 100 tok/s at full 1M context is unlikely. |
| **Qwen2.5-14B-Instruct-1M** | Dense Transformer | 14.7B total, 13.1B non-embedding | 1,010,000 tokens | BF16 ~29 GB, Q8 ~15 GB, Q4 ~8-11 GB | Better quality than 7B, but official full-1M deployment expects 320 GB VRAM. Not a likely 100 tok/s full-context option. |
| **Llama 4 Scout** | MoE Transformer, multimodal | 109B total, 17B active, 16 experts | 10M | BF16 ~218 GB, Q8 ~109 GB, Q4 ~55-75 GB | Fits quantized. Native huge context, but likely far below 100 tok/s locally. |
| **DeepSeek V4 Flash** | MoE with long-context optimized attention | 284B total, 13B active | 1M | BF16 ~568 GB, Q8 ~284 GB, Q4 ~142-190 GB | May fit in aggressive quantization, but not a likely 100 tok/s local model. |
| **DeepSeek V4 Pro** | MoE with compressed/sparse attention | 1.6T total, 49B active | 1M | BF16 ~3.2 TB, Q8 ~1.6 TB, Q4 ~800 GB+ | Not viable on this 512 GB Mac. |
| **GLM-5.2** | Large MoE with long-context optimizations | ~743B total, ~39B active | 1M | BF16 ~1.5 TB, FP8 ~743 GB, Q4 ~370 GB+ | Edge-case fit only with special quantization and little headroom. Too large/slow for the target. |
| **Qwen3-Coder / similar Qwen 3 long-context models** | Dense/MoE depending variant | varies | Native 256K, extensible to 1M via YaRN | varies | Useful local coding models, but not strict native 1M. |

## Why 1M Context Changes the Answer

Short-context speed and 1M-context speed are different workloads.

At short context, a quantized 7B-14B model can plausibly exceed 100 tok/s on this machine with a good MLX or Metal backend. At 1M context, the bottleneck shifts:

- Dense Transformer attention has a large KV-cache and attention cost.
- Prefilling 1M tokens can take a long time even if decoding is later acceptable.
- Some model cards require custom sparse/chunked inference kernels for accuracy and memory.
- CUDA/vLLM guidance does not translate directly to Apple MLX/Metal.

That is why a small model like Qwen2.5-7B-1M can fit in RAM but still not be a clear 100 tok/s answer at full context.

## Recommendation

1. **Benchmark Nemotron 3 Nano 30B-A3B first.**
   - It has the best combination of official 1M support, small active parameter count, and long-context-friendly architecture.
   - Try MLX if an optimized conversion exists; otherwise try llama.cpp/GGUF if a stable quantization supports the architecture correctly.

2. **Use Qwen2.5-7B-Instruct-1M or Qwen2.5-14B-Instruct-1M as the strict dense baselines.**
   - They are officially 1M-context models.
   - Expect much lower speed at full context than at short context.

3. **Use Llama 4 Scout only if native very-large context matters more than speed.**
   - It supports 10M context and should fit quantized, but 100 tok/s locally is not a realistic expectation.

4. **Do not treat YaRN/RoPE extension as equivalent to native 1M.**
   - Extended-context models can be useful, but they do not satisfy the strict requirement unless the model was trained/evaluated for that range and the runtime supports it well.

## Practical Benchmark Plan

Measure these separately:

1. **Load memory**
   - Resident memory after model load.
   - Maximum context/KV cache allocation.

2. **Short-context decode**
   - 4K prompt, 512 generated tokens.
   - This answers the normal "tokens/sec" question.

3. **Long-context prefill**
   - 128K, 256K, 512K, and 1M prompt sizes.
   - Record time-to-first-token.

4. **Long-context decode**
   - Decode 512 tokens after each long prompt size.
   - This is the real target metric.

5. **Recall/accuracy**
   - Include needle-in-haystack or RULER-style checks.
   - A model that accepts 1M tokens but fails retrieval is not useful.

## Source Links

- [Apple Mac Studio technical specifications](https://www.apple.com/mac-studio/specs/)
- [Apple Support: Mac Studio 2025 technical specifications](https://support.apple.com/en-us/122211)
- [NVIDIA Nemotron 3 Nano 30B-A3B model page](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b)
- [NVIDIA Nemotron 3 Nano 30B-A3B Hugging Face model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [NVIDIA Nemotron 3 research page](https://research.nvidia.com/labs/nemotron/Nemotron-3/)
- [Qwen2.5-1M official blog](https://qwenlm.github.io/blog/qwen2.5-1m/)
- [Qwen2.5-7B-Instruct-1M model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)
- [Qwen2.5-14B-Instruct-1M model card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)
- [Meta Llama 4 announcement](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Meta Llama 4 model cards and prompt formats](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama4/)
- [DeepSeek V4 Pro Hugging Face model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek V4 preview release](https://api-docs.deepseek.com/news/news260424)
- [GLM-5.2 Hugging Face model card](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5.2 vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.2)
