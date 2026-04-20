# GLM-5.1 UD-IQ3_XXS Benchmark Report

**Date:** 2026-04-08
**Machine:** Mac Studio M3 Ultra, 512 GB unified memory, 819.2 GB/s bandwidth
**Model:** GLM-5.1 (754B total params, 40B active per token, MoE architecture)
**Quantization:** UD-IQ3_XXS (268 GB, 7 GGUF shards)
**Source:** https://huggingface.co/unsloth/GLM-5.1-GGUF

---

## Architecture

| Param | Value |
|---|---|
| Total parameters | 754B |
| Active parameters/token | 40B |
| Architecture | glm_moe_dsa (MoE + Dynamic Sparse Attention) |
| Layers | 79 |
| Embedding dim | 6144 |
| Attention heads | 64 |
| KV heads | 1 (MLA - Multi-head Latent Attention) |
| Head dim (K) | 576 |
| Head dim (V) | 512 |
| Training context | 202,752 tokens |
| RoPE freq base | 1,000,000 |

## Server Configurations Tested

### 1. Baseline (TurboQuant fork, default flags)

```
Server: llama-cpp-turboquant (TheTom fork, feature/turboquant-kv-cache branch)
Build:  version 8807 (eea498c42)
Flags:  -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 32768
```

### 2. Optimized (TurboQuant fork, tuned flags)

```
Server: llama-cpp-turboquant (same fork)
Flags:  -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 8192 -b 2048 -ub 512 -t 16 -np 1
```

### 3. Mainline llama.cpp, 8K context

```
Server: llama.cpp mainline (latest as of 2026-04-08)
Flags:  -fa on -ngl 99 -c 8192 -b 2048 -ub 512 -t 16 -np 1
```

### 4. Mainline llama.cpp, 200K context

```
Server: llama.cpp mainline
Flags:  -fa on -ngl 99 -c 202752 -b 2048 -ub 512 -t 16 -np 1
```

### 5. Mainline llama.cpp, 1M context

```
Server: llama.cpp mainline
Flags:  -fa on -ngl 99 -c 1000000 -b 2048 -ub 512 -t 16 -np 1
```

## Benchmark Results

**Prompt:** "Write a detailed explanation of how transformers work in machine learning, including attention mechanisms, positional encoding, and the encoder-decoder architecture."
**Max tokens:** 512

| Config | Context | Prefill (tok/s) | Decode (tok/s) | KV Cache Size |
|---|---|---|---|---|
| 1. Baseline (TQ fork, defaults) | 32K | 0.83 | 4.62 | 2.7 GB |
| 2. Optimized (TQ fork, tuned) | 8K | 45.95 | 15.54 | 0.7 GB |
| 3. Mainline, 8K | 8K | 43.90 | 17.55 | 0.7 GB |
| 4. Mainline, 200K | 200K | 10.22 | 17.49 | 17.0 GB |
| 5. Mainline, 1M | 1M | 9.97 | 17.55 | 83.7 GB |

## Key Findings

### 1. Server flag tuning: 3.4x decode speedup (4.6 -> 15.5 tok/s)

The biggest gains came from:
- `-c 8192` instead of 32768 (reduced KV allocation overhead)
- `-np 1` instead of default 4 (single slot, less memory fragmentation)
- `-t 16` (explicit thread count matching M3 Ultra P-cores)
- `-b 2048 -ub 512` (explicit batch sizes)

### 2. Mainline vs TurboQuant fork: +13% decode speed

Mainline llama.cpp (17.55 tok/s) outperformed the TurboQuant fork (15.54 tok/s) by ~13%. The TurboQuant fork is based on an older llama.cpp revision and likely missing recent MoE Metal kernel optimizations.

### 3. Context length has zero decode impact

Decode speed remained constant at ~17.5 tok/s across 8K, 200K, and 1M context allocations. The MLA architecture (n_head_kv=1) makes KV cache extremely compact, so larger allocations don't impact generation bandwidth.

Prefill dropped from ~44 tok/s (8K) to ~10 tok/s (200K+) due to larger KV buffer initialization, but stabilized between 200K and 1M.

### 4. MLA makes 200K context trivially cheap

With n_head_kv=1, KV cache scales at only ~87 MB per 1K tokens. This means:
- 200K context = 17 GB KV cache
- 1M context = 84 GB KV cache
- 475 GB GPU memory minus 252 GB model = 223 GB headroom

### 5. 1M context exceeds training window

Model was trained on 202,752 tokens. Running at 1M context produces `n_ctx_seq > n_ctx_train` warning. No YaRN scaling is configured. Accuracy degrades beyond 200K positions due to RoPE extrapolation. Recommended max context: 202,752.

### 6. DSA indexer not implemented in llama.cpp

PR #19460 added GLM MoE DSA architecture support but the DSA indexer (which selects important KV positions for sparse attention) is not yet implemented. The indexer tensors are loaded into memory but unused. This means:
- Quality is "suboptimal" per the PR author
- A future PR implementing the indexer could improve both quality and prefill speed
- No GGUFs need reconversion when indexer support lands

## Memory Breakdown (Final 200K Config)

| Component | Size |
|---|---|
| Model weights (GPU) | 252.2 GB |
| KV cache (GPU, FP16) | 17.0 GB |
| Compute buffers | 0.4 GB |
| CPU buffers | 0.7 GB |
| **Total used** | **~270 GB** |
| **Free headroom** | **~205 GB** |

## Final Production Config

```bash
export GGML_METAL_RESIDENCY_KEEP_ALIVE_S=86400

/Users/macmini/projects/llama-cpp-mainline/build/bin/llama-server \
  -m /Users/macmini/models/GLM-5.1-UD-IQ3_XXS/UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf \
  -fa on -ngl 99 \
  -c 202752 \
  -b 2048 -ub 2048 \
  -t 8 \
  -np 1 \
  --mlock \
  --host 0.0.0.0 --port 1234
```

**Performance:** 17.3 tok/s decode, 46.3 tok/s prefill (short prompt), 200K native context

## TurboQuant KV Cache Notes

TurboQuant (turbo3) compresses KV cache ~5x (FP16 -> 3.25 bits). For GLM-5.1 on this machine:
- Not needed at 200K context (17 GB FP16 KV fits easily)
- Would be beneficial if serving multiple concurrent requests (multiple slots)
- TurboQuant fork is ~13% slower on decode due to older llama.cpp base
- Recommendation: use mainline until TurboQuant is merged upstream or fork is rebased

## Software Versions

| Component | Version |
|---|---|
| llama.cpp (mainline) | Latest as of 2026-04-08 |
| llama.cpp (TQ fork) | v8807 (eea498c42), branch feature/turboquant-kv-cache |
| macOS | Darwin 25.3.0 |
| Xcode | AppleClang 17.0.0 |
| CMake | 4.2.3 |
| Metal GPU family | Apple9, Metal4 |

## Hermes Agent Integration

Hermes agent config at `/Users/macmini/.hermes/config.yaml`:
- Default model: `GLM-5.1-UD-IQ3_XXS`
- Provider: `llama-cpp-mainline`
- Port: 1234 (OpenAI-compatible API)
- Context: 202,752 tokens
- Launch agent: `launchctl load /Users/macmini/Library/LaunchAgents/ai.hermes.gateway.plist`

## DSA (Dynamic Sparse Attention) Status

A custom Metal SCATTER kernel was implemented and all 11 files modified to enable DSA for GLM-5.1 on Apple Silicon. The DSA branch is functional but adds ~42-45% overhead at <16K context due to per-layer indexer computation. DSA only benefits at 50K+ filled context.

- DSA branch: `/Users/macmini/projects/llama-cpp-dsa/`
- SCATTER kernel: 8/8 backend tests passed
- Production recommendation: **do not use DSA** for typical conversations
- Full details: `experiments_glm5_1_08apr2026.md` (EXP-10, EXP-11)
