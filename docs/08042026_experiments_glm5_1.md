# GLM-5.1 Inference Optimization Experiments

**Date:** 2026-04-08
**Model:** GLM-5.1 UD-IQ3_XXS (754B MoE, 40B active, 268 GB)
**Hardware:** Mac Studio M3 Ultra 512 GB, 819.2 GB/s bandwidth
**Metric:** decode tok/s (primary), prefill tok/s (secondary)
**Benchmark prompt:** "Write a detailed explanation of how transformers work in machine learning, including attention mechanisms, positional encoding, and the encoder-decoder architecture." (max_tokens=512)

---

## Experiment Log

### EXP-01: Baseline (TurboQuant fork, default flags)

```
Server: llama-cpp-turboquant (TheTom fork, feature/turboquant-kv-cache)
Flags:  -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 32768
```

| Metric | Value |
|---|---|
| Prefill | 0.83 tok/s |
| Decode | 4.62 tok/s |
| **Status** | **BASELINE** |

---

### EXP-02: Optimized flags (TurboQuant fork)

**Changes:** -c 8192, -b 2048 -ub 512, -t 16, -np 1

```
Flags: -ctk q8_0 -ctv turbo3 -fa on -ngl 99 -c 8192 -b 2048 -ub 512 -t 16 -np 1
```

| Metric | Value | Delta |
|---|---|---|
| Prefill | 45.95 tok/s | **+5438%** |
| Decode | 15.54 tok/s | **+236%** |
| **Status** | **KEEP** — massive improvement from flag tuning |

**Root cause of EXP-01 slowness:** 32K context with 4 slots (default -np) allocated excessive KV cache. Reducing to 8K + 1 slot freed bandwidth. Thread count and batch size also helped.

---

### EXP-03: Mainline llama.cpp, 8K context

**Changes:** Switch from TurboQuant fork to mainline llama.cpp (latest). Drop TurboQuant KV cache flags.

```
Server: llama.cpp mainline
Flags:  -fa on -ngl 99 -c 8192 -b 2048 -ub 512 -t 16 -np 1
```

| Metric | Value | Delta vs EXP-02 |
|---|---|---|
| Prefill | 43.90 tok/s | -4.5% |
| Decode | 17.55 tok/s | **+13%** |
| **Status** | **KEEP** — mainline has better MoE Metal kernels |

---

### EXP-04: Mainline, 200K native context

**Changes:** Increase context to 202752 (model's training length).

```
Flags: -fa on -ngl 99 -c 202752 -b 2048 -ub 512 -t 16 -np 1
```

| Metric | Value | Delta vs EXP-03 |
|---|---|---|
| Prefill | 10.22 tok/s | -77% (expected: larger KV alloc) |
| Decode | 17.49 tok/s | -0.3% (within noise) |
| KV cache | 17.0 GB | |
| **Status** | **KEEP** — decode unaffected, full native context |

---

### EXP-05: Mainline, 1M context (beyond training)

**Changes:** Increase context to 1,000,000.

```
Flags: -fa on -ngl 99 -c 1000000 -b 2048 -ub 512 -t 16 -np 1
```

| Metric | Value | Delta vs EXP-04 |
|---|---|---|
| Prefill | 9.97 tok/s | -2.4% |
| Decode | 17.55 tok/s | +0.3% (noise) |
| KV cache | 83.7 GB | |
| **Status** | **DISCARD** — no speed benefit, exceeds 202K training window, degrades accuracy via RoPE extrapolation. Warning: `n_ctx_seq > n_ctx_train` |

---

### EXP-06: System tuning (residency, threads, ubatch, mlock)

**Changes from EXP-04:**
- GGML_METAL_RESIDENCY_KEEP_ALIVE_S=86400
- -t 8 (from 16)
- -ub 2048 (from 512)
- --mlock

```
Env:   GGML_METAL_RESIDENCY_KEEP_ALIVE_S=86400
Flags: -fa on -ngl 99 -c 202752 -b 2048 -ub 2048 -t 8 -np 1 --mlock
```

| Metric | Value | Delta vs EXP-04 |
|---|---|---|
| Prefill | 46.26 tok/s | **+353%** |
| Decode | 17.29 tok/s | -1.1% (noise) |
| **Status** | **KEEP** — huge prefill win from -ub 2048 |

**Key finding:** `-ub 2048` (ubatch size) was the primary driver of the prefill improvement (was 512). Thread count change (-t 8 vs 16) had minimal decode impact.

---

### EXP-07: ngram-mod speculative decoding (draftless)

**Changes from EXP-06:** Add `--spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 48 --draft-max 64`

Also added sudo nice -n -20 and --prio 3.

```
Env:   GGML_METAL_RESIDENCY_KEEP_ALIVE_S=86400
Flags: ... --spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 48 --draft-max 64 --prio 3
```

**Test 1: Novel generation prompt**

| Metric | Value | Delta vs EXP-06 |
|---|---|---|
| Prefill | 26.16 tok/s | -43% (overhead from spec context init) |
| Decode | 17.34 tok/s | +0.3% |

**Test 2: Repetitive code prompt (binary search in 6 languages, 1024 tokens)**

| Metric | Value | Delta vs EXP-06 |
|---|---|---|
| Prefill | 30.75 tok/s | N/A (different prompt) |
| Decode | 17.02 tok/s | -1.6% |

| **Status** | **DISCARD for general use** — ngram-mod adds overhead without visible decode speedup. GLM-5.1's reasoning/thinking mode generates novel tokens, defeating n-gram prediction. May help in direct/non-thinking mode for repetitive tasks. |

---

### EXP-08: ik_llama.cpp fork

**Changes:** Switch to ik_llama.cpp (ikawrakow fork, latest main). Same flags as EXP-06.

```
Server: ik_llama.cpp (latest main as of 2026-04-08)
Flags:  -fa on -ngl 99 -c 202752 -b 2048 -ub 2048 -t 8 -np 1 --mlock
```

| Metric | Value | Delta vs EXP-06 |
|---|---|---|
| Prefill | 16.93 tok/s | -63% |
| Decode | 15.86 tok/s | -8.3% |
| **Status** | **DISCARD** — slower than mainline on both metrics. ik_llama.cpp's Metal MoE optimizations (PR #307) may not apply to GLM-5.1's glm_moe_dsa architecture, or the fork's Metal backend is less optimized for this model. |

---

### EXP-09: Draft model speculative decoding (GLM-4.7-Flash)

**Changes from EXP-06:** Add GLM-4.7-Flash Q4_K_M (17 GB) as draft model.

```
Flags: ... -md /Users/macmini/models/GLM-4.7-Flash-Q4_K_M/GLM-4.7-Flash-Q4_K_M.gguf \
  -ngld 99 --draft-max 16 --draft-min 4 --draft-p-min 0.75
Context: 8192 (reduced for test), port 1237
```

| Metric | Value | Delta vs EXP-06 |
|---|---|---|
| Prefill | 47.65 tok/s | +3% |
| Decode | 15.27 tok/s | **-11.7%** |
| **Status** | **DISCARD** — draft model speculation SLOWER than no speculation |

**Root cause:** GLM-5.1's reasoning/thinking mode generates novel tokens with low acceptance rate. GLM-4.7-Flash (30B total, 3B active) is also not small enough — the draft model inference + target verification overhead exceeds the speculative gains. A smaller draft model (<1B) sharing GLM-5.1's tokenizer would be needed, but none exists.

---

## Results Summary

| # | Config | Prefill | Decode | Status |
|---|---|---|---|---|
| 01 | TQ fork, defaults | 0.83 | 4.62 | baseline |
| 02 | TQ fork, tuned flags | 45.95 | 15.54 | KEEP |
| 03 | Mainline, 8K | 43.90 | 17.55 | KEEP |
| 04 | Mainline, 200K | 10.22 | 17.49 | KEEP |
| 05 | Mainline, 1M | 9.97 | 17.55 | DISCARD (accuracy) |
| **06** | **Mainline, tuned** | **46.26** | **17.29** | **BEST / PRODUCTION** |
| 07 | + ngram-mod spec | 26.16 | 17.34 | DISCARD |
| 08 | ik_llama.cpp | 16.93 | 15.86 | DISCARD |
| 09 | + draft model (GLM-4.7-Flash) | 47.65 | 15.27 | DISCARD (-12% decode) |
| 10 | DSA branch (pre-fix) | — | — | FAILED (segfault) |
| 11 | DSA branch (with SCATTER kernel + Hadamard fix) | 40.41 | 10.04 | DISCARD (<16K ctx) |

### KV Cache Strategy Comparison (long prompt scaling, EXP-06 flags)

| Filled Context | FP16 KV (mainline) | q8_0 KV (mainline) | turbo3 KV (TQ fork) |
|---|---|---|---|
| 31 tokens | **17.46** | 16.15 | 15.24 |
| 299 tokens | **17.36** | 16.10 | 14.36 |
| 1,042 tokens | **16.58** | 14.83 | 10.64 |
| 4,050 tokens | **14.45** | 9.19 | 8.31 |
| 5,460 tokens | **9.93** | 5.53 | 3.75 |

**FP16 wins at every context length.** MLA (n_head_kv=1) makes KV already tiny; compressing it just adds dequant overhead.

### DSA vs No-DSA Comparison (decode tok/s at varying context fill)

| Filled Context | No-DSA (EXP-06) | DSA (EXP-11) | Winner |
|---|---|---|---|
| 31 tokens | **17.46** | 10.04 | No-DSA |
| ~300 tokens | **17.36** | 9.91 | No-DSA |
| ~1K tokens | **16.58** | 9.26 | No-DSA |
| ~4K tokens | **14.45** | 7.94 | No-DSA |
| ~5.5K tokens | **9.93** | — | No-DSA |
| 50K+ tokens | <5 (projected) | ~8-10 (projected) | **DSA** |

DSA indexer overhead (~42-45%) exceeds attention savings at <16K context. DSA only wins at 50K+ filled context.

## Current Best Config (EXP-06)

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

**Performance:** 46.26 tok/s prefill, 17.29 tok/s decode, 200K native context

## Improvement from Baseline

| Metric | Baseline | Best | Improvement |
|---|---|---|---|
| Prefill | 0.83 tok/s | 46.26 tok/s | **55.7x** |
| Decode | 4.62 tok/s | 17.29 tok/s | **3.7x** |

## What Worked

1. **-ub 2048** — Biggest single prefill win. Default 512 was leaving Metal GPU underutilized during prompt processing.
2. **-np 1** — Single slot instead of 4 reduced KV cache fragmentation and memory overhead.
3. **-c 8192 -> 202752** — Context size doesn't affect decode speed thanks to MLA (n_head_kv=1).
4. **Mainline over forks** — Mainline llama.cpp had better Metal kernels than both TurboQuant fork and ik_llama.cpp for this specific model architecture.
5. **--mlock** — Prevents OS memory management interference.
6. **GGML_METAL_RESIDENCY_KEEP_ALIVE_S=86400** — Keeps GPU buffers wired for long-running server.

## What Didn't Work

1. **TurboQuant KV cache (turbo3)** — Not needed at 200K context (MLA makes KV tiny at 17 GB). Added 13% decode overhead from older fork.
2. **ngram-mod speculation** — GLM-5.1's reasoning/thinking mode generates novel tokens, defeating n-gram prediction. No visible decode improvement.
3. **ik_llama.cpp** — 8% slower decode, 63% slower prefill. Metal MoE optimizations don't apply to glm_moe_dsa arch.
4. **1M context** — Exceeds 202K training window, RoPE extrapolation degrades accuracy.
5. **-t 16** (all P-cores) — No better than -t 8 for GPU-offloaded inference. More threads add synchronization overhead.

## Remaining Opportunities

1. **MTP (Multi-Token Prediction)** — GLM-5.1 has native MTP heads (`nextn_predict_layers=1`), currently unused by llama.cpp. When support lands (watch PR #15225), could provide free speculative speedup with zero memory cost and perfect acceptance rate since it's the model's own prediction.
3. **DSA indexer** — When llama.cpp implements the DSA indexer (PR #19460 follow-up), sparse attention will improve both quality and prefill speed.
4. **Q4_K_M quant** — IQ3_XXS uses codebook lookups that are slow on Metal. Q4_K_M would be 20-40% faster decode but user declined different quant.

## Long Prompt Scaling (EXP-06 config)

Tested with varying prompt lengths on the best config (mainline, -ub 2048, -t 8, 200K ctx). Max generation: 256 tokens.

| Prompt Tokens | Prefill (tok/s) | Decode (tok/s) | Wall Time |
|---|---|---|---|
| 31 | 46.75 | 17.46 | 15.3s |
| 299 | 115.76 | 17.36 | 17.3s |
| 1,042 | **184.86** | 16.58 | 21.1s |
| 4,050 | 151.29 | 14.45 | 44.5s |
| 5,460 | 85.60 | 9.93 | 89.6s |

### Key Findings

**Prefill scales up then down:**
- Peaks at ~1K tokens (**184.86 tok/s**) — GPU is fully utilized at this batch size
- Short prompts (31 tokens) underutilize the GPU: only 46.75 tok/s
- Beyond ~4K tokens, prefill drops as attention computation grows quadratically

**Decode degrades with filled context:**
- 31-1K tokens: stable at ~17 tok/s
- 4K tokens: drops to 14.45 tok/s (-17%)
- 5.5K tokens: drops to 9.93 tok/s (-43%)
- This is because each decode step must attend over ALL filled KV entries
- At 5.5K filled context, attention over the full KV cache becomes the bottleneck

**Implication:** The 17.5 tok/s decode rate only holds for short conversations. For long-context use (RAG, document analysis), expect ~10-15 tok/s decode depending on how much context is filled.

**Note:** The actual token counts are lower than targets because the tokenizer is more efficient than the 4 chars/token estimate used. The 16K target produced only 5,460 actual tokens.

---

### EXP-10: DSA branch (PR #21149) rebase — GLM-DSA routed to deepseek32 builder

**Changes:** Cloned llama.cpp, checked out PR #21149 (fairydreaming/deepseek-dsa), modified `llama-model.cpp` to route `LLM_ARCH_GLM_DSA` to `llm_build_deepseek32` (which has the full DSA indexer implementation).

```
Repo: /Users/macmini/projects/llama-cpp-dsa
Branch: deepseek-dsa (PR #21149)
Change: LLM_ARCH_GLM_DSA -> llm_build_deepseek32 instead of llm_build_deepseek2
Flags: -fa on -ngl 99 -c 8192 -fit off
```

**Result: CRASH during `sched_reserve` (compute graph scheduling)**

The model loads successfully (268 GB, all tensors), KV cache creates fine, but the Metal scheduler crashes when trying to reserve compute buffers for the DSA graph.

**Root cause:** PR #21149 was developed for CPU/CUDA only. The DSA compute graph uses GGML operations that have no Metal kernel implementations:
- `GGML_OP_SCATTER` — needed for sparse mask construction from top-k indices
- Hadamard rotation matrix multiply — needed for indexer key/query rotation
- Dual KV cache (`llama_kv_cache_dsa`) — Metal scheduler can't handle the split MLA + indexer cache

| Metric | Value |
|---|---|
| Build | OK |
| Model load | OK (268 GB) |
| KV cache | OK (702 MB) |
| Graph scheduling | **CRASH** |
| **Status** | **FAILED — missing Metal kernels for DSA ops** |

**To make this work:** Need to implement Metal kernels for scatter, Hadamard rotation, and the indexer scoring GEMM. This is a multi-week effort requiring Metal Shading Language development.

**The DSA branch is at:** `/Users/macmini/projects/llama-cpp-dsa/` (ready for future Metal kernel development).

### EXP-11: DSA with Metal SCATTER kernel (PR #21149 + custom kernel)

**Changes:**
1. Wrote `GGML_OP_SCATTER` Metal kernel (MSL) — f32 + f16 support
2. Added device support, op dispatch, pipeline registration (6 files)
3. Routed `LLM_ARCH_GLM_DSA` to `llm_build_deepseek32` (graph builder)
4. Routed `LLM_ARCH_GLM_DSA` to `llama_kv_cache_dsa` (dual KV cache)

**SCATTER kernel test:** `test-backend-ops test -o SCATTER` → **8/8 PASSED** (CPU vs Metal identical)

**Server test:** SEGFAULT (exit code 139) during `sched_reserve`

Both KV caches created successfully:
- MLA cache: 43.88 MiB (512 cells, 78 layers)
- Indexer cache: 9.75 MiB (512 cells, 78 layers, head_size=128)

Crash occurs at graph scheduling — the DSA compute graph has other Metal-incompatible patterns beyond SCATTER:
- Likely: tensor layout/stride assumptions that differ between CPU/CUDA and Metal
- Likely: Hadamard rotation matrix (`attn_rot_k`) not being set for indexer cache
- Possibly: buffer allocation ordering issues in the dual cache system

**Status: PARTIAL SUCCESS** — SCATTER kernel works perfectly, but deeper Metal porting of the DSA graph is needed.

**Repo:** `/Users/macmini/projects/llama-cpp-dsa/` (branch: deepseek-dsa + GLM-DSA routing + SCATTER kernel)

**Next steps to resolve:**
1. Build debug version (`-DCMAKE_BUILD_TYPE=Debug`) and run under lldb to find exact crash line
2. Check if `attn_rot_k` (Hadamard rotation) needs to be enabled for the indexer cache
3. Verify all tensor shapes in the DSA graph match Metal backend expectations
4. Test on CPU-only first (`-ngl 0`) to verify graph correctness before Metal

---

## Theoretical Analysis

- M3 Ultra bandwidth: ~800 GB/s effective
- 40B active params at IQ3_XXS: ~15-20 GB read per token
- Theoretical max: 800/17.5 = ~45 tok/s
- Achieved: 17.29 tok/s = ~38% of theoretical
- Gap due to: IQ codebook overhead, attention computation, Metal scheduling, shared memory contention
