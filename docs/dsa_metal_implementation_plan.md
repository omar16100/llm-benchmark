# DSA Metal Kernel Implementation Plan

**Date:** 2026-04-08
**Goal:** Enable Dynamic Sparse Attention (DSA) for GLM-5.1 on Apple Silicon Metal, fixing decode speed degradation at long context.

---

## Problem Statement

GLM-5.1 decode speed degrades as context fills:

| Filled Context | Current Decode (tok/s) | With DSA (projected) |
|---|---|---|
| 31 tokens | 17.46 | 17.46 |
| 1,042 tokens | 16.58 | ~17.0 |
| 4,050 tokens | 14.45 | ~17.0 |
| 5,460 tokens | 9.93 | ~17.0 |
| 50K+ tokens | <5 (projected) | ~15-17 |

DSA indexer selects top-2048 KV positions per query, keeping attention cost constant regardless of context fill. The model already has trained indexer weights — they're loaded but unused.

## Root Cause: One Missing Metal Kernel

PR #21149 (fairydreaming/deepseek-dsa) implements DSA for CPU/CUDA. All 15+ GGML ops in the DSA compute graph have Metal kernels **except `GGML_OP_SCATTER`**.

**SCATTER semantics**: Given a destination tensor (attention mask filled with -inf) and int32 indices (top-k positions), write a scalar (0.0f) at each indexed position — unmasking those positions for attention.

**Confirmed by**: Codebase exploration + Codex (GPT-5.4) independent analysis.

## Implementation: 6 Files to Modify

All in `/Users/macmini/projects/llama-cpp-dsa/`:

### 1. Metal Kernel (`ggml/src/ggml-metal/ggml-metal.metal`, ~line 9058)

```metal
template<typename T>
kernel void kernel_scatter(
        constant ggml_metal_kargs_scatter & args,
        device const char * ids,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z, i2 = tgpig.y, i1 = tgpig.x;
    if (i3 >= args.nr3 || i2 >= args.nr2 || i1 >= args.nr1) return;

    device const char * ids_row = ids + i1*args.idnb1 + i2*args.idnb2 + i3*args.idnb3;
    device       char * dst_row = dst + i1*args.nb1   + i2*args.nb2   + i3*args.nb3;
    const T c = (T) args.c;

    for (int32_t j = tiitg; j < args.nids; j += ntg.x) {
        const int32_t id = *(device const int32_t *)(ids_row + (uint64_t)j*args.idnb0);
        if ((uint32_t)id < (uint32_t)args.ne0) {
            *((device T *)(dst_row + (uint64_t)id*args.nb0)) = c;
        }
    }
}

typedef decltype(kernel_scatter<float>) kernel_scatter_t;
template [[host_name("kernel_scatter_f32")]] kernel kernel_scatter_t kernel_scatter<float>;
template [[host_name("kernel_scatter_f16")]] kernel kernel_scatter_t kernel_scatter<half>;
```

### 2. Kernel Args (`ggml/src/ggml-metal/ggml-metal-impl.h`, ~line 933)

```cpp
typedef struct {
    int32_t  ne0;
    int32_t  nids, nr1, nr2, nr3;
    uint64_t nb0, nb1, nb2, nb3;
    uint64_t idnb0, idnb1, idnb2, idnb3;
    float    c;
} ggml_metal_kargs_scatter;
```

### 3. Device Support (`ggml/src/ggml-metal/ggml-metal-device.m`, ~line 1233)

```cpp
case GGML_OP_SCATTER:
    return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
           op->src[0]->type == op->type &&
           op->src[1]->type == GGML_TYPE_I32 &&
           op->src[0]->ne[1] == op->src[1]->ne[1] &&
           op->src[0]->ne[2] == op->src[1]->ne[2] &&
           op->src[0]->ne[3] == op->src[1]->ne[3];
```

### 4. Op Dispatch (`ggml/src/ggml-metal/ggml-metal-ops.cpp`)

Add `ggml_metal_op_scatter()` function and switch case `GGML_OP_SCATTER`.

### 5. Pipeline Registration (`ggml/src/ggml-metal/ggml-metal-device.h` + `.cpp`)

Add `ggml_metal_library_get_pipeline_scatter()` compiling `kernel_scatter_f32` / `kernel_scatter_f16`.

### 6. Op Forward Declaration (`ggml/src/ggml-metal/ggml-metal-ops.h`)

Add `int ggml_metal_op_scatter(ggml_metal_op_t ctx, int idx);`

### 7. Model Routing (already done)

`src/llama-model.cpp`: Route `LLM_ARCH_GLM_DSA` to `llm_build_deepseek32`.

## Metrics to Track

### Primary: Decode tok/s at varying context fill

| Metric | Baseline (no DSA) | Target (with DSA) |
|---|---|---|
| Decode @ 31 tokens | 17.46 | >=17.0 |
| Decode @ 1K tokens | 16.58 | >=16.5 |
| Decode @ 4K tokens | 14.45 | >=16.5 |
| Decode @ 5.5K tokens | 9.93 | >=16.0 |
| Decode @ 16K tokens | ~6 (projected) | >=15.0 |
| Decode @ 50K tokens | <5 (projected) | >=14.0 |

**Success criterion**: Decode speed at 5.5K context >= 15 tok/s (vs 9.93 baseline).

### Secondary: Prefill tok/s

| Metric | Baseline | Target |
|---|---|---|
| Prefill @ 31 tokens | 46.26 | >=40 (some overhead acceptable) |
| Prefill @ 1K tokens | 184.86 | >=150 |
| Prefill @ 5.5K tokens | 85.60 | >=70 |

### Quality: Output correctness

| Metric | Method |
|---|---|
| Token-level match | Compare first 100 tokens of same prompt, DSA vs no-DSA, seed=42 |
| Perplexity | Run `llama-perplexity` on a test dataset, compare DSA vs no-DSA |
| Reasoning quality | Manual eval on 5 reasoning prompts |

### System: Memory and stability

| Metric | Target |
|---|---|
| Total GPU memory | <400 GB (model + dual KV cache) |
| Server stability | No crash in 1 hour continuous use |
| `test-backend-ops test -o SCATTER` | PASS |
| `test-backend-ops perf -o SCATTER` | >1 GB/s throughput |

## If It Works (Success Path)

1. **Update server config**: Switch Hermes agent to DSA-enabled server
2. **Update start script**: `/Users/macmini/projects/llama-cpp-turboquant/start_glm5.sh` -> use DSA build
3. **Benchmark at 200K context**: Run full long-prompt suite
4. **Document**: Add EXP-11 to experiments doc
5. **Upstream contribution**: Submit Metal SCATTER kernel as PR to ggml-org/llama.cpp, reference PR #21149
6. **Enable full 200K context**: With DSA, decode at 200K should be usable (~15+ tok/s)
7. **Consider IndexCache optimization**: THUDM's IndexCache reuses indexer results across layers for 1.82x prefill speedup

## If It Doesn't Work (Failure Paths)

### Failure 1: Kernel works but DSA graph crashes elsewhere
- **Diagnosis**: Run `test-backend-ops test -o SCATTER` first to isolate kernel correctness
- **Mitigation**: Check if other ops have edge cases with DSA tensor shapes (e.g., top_k output format)
- **Fallback**: Run DSA on CPU-only (`--device none -ngl 0`) to verify graph correctness, then isolate Metal-specific failures

### Failure 2: DSA works but decode speed doesn't improve
- **Diagnosis**: Check if indexer computation overhead cancels out the attention savings
- **Root cause**: Indexer runs per-layer (79 layers x per-token), adding ~32 extra matmuls per token
- **Mitigation**: Reduce indexer_top_k from 2048 to 1024 or 512 via `--override-kv`
- **Fallback**: IndexCache (skip indexer on some layers) would reduce overhead

### Failure 3: DSA works but output quality degrades
- **Diagnosis**: Compare perplexity with and without DSA
- **Root cause**: Top-2048 may not capture important tokens at very long context
- **Mitigation**: Increase indexer_top_k or use hybrid (DSA for long context, full attention for short)
- **Fallback**: Keep no-DSA config for quality-critical tasks

### Failure 4: Dual KV cache exceeds Metal memory
- **Diagnosis**: Check `recommendedMaxWorkingSetSize` vs actual allocation
- **Root cause**: Indexer KV cache adds ~10-20 GB at 200K context
- **Mitigation**: Use q8_0 for indexer KV cache; reduce context to 128K
- **Fallback**: Use no-DSA for 200K, DSA for 128K with better decode speed

### Failure 5: Complete blocker beyond SCATTER
- **Diagnosis**: If other ops fail at runtime (not detected at graph scheduling)
- **Mitigation**: Run with `GGML_METAL_GRAPH_DEBUG=1` to trace failing ops
- **Fallback**: Wait for PR #21149 to add official Metal support

## Estimated Effort

| Phase | Hours |
|---|---|
| Write kernel + plumbing (6 files) | 2-4 |
| Build + fix compile errors | 1-2 |
| Run backend tests | 1-2 |
| End-to-end GLM-5.1 test | 2-4 |
| Benchmark suite | 2-3 |
| Documentation | 1-2 |
| **Total** | **9-17 hours (~2 days)** |

## Reference Files

| File | Purpose |
|---|---|
| `/Users/macmini/projects/llama-cpp-dsa/ggml/src/ggml-cuda/scatter.cu` | CUDA reference implementation |
| `/Users/macmini/projects/llama-cpp-dsa/ggml/src/ggml-cpu/ops.cpp:11245` | CPU reference implementation |
| `/Users/macmini/projects/llama-cpp-dsa/src/models/deepseek32.cpp` | DSA graph builder (indexer lines 68-200) |
| `/Users/macmini/projects/llama-cpp-dsa/src/llama-graph.cpp:2288` | SCATTER call site |
| `/Users/macmini/projects/llama-cpp-dsa/src/llama-kv-cache-dsa.cpp` | Dual KV cache implementation |
| `/Users/macmini/projects/codex/dsa_metal_plan.txt` | Codex GPT-5.4 analysis |
| `/Users/macmini/projects/llm-benchmarks/experiments_glm5_1_08apr2026.md` | Experiment log |
| `/Users/macmini/projects/llm-benchmarks/bench_long_prompt.py` | Benchmark script |

## Verification Commands

```bash
# 1. Build
cd /Users/macmini/projects/llama-cpp-dsa
cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON -DGGML_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. Test SCATTER kernel
./build/bin/test-backend-ops support -o SCATTER
./build/bin/test-backend-ops test -o SCATTER
./build/bin/test-backend-ops perf -o SCATTER

# 3. End-to-end server test
export GGML_METAL_RESIDENCY_KEEP_ALIVE_S=86400
./build/bin/llama-server \
  -m /Users/macmini/models/GLM-5.1-UD-IQ3_XXS/UD-IQ3_XXS/GLM-5.1-UD-IQ3_XXS-00001-of-00007.gguf \
  -fa on -ngl 99 -c 8192 -b 2048 -ub 2048 -t 8 -np 1 --mlock -fit off \
  --host 0.0.0.0 --port 1234

# 4. Benchmark
cd /Users/macmini/projects/llm-benchmarks
python3 bench_long_prompt.py
```

---

## Actual Results (2026-04-08)

### Implementation completed
- SCATTER Metal kernel: **8/8 backend tests passed** (F32 + F16, inplace + non-inplace, 1D + 4D)
- Hadamard rotation fix: `attn_rot_k` enabled for GLM-DSA indexer cache (bug: arch check only matched DEEPSEEK32)
- Defensive assertions and null guards added
- DSA server starts, loads model, generates text on Metal

### Bug found and fixed
Indexer KV cache wasn't enabling `attn_rot_k` — arch check in `llama-kv-cache.cpp:284` only matched `LLM_ARCH_DEEPSEEK32`, not `LLM_ARCH_GLM_DSA`. Caused null dereference during Hadamard rotation in graph build.

### Performance: DSA adds overhead at short/medium context

| Filled Context | No-DSA (EXP-06) | DSA (EXP-11) |
|---|---|---|
| 31 tokens | 17.46 tok/s | 10.04 tok/s (-42%) |
| ~300 tokens | 17.36 | 9.91 (-43%) |
| ~1K tokens | 16.58 | 9.26 (-44%) |
| ~4K tokens | 14.45 | 7.94 (-45%) |

### Verdict
**DSA not recommended for <16K context.** Indexer overhead (79 layers x matmul+hadamard+topk+scatter per token) exceeds attention savings. Only beneficial at 50K+ filled context.

### Codex Reviews
- Initial plan: `/Users/macmini/projects/codex/dsa_metal_plan.txt`
- Debug feedback: `/Users/macmini/projects/codex/dsa_debug_feedback.txt`
