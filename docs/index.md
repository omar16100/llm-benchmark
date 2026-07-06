# LLM Benchmark Documentation Index

| File | Category | Description |
|------|----------|-------------|
| [benchmark_design.md](benchmark_design.md) | evergreen | Benchmark suite design, categories, scoring methodology |
| [longctx_bench.md](longctx_bench.md) | evergreen | Long-context needle-in-haystack + prefill-throughput bench (bench_longctx.py, bench_common.py) for any OpenAI-compatible server |
| [results.md](results.md) | evergreen | Multi-model leaderboard (MiniMax-M2.7, Qwen3.5-122B/397B, GLM-4.7-Flash, SuperGemma4) + bench history |
| [c4model.md](c4model.md) | evergreen | C4 architecture model |
| [evaluation_frameworks.md](evaluation_frameworks.md) | evergreen | Eval frameworks: lm-eval, LiveCodeBench, DeepEval, bigcode |
| [local_llm_1m_context_findings.md](local_llm_1m_context_findings.md) | dated | M3 Ultra local model survey for native 1M-context + 100 tok/s target |
| [06072026_kimi_linear_1m_verification.md](06072026_kimi_linear_1m_verification.md) | dated | First verified full-1M NIAH pass on M3 Ultra: Kimi-Linear-48B-A3B, 8/8 recall at 1,048,692 tokens (bench_niah_mlx.py) |
| [19042026_bengali_ocr_finetuning.md](19042026_bengali_ocr_finetuning.md) | dated | Bengali OCR fine-tuning experiment plan, decision log, mlx-vlm fix |
| [08042026_experiments_glm5_1.md](08042026_experiments_glm5_1.md) | dated | GLM-5.1 experiment notes (from earlier session, merged from llm-benchmarks/) |
| [08042026_glm5_1_benchmark.md](08042026_glm5_1_benchmark.md) | dated | GLM-5.1 benchmark results (from earlier session) |
| [dsa_metal_implementation_plan.md](dsa_metal_implementation_plan.md) | evergreen | Dense Sparse Attention (DSA) Metal implementation plan for GLM-5.1 on Apple Silicon |

## Naming Conventions
- Dated docs: `DDMMYYYY_topic.md`
- Evergreen docs: `topic.md`

## Required Sections per Category
- evergreen: Overview, Details, Usage
- dated: Context, Changes, Impact
