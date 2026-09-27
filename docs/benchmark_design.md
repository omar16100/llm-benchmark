# Benchmark Design

## Overview
Head-to-head comparison of Gemma 4 31B (bf16) vs Qwen 3.5 27B (Q8_0) served via LM Studio on Apple Silicon.

## Details

### Models
- **gemma-4-31b**: 31B dense, bf16 GGUF weights of 57.18 GiB (61.4 GB), sliding-window (1024 tokens) plus global attention layers, vision
- **qwen3.5-27b**: 27B dense, Q8_0 GGUF weights of 26.62 GiB (28.6 GB), hybrid linear-attention (DeltaNet) layers with full attention every 4th layer, vision

Weight sizes and attention layout are from the GGUF metadata in the LM Studio load log of 5 Apr 2026.
The run of that day was recovered from server logs: see
[27092026_recovered_gemma4_qwen35_run.md](27092026_recovered_gemma4_qwen35_run.md).

### Categories (26 prompts)
- Reasoning (R1-R5): logic puzzles, incident RCA, sprint planning, ETL diagnosis
- Coding (C1-C5): function generation, bug fixing, SQL, LRU cache, nginx parser
- Math (M1-M4): probability, CRT, combinatorics, geometry
- Instruction Following (I1-I4): JSON format, bullet constraints, sorting, word count
- Creative (W1-W4): microstory, landing page, dialogue, poetry
- Tool Use (T1-T4): refund policy, weather+calendar, debugging, SQL analytics

### Scoring
- **Programmatic**: exact match, unit tests, constraint checks, tool trace validation
- **Claude judge**: blind pairwise A/B with swapped order, rubric-based (correctness 0-5, instruction 0-3, completeness 0-1, style 0-1)

### Execution
- 2 warmup + 3 scored repeats per prompt per model
- Deterministic: temperature=0, seed=42
- Creative: temperature=0.8, seeds 41/42/43
- Single concurrency

## Usage

```bash
# run benchmarks
uv run python run_bench.py

# run Claude judge (shells out to the local `claude -p` CLI and its login)
uv run python judge_claude.py
```

Results: `results/runs.csv`, `results/transcripts.jsonl`, `results/judged_results.csv`
