# Benchmark Design

## Overview
Head-to-head comparison of Gemma 4 31B (bf16) vs Qwen 3.5 27B (Q8_0) served via LM Studio on Apple Silicon.

## Details

### Models
- **gemma-4-31b**: 31B dense, bf16, 61.4GB RAM, quadratic attention, vision
- **qwen3.5-27b**: 27B dense, Q8_0, 28.6GB RAM, DeltaNet linear attention, vision

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

# run Claude judge (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=... uv run python judge_claude.py
```

Results: `results/runs.csv`, `results/transcripts.jsonl`, `results/judged_results.csv`
