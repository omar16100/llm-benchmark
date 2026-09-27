# LLM Benchmark Results

## Overview

Multi-model comparison on **Mac Studio M3 Ultra** (512 GB unified memory, 32-core CPU, 800 GB/s memory bandwidth). The benchmark runner (`run_bench.py`) sends 26 prompts × 3 repeats × 2 warmups across 6 categories (coding, creative, instruction, math, reasoning, tool_use) and captures streaming chat completions via an OpenAI-compatible API.

Two inference backends are in use:
- **LM Studio** on `localhost:1234` (default) — wraps llama.cpp and MLX.
- **`mlx_lm.server`** on `localhost:8081` (via `BENCH_BASE_URL` env var) — used when LM Studio doesn't support a model (e.g. Gemma 4 architecture) or for raw-MLX speed comparisons.

Each row in `results/runs.csv` is tagged with a `bench_run_id` (UUID per `run_bench.py` invocation) and a `valid` flag. Only `valid=True` rows contribute to the leaderboard and per-category scores.

## Leaderboard (as of 2026-04-20)

Ordered by average score over valid rows. `Valid/Total` and `Overall /5` are reproducible from [../data/runs_20apr2026.csv](../data/runs_20apr2026.csv) with `uv run python scripts/runs_data.py summary data/runs_20apr2026.csv` (the README sample results table). `tok/s (real)` and `Disk` come from local files that are not published (`results/transcripts.jsonl`, the model directories), so they cannot be checked from this repo. `tok/s (real)` is derived from `results/transcripts.jsonl` as `(response_chars + reasoning_chars) / 4 / gen_s`, which corrects the under-count of thinking tokens in the raw CSV column.

| Rank | Model | Quant | Valid/Total | Overall /5 | tok/s (real) | Disk |
|---:|---|---|---:|---:|---:|---:|
| 1 | SuperGemma4-26B (mlx_lm.server) | MLX 4-bit | 54/78 | 4.41 | 63.3 | 14 GB |
| 2 | GLM-5.1 | MLX 3.6-bit | 75/78 | 4.37 | 6.6 | 382 GB |
| 3 | MiniMax-M2.7 (LM Studio) | MLX 4-bit mxfp4 | 69/78 | 4.16 | 36.3 | 121 GB |
| 4 | Qwen3.5-122B-A10B Q8 | GGUF Q8_0 | 72/78 | 4.09 | 22.9 | 121 GB |
| 5 | M2.7 (mlx_lm.server) | MLX 4-bit mxfp4 | 60/78 | 3.73 | 34.9 | 121 GB |
| 6 | Qwen3.5-122B-A10B | GGUF Q4_K_M | 51/78 | 3.58 | 25.4 | 71 GB |
| 7 | Qwen3.5-397B-A17B | MLX Q4 | 57/78 | 2.78 | 26.0 | 224 GB |
| 8 | GLM-4.7-Flash (32K) | GGUF Q4_K_M | 30/78 | 2.39 | 55.2 | 18 GB |
| 9 | GLM-4.7-Flash | GGUF Q4_K_M | 24/78 | 2.14 | 62.3 | 18 GB |
| n/a | Qwen3.5-27B-Claude-Opus-Distilled | GGUF Q8_0 | 3/3 | 0.00 *(smoke)* | 14.0 | 27 GB |

**SuperGemma4-26B overtook MiniMax-M2.7 on 2026-04-20**: +6% quality (4.41 vs 4.16), +74% faster (63 vs 36 tok/s), 8.6x smaller on disk (14 vs 121 GB). Perfect 5.00 on reasoning, coding and instruction. Weaker on tool_use (3.75 vs 4.06) and valid-row rate (69% vs 88%).

Before the SuperGemma4-26B and GLM-5.1 runs, MiniMax-M2.7 led every category. 122B-A10B (Q4_K_M) trails it by ~0.6 overall but matches on instruction and coding. 397B and GLM-Flash are held back by validity rates (many runs length-truncated before the patched 32K-token policy; rerun pending at the time).

### Per-category leaders (all runs in data/runs_20apr2026.csv)

Highest mean score over valid, programmatically scored rows; `n` is that model's scored rows in the category.

| Category | Leader(s) | Score | n |
|---|---|---:|---:|
| coding | 7-way tie: SuperGemma4-26B, GLM-5.1, MiniMax-M2.7 (both runtimes), Qwen3.5-122B-A10B (Q8 and Q4_K_M), Qwen3.5-397B | 5.00 | 6 to 12 |
| creative | GLM-5.1, MiniMax-M2.7 (LM Studio), Qwen3.5-122B-A10B Q8 | 5.00 | 9 each |
| instruction | 7-way tie (same models as coding) | 5.00 | 6 to 12 |
| math | SuperGemma4-26B, MiniMax-M2.7 (both runtimes) | 3.33 | 9 each |
| reasoning | SuperGemma4-26B | 5.00 | 6 |
| tool_use | GLM-5.1 | 4.17 | 9 |

## Per-model detail

### MiniMax-M2.7 MLX-4bit-mxfp4 (LM Studio)

- **bench_run_id**: `dd67f270-c3d1-44d5-b91f-753237a5d8e0`
- **Weights**: `mlx-community/MiniMax-M2.7-4bit-mxfp4` (121 GB)
- **69/78 valid** (9 length-truncated — `reasoning R2` × 3, `math M3` × 3, etc.)
- **Perfect 5.0** on coding, creative, instruction
- **Weak**: math 3.33, reasoning 2.80
- First fully clean bench under the patched runner — landmark data point.

### Qwen3.5-122B-A10B Q4_K_M (GGUF, LM Studio)

- **bench_run_id**: `b8579c32-0403-4a26-a3f2-4f0a31ab45dd`
- **Weights**: `unsloth/Qwen3.5-122B-A10B-GGUF` (71 GB, 3 shards, now deleted to reclaim disk)
- **51/78 valid** (27 length-truncated)
- coding 5.00, instruction 5.00, creative 4.56, tool_use 3.76 — solid generalist
- **math 0.00** ⚠️ complete failure, consistent with other Qwen3.5 models on M1 (see anomalies)

### Qwen3.5-397B-A17B MLX-Q4 (LM Studio)

- **bench_run_id**: `6ad21732-c0d0-4f7f-bd71-5f98d6e75b4d`
- **Weights**: `mlx-community/Qwen3.5-397B-A17B-4bit` (224 GB, still on disk)
- **57/78 valid** (21 length-truncated)
- coding 5.00, instruction 5.00, creative 4.78 — strong on structure
- **tool_use 0.00** ⚠️ — scoring path didn't register any tool calls; could be an MLX-backend streaming quirk for tool-call deltas. Needs investigation.
- math 1.00 — same pattern as 122B.

### GLM-4.7-Flash Q4_K_M (GGUF, LM Studio)

- **bench_run_id**: `5ab520af-d36d-47a2-9fa2-16bd42c84e4d`
- **Weights**: `unsloth/GLM-4.7-Flash-GGUF` (17 GB)
- **Only 24/78 valid** (51 `length`, 3 `empty`)
- 62.3 tok/s — **fastest in the stack** (3B active parameters)
- Thinking-trace behaviour burns through the token budget. Rerun with `MAX_RESPONSE_TOKENS = 32768` is queued (task #24).

### MiniMax-M2.7 mlx_lm.server A/B (COMPLETE)

- **bench_run_id**: `b7aae11e-5b02-4055-8cc1-d895fc406d21`
- **Weights**: same `mlx-community/MiniMax-M2.7-4bit-mxfp4`, but served via `python -m mlx_lm server --port 8081`
- **60/78 valid**, overall **3.73/5**, **tok/s 34.9** (vs LM Studio's 36.3 on identical weights).
- **Result: LM Studio wins on every category that differed.** Reasoning regressed 2.80 → 0.60, tool_use 4.06 → 2.81. Coding, instruction, math, creative all ≈ tie. tok/s was a wash (~-4%).
- **Research agent's +30% speedup prediction did NOT hold** for `mlx_lm.server` in HTTP mode. Likely the community benchmarks were done with `mlx_lm.generate` CLI, which bypasses the server's HTTP/JSON overhead.
- **Decision: LM Studio stays the default runtime for M2.7** — runtime switch is not worth quality regression.

### SuperGemma4-26B (earlier attempts, before the valid 20 Apr run above)

- LM Studio's MLX engine returns `"Gemma 4 support is not ready yet, stay tuned!"` — model is unloadable there.
- Switched to `mlx_lm.server` on port 8081. Immediate issue: `delta.reasoning` field (not `delta.reasoning_content`) — runner patched to accept both.
- Second issue: the model emits ≥12K chars of reasoning without transitioning to content even at `thinking_token_multiplier: 32`. Entire bench was `valid=False`.
- Now queued again under the `MAX_RESPONSE_TOKENS = 32768` policy and will retry when the current M2.7 A/B completes.

### Qwen3.5-27B-Claude-Opus-Distilled Q8 (3-row smoke)

- **bench_run_id**: `f4949f59-029c-4f96-b903-fd91a176e546`
- Only the M1 math case was run; all 3 repeats scored `0.0` because the model answered `1/2` while `cases.json` says `expected: "5/12"`. The model's math is correct (`P(exactly one blue in 2 of 9, 3 blue) = C(3,1)·C(6,1)/C(9,2) = 18/36 = 1/2`). **`cases.json` M1 is almost certainly mis-specified.** Audit pending (task #22).
- Full 78-run rerun is queued.

## Known anomalies

- **tool_use = 0.0 on 397B MLX-Q4**: 12 clean valid rows, 0 score across all. Either the MLX runtime isn't emitting OpenAI-style tool-call deltas in streaming, or the scoring is rejecting the tool trace shape. Investigate before drawing architectural conclusions.
- **math M1 expected wrong**: `"5/12"` doesn't match the stated problem. Audit cases.json.
- **math in general drags every Qwen3.5 model** (0.0–1.0). Either cases.json has more wrong expecteds, or these models genuinely struggle with multi-step probability/arithmetic at this quantization level.

## New bench policy (2026-04-14)

`run_bench.py` no longer scales `max_tokens` by case budget × multiplier. Every scored request (including tool-use turns) now uses the line below; warmup calls use `min(case max_tokens, 64)`:

```python
MAX_RESPONSE_TOKENS = 32768
```

Rationale: length-truncation produces `valid=False` rows with no scoring signal — 51/78 invalid on GLM-Flash, 27/78 on 122B, 21/78 on 397B. With a generous ceiling, only runaway loops trigger invalidation. Captured as a durable preference in `memory/feedback_always_max_tokens.md`.

## Data-integrity history

### 2026-04-13 — pre-fix
- `run_bench.py:503,507` opened outputs in `"w"` mode, wiping all history per invocation. Lost the 78-run MiniMax-M2.5 sweep and 78+ runs of Qwen3.5-397B PLD.
- Fixed to append mode with header-skip; added periodic 10-min snapshots under `results/snapshots/`.

### 2026-04-13: Codex review (review output kept locally, not published)
Five bugs surfaced and patched:

1. **Transport errors were silently scored as data.** The `except` in `stream_completion` synthesised a zero-valued result. Replaced with retry (3 × exponential backoff) + `TransportError` raise. Aborted benches no longer persist fake rows.
2. **Empty `response_text` scored non-zero on 36/55 rows.** Constraint scoring subtracted penalties from a base of 5.0; empty strings still came out positive. Added `is_invalid_result()` and `_scoreable_text()` (falls back to `reasoning_text` for thinking models that put answers there).
3. **Cross-session mashup in judge/aggregate.** Added `bench_run_id` UUID column per invocation so rows can be sliced by run.
4. **CSV/JSONL divergence** (different row counts after crash). Write order is now JSONL, flush, CSV, flush (the code calls `flush()`, not `os.fsync`), so transcripts ≥ csv after a process crash.
5. **Sanity early-abort.** If the first 3 scored runs all produce empty `response+reasoning`, the bench raises — catches misconfigured chat templates or wrong `thinking` flag.

### 2026-04-14 — streaming alias + max-token policy
- `mlx_lm.server` streams reasoning in `delta.reasoning`; LM Studio uses `delta.reasoning_content`. Streamer accepts both aliases.
- `MAX_RESPONSE_TOKENS = 32768` replaces the multiplier-based budget.

### Test coverage
28 unit tests in `tests/test_scoring.py`, including:
- Empty responses never score non-zero on `constraint_check` / `exact` / `keywords`.
- `is_invalid_result` flags empty-both, `error`, and `length` (non-creative).
- `_scoreable_text` prefers response but falls back to reasoning.
- Thinking-model scoring finds embedded answers in reasoning.

## Performance caveat: tok/s under-counts for thinking models

`results/runs.csv.tok_per_s` only divides `response_text_chars/4` by `gen_s`. Thinking models emit the bulk of their tokens in `reasoning_content`, so the CSV number is artificially low (0.4–6 tok/s for some categories). The `tok/s (real)` column in this doc reconstructs the honest throughput from `results/transcripts.jsonl`. Fixing the CSV calc is a pending runner change (task #23).

## Pending

1. **M2.7 mlx_lm.server A/B** — in flight; target is to quantify LM Studio overhead.
2. **122B-Q8 download + bench** — requires deleting 397B weights first (124 GB headroom needed).
3. **Qwen3.5-397B tool_use investigation** — why 0.0 across the board.
4. **GLM-4.7-Flash rerun** — under the 32K-token policy; expect a real quality signal.
5. **SuperGemma4 retry** — same policy; expect some valid rows this time.
6. **Full Qwen3.5-27B-Claude-Opus-Distilled bench** — replace the 3-row smoke.
7. **`cases.json` audit** — at minimum M1 is wrong; others suspect.
8. **Claude-as-judge pass** — for categories where programmatic scoring can't decide (`needs_judge` rows).
9. **Statistical tests** — sign test, Wilcoxon, bootstrap CIs on the pairwise category results.

## Caveats

- Sample size per category is small (3–5 prompts × 3 repeats = 9–15 valid rows per model-category cell). Treat per-category conclusions as directional.
- At the time of these April 2026 runs, LM Studio did not load Gemma 4, so all Gemma 4 comparisons here went through `mlx_lm.server`, a different runtime whose overhead profile was still being characterised.
- Thinking-model runs with `MAX_RESPONSE_TOKENS = 32768` are slow (multi-minute per request). Benches that would take 30 min previously now take 2–4h.
- Disk budget (926 GB total, frequently under 50 GB free during active benching) is the limiting factor for bigger experiments (Kimi K2.5, GLM-5.1, 122B-Q8).
