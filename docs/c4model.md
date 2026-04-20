# C4 Model — LLM Benchmark

## Overview

The benchmark is a single-user local tool that exercises LLMs served by two alternate local inference runtimes (LM Studio and `mlx_lm.server`), scores responses with a mix of programmatic rules and the Claude API, and aggregates results from several evaluation frameworks.

## Level 1: System Context

- **User** — runs the benchmark from the CLI on a Mac Studio M3 Ultra (512 GB unified memory).
- **LM Studio** (`localhost:1234`) — default OpenAI-compatible inference server. Wraps llama.cpp (GGUF) and an MLX engine. Primary path for Qwen3.5, MiniMax, GLM, etc. **Does not yet support Gemma 4**.
- **`mlx_lm.server`** (`localhost:8081`) — alternate inference sidecar, launched via `python -m mlx_lm server --model <path> --port 8081`. Used for:
  - Models unsupported by LM Studio's MLX engine (e.g. Gemma 4 family, including SuperGemma4).
  - A/B speed comparisons against LM Studio on the same weights (`mlx_lm.server` is typically ~15–30% faster on large MoE models per community benchmarks and our in-progress A/B test).
- **Claude API** — external LLM-as-judge for responses not decidable by programmatic rules.
- **Hugging Face Hub** — source of all model weights, fetched via the `hf` CLI with a stored token.

## Level 2: Containers

- **`run_bench.py`** — custom benchmark runner. Sends prompts to whatever API is pointed at by `BENCH_BASE_URL` (defaults to LM Studio on `1234`). Captures streaming metrics, writes `results/runs.csv` (one row per scored run) and `results/transcripts.jsonl` (full response text + reasoning). Both files are opened in append mode; each invocation tags its rows with a `bench_run_id` UUID for session isolation.
- **`run_eval.py`** — unified wrapper over lm-evaluation-harness, LiveCodeBench, DeepEval, and bigcode-evaluation-harness. Reads a model registry and routes jobs to the right harness.
- **`aggregate_results.py`** — merges outputs from all frameworks into `results/aggregate.csv`.
- **`judge_claude.py`** — reads `transcripts.jsonl`, issues blind pairwise comparisons to the Claude API, writes judged CSV.
- **`cases.json`** — the 26-prompt suite (6 categories).
- **`eval_frameworks/`** — cloned repos of external eval suites.
- **`results/`** — output directory. Contains `runs.csv`, `transcripts.jsonl`, per-framework subdirs, and `snapshots/` (10-minute rolling backups, up to 12 retained).
- **Snapshot watcher** — shell loop `/tmp/llm-bench-logs/snapshot_runs.sh` that `cp`s the results files to `results/snapshots/` every 600s. Guards against mid-write corruption.

Environment contract for pointing the runner at an alternate server:
- `BENCH_BASE_URL` — default `http://localhost:1234/v1`
- `BENCH_API_KEY` — default `lm-studio`

## Level 3: Components

### `run_bench.py`

Module-level constants:
- `MAX_RESPONSE_TOKENS = 32768` — every request uses this ceiling (policy 2026-04-14). No per-case `max_tokens` scaling.
- `MAX_TRANSPORT_RETRIES = 3`, `RETRY_BACKOFF_S = 2.0` — transport retry policy.
- `SANITY_CHECK_AFTER = 3` — early abort if first N scored runs all produce empty `response+reasoning`.

Key functions:
- `stream_completion()` — httpx-streamed OpenAI chat completion. Captures `delta.content`, `delta.reasoning_content` (LM Studio), `delta.reasoning` (`mlx_lm.server`), and `delta.tool_calls`. Raises `TransportError` on connection failure (does **not** synthesise a fake result).
- `stream_completion_retried()` — wraps the above with retry + exponential backoff; raises after `MAX_TRANSPORT_RETRIES`.
- `run_tool_use_case()` — multi-turn tool-use handler (up to 4 turns) with stub `tool_responses` per case.
- `score_exact / score_keywords / score_unit_tests / score_constraint_check / score_tool_trace` — programmatic scorers.
- `_scoreable_text(result)` — returns `response_text` if non-empty, else `reasoning_text`. Lets thinking models whose answers land only in `reasoning_content` still be scored.
- `is_invalid_result(result, case)` — returns `True` for `finish_reason == "error"`, empty scoreable text, or `finish_reason == "length"` on non-creative tasks. Causes `score_case()` to return `(None, "invalid")`.
- `score_case()` — router; returns `(score, score_type)` where `score_type="invalid"` means the row is not counted in leaderboard aggregations.
- `run_benchmark()` — main loop. Generates one `bench_run_id` UUID per invocation, iterates cases × models × repeats, writes **JSONL first**, then CSV (so on crash transcripts ≥ csv). Triggers sanity early-abort if the first 3 scored runs are all empty.

CSV schema (21 columns, in order):
```
bench_run_id, run_id, model_label, served_model, quant, category, prompt_id,
repeat, seed, temperature, top_p, max_tokens, ttft_s, gen_s, total_s,
output_tokens_approx, tok_per_s, finish_reason, valid, score_raw, score_type
```

### `judge_claude.py`
- `call_judge()` — Claude API call with rubric system prompt.
- `run_judge()` — pairwise blind evaluation with order randomization.
- Groups transcripts by `prompt_id` + `model_label` across the whole file; to avoid cross-session mashup, filter by `bench_run_id` before invoking.

### `run_eval.py`
- `run_lm_eval()` / `run_bigcode()` / `run_livecodebench()` / `run_deepeval()` — thin wrappers over each harness, targeting the same `BASE_URL`/`API_KEY`.

### `aggregate_results.py`
- `collect_custom_benchmark()` — reads `results/runs.csv`; should filter `valid=True` when computing averages.
- `collect_lm_eval()` / `collect_bigcode()` / `collect_deepeval()` — framework-specific readers.
- `aggregate()` — merges into `results/aggregate.csv`.

## Data Flow

```
cases.json ──► run_bench.py ──► [BENCH_BASE_URL]/v1/chat/completions
                                     │
                                     ├── LM Studio (localhost:1234)
                                     │        └─ GGUF via llama.cpp / MLX engine
                                     │
                                     └── mlx_lm.server (localhost:8081)
                                              └─ raw MLX (required for Gemma 4;
                                                 optional for speed A/B on any MLX model)
                                     │
                   per run:  JSONL ►  results/transcripts.jsonl     (written first + fsync)
                             CSV   ►  results/runs.csv              (written second + fsync)
                                     │
   snapshot_runs.sh (every 600s) ─► results/snapshots/{runs,transcripts}_<ts>.{csv,jsonl}
                                     │
                                     ├── judge_claude.py ─► Claude API ─► results/judged_results.csv
                                     └── aggregate_results.py ──────────► results/aggregate.csv

run_eval.py ──┬── lm-eval ─────► [BENCH_BASE_URL] ─► results/lm-eval/
              ├── bigcode-eval ─► [BENCH_BASE_URL] ─► results/bigcode/
              ├── LiveCodeBench ► [BENCH_BASE_URL] ─► results/livecodebench/
              └── DeepEval ─────► [BENCH_BASE_URL] ─► results/deepeval/
```

## Invariants worth preserving

- **Only `valid=True` rows enter the leaderboard.** Every analysis script should filter accordingly.
- **Append-only writes.** Never open `results/runs.csv` or `results/transcripts.jsonl` in truncate mode. `.gitignore` has `!`-unignore entries for both, so accidental wipes are git-recoverable.
- **Never restart LM Studio mid-bench.** Kills in-flight streams; pre-fix it left zero-score transport-error rows. See `memory/feedback_lm_studio_restart.md`.
- **Never cap `max_tokens` artificially.** See `memory/feedback_always_max_tokens.md`.
- **One `bench_run_id` per invocation.** Always filter by it when comparing two runs of the same `model_label`.

## Usage

Default LM Studio path:
```bash
uv run python run_bench.py
```

Targeting `mlx_lm.server`:
```bash
# 1. Start the server (foreground or backgrounded with &)
python -m mlx_lm server --model /path/to/model --port 8081 --host 127.0.0.1 &

# 2. Point the runner at it
BENCH_BASE_URL=http://localhost:8081/v1 BENCH_API_KEY=none \
  uv run python run_bench.py
```

Test coverage:
```bash
uv run pytest tests/ -q   # 28 passing as of 2026-04-14
```
