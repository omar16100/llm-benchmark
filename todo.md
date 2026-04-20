# TODO

## Done
- [x] Project setup (uv, git, dependencies)
- [x] cases.json — 26 prompts across 6 categories
- [x] run_bench.py — benchmark runner with streaming, timing, programmatic scoring
- [x] judge_claude.py — blind pairwise Claude-as-judge
- [x] tests/test_scoring.py — 19 unit tests for scoring functions (all passing)
- [x] docs (index, benchmark_design, c4model)

## Pending (revised 2026-04-13 14:50)

Disk-aware execution sequence. Each step ends in a known disk + RAM state.

### In flight
- [~] Bench Qwen3.5-27B-Claude-Opus-Distilled Q8 (PID 50343, ~62%, ETA 38 min)
- [~] Download MiniMax-M2.7 mlx-community/4-bit-mxfp4 (PID 51740, ~74%, ETA 15 min)

### Queued (in order, with disk + RAM state expected after each)
1. **MiniMax-M2.7** — needs unload of Qwen3.5-397B from RAM first; bench then unload. Disk after: ~47 GB free.
2. **Qwen3.5-397B rerun** — restore lost baseline. Weights still on disk; reload into RAM. Disk unchanged.
3. **GLM-4.7-Flash Q4_K_M** — small; load alongside 397B if RAM allows, otherwise unload first. Disk unchanged.
4. **Qwen3.5-122B-A10B Q4_K_M** — 71 GB on disk. Load, bench, **then delete Q4 weights** (frees 71 GB).
5. **Delete `~/models/Qwen3.5-397B-A17B-MLX-4bit`** (frees ~224 GB) — required because Q8 won't fit in 47+71=118 GB; need 224+71+47=342 GB free total.
6. **Download + bench Qwen3.5-122B-A10B Q8_0** (~122 GB).
7. **GLM-5.1 / Kimi K2.5** — only after step 6, when 200+ GB is free.

### Post-processing (run after each new bench lands)
- [ ] Run Claude judge on results (judge_claude.py)
- [ ] Statistical analysis (sign test, Wilcoxon, bootstrap CIs)
- [ ] Update docs/results.md with multi-model comparison

### Mitigations active
- [x] Append-mode CSV/JSONL writes (run_bench.py:506-516)
- [x] Results files unignored from .gitignore (git-recoverable)
- [x] HF_TOKEN authed
- [x] **Transport errors retry then raise** — never persisted as fake rows (run_bench.py:159-179)
- [x] **Thinking-model scoring fallback to reasoning_text** when response_text empty (run_bench.py `_scoreable_text`)
- [x] **Validity flag per row** — `valid=False` if response/reasoning both empty or finish_reason in {error, length} for non-creative
- [x] **`bench_run_id` UUID per invocation** — tags each row to its run, enables session isolation without per-session dirs
- [x] **Sanity early-abort** — bails if first 3 scored runs all empty (catches misconfigured thinking models)
- [x] **JSONL written before CSV per row** — on crash transcripts ≥ csv (re-derivable)
- [x] Periodic 10-min snapshots → `results/snapshots/` (last 12 retained)
- [x] 28 unit tests cover empty-text invalidation + thinking model scoring
- [ ] Atomic write across CSV+JSONL pair (codex #4 — current ordering helps but not transactional)
- [ ] Per-session output dirs (deferred — flat files + bench_run_id sufficient)
- [ ] Disk-free guard before download/load (deferred — manual sequencing suffices)
- [ ] Driver script for automatic handoff (deferred)

### Codex review findings (full output: /Users/macmini/projects/codex/llm_bench_mitigation_review_13apr2026.txt)
- Confirmed transport-error rows were silently scored as data; **fixed**
- Confirmed empty responses scored non-zero (codex said 12; verified 36 of 55); **fixed via _scoreable_text + is_invalid_result**
- Confirmed CSV/JSONL divergence; **mitigated via JSONL-first write order**
- Concurrent downloads-during-bench claim — **acknowledged** (no concurrent dl during current M2.7 bench), claim still untested for actual tok/s impact
- Per-session dirs / SQLite / driver script — deferred as overkill at current scale

### Done
- [x] Set up lm-evaluation-harness v0.4.11 (MMLU, GPQA, GSM8K, IFEval, HumanEval)
- [x] Set up LiveCodeBench (cloned to eval_frameworks/)
- [x] Set up DeepEval v3.9.5 (LLM-as-judge metrics)
- [x] Set up bigcode-evaluation-harness (cloned to eval_frameworks/)
- [x] Create unified run_eval.py wrapper
- [x] Create aggregate_results.py for cross-framework comparison
- [x] Add docs/evaluation_frameworks.md
- [x] Smoke test lm-eval IFEval against qwen3.5-27b (54% on 50 samples)
- [x] Qwen3.5-397B vMLX + PLD 78-run sweep → 32.3 tok/s, 3.95/5 overall

## Done (recent)
- [x] Run benchmark suite (156 runs completed)
- [x] Fix Qwen reasoning_content streaming capture
- [x] Add thinking_token_multiplier for thinking models
- [x] Write README.md
- [x] Add .gitignore and LICENSE (MIT)
- [x] Create docs/results.md with findings
