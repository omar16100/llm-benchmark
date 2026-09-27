# README results backing, CI, and 256K NIAH harnesses (27 Sep 2026)

## Context

An open source maintenance pass found that the README's "Sample Results: Gemma 4 31B vs
Qwen 3.5 27B" table had no backing file in the repository: `results/` is gitignored, and the
`.gitignore` lines meant to re-include `results/runs.csv` and `results/transcripts.jsonl`
never worked (git cannot re-include a file inside an excluded directory). The README also
stated a hardware line ("Mac Mini (192GB RAM)") that conflicts with other docs, a head-to-head
tally (9 + 4 + 11 = 24) that does not add up to the 26 prompts it describes, a stale test count
(39), and a thinking-model max_tokens multiplier the runner no longer uses. The repo had no CI.

## Changes

Part 1 (PR 1: results backing and CI):
- The local run data (`results/runs.csv`, 705 rows, last written 20 Apr 2026) contains no
  Gemma 4 31B or Qwen 3.5 27B rows, and no other local file or commit does either (the numbers
  were first added in the initial commit, with no data file). The Gemma 4 vs Qwen 3.5 table,
  its performance lines, the head-to-head tally, and the hardware line were removed.
- `scripts/runs_data.py` exports a publishable copy of `runs.csv` (only change: absolute
  `served_model` paths reduced to directory names; nothing else in the file is personal) to
  `data/runs_20apr2026.csv`, and generates the README's new sample results table from it.
  `tests/test_runs_data.py` asserts the README table is exactly what the script produces
  from the committed data, so the table cannot drift from its source.
- Hardware is not recorded anywhere in the run data (no column in `runs.csv`, and the
  lm-eval results record no hardware), so the README makes no hardware claim.
- README corrections: max_tokens policy (`MAX_RESPONSE_TOKENS = 32768` for every scored request),
  judge invocation (`claude -p`, not an API key), pairwise judging description, test count,
  em dashes removed.
- `.gitignore`: dead `!results/...` negations replaced by a comment pointing at `data/`.
- CI: `.github/workflows/ci.yml` (ubuntu, `astral-sh/setup-uv@v10.2.0`,
  `uv run --locked pytest tests/`). Tests that need a model directory, a live server, or MLX
  skip themselves.
- GitHub description: drop "Includes Gemma 4 vs Qwen 3.5 comparison" (no backing data).

Part 2 (PR 2: 256K NIAH harnesses):
- Carried over the uncommitted `bench_niah_mlx.py` changes (decode slope over per-token
  timestamps, `<think>` split before scoring, KV-cache quantization flags, alternate loaders)
  and the untracked harness files, tests moved under `tests/`.
- Absolute local paths replaced with flags, environment variables, or paths relative to
  `__file__` (`NIAH_MODEL_DIR`, `BONSAI_FAST_DIR`, `GLM53_FLASH_MLX_DIR`, `NIAH_256K_PROMPT`,
  `LCB_DIR`, `NEMOTRON_MLX8_DIR`, `NIAH_TOKENIZER_DIR`); all default hosts stay `127.0.0.1`.
- Comments and docstrings that cited results from the private 256K comparison (model scores,
  memory and speed figures, internal finding and cell numbers) were rewritten to state the
  reasoning without those numbers. Public facts kept were rechecked: Qwen2.5-7B-Instruct-1M
  `max_position_embeddings` = 1,010,000 (its `config.json`), LiveCodeBench `release_v6` = 1055
  problems dated 2023-05-07 to 2025-04-06, dataset last modified 2025-06-05.
- A tautological test that only asserted private memory figures against each other was
  dropped; `tests/test_niah_clients.py` (scorer, regex, and slope agreement across clients, env
  and loopback checks) and `tests/test_bench_niah_mlx.py` (marker `mlx`) were added.
- Documented in `docs/niah_harnesses.md` and a README section.

## Impact

Every number in the README results section now traces to `data/runs_20apr2026.csv` through a
committed script and a test. `docs/results.md` keeps its per-model notes; its `tok/s (real)`
column was derived from `results/transcripts.jsonl` (16 MB, not published).

## Status log

- 27/09/2026: part 1 implemented; 46 tests pass locally, 11 skipped (10 DeepEval tests without an endpoint, plus the NIAH haystack module without tiktoken).
- 27/09/2026: part 1 merged (PR #4), CI green. Part 2: 131 tests pass locally, 12 skipped.
- 27/09/2026: Codex review applied: docs/results.md per-category leaders regenerated from the data, SuperGemma4 coding/tool_use mix-up and stale "wins every category" / "no valid data yet" lines corrected; judge_constraint, warmup token cap, and fsync wording corrected.
- 27/09/2026: part 2 Codex reviews (3 rounds) applied: test fixtures replaced with synthetic
  values (no private-run numbers), `LCB_DIR` and `NEMOTRON_MLX8_DIR` made absolute before use,
  per-harness guard / decode / scorer differences documented instead of claiming uniformity,
  `make_1m_prompt.py` depth behaviour documented (evenly spaced, not copied from the source),
  and `lcb_local_runner.py --limit` (previously parsed but ignored) now errors instead of
  silently running the full problem set.
- 27/09/2026: `mlx_raw_niah_client.py` now records `raw_prompt: false` and a matching engine
  label when `--chat-template` is used. Left as is (owner's choice): `macmon_clamp_report.py`
  uses the upper-middle sample as the median and `round` (not `ceil`) for nearest-rank p10.
