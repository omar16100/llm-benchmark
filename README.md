# llm-benchmark

Local LLM benchmark suite comparing models served via LM Studio OpenAI-compatible API. Designed for Apple Silicon, but works with any LM Studio endpoint.

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/omar16100/llm-benchmark
cd llm-benchmark
uv sync

# 2. Load your models in LM Studio
lms server start -p 1234
lms load your-model-1 --identifier model-1
lms load your-model-2 --identifier model-2

# 3. Edit MODELS dict in run_bench.py to match your loaded identifiers

# 4. Run benchmark
uv run python run_bench.py

# 5. (Optional) Claude-as-judge scoring: shells out to the `claude` CLI
#    (`claude -p`), which uses that CLI's own login
uv run python judge_claude.py
```

## What It Does

Runs 26 prompts across 6 categories against each model configured in `MODELS` (`run_bench.py`). Captures:
- **Quality**: exact match, unit tests, constraint checks, tool-call validation, Claude-as-judge
- **Performance**: time-to-first-token, generation time, total time, approximate tokens/second
- **Pairwise judging**: `judge_claude.py` compares two models blind, per prompt (A, B, or tie)

### Categories

| Code | Category | Count | Scoring |
|---|---|---|---|
| R | reasoning | 5 | exact, keyword |
| C | coding | 5 | unit tests, judge |
| M | math | 4 | exact |
| I | instruction | 4 | exact, constraint |
| W | creative | 4 | judge, constraint |
| T | tool_use | 4 | tool trace |

## Sample Results (run data from 20 Apr 2026)

The table below is generated from [data/runs_20apr2026.csv](data/runs_20apr2026.csv), a copy of a local `results/runs.csv` (705 rows) exported with `scripts/runs_data.py` (the only change is that absolute model paths are reduced to their directory names):

```bash
uv run python scripts/runs_data.py summary data/runs_20apr2026.csv
```

| model_label | quant | valid/total | scored | mean /5 | reasoning | coding | math | instruction | creative | tool_use | median total_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| supergemma4_26b_mlx4_v2 | MLX-4bit | 54/78 | 48 | 4.41 | 5.00 | 5.00 | 3.33 | 5.00 | 4.78 | 3.75 | 18.0 |
| glm_51_mlx_36bit | MLX-3.6bit | 75/78 | 69 | 4.37 | 4.00 | 5.00 | 3.25 | 5.00 | 5.00 | 4.17 | 131.3 |
| minimax_m27_mlx4_mxfp4 | MLX-4bit-mxfp4 | 69/78 | 63 | 4.16 | 2.80 | 5.00 | 3.33 | 5.00 | 5.00 | 4.06 | 18.9 |
| qwen35_122b_a10b_q8 | GGUF-Q8_0 | 72/78 | 66 | 4.09 | 3.00 | 5.00 | 2.50 | 5.00 | 5.00 | 3.76 | 61.4 |
| minimax_m27_mlx4_mxfp4_mlxlm | MLX-4bit-mxfp4 | 60/78 | 54 | 3.73 | 0.60 | 5.00 | 3.33 | 5.00 | 4.89 | 2.81 | 15.6 |
| qwen35_122b_a10b_q4km | GGUF-Q4_K_M | 51/78 | 45 | 3.58 | 2.50 | 5.00 | 0.00 | 5.00 | 4.56 | 3.76 | 41.1 |
| qwen35_397b_mlx4 | MLX-Q4 | 57/78 | 51 | 2.78 | 2.50 | 5.00 | 1.00 | 5.00 | 4.78 | 0.00 | 41.7 |
| glm_47_flash_q4km_32k | GGUF-Q4_K_M | 30/78 | 27 | 2.39 | 2.75 | n/a | n/a | 0.00 | 3.67 | 2.50 | 48.1 |
| glm_47_flash_q4km | GGUF-Q4_K_M | 24/78 | 21 | 2.14 | n/a | n/a | n/a | 0.00 | 3.33 | 2.50 | 8.9 |
| qwen35_27b_claude_opus_distilled_q8 | GGUF-Q8_0 | 3/3 | 3 | 0.00 | n/a | n/a | 0.00 | n/a | n/a | n/a | 50.3 |

How to read it:
- `mean /5` and the category columns average `score_raw` over valid rows that have a programmatic score. Rows scored `needs_judge` have none, so `scored` can be lower than `valid`; `n/a` means no scored rows in that category.
- A row is invalid when the response errored, was empty, or was length-truncated on a non-creative prompt (`is_invalid_result` in `run_bench.py`). A full run is 26 prompts x 3 repeats = 78 rows; `qwen35_27b_claude_opus_distilled_q8` is a 3-row smoke run.
- `median total_s` is the median of the per-row `total_s` over valid rows (wall time of one scored run; for tool-use cases, the sum over its turns). Throughput is not shown because `tok_per_s` is not comparable across these runs: rows written before the runner started counting reasoning characters (commit fd9c240) under-count thinking models (see [docs/results.md](docs/results.md)).
- Models ran on different runtimes (LM Studio, `mlx_lm.server`) and quantizations, so this is a deployment comparison, not an architecture ranking (see Fair Comparison Caveats below). Hardware is not recorded in the run data.

Per-model notes and run history: [docs/results.md](docs/results.md).

## Architecture

```
cases.json            26 prompts with metadata, expected answers, constraints
run_bench.py          Benchmark runner: streams to LM Studio, scores, writes CSV
judge_claude.py       Blind pairwise Claude-as-judge scorer
bench_longctx.py      Long-context needle + prefill-throughput eval (any OpenAI-compatible server)
bench_long_prompt.py  Prompt-length sweep; prefill/decode tok/s from server timings
bench_common.py       Shared OpenAI-compatible chat call + server timing extraction
scripts/runs_data.py  Export a publishable runs CSV and generate the sample results table
data/                 Published run data (runs_20apr2026.csv)
tests/                Unit tests: 46 run without a model or server; the DeepEval tests
                      need a live endpoint and the NIAH haystack tests need tiktoken
                      plus a local tokenizer, so they skip otherwise
results/              Local outputs, gitignored
  runs.csv          Per-run metrics (timing, scores)
  transcripts.jsonl Full responses + reasoning traces
```

### Streaming & Thinking Models

The runner uses raw `httpx` streaming to capture reasoning (`reasoning_content` from LM Studio, `reasoning` from `mlx_lm.server`) separately from `content`. Every scored request (including each tool-use turn) uses `MAX_RESPONSE_TOKENS = 32768` because reasoning tokens share the budget with the final answer; warmup calls use `min(case max_tokens, 64)`. The `max_tokens` column in `runs.csv` records the case's nominal budget from `cases.json`.

## Long-Context Needle Benchmark

`bench_longctx.py` measures long-context retrieval and prefill throughput against any OpenAI-compatible endpoint (llama-server, LM Studio, vLLM). It builds a haystack of roughly N tokens, inserts a unique needle at each requested depth, asks for it back, and checks exact recall. Prefill and decode tok/s come from the server `timings` block when present (true server-side throughput), so client wall clock is never reported as prefill.

```bash
uv run python bench_longctx.py \
  --base-url http://127.0.0.1:8081 --model glm-5.2 \
  --target-tokens 2000 8000 32000 --depths 25 50 90 \
  --no-thinking --stream --json out.json --csv out.csv
```

Each result row records: `target_tokens`, `depth_pct`, `prompt_tokens`, `prefill_tps`, `decode_tps`, `ttft_s` (with `--stream`), `end_to_end_s`, `recall` (PASS/FAIL), and a free-text `server` label. Use `--no-thinking` for GLM and Qwen reasoning models (it sends `chat_template_kwargs.enable_thinking=false`); omit it for servers that reject unknown template kwargs. `bench_common.py` holds the shared endpoint call and timing extraction, reused by `bench_long_prompt.py`.

By default each cell disables server prompt caching (`cache_prompt: false`, llama.cpp) so prefill is measured cold and is comparable across cells; pass `--cache-prompt` to keep caching on. `prompt_tokens` is the full context size from `usage`, not the server-evaluated subset. The base URL is accepted with or without a `/v1` suffix and falls back to `BENCH_BASE_URL` / `BENCH_API_KEY` (the same contract as `run_bench.py`). Full reference: [docs/longctx_bench.md](docs/longctx_bench.md).

## Adding New Prompts

Edit `cases.json`:

```json
{
  "id": "R6",
  "category": "reasoning",
  "scoring": "exact",
  "temperature": 0.0,
  "max_tokens": 256,
  "prompt": "Your question here",
  "expected": "expected answer"
}
```

Scoring types:
- `exact`: requires `expected` field, checks exact match (5pt) or substring (3pt)
- `judge_keyword`: requires `expected_keywords` list
- `unit_tests`: requires `test_code` (Python assertions)
- `constraint_check`: requires `constraints` dict (word_count, required_words, forbidden_chars, line_count, word_range, etc.)
- `tool_trace_exact` / `tool_trace_judge`: for tool_use cases with mock responses
- `judge`: deferred to the Claude judge (`score_type` `needs_judge`)
- `judge_constraint`: constraint score when `constraints` is present, otherwise deferred to the Claude judge

## Adding New Models

Edit `MODELS` dict in `run_bench.py`:

```python
MODELS = {
    "my_model": {
        "served_model": "lmstudio-identifier",
        "quant": "Q4_K_M",
        "thinking": False,
    },
    "thinking_model": {
        "served_model": "another-id",
        "quant": "Q8_0",
        "thinking": True,
    },
}
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Methodology

- 2 warmup calls discarded, 3 scored repeats per prompt per model
- Deterministic tasks: `temperature=0, top_p=1, seed=42`
- Creative tasks: `temperature=0.8, top_p=0.95, seeds 41/42/43`
- Concurrency: 1 request at a time

### Fair Comparison Caveats

This is a **deployment comparison**, not pure architecture. Comparing models at different quantizations (bf16, Q8_0, Q4, MLX 4-bit) and runtimes folds together:
- Model quality
- Quantization precision
- Memory bandwidth
- Runtime stack

Reported conclusions should be framed as "practical winner on this hardware," not "model A is better than model B."

## License

MIT
