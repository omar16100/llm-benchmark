# Long-Context Needle Benchmark

## Overview

`bench_longctx.py` measures two things that the main suite (`run_bench.py`) does not: long-context
**retrieval accuracy** and **prefill throughput** as context grows. It builds a synthetic haystack of a
target token size, inserts a unique "needle" fact at a chosen depth, asks the model to retrieve it, and
checks exact recall. It runs a grid over (context length x needle depth) and reports server-measured
prefill and decode throughput per cell.

It targets any OpenAI-compatible `/v1/chat/completions` endpoint (llama-server, LM Studio, vLLM, and
`mlx_lm.server`), so it works against the same servers as the rest of the suite. It depends only on
`requests`.

Three files make up this feature:

| File | Role |
|------|------|
| `bench_longctx.py` | The grid runner: needle construction, recall, per-cell metrics, JSON/CSV output. |
| `bench_long_prompt.py` | A simpler prompt-length sweep (no needle), kept for quick prefill/decode curves. |
| `bench_common.py` | Shared OpenAI-compatible chat call and server-timing extraction, used by both. |

## Details

### The grid and the needle
For each `(target_tokens, depth)` cell, `build_prompt()` fills a neutral filler corpus to roughly the
target size and splices in a unique needle of the form `IMPORTANT FACT: the authorization code for vault
seven is <CODE>`. The code is unique per cell (`QX{N}K-{depth}-{idx}`). The model is asked to return only
the code.

- **Sizing.** Word count is `target_tokens / tokens_per_word` (`--tokens-per-word`, default 1.4 for
  English prose; tune per tokenizer). The reported `prompt_tokens` is the true context size from the
  server `usage`, not the (possibly cache-reduced) evaluated count.
- **Depth.** `--depths` are percentages in `0..100`; the needle is placed at that fraction of the corpus.
- **Recall.** Exact match with non-code boundaries (regex `(?<![A-Z0-9-])CODE(?![A-Z0-9-])`), so a longer
  token that merely contains the code (for example `QX0K-50-700012`) does not count as a hit.

### Metrics (server-measured, not client wall clock)
`bench_common` reads llama.cpp / LM Studio's non-standard `timings` block:

- `prefill_tps` from `timings.prompt_per_second` (true prefill throughput). If the server does not expose
  `timings`, this is reported as `n/a` rather than derived from wall clock.
- `decode_tps` from `timings.predicted_per_second`.
- `ttft_s` (client-observed time to first non-empty token) only when `--stream` is set.
- `end_to_end_s` is the full request wall time, reported as such (never relabeled as prefill).

### Prompt caching
By default each cell sends `cache_prompt: false` (a llama.cpp field) so prefill is measured cold and is
comparable across cells. Without this, a server's cross-request prefix caching reuses the shared filler
and skews both the evaluated-token count and `prefill_tps`. Pass `--cache-prompt` to keep caching on.

### Server compatibility
- Base URL is accepted with or without a trailing `/v1`, matching the suite's `BENCH_BASE_URL` contract.
- `--no-thinking` sends `chat_template_kwargs.enable_thinking=false` (GLM and Qwen reasoning models). Omit
  it for servers that reject unknown template kwargs.
- Every row carries a free-text `--server-label` so result files record which server and version produced
  them.

### Output schema
Console prints one line per cell plus a recall summary. `--json` and `--csv` write rows with fields:
`target_tokens`, `depth_pct`, `prompt_tokens`, `prefill_tps`, `decode_tps`, `ttft_s`, `end_to_end_s`,
`recall` (PASS / FAIL / ERROR), `got` (first 48 chars of the answer), `server`.

## Usage

```bash
# Grid over three context sizes and three depths against llama-server holding GLM-5.2
uv run python bench_longctx.py \
  --base-url http://127.0.0.1:8081 --model glm-5.2 \
  --target-tokens 2000 8000 32000 --depths 25 50 90 \
  --no-thinking --stream --json out.json --csv out.csv

# Reuse the suite's env contract instead of flags
BENCH_BASE_URL=http://localhost:1234/v1 BENCH_API_KEY=lm-studio \
  uv run python bench_longctx.py --model my-model --target-tokens 4000 16000 --depths 50
```

Key flags: `--target-tokens` (one or more sizes), `--depths` (0..100), `--tokens-per-word`,
`--max-tokens`, `--no-thinking`, `--stream` (for TTFT), `--cache-prompt` (keep server caching on),
`--server-label`, `--json`, `--csv`.

Tests: `uv run pytest tests/test_bench_longctx.py -q` (no network; mocks the endpoint).

The simpler companion, `bench_long_prompt.py`, sweeps a fixed set of prompt lengths with no needle and
prints a prefill/decode/wall table, useful for a quick throughput curve.
