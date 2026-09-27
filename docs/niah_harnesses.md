# NIAH harnesses for long-context runs (256K and 1M)

## Overview

A set of harnesses for multi-needle needle-in-a-haystack (NIAH) runs at 256K tokens and
beyond against local runtimes. They share one needle format and a common approach to scoring,
decode measurement, and guards; the per-harness differences are listed below and matter when
comparing a result from one engine with another. They
were built for a cross-engine 256K comparison kept in a private repository; none of its
results are published here, and nothing in this doc should be read as a benchmark claim.

Also included: a LiveCodeBench pass@1 wrapper with the same guard philosophy, a GPU clock
and memory reporter for macmon traces, and a 1M-token prompt builder.

## Details

### Harnesses

| file | engine and transport | prompt form | needs |
|---|---|---|---|
| `bench_niah_mlx.py` | mlx-lm, in-process | haystack built by `niah_haystack.py`, model chat template | `mlx`, `mlx-lm` |
| `mlx_raw_niah_client.py` | mlx-lm, in-process | raw prompt file (optional `--chat-template`) | `mlx`, `mlx-lm` |
| `llamacpp_niah_client.py` | `llama-server` over HTTP (`/completion`) | raw prompt file | stdlib only |
| `omlx_niah_client.py` | oMLX server, `/v1/completions` streaming | templated haystack (same code path as `bench_niah_mlx.py`) or `--prompt-file` raw | `mlx-lm` (for the tokenizer) |
| `dspark_niah_client.py` | mlx-dspark server, `/v1/chat/completions` streaming | raw prompt file, chat template applied by the server | stdlib only |
| `glm53_niah_client.py` | `glm53_flash_mlx` fork of mlx-vlm (for `glm5_next` models), in-process | raw prompt file through the model's chat template | `mlx`, `mlx-vlm`, the fork |
| `make_1m_prompt.py` | none | builds a ~995K-token raw prompt from an existing prompt's needles | `transformers` |
| `macmon_clamp_report.py` | none | summarises a macmon JSONL trace | stdlib only |
| `lcb_local_runner.py` + `lcb_guards.py` | LiveCodeBench against a local OpenAI-compatible server | LiveCodeBench problems | a LiveCodeBench clone and its own venv |
| `scripts/run_nemotron_mlx8_token_test.sh` | `mlx_lm.benchmark` | synthetic 100K-token prompt | `mlx-lm` in `.venv` |

All default endpoints are on `127.0.0.1` (llama-server 8081, oMLX 8082, LiveCodeBench target
8090, mlx-dspark 8091).

### Prompt format and scoring

Every needle is the sentence `IMPORTANT RECORD: the secret access code for <City> is <code>.`
with an 8-digit code, matched by the same regex in every raw-prompt client. `niah_haystack.py`
builds a token-accurate haystack with 8 needles at depths 0.03 to 0.90. `make_1m_prompt.py`
copies the needle wording and codes from a source prompt into a ~995K-token prompt with its own
filler and question and the needles at evenly spaced depths (1/9 to 8/9), so it does not keep a
source prompt's depths; it refuses to exceed 1,010,000 tokens (the `max_position_embeddings`
in Qwen2.5-7B-Instruct-1M's `config.json`).

Two scores per run: `retrieval` (the code appears anywhere in the output) and `assoc`. The
raw-prompt clients (`mlx_raw`, `llamacpp`, `omlx`, `dspark`, `glm53`) count `assoc` only when
the code and its city are on the same line; they keep their own copies of that scorer so each
file runs standalone, and `tests/test_niah_clients.py` pins that the copies agree.
`bench_niah_mlx.py` uses `niah_haystack.score_answer` instead, which also accepts the code
within 60 characters after a city mention across a line break, so its `assoc` can be higher on
the same text (for example `Reykjavik` and its code on consecutive lines). Compare `assoc`
across the two families with that in mind.

### Decode metric and timestamps

| harness | decode figure | per-token timestamps saved |
|---|---|---|
| `bench_niah_mlx.py` | mlx-lm's mean `generation_tps` and a least-squares slope (`generation_tps_slope`) from the same generation | yes (`token_times`) |
| `mlx_raw_niah_client.py`, `glm53_niah_client.py` | least-squares slope over token times after prefill | yes (`token_times`) |
| `omlx_niah_client.py`, `dspark_niah_client.py` | least-squares slope over streamed arrival times after dropping `--warmup` tokens (oMLX drops at most enough to keep 3 points; dspark returns 0 when too few remain, and also records the server's own figure) | no |
| `llamacpp_niah_client.py` | server timings: `predicted_n / (predicted_ms / 1000)` tokens/s | no |

The slope is mean-centered (so epoch-scale timestamps do not cancel) and excludes prefill
instead of dividing total tokens by total time. `llamacpp_niah_client.py` is the exception: its
figure comes from the server's own timings block, not a slope.

### Guards

| harness | prompt-token truncation check | reasoning-truncation check |
|---|---|---|
| `bench_niah_mlx.py` | raises when mlx-lm's `prompt_tokens` differs from the tokens built (the length is recorded as an error) | when the PROMPT opens `<think>` and generation hit `--max-gen` without `</think>`: `answer_inconclusive`, scored 0; scoring always uses the text after the last `</think>`, with the whole-output score kept alongside |
| `mlx_raw_niah_client.py` | `truncated` when mlx-lm's served count differs from the tokenizer count; exit code 3 | output-keyed: opened `<think>`, never closed it, hit the cap: `answer_inconclusive` flag (the numeric score is still recorded) |
| `glm53_niah_client.py` | none (records the tokenizer count only) | output-keyed flag, as in `mlx_raw_niah_client.py` |
| `llamacpp_niah_client.py` | `truncated` when `/tokenize` count differs from `timings.prompt_n`; exit code 3 | none |
| `omlx_niah_client.py` | `truncated` when usage reports a different prompt count than was built (`guard: direct`); `None` and `INDIRECT` when the server sends no usage; exit code 3 on a mismatch | none |
| `dspark_niah_client.py` | records the served prompt count and `guard: direct` when usage is present; there is no client-side count, so `truncated` is `False` whenever usage is present and `None` otherwise | none; scores content plus reasoning text together |

The oMLX and mlx-dspark clients request `stream_options.include_usage` and retry without it if
the server rejects the field. The output-keyed rule exists because a model can open `<think>`
with no chat template at all; the prompt-keyed rule in `bench_niah_mlx.py` misses that case.
None of the clients refuses to write a score: check `truncated`, `guard`, and
`answer_inconclusive` before using one.

- **LiveCodeBench.** `lcb_guards.py` separates truncated, inconclusive, no-extractable-answer,
  and refusal completions from real failures, and reports pass@1 alongside pass@1 excluding
  truncated and excluding no-answer completions.
  `lcb_local_runner.py` counts null and empty-extraction completions and can log per-request
  `finish_reason` and token counts (`--metadata-out`) so a truncated run can be repaired.
  Note: LiveCodeBench `release_v6` holds 1055 problems dated 2023-05-07 to 2025-04-06 (checked
  27 Sep 2026), so it cannot score recent models only on post-cutoff problems.

### Environment variables

| variable | used by | meaning |
|---|---|---|
| `NIAH_MODEL_DIR` | `bench_niah_mlx.py` | default for `--model-dir` |
| `BONSAI_FAST_DIR` | `bench_niah_mlx.py --bonsai-fast` | directory containing `bonsai_fast.py` |
| `GLM53_FLASH_MLX_DIR` | `glm53_niah_client.py` | default for `--fork-dir` (checkout of the fork) |
| `NIAH_256K_PROMPT` | `make_1m_prompt.py` | default for `--source-prompt` |
| `LCB_DIR` | `lcb_local_runner.py` | LiveCodeBench clone (default `eval_frameworks/LiveCodeBench`) |
| `NEMOTRON_MLX8_DIR` | `scripts/run_nemotron_mlx8_token_test.sh` | model directory (required) |
| `NIAH_TOKENIZER_DIR` | `tests/test_niah_haystack.py` | Kimi-Linear model dir for the tokenizer tests |

### Tests

`tests/test_niah_guards.py`, `tests/test_niah_clients.py`, `tests/test_lcb_guards.py`, and
`tests/test_macmon_clamp_report.py` need no model, GPU, or network and run in CI.
`tests/test_bench_niah_mlx.py` is marked `mlx` and skips unless mlx and mlx-lm are installed.

## Usage

Build a raw 256K prompt file with the repo's haystack builder (any tokenizer works; the token
count is measured with it):

```bash
uv run --with mlx-lm python - <<'PY'
from pathlib import Path
from mlx_lm.utils import load_tokenizer
import niah_haystack
tok = load_tokenizer(Path("/path/to/model"))
h = niah_haystack.build_haystack(tok, 262_144, seed=20260706)
Path("niah_256k_prompt.txt").write_text(h.prompt_text)
PY
```

Run the same file through different engines:

```bash
# mlx-lm in-process, raw prompt
uv run --with mlx-lm python mlx_raw_niah_client.py --model-dir /path/to/model \
  --prompt-file niah_256k_prompt.txt --out mlx_raw.json

# llama-server already running on 127.0.0.1:8081
uv run python llamacpp_niah_client.py --prompt-file niah_256k_prompt.txt --out llamacpp.json

# templated haystack ladder, in-process
NIAH_MODEL_DIR=/path/to/model uv run --with mlx-lm python bench_niah_mlx.py \
  --lengths control_2k,128k,256k --max-gen 6000
```

Scale to ~1M tokens and summarise a macmon trace:

```bash
uv run python make_1m_prompt.py --model-dir /path/to/model \
  --source-prompt niah_256k_prompt.txt --out niah_1m_prompt.txt
python3 macmon_clamp_report.py macmon.jsonl my_run
```

Use a generous `--max-gen` / `--max-tokens` for reasoning models; where a harness has a
reasoning-truncation check, it flags a run whose budget ran out mid-reasoning (see Guards).
