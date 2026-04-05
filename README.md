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

# 5. (Optional) Claude-as-judge scoring
export ANTHROPIC_API_KEY=sk-ant-...
uv run python judge_claude.py
```

## What It Does

Runs 26 prompts across 6 categories against two models head-to-head. Captures:
- **Quality**: exact match, unit tests, constraint checks, tool-call validation, Claude-as-judge
- **Performance**: tokens/second, time-to-first-token, total time
- **Statistical comparison**: paired wins/losses per prompt

### Categories

| Code | Category | Count | Scoring |
|---|---|---|---|
| R | reasoning | 5 | exact, keyword |
| C | coding | 5 | unit tests, judge |
| M | math | 4 | exact |
| I | instruction | 4 | exact, constraint |
| W | creative | 4 | judge, constraint |
| T | tool_use | 4 | tool trace |

## Sample Results: Gemma 4 31B vs Qwen 3.5 27B

Tested on Mac Mini (192GB RAM, Apple Silicon):

| Category | Gemma4 (bf16) | Qwen3.5 (Q8) |
|---|---|---|
| reasoning | **3.60** | 2.60 |
| coding | **5.00** | **5.00** |
| math | **1.25** | 0.75 |
| instruction | **5.00** | 3.00 |
| creative | **5.00** | 3.00 |
| tool_use | **3.12** | 2.50 |
| **Overall** | **3.77** | 2.79 |

**Performance** (median):
- Gemma4 bf16: 8.2 tok/s, 8.0s total per prompt
- Qwen3.5 Q8: 3.4 tok/s, 65.6s total per prompt (thinking mode overhead)

**Head-to-head**: Gemma4 wins 9, Qwen3.5 wins 4, Ties 11

Full analysis in [docs/results.md](docs/results.md).

## Architecture

```
cases.json          26 prompts with metadata, expected answers, constraints
run_bench.py        Benchmark runner: streams to LM Studio, scores, writes CSV
judge_claude.py     Blind pairwise Claude-as-judge scorer
tests/              Unit tests for scoring functions (19 tests)
results/
  runs.csv          Per-run metrics (timing, scores)
  transcripts.jsonl Full responses + reasoning traces
```

### Streaming & Thinking Models

The runner uses raw `httpx` streaming to capture `reasoning_content` (Qwen thinking mode) separately from `content`. Thinking models get an 8x `max_tokens` multiplier since reasoning tokens share the budget with the final answer.

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
- `exact` — requires `expected` field, checks exact match (5pt) or substring (3pt)
- `judge_keyword` — requires `expected_keywords` list
- `unit_tests` — requires `test_code` (Python assertions)
- `constraint_check` — requires `constraints` dict (word_count, required_words, forbidden_chars, line_count, word_range, etc.)
- `tool_trace_exact` / `tool_trace_judge` — for tool_use cases with mock responses
- `judge` / `judge_constraint` — defers to Claude judge

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
        "thinking_token_multiplier": 8,
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

This is a **deployment comparison**, not pure architecture. Comparing bf16 vs Q8_0 folds together:
- Model quality
- Quantization precision
- Memory bandwidth
- Runtime stack

Reported conclusions should be framed as "practical winner on this hardware," not "model A is better than model B."

## License

MIT
