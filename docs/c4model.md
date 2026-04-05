# C4 Model — LLM Benchmark

## Level 1: System Context
- **User** runs benchmark CLI
- **LM Studio** serves local LLM models (OpenAI-compatible API, port 1234)
- **Claude API** provides judge scoring (external)

## Level 2: Container
- `run_bench.py` — benchmark runner, sends prompts to LM Studio, captures metrics, writes CSV/JSONL
- `judge_claude.py` — reads transcripts, sends blind pairwise comparisons to Claude API, writes judged CSV
- `cases.json` — prompt suite (26 prompts, 6 categories)
- `results/` — output directory (runs.csv, transcripts.jsonl, judged_results.csv)

## Level 3: Component
### run_bench.py
- `stream_completion()` — streaming OpenAI chat completion with timing capture
- `run_tool_use_case()` — multi-turn tool use handler with stub responses
- `score_exact/keywords/unit_tests/constraint_check/tool_trace()` — programmatic scorers
- `run_benchmark()` — main loop: warmup + scored repeats for each model x prompt

### judge_claude.py
- `call_judge()` — Claude API call with rubric system prompt
- `run_judge()` — pairwise blind evaluation with order randomization
- `print_summary()` — win/tie/loss aggregation by category

## Data Flow
```
cases.json → run_bench.py → LM Studio API → results/runs.csv + transcripts.jsonl
                                                    ↓
                                           judge_claude.py → Claude API → results/judged_results.csv
```
