# Evaluation Frameworks

## Overview
This project uses multiple evaluation frameworks to benchmark LLMs served via LM Studio on a Mac Studio M3 Ultra 512GB. Results from all frameworks are aggregated via `aggregate_results.py`.

## Frameworks

### 1. Custom Benchmark (`run_bench.py`)
- 26 prompts across 6 categories: reasoning, coding, math, instruction, creative, tool_use
- Scoring: exact match, keyword, unit tests, constraint checks, tool trace, Claude-as-judge
- Captures timing: TTFT, generation speed, tok/s

### 2. lm-evaluation-harness (EleutherAI)
- Repo: https://github.com/EleutherAI/lm-evaluation-harness
- Powers HuggingFace Open LLM Leaderboard
- Tasks: MMLU, GPQA Diamond, GSM8K, IFEval, HumanEval, MATH
- Connection: `local-completions` model type → LM Studio API at localhost:1234

### 3. LiveCodeBench
- Repo: https://github.com/LiveCodeBench/LiveCodeBench
- 1000+ contamination-free coding problems from LeetCode/AtCoder/CodeForces
- Avoids training-set contamination that affects HumanEval scores
- Cloned to `eval_frameworks/LiveCodeBench/`

### 4. DeepEval
- Repo: https://github.com/confident-ai/deepeval
- 14+ LLM-as-judge metrics: answer relevancy, hallucination, G-Eval task completion
- Pytest-style test runner
- Test cases in `tests/test_deepeval.py`

### 5. bigcode-evaluation-harness
- Repo: https://github.com/bigcode-project/bigcode-evaluation-harness
- Code-specific: HumanEval+, MBPP+, DS-1000, APPS
- More thorough code evaluation than custom test cases
- Cloned to `eval_frameworks/bigcode-evaluation-harness/`

## Usage

```bash
# unified runner
uv run python run_eval.py --framework lm-eval --model qwen35_122b_a10b_q8 --tasks mmlu,gsm8k
uv run python run_eval.py --framework deepeval --model qwen35_122b_a10b_q8
uv run python run_eval.py --framework bigcode --model qwen35_122b_a10b_q8 --tasks humaneval
uv run python run_eval.py --framework livecodebench --model qwen35_122b_a10b_q8

# list models/tasks
uv run python run_eval.py --list-models
uv run python run_eval.py --list-tasks --framework lm-eval

# aggregate all results
uv run python aggregate_results.py
```

## Details
- Results saved to `results/<framework>/<model>/<timestamp>/`
- `aggregate_results.py` merges all into `results/aggregate.csv`
- Code execution benchmarks (bigcode, LiveCodeBench) execute model-generated code — consider Docker sandboxing for safety
