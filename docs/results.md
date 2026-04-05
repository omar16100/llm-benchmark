# Results: Gemma 4 31B vs Qwen 3.5 27B

## Overview
Head-to-head comparison on Mac Mini (Apple Silicon, 192GB RAM), both served via LM Studio on port 1234.

## Models

| | Gemma 4 31B (bf16) | Qwen 3.5 27B (Q8_0) |
|---|---|---|
| Params | 31B dense | 27B dense |
| RAM footprint | 61.4 GB | 28.6 GB |
| Context | 256K native | 262K (extensible to 1M) |
| Architecture | Sliding-window + global attention | Hybrid DeltaNet linear (3/4) + GQA |
| Thinking mode | No | Yes (reasoning_content) |

## Quality Scores (0-5)

| Category | Gemma4 | Qwen3.5 | Winner |
|---|---|---|---|
| reasoning | 3.60 | 2.60 | Gemma4 |
| coding | 5.00 | 5.00 | Tie |
| math | 1.25 | 0.75 | Gemma4 |
| instruction | 5.00 | 3.00 | Gemma4 |
| creative | 5.00 | 3.00 | Gemma4 |
| tool_use | 3.12 | 2.50 | Gemma4 |
| **Overall** | **3.77** | 2.79 | **Gemma4** |

## Performance (Median)

| | Gemma4 bf16 | Qwen3.5 Q8 |
|---|---|---|
| TTFT (s) | 1.84 | 0.66 |
| tok/s | 8.2 | 3.4 |
| Gen time (s) | 6.3 | 63.1 |
| Total time (s) | 8.0 | 65.6 |

## Head-to-Head Wins

Per-prompt comparison (best of 3 repeats each):

- **Gemma4**: 9 wins
- **Qwen3.5**: 4 wins
- **Ties**: 11

### By Category
| Category | Gemma | Qwen | Tie |
|---|---|---|---|
| reasoning | 2 | 2 | 1 |
| coding | 0 | 0 | 4 |
| math | 1 | 1 | 2 |
| instruction | 4 | 0 | 0 |
| creative | 1 | 0 | 2 |
| tool_use | 1 | 1 | 2 |

## Key Findings

### 1. Thinking Mode Is Expensive
Qwen 3.5's thinking mode makes it 8x slower end-to-end (65.6s vs 8.0s per prompt). The reasoning tokens share the max_tokens budget with the answer, requiring an 8x budget multiplier to avoid truncation.

### 2. Gemma Wins Instruction Following (4-0)
Gemma 4 reliably followed exact format constraints (JSON key ordering, word counts, bullet formats). Qwen's thinking added preamble that broke strict formats.

### 3. Coding Is a Tie
Both models scored 5.0 on all 4 coding prompts that passed unit tests. No differentiation at this difficulty level.

### 4. TTFT Advantage Goes to Qwen
Qwen starts streaming thinking tokens in 0.66s vs Gemma's 1.84s. Useful for perceived responsiveness, but total time is dominated by generation.

## Recommendation

**Use Gemma 4 31B as daily driver on this hardware.** Better quality across categories (except tied coding), 8x faster end-to-end, and dominates instruction following which matters for agent use cases.

**Switch to Qwen 3.5 27B when**: you specifically need long-context or linear-attention efficiency at very long contexts (>64K tokens).

## Caveats

This is a **deployment comparison**, not a pure architecture comparison. The bf16 vs Q8_0 setup folds together:
- Model quality
- Quantization impact
- Memory bandwidth utilization
- Runtime backend (both use llama.cpp via LM Studio)

With only 26 prompts, avoid broad claims. The wins within categories have small sample sizes (4-5 prompts each).
