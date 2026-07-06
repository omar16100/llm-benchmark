# Kimi-Linear-48B-A3B native 1M-context verification

Date: 2026-07-06

## Context

The 2026-07-04 survey ([local_llm_1m_context_findings.md](local_llm_1m_context_findings.md))
concluded that no local model on this M3 Ultra had been shown to run at both native 1M
context and useful decode speed, and the best candidate (Nemotron-3-Nano) had an
*interrupted* 1M run with no valid measurement (see
`results/1m_candidates/2026-07-04_attempts.csv`).

This doc records the first model that actually completes a verified full-1M
needle-in-haystack run on this machine: **`mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit`**
(48B total, ~3B active; 27 layers = 20 KDA linear-attention + 7 MLA full-attention;
`rope_scaling: null`, `mla_use_nope: true`, `model_max_length: 1048576` in config.json).
Runtime: mlx-lm 0.31.3, isolated uv venv, Python 3.12. Machine: Mac Studio M3 Ultra,
512 GB, iogpu wired limit 490 GB.

Every number below is reproducible from the raw results file
`results/1m_candidates/2026-07-06_kimi_linear_48b_a3b_niah.json` (and the summary
`2026-07-06_kimi_linear_niah.csv`); the runner is `bench_niah_mlx.py` + `niah_haystack.py`.

## Changes

New method (mlx-lm in-process, not the HTTP `bench_longctx.py` path) because Kimi-Linear's
1M path needs the native mlx-lm `kimi_linear` model class, and because the load-bearing
check is a **truncation guard**: the harness passes pre-tokenized ids and asserts
`prompt_tokens == tokens built`. This catches silent context capping (LM Studio caps this
model at 4K because it ships `model_max_length` instead of `max_position_embeddings`).
Test: 8 unique needles at 8 depths (3%..90%) per length, one greedy generation, scored on
exact code-to-city association. Unit tests: `tests/test_niah_haystack.py`, 9/9 pass.

### Measured ladder (2K to 1M), all recall 8/8, all full-context verified

| Context | Prompt tokens | Saw all? | Recall | Prefill t/s | Decode t/s | Peak RAM | Prefill wall |
|---------|--------------:|:--------:|:------:|------------:|-----------:|---------:|-------------:|
| 2K control | 2,113 | yes | 8/8 | 1,201 | 86.3 | 54.1 GB | 3.0 s |
| 8K | 8,113 | yes | 8/8 | 2,518 | 87.4 | 55.0 GB | 4.5 s |
| 32K | 32,113 | yes | 8/8 | 2,077 | 82.3 | 58.6 GB | 16.4 s |
| 128K | 131,186 | yes | 8/8 | 986 | 58.6 | 74.3 GB | 134.4 s |
| 256K | 262,258 | yes | 8/8 | 179* | 14.7 | 95.1 GB | 1,470.5 s* |
| 512K | 524,404 | yes | 8/8 | 120 | 9.9 | 136.8 GB | 4,398.5 s |
| 1M | 1,048,692 | yes | 8/8 | 58 | 5.1 | 219.9 GB | 18,019.7 s |

\* 256K prefill/decode/wall are inflated by concurrent debug probing during that run; the
clean prefill trend is set by 128K, 512K, and 1M.

## Impact

- **Native 1M is real for this model.** Perfect 8/8 multi-needle recall at a genuine
  1,048,692-token prompt, with the guard proving full ingestion (no silent cap), coherent
  output. Retrieval does not degrade with length in this test. The 1M is architectural
  (NoPE MLA + KDA linear attention), not YaRN/DCA extension, so there is no positional
  extrapolation cliff.
- **It works because the mechanism is linear attention, not sparse attention.** mlx-lm
  implements the `kimi_linear` KDA path; the sparse-attention 1M paths that other models
  (GLM DSA, MiniMax MSA) rely on are still absent on Apple Silicon.
- **Memory is a non-issue:** 219.9 GB peak at 1M, well inside the 490 GB wired limit.
- **The cost is prefill time.** O(n^2) in the 7 MLA layers: ~5.0 h to ingest a 1M prompt
  (58 t/s), and decode falls to 5.1 t/s at 1M. So it clears the native-1M bar but not the
  100 t/s bar. Practical use: load a large corpus once and query deeply, not interactive.
- Refines the 2026-07-04 bottom line: the native-1M half of the target is now demonstrably
  achievable locally; the speed half is not, for the same bandwidth reasons.

## Reproduce

```bash
# needs mlx-lm, transformers<5, tiktoken, blobfile; weights at
# /Users/macmini/models/Kimi-Linear-48B-A3B-Instruct-8bit
python bench_niah_mlx.py --lengths control_2k,8k,32k,128k,256k   # ~30 min
python bench_niah_mlx.py --lengths 512k                          # ~75 min
python bench_niah_mlx.py --lengths 1m                            # ~5 h
```

Chart: [06072026_kimi_linear_scaling_curves.html](06072026_kimi_linear_scaling_curves.html).
