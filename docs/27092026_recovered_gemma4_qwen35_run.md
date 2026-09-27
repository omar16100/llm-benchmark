# Recovered data: Gemma 4 31B vs Qwen 3.5 27B run of 5 Apr 2026 (27 Sep 2026)

## Context

On 5 Apr 2026 `run_bench.py` compared Gemma 4 31B (bf16 GGUF) with Qwen 3.5 27B (Q8_0 GGUF),
both served by LM Studio. The README of the initial commit (`ca4a63a`, 16:07 +0800) and the blog
post <https://omarshabab.com/llm-benchmark/> quoted quality scores, a head-to-head tally and
speed medians from that run. The run's `results/runs.csv` and `results/transcripts.jsonl` were
never committed: `results/` is gitignored, and `run_bench.py` at the time opened both files in
`"w"` mode, so later runs overwrote them. The README table was removed on 27 Sep 2026 for lack of
backing data (see [27092026_readme_results_backing_plan.md](27092026_readme_results_backing_plan.md)).

This doc records what could be recovered from logs that survived on the benchmark machine, how it
was recovered, how it compares with every published claim, and what cannot be recovered.

## Changes

- `scripts/recover_lmstudio_log.py`: parses LM Studio server logs and the harness's own
  `bench.log`, keeps only requests whose messages are exactly what LM Studio logs for a
  `cases.json` case and that the harness log ties to a harness call, pairs every request with its
  llama.cpp slot timings, and scores a generation only where the logs determine the score.
- `scripts/recovered_run_summary.py`: prints the tables below from the committed CSVs, or
  writes them into this doc.
- `data/recovered_05apr2026_gemma4_qwen35.csv`: one row per benchmark HTTP request (647 rows:
  warmups, scored turns, 2 ad hoc requests).
- `data/recovered_05apr2026_gemma4_qwen35_generations.csv`: one row per scored generation
  (329 rows: 156 in each complete run, 17 in two aborted runs).
- `data/bench_log_05apr2026.txt`: the 5 Apr 2026 lines of the harness `bench.log`, verbatim.
- Tests: `tests/test_recover_lmstudio_log.py` (synthetic logs, including a non-benchmark request
  whose content must not reach the outputs, and the edge cases below) and
  `tests/test_recovered_run_summary.py` (pins the numbers quoted here and in the corrected post,
  re-derives the `bench.log` fields of the generations CSV from the committed excerpt, and checks
  that the tables below are the script's output).

## Sources checked

| Source | Result |
|---|---|
| LM Studio server log `2026-04-05.1.log` | Request bodies and llama.cpp timings for every benchmark request. No response text: streamed responses are not logged. Also holds unrelated requests from other local clients (not extracted). |
| LM Studio server logs `2026-04-06.1.log`, `2026-04-07.1.log` | Chat requests to `qwen3.5-27b`, none of them a `cases.json` prompt (lm-eval traffic). No rerun of this benchmark. |
| Harness `bench.log` (gitignored, local) | Millisecond timestamps for every warmup group and scored generation of the four 5 Apr invocations, and the scoring messages `run_bench.py` logged. The 5 Apr part is committed as `data/bench_log_05apr2026.txt`. |
| Local `results/` (`runs.csv`, its `.bak` copies, `transcripts.jsonl` and its backups, `snapshots/`, `judged_results.csv`, `lm-eval/`) | No Gemma 4 31B or Qwen 3.5 27B rows. |
| Git history of this repo (all refs, reflog, stash, unreachable objects) | The model labels appear only in the `MODELS` dict of `ca4a63a:run_bench.py`. No results. |
| Other local project folders and agent session records | No per-call data from this run. The session that ran the benchmark and wrote the README and post has no surviving transcript. |

## Provenance

These are local observations: they come from files that are not published (the server logs also
hold unrelated traffic from other local clients) and cannot be re-derived from this repo. The
committed CSVs and excerpt are what the published tables are computed from.

| Local file (27 Sep 2026) | Size | sha256 |
|---|---|---|
| `~/.lmstudio/server-logs/2026-04/2026-04-05.1.log` | 3,156,209 bytes, 46,501 lines | `4857323cf57fbdb203f557d632ea03176dc1da4b9cfde53b048f2bee785cfbde` |
| `~/.lmstudio/server-logs/2026-04/2026-04-06.1.log` | 7,171,900 bytes | `2e0429e135e565abe04b73ec380ebf27df1685c99709dc0960f22e31f28cc1b3` |
| `~/.lmstudio/server-logs/2026-04/2026-04-07.1.log` | 3,935,808 bytes | `c4273cfa4b050f95aa05cadce62b279bc28ccbeeaf2199f1852d428b723acd44` |
| `bench.log` in the working copy (last written 20 Apr 2026) | 6,358 lines | `14af6289e502045bb2931a58ed24a2bfbcf8ef1c151241d5d6c282fd4f6b3213` |

Committed: `data/bench_log_05apr2026.txt` is lines 1 to 1133 of that `bench.log` (1,133 lines,
sha256 `830ae085f68364fae0c4e9b48b04b6edd48d984efc22ea8df7bdaf3460b00dc4`). It holds only
harness log lines: benchmark calls, their scoring messages, and three httpx lines of ad hoc
requests made through the same logger (localhost URL and a timestamp, no content).

Local observations from the 5 Apr server log:
- 653 chat requests: 647 benchmark requests (the calls CSV) and 6 others (2 before the first run,
  4 short requests to `qwen3.5-27b` between runs), which are only counted. The 6 and 7 Apr logs
  hold 317 and 358 chat requests, none a benchmark request.
- Benchmark requests by line (the `lmstudio_line` column): run A 1666 to 22130, run B 22433 to
  23540, run C 23710 to 24076, ad hoc 24148 and 24228, run D 24324 to 46386. Times in both logs
  are local time (UTC+08:00).
- Model load lines: `using device MTL0 (Apple M3 Ultra)`, `recommendedMaxWorkingSetSize =
  498216.21 MB`, both models loaded with `n_ctx = 32768` and `n_parallel = 4`, server on port
  1234, GGUF file sizes 57.18 GiB (Gemma 4 31B bf16, 61.4 GB) and 26.62 GiB (Qwen 3.5 27B Q8_0,
  28.6 GB). The machine reports `hw.model` Mac15,14 (Mac Studio) and 512 GiB of memory.

Harness version: the request bodies match `run_bench.py` as committed in `ca4a63a` (warmups at
`max_tokens` 64 and seed 42 without tools, scored requests with `top_p` 1, Qwen's scored budget 8x
the case `max_tokens` in run D). The scorers used below (`score_unit_tests`,
`score_constraint_check`, `score_tool_trace`) are unchanged between `ca4a63a` and `main`. The
`score_case` router changed: at `ca4a63a` it scored every row, while `main` marks empty and
length-truncated rows invalid. Scores here follow the router as it ran on 5 Apr.

## Method

1. **Parse.** For each `Received request: POST to /v1/chat/completions with body {` line the JSON
   body is read up to the closing `}` line. LM Studio logs strings longer than 120 UTF-16 units
   as their first 50 and last 50 units around `... <Truncated in logs> ...`. The request's
   model starts it (`Running chat completion`), the next slot launch binds it to a llama.cpp task,
   and the task's prompt size, `print_timing` block and release line are read while the task is
   live. The `server assigned slot S to task T` line names the request that the model's next
   `Finished streaming response` belongs to. Any binding that is not unique (two requests waiting
   for a launch, a task id live twice, finish lines without a named task) is marked ambiguous and
   its timings are left empty. A request in flight together with another request is flagged
   `overlapped`.
2. **Classify.** A request is a benchmark candidate only if every string is exactly what LM
   Studio logs for the case: the first non-system message is one case prompt, a system message is
   that case's system prompt, tools are exactly the case's tool definitions, and every later
   message is one tool call to a case tool followed by its canned tool response (or `"{}"`).
   Everything else is counted and its body dropped before anything is stored. Two prompts that
   share their first and last 50 characters log identically, so classification alone never
   exports a request: step 3 must tie it to the harness.
3. **Align.** `bench.log` has a line at the start of every warmup group and scored generation,
   one httpx line per HTTP request, and the scoring messages. For each harness call the
   candidates are benchmark requests of the same model and case, inside the call's time span,
   with the configuration the harness sends for it (seed, temperature, `top_p` 1, streaming, a
   budget that is a whole multiple of the case budget, the case's system prompt and tools on
   scored turns, warmup shape). Candidates of an ambiguous call are excluded from every later
   call too. They are paired with the call's httpx lines
   in order: the first request right after the call's log line, later tool-use turns extending the
   previous turn's messages, each request at most 1.5 s before its httpx line. Two best pairings
   with different requests make the call ambiguous, and none of its candidates is exported. A
   harness httpx line left over is explained by the one request, benchmark (exported as ad hoc)
   or not (not exported), that could have produced it; if more than one could, none is exported.
   A benchmark candidate with no harness line at all is dropped: none in this data.
4. **Complete.** A scored generation is complete only if the harness logged the next call (or
   `benchmark complete`) after it, every httpx line inside it is paired or explained by traffic
   of another case, and every turn has a unique binding and all of its server measurements
   (prompt size, prompt eval, eval, total, finish line). Only complete generations get a score or
   a time.
5. **Time.** `total_s` of a generation is the `bench.log` time from its `[i/156]` line to the
   next harness line (all tool-use turns included). It includes the harness's scoring step, which
   for unit-test prompts is a Python subprocess. Server-side figures come from llama.cpp:
   `prefill_s` is the first request's prompt eval time, `gen_s` the summed eval time, and decode
   tok/s is completion tokens over `gen_s`. The performance tables leave out generations that
   overlapped another request (one in run A, none in run D).
6. **Score.** Only where the logs determine the score, using the harness's own functions:
   unit-test prompts (C1, C2, C4, C5): `score_unit_tests` logs a warning on every failure and
   nothing on a pass. I4: its three constraint types are exactly the ones `score_constraint_check`
   logs, so the score is 5 minus the logged penalties (a test feeds the harness's own log output
   back through this rule). T1 and T2: the harness's `expected tool ... not called` line decides a
   missing call; otherwise `run_bench.score_tool_trace` runs on the arguments replayed in the next
   turn's request. Complete arguments go to it verbatim. Truncated arguments give a score only
   when every compared key is visible (a string with its closing quote, a number followed by a
   delimiter) or its visible prefix already differs, and that score rests on the assumption that
   the hidden part is well-formed JSON that does not repeat a compared key. Without it, a called
   tool is only known to score between 2.5 and 5.0 (both bounds are in the CSV).
7. **finish_reason** is not logged for streamed responses. It is inferred: `length` when the
   completion used the whole `max_tokens` budget (llama.cpp stops at exactly that count),
   `tool_calls` when another turn of the same generation followed, `stop_or_tool_calls` for the
   harness's fourth tool-use turn, empty for an interrupted generation or a request without a
   finish line, otherwise `stop`.

## Recovered tables

Generated with:

```bash
uv run python scripts/recovered_run_summary.py \
    data/recovered_05apr2026_gemma4_qwen35_generations.csv \
    --calls data/recovered_05apr2026_gemma4_qwen35.csv \
    --update-doc docs/27092026_recovered_gemma4_qwen35_run.md
```

`tests/test_recovered_run_summary.py` asserts that the block below is exactly the script's output.

<!-- recovered_run_summary:start -->
### Runs

| run | status | start | end | scored generations (Gemma / Qwen) | complete | warmup requests | HTTP requests | Qwen scored max_tokens | length-truncated generations (Gemma / Qwen) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| A | complete | 10:46:37 | 12:18:31 | 78 / 78 | 78 / 78 | 104 | 308 | 256, 512, 768 | 3 / 62 |
| B | aborted | 12:19:19 | 12:23:55 | 6 / 6 | 6 / 5 | 8 | 20 | 256, 512 | 0 / 5 |
| C | aborted | 12:24:16 | 12:26:01 | 3 / 2 | 3 / 1 | 4 | 9 | 512 | 0 / 1 |
| D | complete | 12:29:27 | 14:54:17 | 78 / 78 | 78 / 78 | 104 | 308 | 2048, 4096, 6144 | 3 / 6 |

### Performance, run A (complete generations)

| model | n | overlapped, left out | median total_s | mean total_s | sum total_s | median prefill_s (turn 1) | median gen_s | median decode tok/s | median completion tokens | length-truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4_31b_bf16 | 78 | 0 | 9.03 | 18.98 | 1480 | 1.264 | 7.83 | 6.51 | 50.5 | 3 |
| qwen35_27b_q8 | 77 | 1 | 37.77 | 41.76 | 3215 | 0.424 | 37.48 | 10.00 | 256.0 | 62 |

Qwen / Gemma: median total_s 4.18x, sum total_s 2.17x, median completion tokens 5.07x, median decode tok/s 1.54x.

### Recovered scores, run A

| case | category | scoring | Gemma r1, r2, r3 | Qwen r1, r2, r3 | best of 3 | evidence (Gemma; Qwen) |
|---|---|---|---|---|---|---|
| C1 | coding | unit_tests | 5.0, 5.0, 5.0 | 5.0, 5.0, 5.0 | tie | no unit test failure logged (pass); no unit test failure logged (pass) |
| C2 | coding | unit_tests | 5.0, 5.0, 5.0 | 1.0, 1.0, 1.0 | Gemma | no unit test failure logged (pass); harness logged 'unit test failed' |
| C4 | coding | unit_tests | 5.0, 5.0, 5.0 | 5.0, 5.0, 5.0 | tie | no unit test failure logged (pass); no unit test failure logged (pass) |
| C5 | coding | unit_tests | 5.0, 5.0, 5.0 | 5.0, 5.0, 5.0 | tie | no unit test failure logged (pass); no unit test failure logged (pass) |
| I4 | instruction | constraint_check | 5.0, 5.0, 5.0 | 0.0, 0.0, 0.0 | Gemma | no constraint failure logged; constraint fail: word_count 0 != 17; constraint fail: missing word 'index'; constraint fail: missing word 'latency'; constraint fail: missing word 'consistency' |
| T1 | tool_use | tool_trace_exact | 5.0*, 5.0*, 5.0* | 0.0, 0.0, 0.0 | Gemma | create_refund(order_id='A1009', amount_usd=84.0) visible in the truncated arguments; harness logged 'create_refund' not called, got: [] |
| T2 | tool_use | tool_trace_exact | 2.5 to 5.0, 2.5 to 5.0, 2.5 to 5.0 | 2.5*, 2.5*, 2.5* | tie or Gemma* | 'create_calendar_event' called; argument(s) start_iso cut from the log, so the harness score is between 2.5 and 5.0; create_calendar_event(start_iso='2026-05-1...') visible in the truncated arguments |

\* assumes the part of the tool-call arguments that LM Studio cut from its log is well-formed JSON that does not repeat a compared key. Without that assumption such a score is only known to lie between 2.5 and 5.0.

### Category means where every programmatic prompt was recovered, run A

| category | programmatic prompts | recovered (Gemma / Qwen) | Gemma | Qwen |
|---|---|---|---:|---:|
| reasoning | R1, R2, R3, R4, R5 | 0 / 0 | n/a | n/a |
| coding | C1, C2, C4, C5 | 4 / 4 | 5.00 | 4.00 |
| math | M1, M2, M3, M4 | 0 / 0 | n/a | n/a |
| instruction | I1, I2, I3, I4 | 1 / 1 | n/a | n/a |
| creative | W2, W3, W4 | 0 / 0 | n/a | n/a |
| tool_use | T1, T2, T3, T4 | 1 / 2 | n/a | n/a |

### Performance, run D (complete generations)

| model | n | overlapped, left out | median total_s | mean total_s | sum total_s | median prefill_s (turn 1) | median gen_s | median decode tok/s | median completion tokens | length-truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4_31b_bf16 | 78 | 0 | 8.00 | 15.81 | 1233 | 1.238 | 7.00 | 7.85 | 50.5 | 3 |
| qwen35_27b_q8 | 78 | 0 | 65.63 | 87.09 | 6793 | 0.274 | 63.51 | 10.60 | 764.0 | 6 |

Qwen / Gemma: median total_s 8.20x, sum total_s 5.51x, median completion tokens 15.13x, median decode tok/s 1.35x.

### Recovered scores, run D

| case | category | scoring | Gemma r1, r2, r3 | Qwen r1, r2, r3 | best of 3 | evidence (Gemma; Qwen) |
|---|---|---|---|---|---|---|
| C1 | coding | unit_tests | 5.0, 5.0, 5.0 | 5.0, 5.0, 5.0 | tie | no unit test failure logged (pass); no unit test failure logged (pass) |
| C2 | coding | unit_tests | 5.0, 5.0, 5.0 | 5.0, 5.0, 5.0 | tie | no unit test failure logged (pass); no unit test failure logged (pass) |
| C4 | coding | unit_tests | 5.0, 5.0, 5.0 | 5.0, 5.0, 5.0 | tie | no unit test failure logged (pass); no unit test failure logged (pass) |
| C5 | coding | unit_tests | 5.0, 5.0, 5.0 | 5.0, 5.0, 5.0 | tie | no unit test failure logged (pass); no unit test failure logged (pass) |
| I4 | instruction | constraint_check | 5.0, 5.0, 5.0 | 2.0, 2.0, 2.0 | Gemma | no constraint failure logged; constraint fail: word_count 235 != 17; constraint fail: forbidden char ',' |
| T1 | tool_use | tool_trace_exact | 5.0*, 5.0*, 5.0* | 0.0, 0.0, 0.0 | Gemma | create_refund(order_id='A1009', amount_usd=84.0) visible in the truncated arguments; harness logged 'create_refund' not called, got: [] |
| T2 | tool_use | tool_trace_exact | 2.5 to 5.0, 2.5 to 5.0, 2.5 to 5.0 | 2.5*, 2.5*, 2.5* | tie or Gemma* | 'create_calendar_event' called; argument(s) start_iso cut from the log, so the harness score is between 2.5 and 5.0; create_calendar_event(start_iso='2026-05-1...') visible in the truncated arguments |

\* assumes the part of the tool-call arguments that LM Studio cut from its log is well-formed JSON that does not repeat a compared key. Without that assumption such a score is only known to lie between 2.5 and 5.0.

### Category means where every programmatic prompt was recovered, run D

| category | programmatic prompts | recovered (Gemma / Qwen) | Gemma | Qwen |
|---|---|---|---:|---:|
| reasoning | R1, R2, R3, R4, R5 | 0 / 0 | n/a | n/a |
| coding | C1, C2, C4, C5 | 4 / 4 | 5.00 | 5.00 |
| math | M1, M2, M3, M4 | 0 / 0 | n/a | n/a |
| instruction | I1, I2, I3, I4 | 1 / 1 | n/a | n/a |
| creative | W2, W3, W4 | 0 / 0 | n/a | n/a |
| tool_use | T1, T2, T3, T4 | 1 / 2 | n/a | n/a |
<!-- recovered_run_summary:end -->

Reading the tables: runs B and C were started and stopped by hand between A and D (their "end"
is the end of their last finished generation). The two ad hoc requests in the calls CSV, at
12:26:43 and 12:28:28, sent prompt R1 to `qwen3.5-27b` with `max_tokens` 512 (it used all 512
tokens) and 4096 (it stopped after 614 tokens): a manual check of the larger budget before run D.

## Which run the post used

Run D (12:29:27 to 14:54:17). Its median totals, 8.00 s for Gemma and 65.63 s for Qwen, match the
published 8.0 s and 65.6 s, and all four unit-tested coding prompts pass for both models, which
matches the published coding score of 5.00 for both (in run A Qwen fails C2 in all three repeats,
giving 4.00). The post calls run A "my first run".

## Comparison with the published claims

Claims as published on 27 Sep 2026 in the post (and the initial README).

| Claim | Recovered | Verdict |
|---|---|---|
| 26 prompts, 6 categories | 26 case ids in 6 categories in run D | reproduced |
| Deterministic prompts at temperature 0, seed 42 | every scored non-creative request | reproduced |
| Creative prompts at temperature 0.8, three seeds | seeds 41, 42, 43 on W1 to W4 | reproduced (`top_p` was 1, not the 0.95 the README stated) |
| 2 warmups discarded, 3 scored repeats per model per prompt, 156 generations | 104 warmup requests, 78 + 78 scored generations in run D | reproduced |
| LM Studio on port 1234, 32K context for both | `n_ctx_slot = 32768` on every request (port: local observation) | reproduced |
| 61.4 GB and 28.6 GB "in RAM" | GGUF weight files of 57.18 GiB (61.4 GB) and 26.62 GiB (28.6 GB), local observation | reproduced as weight size, not measured RAM |
| Hardware "Mac Studio M3" (512 GB) | Apple M3 Ultra in the load log; Mac15,14, 512 GiB (local observations) | corrected to M3 Ultra |
| Quality: coding 5.00 / 5.00 | C1, C2, C4, C5: 5.0 in all 3 repeats for both | reproduced |
| Quality: reasoning 3.60 / 2.60, math 1.25 / 0.75, instruction 5.00 / 3.00, creative 5.00 / 3.00, tool_use 3.12 / 2.50, overall 3.77 / 2.79 | response text not logged; determined: I4 (Gemma 5.0, Qwen 2.0), T1 (Qwen 0.0; Gemma 5.0 under the argument assumption, at least 2.5 without it), T2 (Qwen 2.5 under the assumption; Gemma 2.5 to 5.0) | not recoverable |
| Head-to-head best of 3: Gemma 9, ties 11, Qwen 4 | decided for 6 of 24 prompts: 4 ties (C1, C2, C4, C5), 2 Gemma wins (I4, T1); T2 a tie or a Gemma win under the assumption | not recoverable |
| Per category: coding 0 / 0 / 4 | 4 ties | reproduced |
| Per category: instruction 4 / 0 / 0, tool_use 1 / 1 / 2, others | only I4 and T1 decided, both Gemma wins, consistent with the published rows | not recoverable |
| Median TTFT 1.84 s / 0.66 s | client TTFT not in any log; server prefill median 1.24 s / 0.27 s | not recoverable (same direction) |
| Median tok/s 8.2 / 3.4, "Gemma 2.4x faster generation" | server decode median 7.85 / 10.60 tok/s: Qwen decodes 1.35x faster per token | contradicted |
| Median total 8.0 s / 65.6 s | 8.00 s / 65.63 s | reproduced |
| "~8x faster end-to-end" | median ratio 8.20x; ratio of summed totals 5.51x | reproduced for medians |
| "Gemma averaged 8" seconds | mean 15.81 s, median 8.00 s | corrected (median, not average) |
| First run "capped Qwen at 512 tokens", which "never output a final answer" (`finish_reason: length`, empty content) | run A gave Qwen each prompt's own budget (256 for 12 prompts, 512 for 12, 768 for 2); 62 of 78 generations used all of it. The harness logged an empty answer for I4 (0 words, 3 of 3), but Qwen passed the unit tests of C1, C4 and C5 in run A, so it did answer there | partly reproduced |
| Budget multiplied 8x | run D budgets 2048, 4096, 6144 (8x); 6 of 78 Qwen generations still used all of it (I2 x3, W1 x2, W4 x1) | reproduced |
| Instruction example "3 bullet points, 4 words each" (I2) nailed by Gemma | I2 not recoverable; in run D all 3 Qwen I2 generations ran out of budget | not recoverable |
| Coding list "LRU cache, nginx parser, range compactor, SQL query: everything 5/5" | the four unit-tested prompts are C1 (range compactor), C2 (rotate bug fix), C4 (LRU cache), C5 (nginx parser); the SQL prompt C3 is judge-only and no score for it was recovered | corrected |
| "Qwen's only speed win: TTFT" | Qwen also decodes faster per token; it loses end to end because it generates 15.13x more tokens (median 764 vs 50.5) | contradicted |
| Claude-as-judge for subjective outputs | no judge output for this run survives; no score for C3 or W1 was recovered | not recoverable |

## Not recoverable

- Response text (content and `reasoning_content`) of every streamed response, so the scores of
  R1 to R5, M1 to M4, I1 to I3, W1 to W4, C3, T3, T4 and Gemma's T2 (its `start_iso` argument is cut
  from the log; the harness score is 2.5 to 5.0). With it, every category mean except coding, the
  overall means, and the head-to-head totals.
- Re-judging: `judge_claude.py` uses the local `claude` CLI, but there is no text to judge.
- Client-side TTFT and the harness's `tok_per_s` and `gen_s` columns.
- The logged `finish_reason` (inferred instead, see Method).

## Other findings

- The harness's `tok_per_s` at the time counted only final-answer characters (`len(content) // 4`)
  and ignored reasoning tokens, which is why Qwen showed 3.4 tok/s. `run_bench.py` has since been
  changed to count both channels.
- The M1 answer key in `ca4a63a` was `5/12`; the correct probability of exactly one blue ball is
  `3 x 6 / C(9,2) = 18/36 = 1/2`, fixed in `fd9c240` (20 Apr 2026). The published math scores used
  the wrong key for M1.
- T2's `expected_args_contain` is `{"start_iso": "15:00"}`, compared by exact string equality, so
  an ISO timestamp such as `2026-05-12T15:00:00` can never earn the argument half of the score.
- In run D Gemma used its whole 256-token budget on M3 in all three repeats (Gemma's budget was not
  multiplied).
- Under `main`'s router, run A's Qwen C4 (all three repeats used the whole 768-token budget, yet
  passed the unit tests) and I4 rows would be invalid rather than scored.

## Usage

Re-create the CSVs on the benchmark machine (needs the local logs):

```bash
uv run python scripts/recover_lmstudio_log.py \
    --lmstudio-log ~/.lmstudio/server-logs/2026-04/2026-04-05.1.log \
    --bench-log bench.log \
    --calls-csv data/recovered_05apr2026_gemma4_qwen35.csv \
    --generations-csv data/recovered_05apr2026_gemma4_qwen35_generations.csv
```

Anyone can re-derive the tables and run the checks from the committed files:

```bash
uv run python scripts/recovered_run_summary.py \
    data/recovered_05apr2026_gemma4_qwen35_generations.csv \
    --calls data/recovered_05apr2026_gemma4_qwen35.csv
uv run pytest tests/test_recover_lmstudio_log.py tests/test_recovered_run_summary.py
```

## Impact

The speed comparison of the published run is now backed by committed data, and three published
claims are shown to be wrong (per-token speed, "only speed win", "averaged 8"). Most quality
numbers cannot be backed and are being removed from the post; the corrected draft links here.
