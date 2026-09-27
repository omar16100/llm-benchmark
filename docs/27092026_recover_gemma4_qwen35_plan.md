# Plan: recover the 5 Apr 2026 Gemma 4 vs Qwen 3.5 run from server logs (27 Sep 2026)

## Context

The blog post <https://omarshabab.com/llm-benchmark/> and the initial README quoted numbers from a
5 Apr 2026 run whose results files were never committed (the README table was removed earlier on
27 Sep 2026 for that reason). LM Studio server logs and the harness `bench.log` from that day
survive on the benchmark machine. The server logs also hold unrelated traffic that must not leave
the machine.

## Changes

Plan (status in brackets):
1. [done] Read the harness as it ran (`ca4a63a`): request shape, warmups, scoring functions.
2. [done] Parser `scripts/recover_lmstudio_log.py`: benchmark-only classification by exact match
   against `cases.json`, request to llama.cpp task pairing, alignment with `bench.log`, scoring
   only where the logs determine it, with the harness's own scorers.
3. [done] Data under `data/` (calls CSV, generations CSV, 5 Apr excerpt of `bench.log`); summary
   script `scripts/recovered_run_summary.py`; fixture-based tests.
4. [done] Findings doc [27092026_recovered_gemma4_qwen35_run.md](27092026_recovered_gemma4_qwen35_run.md).
5. [done] Codex review, four rounds (round 4: no blocker or major), personal-data scan and gitleaks; PR, CI and squash-merge follow.
6. [pending] Corrected blog post drafted on a separate branch of the site repo (not merged here).

Decisions:
- No response text survives (LM Studio does not log streamed responses), so re-scoring is
  limited to prompts whose score the harness logged (unit tests, I4 constraints) or that the
  logged tool-call arguments decide (T1, T2). Everything else is reported as unrecoverable
  rather than estimated.
- Re-judging with `judge_claude.py` was not run: there is no text to judge.
- `finish_reason` is inferred from token counts and labelled as inferred.
- The `bench.log` excerpt is committed because it holds only harness log lines (benchmark calls,
  scoring messages, and three httpx lines of ad hoc requests with no content); the LM Studio logs
  are not, and only their sha256 and line ranges are recorded.
- A benchmark request is exported only with harness provenance (paired with an httpx line of the
  harness call that sent it, or of the same second for ad hoc requests); classification alone is
  not enough, because two prompts sharing their first and last 50 characters log identically.
- Scores that rest on truncated tool-call arguments carry an explicit assumption and bounds.

Codex review round 1 (`oss_sweep_llm-benchmark_recover_gemma4_qwen35.txt`): all nine findings
applied: canonical 50 + 50 truncation matching and harness provenance before export; numbers cut at
the truncation marker treated as unknown, escaped prefixes unknown, complete arguments scored
verbatim, assumption and bounds recorded; finish lines paired through `server assigned slot`,
live-only task ids, ambiguous bindings blanked; alignment checks seed, temperature and budget and
pairs in order with the first request right after the call line; a generation is complete only
when closed and fully paired; fourth tool-use turn finish left open; doc separates local
observations from committed data; excerpt described accurately.

Codex review round 2 (`oss_sweep_llm-benchmark_recover_gemma4_qwen35_r2.txt`): one blocker and
three majors, all applied: candidates of an ambiguous call are never exported and an ad hoc
request must be the only request that could have produced its harness line; two request bodies
waiting for one start line are both marked ambiguous; scored turns must carry the case's system
prompt and its exact tool definitions; a generation needs every server measurement of every turn
to be complete. Regression tests added for each. The real 5 Apr output is byte-identical before
and after these fixes.

Codex review round 3 (`oss_sweep_llm-benchmark_recover_gemma4_qwen35_r3.txt`): one blocker, one
major, one minor, all applied: tainted candidates are excluded from every later call, not only from
the ad hoc fallback; warmups and scored turns must carry the harness's `top_p` (1) and `stream`
(true); a request without a finish line gets an unknown finish reason instead of `stop`. Tests added;
real output again byte-identical.

Codex review round 4 (`oss_sweep_llm-benchmark_recover_gemma4_qwen35_r4.txt`): all three fixes
verified, no blocker or major left. Its one minor (the taint regression test did not carry the
private argument) was fixed; the test now fails when the taint guard is removed.

Deviations: the task brief expected `Generated prediction` JSON with response text in the logs;
those lines exist only for non-streamed requests, and the benchmark streamed every request.

## Impact

See the findings doc. Tests: `tests/test_recover_lmstudio_log.py`,
`tests/test_recovered_run_summary.py`.
