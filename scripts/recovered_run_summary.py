#!/usr/bin/env python3
"""Summarise the recovered 5 Apr 2026 Gemma 4 vs Qwen 3.5 runs as markdown tables.

Reads the two CSVs written by scripts/recover_lmstudio_log.py and prints the tables used in
docs/27092026_recovered_gemma4_qwen35_run.md:

  runs         one row per harness invocation found in the logs
  performance  per complete run and model: medians of client total time (bench.log),
               server prefill (turn 1), server generation time and decode rate
  scores       every prompt whose score the logs determine, per repeat, plus best-of-3
  categories   a category mean only where every programmatically scored prompt of the
               category was recovered

Definitions:
  total_s          harness bench.log time from the "[i/N]" line of a scored generation to
                   the next harness log line (all tool-use turns included). Includes the
                   harness's scoring time, a Python subprocess for unit-test prompts.
  prefill_s        llama.cpp "prompt eval time" of the first request of the generation.
                   A server-side lower bound on time to first token, not the client TTFT.
  gen_s            sum of llama.cpp "eval time" over the generation's requests.
  decode tok/s     completion tokens / gen_s, per generation, then the median.

Usage:
  uv run python scripts/recovered_run_summary.py \\
      data/recovered_05apr2026_gemma4_qwen35_generations.csv \\
      --calls data/recovered_05apr2026_gemma4_qwen35.csv \\
      [--update-doc docs/27092026_recovered_gemma4_qwen35_run.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("recovered_run_summary")

REPO_ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["reasoning", "coding", "math", "instruction", "creative", "tool_use"]
MODEL_ORDER = ["gemma4_31b_bf16", "qwen35_27b_q8"]
JUDGE_ONLY = {"judge"}  # scoring types with no programmatic score in run_bench.py


def _f(value):
    return None if value in (None, "") else float(value)


def _fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_csv(path: Path) -> list:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    log.info("loaded %d rows from %s", len(rows), path.name)
    return rows


def run_inventory(gens: list, calls: list) -> list:
    """One dict per run label with counts and time span."""
    out = []
    for run in sorted({g["run"] for g in gens}):
        g_run = [g for g in gens if g["run"] == run]
        c_run = [c for c in calls if c["run"] == run]
        per_model = {m: sum(g["model_label"] == m for g in g_run) for m in MODEL_ORDER}
        complete_gen = {m: sum(g["model_label"] == m and g["complete"] == "True" for g in g_run)
                        for m in MODEL_ORDER}
        truncated = {m: sum(g["model_label"] == m and int(g["length_truncated_turns"] or 0) > 0
                            for g in g_run) for m in MODEL_ORDER}
        qwen_budget = sorted({int(c["max_tokens"]) for c in c_run
                              if c["model_label"] == "qwen35_27b_q8" and c["call_kind"] == "scored"})
        out.append({
            "run": run, "status": g_run[0]["run_status"],
            "start": min(c["bench_t_request"] for c in c_run if c["bench_t_request"]),
            "end": max((g["bench_t_end"] for g in g_run if g["bench_t_end"]), default=""),
            "generations": per_model, "complete_generations": complete_gen,
            "warmups": sum(c["call_kind"] == "warmup" for c in c_run),
            "http_requests": len(c_run), "length_truncated": truncated,
            "qwen_scored_max_tokens": qwen_budget,
        })
    return out


def performance(gens: list, run: str) -> dict:
    """Per-model medians over the complete generations of one run.

    Generations that overlapped another request on the server are left out (their timings
    include contention) and counted in `excluded_overlapped`.
    """
    out = {}
    for model in MODEL_ORDER:
        done = [g for g in gens if g["run"] == run and g["model_label"] == model
                and g["complete"] == "True"]
        rows = [g for g in done if g["overlapped"] != "True"]
        if not rows:
            continue
        tot = [_f(g["total_s"]) for g in rows]
        out[model] = {
            "n": len(rows),
            "excluded_overlapped": len(done) - len(rows),
            "median_total_s": statistics.median(tot),
            "mean_total_s": statistics.mean(tot),
            "sum_total_s": sum(tot),
            "median_prefill_s": statistics.median(_f(g["server_prefill_s_turn1"]) for g in rows),
            "median_gen_s": statistics.median(_f(g["server_gen_s"]) for g in rows),
            "median_decode_tok_s": statistics.median(_f(g["decode_tok_s"]) for g in rows
                                                     if g["decode_tok_s"]),
            "median_completion_tokens": statistics.median(int(g["completion_tokens"]) for g in rows),
            "length_truncated": sum(int(g["length_truncated_turns"] or 0) > 0 for g in rows),
        }
    return out


def _interval(g: dict, assume: bool = True) -> tuple:
    """(low, high, assumed) of one generation's score; (None, None, False) if unknown.

    With assume=False a score that rests on the truncated-arguments assumption falls back to
    its unconditional bounds.
    """
    score, low, high = _f(g["score"]), _f(g.get("score_low")), _f(g.get("score_high"))
    assumed = bool(g.get("score_assumption"))
    if score is not None and (assume or not assumed):
        return score, score, assumed
    return low, high, False


def compare(best: dict) -> str:
    """Best-of-3 verdict from (low, high) intervals per model."""
    (gl, gh), (ql, qh) = best[MODEL_ORDER[0]], best[MODEL_ORDER[1]]
    if None in (gl, gh, ql, qh):
        return "unknown"
    if gl == gh == ql == qh:
        return "tie"
    if gl > qh:
        return "Gemma"
    if ql > gh:
        return "Qwen"
    if gl >= qh:
        return "tie or Gemma"
    if ql >= gh:
        return "tie or Qwen"
    return "unknown"


def recovered_scores(gens: list, run: str) -> list:
    """Per case with at least one recovered score or bound, per model in repeat order.

    Each repeat is (low, high, assumed): low == high for a determined score, a range where
    only bounds are known, and assumed=True where the score rests on the stated assumption
    about truncated tool-call arguments.
    """
    by_case = defaultdict(lambda: defaultdict(list))
    meta = {}
    for g in gens:
        if g["run"] != run or g["complete"] != "True":
            continue
        by_case[g["case_id"]][g["model_label"]].append((int(g["repeat"]), _interval(g), g))
        meta[g["case_id"]] = (g["category"], g["scoring"])
    out = []
    for case_id in sorted(by_case, key=lambda c: (CATEGORIES.index(meta[c][0]), c)):
        models = by_case[case_id]
        if not any(iv[0] is not None for m in models.values() for _, iv, _ in m):
            continue
        row = {"case_id": case_id, "category": meta[case_id][0], "scoring": meta[case_id][1]}
        best, strict = {}, {}
        for model in MODEL_ORDER:
            reps = sorted(models.get(model, []), key=lambda x: x[0])
            row[model] = [iv for _, iv, _ in reps]
            row[model + "_evidence"] = reps[0][2]["score_evidence"] if reps else ""
            row[model + "_assumed"] = any(iv[2] for _, iv, _ in reps)
            best[model] = _best_of(row[model])
            strict[model] = _best_of([_interval(g, assume=False) for _, _, g in reps])
        verdict = compare(best)
        row["best_of_3"] = verdict if compare(strict) == verdict else verdict + "*"
        out.append(row)
    return out


def _best_of(intervals: list) -> tuple:
    lows, highs = [iv[0] for iv in intervals], [iv[1] for iv in intervals]
    if not intervals or None in lows + highs:
        return None, None
    return max(lows), max(highs)


def category_means(gens: list, run: str, cases: list) -> list:
    """Category mean per model only when every programmatic prompt in it was recovered."""
    out = []
    for cat in CATEGORIES:
        prog = sorted(c["id"] for c in cases
                      if c["category"] == cat and c.get("scoring") not in JUDGE_ONLY)
        row = {"category": cat, "programmatic_prompts": prog}
        for model in MODEL_ORDER:
            rows = [g for g in gens if g["run"] == run and g["model_label"] == model
                    and g["category"] == cat and g["case_id"] in prog and g["complete"] == "True"]
            got = sorted({g["case_id"] for g in rows if g["score"] != ""})
            row[model + "_recovered"] = got
            scores = [_f(g["score"]) for g in rows]
            row[model] = (statistics.mean(scores) if prog and got == prog and None not in scores
                          else None)
        out.append(row)
    return out


def _fmt_scores(intervals: list) -> str:
    """"5.0, 5.0, 5.0"; a range "2.5 to 5.0" where only bounds are known; "*" if assumed."""
    out = []
    for low, high, assumed in intervals:
        if low is None:
            out.append("?")
        elif low == high:
            out.append(f"{low:.1f}{'*' if assumed else ''}")
        else:
            out.append(f"{low:.1f} to {high:.1f}")
    return ", ".join(out)


def to_markdown(gens: list, calls: list, cases: list, runs_for_detail=("A", "D")) -> str:
    lines = ["### Runs", "",
             "| run | status | start | end | scored generations (Gemma / Qwen) | complete | "
             "warmup requests | HTTP requests | Qwen scored max_tokens | length-truncated "
             "generations (Gemma / Qwen) |",
             "|---|---|---|---|---:|---:|---:|---:|---|---:|"]
    for r in run_inventory(gens, calls):
        g, c, t = r["generations"], r["complete_generations"], r["length_truncated"]
        lines.append(
            f"| {r['run']} | {r['status']} | {r['start'][11:19]} | {r['end'][11:19]} | "
            f"{g[MODEL_ORDER[0]]} / {g[MODEL_ORDER[1]]} | {c[MODEL_ORDER[0]]} / {c[MODEL_ORDER[1]]} | "
            f"{r['warmups']} | {r['http_requests']} | "
            f"{', '.join(str(x) for x in r['qwen_scored_max_tokens'])} | "
            f"{t[MODEL_ORDER[0]]} / {t[MODEL_ORDER[1]]} |")
    for run in runs_for_detail:
        perf = performance(gens, run)
        if not perf:
            continue
        lines += ["", f"### Performance, run {run} (complete generations)", "",
                  "| model | n | overlapped, left out | median total_s | mean total_s | "
                  "sum total_s | median prefill_s (turn 1) | median gen_s | median decode tok/s | "
                  "median completion tokens | length-truncated |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model, p in perf.items():
            lines.append(
                f"| {model} | {p['n']} | {p['excluded_overlapped']} | "
                f"{_fmt(p['median_total_s'])} | {_fmt(p['mean_total_s'])} | "
                f"{_fmt(p['sum_total_s'], 0)} | {_fmt(p['median_prefill_s'], 3)} | "
                f"{_fmt(p['median_gen_s'])} | {_fmt(p['median_decode_tok_s'])} | "
                f"{_fmt(p['median_completion_tokens'], 1)} | {p['length_truncated']} |")
        if len(perf) == 2:
            a, b = (perf[m] for m in MODEL_ORDER)
            lines += ["", f"Qwen / Gemma: median total_s {b['median_total_s'] / a['median_total_s']:.2f}x, "
                      f"sum total_s {b['sum_total_s'] / a['sum_total_s']:.2f}x, median completion tokens "
                      f"{b['median_completion_tokens'] / a['median_completion_tokens']:.2f}x, median "
                      f"decode tok/s {b['median_decode_tok_s'] / a['median_decode_tok_s']:.2f}x."]
        lines += ["", f"### Recovered scores, run {run}", "",
                  "| case | category | scoring | Gemma r1, r2, r3 | Qwen r1, r2, r3 | best of 3 | "
                  "evidence (Gemma; Qwen) |", "|---|---|---|---|---|---|---|"]
        assumed_any = False
        for s in recovered_scores(gens, run):
            assumed_any |= s[MODEL_ORDER[0] + "_assumed"] or s[MODEL_ORDER[1] + "_assumed"]
            lines.append(
                f"| {s['case_id']} | {s['category']} | {s['scoring']} | "
                f"{_fmt_scores(s[MODEL_ORDER[0]])} | {_fmt_scores(s[MODEL_ORDER[1]])} | "
                f"{s['best_of_3']} | "
                f"{s[MODEL_ORDER[0] + '_evidence']}; {s[MODEL_ORDER[1] + '_evidence']} |")
        if assumed_any:
            lines += ["", "\\* assumes the part of the tool-call arguments that LM Studio cut from "
                      "its log is well-formed JSON that does not repeat a compared key. Without "
                      "that assumption such a score is only known to lie between 2.5 and 5.0."]
        lines += ["", f"### Category means where every programmatic prompt was recovered, run {run}",
                  "", "| category | programmatic prompts | recovered (Gemma / Qwen) | Gemma | Qwen |",
                  "|---|---|---|---:|---:|"]
        for c in category_means(gens, run, cases):
            lines.append(
                f"| {c['category']} | {', '.join(c['programmatic_prompts'])} | "
                f"{len(c[MODEL_ORDER[0] + '_recovered'])} / {len(c[MODEL_ORDER[1] + '_recovered'])} | "
                f"{_fmt(c[MODEL_ORDER[0]])} | {_fmt(c[MODEL_ORDER[1]])} |")
    return "\n".join(lines)


DOC_START = "<!-- recovered_run_summary:start -->"
DOC_END = "<!-- recovered_run_summary:end -->"


def replace_doc_block(doc_text: str, markdown: str) -> str:
    """Put `markdown` between the start and end markers of a doc."""
    head, rest = doc_text.split(DOC_START, 1)
    _, tail = rest.split(DOC_END, 1)
    return f"{head}{DOC_START}\n{markdown}\n{DOC_END}{tail}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("generations", type=Path)
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--cases", type=Path, default=REPO_ROOT / "cases.json")
    ap.add_argument("--update-doc", type=Path,
                    help="replace the block between the summary markers in this markdown file")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    cases = json.loads(a.cases.read_text())
    md = to_markdown(load_csv(a.generations), load_csv(a.calls), cases)
    if a.update_doc:
        a.update_doc.write_text(replace_doc_block(a.update_doc.read_text(), md))
        log.info("updated %s", a.update_doc.name)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
