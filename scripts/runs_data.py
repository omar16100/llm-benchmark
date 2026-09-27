#!/usr/bin/env python3
"""Export and summarise run_bench.py results for publication.

Two subcommands:

  export   copy a local results/runs.csv (gitignored) to a publishable CSV. The only
           transformation is that an absolute `served_model` path (mlx_lm.server reports
           the model directory as its id) is reduced to its final directory name, because a
           local filesystem path is machine-specific and not part of the result. Every other
           column is copied unchanged.

  summary  print a markdown table per model_label from a runs CSV: valid/total rows,
           programmatically scored rows, mean score, per-category mean score, and median
           total_s. The README's sample results table is generated with this command.

Score definition (same as docs/results.md): mean of `score_raw` over rows with
`valid == True` and a non-empty `score_raw`. Rows scored `needs_judge` carry no
programmatic score and are excluded, so "scored" can be lower than "valid".

Usage:
  uv run python scripts/runs_data.py export results/runs.csv data/runs_20apr2026.csv
  uv run python scripts/runs_data.py summary data/runs_20apr2026.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import PurePosixPath

log = logging.getLogger("runs_data")

CATEGORIES = ["reasoning", "coding", "math", "instruction", "creative", "tool_use"]


def sanitize_served_model(value: str) -> str:
    """Reduce an absolute or home-relative model path to its final component.

    Plain model ids ("minimax-m2.7", "org/model") are returned unchanged.
    """
    if value.startswith("/") or value.startswith("~"):
        name = PurePosixPath(value.rstrip("/")).name
        return name or value
    return value


def export_runs(src: str, dst: str) -> int:
    """Copy src to dst with served_model sanitised. Returns the row count."""
    with open(src, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "served_model" not in fieldnames:
        raise SystemExit(f"{src}: no served_model column, not a run_bench.py runs.csv")
    changed = 0
    for row in rows:
        clean = sanitize_served_model(row["served_model"])
        if clean != row["served_model"]:
            changed += 1
            row["served_model"] = clean
    with open(dst, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    log.info("exported %d rows from %s to %s (%d served_model paths reduced)",
             len(rows), src, dst, changed)
    return len(rows)


def is_valid(row: dict) -> bool:
    return str(row.get("valid", "")).strip().lower() == "true"


def scored_value(row: dict):
    """Programmatic score of a valid row, or None when there is none."""
    raw = (row.get("score_raw") or "").strip()
    if not is_valid(row) or raw in ("", "None"):
        return None
    return float(raw)


def summarise(rows: list) -> list:
    """Per-model summary dicts, sorted by mean score (highest first)."""
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_label"]].append(row)
    out = []
    for label, model_rows in by_model.items():
        valid = [r for r in model_rows if is_valid(r)]
        scores = [s for s in (scored_value(r) for r in model_rows) if s is not None]
        per_cat = {}
        for cat in CATEGORIES:
            vals = [s for s in (scored_value(r) for r in model_rows if r["category"] == cat)
                    if s is not None]
            per_cat[cat] = statistics.mean(vals) if vals else None
        totals = [float(r["total_s"]) for r in valid if (r.get("total_s") or "") != ""]
        out.append({
            "model_label": label,
            "quant": sorted({r["quant"] for r in model_rows})[0],
            "n_rows": len(model_rows),
            "n_valid": len(valid),
            "n_scored": len(scores),
            "mean_score": statistics.mean(scores) if scores else None,
            "per_category": per_cat,
            "median_total_s": statistics.median(totals) if totals else None,
        })
    out.sort(key=lambda s: (s["mean_score"] is None, -(s["mean_score"] or 0.0), s["model_label"]))
    return out


def _fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def to_markdown(summary: list) -> str:
    head = (["model_label", "quant", "valid/total", "scored", "mean /5"] + CATEGORIES
            + ["median total_s"])
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|"]
    for s in summary:
        cells = [s["model_label"], s["quant"], f"{s['n_valid']}/{s['n_rows']}",
                 str(s["n_scored"]), _fmt(s["mean_score"])]
        cells += [_fmt(s["per_category"][c]) for c in CATEGORIES]
        cells.append(_fmt(s["median_total_s"], 1))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_rows(path: str) -> list:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    log.info("loaded %d rows from %s", len(rows), path)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write a publishable copy of a runs.csv")
    ex.add_argument("src")
    ex.add_argument("dst")
    sm = sub.add_parser("summary", help="print the per-model markdown table")
    sm.add_argument("src")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if a.cmd == "export":
        export_runs(a.src, a.dst)
    else:
        print(to_markdown(summarise(load_rows(a.src))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
