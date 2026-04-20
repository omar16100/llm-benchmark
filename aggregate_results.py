"""
Aggregate results from all evaluation frameworks into a unified comparison table.

Reads from:
- results/runs.csv (custom benchmark)
- results/lm-eval/<model>/<timestamp>/results.json
- results/bigcode/<model>/<timestamp>/metrics.json
- results/deepeval/<model>/<timestamp>/deepeval_results.xml
- results/livecodebench/<model>/<timestamp>/

Outputs:
- results/aggregate.csv — unified comparison table
- Console summary table
"""

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("eval.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
AGGREGATE_CSV = RESULTS_DIR / "aggregate.csv"


def collect_custom_benchmark():
    """Collect scores from custom benchmark runs.csv."""
    rows = []
    runs_csv = RESULTS_DIR / "runs.csv"
    if not runs_csv.exists():
        log.warning("custom benchmark results not found: %s", runs_csv)
        return rows

    with open(runs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("score_raw") and row["score_raw"] != "None":
                rows.append({
                    "model": row["model_label"],
                    "framework": "custom",
                    "task": row["prompt_id"],
                    "category": row["category"],
                    "score": float(row["score_raw"]),
                    "score_type": row.get("score_type", ""),
                    "tok_per_s": float(row.get("tok_per_s", 0)),
                })
    log.info("collected %d custom benchmark results", len(rows))
    return rows


def collect_lm_eval():
    """Collect scores from lm-evaluation-harness results."""
    rows = []
    lm_eval_dir = RESULTS_DIR / "lm-eval"
    if not lm_eval_dir.exists():
        return rows

    for model_dir in lm_eval_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_label = model_dir.name

        # find latest run
        run_dirs = sorted(model_dir.iterdir(), reverse=True)
        if not run_dirs:
            continue
        latest = run_dirs[0]

        # lm-eval outputs results in a nested JSON structure
        for json_file in latest.rglob("results.json"):
            try:
                data = json.loads(json_file.read_text())
                results = data.get("results", {})
                for task_name, metrics in results.items():
                    for metric_name, value in metrics.items():
                        if metric_name.endswith(",none") or metric_name in ("alias",):
                            continue
                        if isinstance(value, (int, float)):
                            rows.append({
                                "model": model_label,
                                "framework": "lm-eval",
                                "task": task_name,
                                "category": metric_name,
                                "score": round(value, 4),
                                "score_type": "accuracy",
                                "tok_per_s": 0,
                            })
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("failed to parse %s: %s", json_file, e)

    log.info("collected %d lm-eval results", len(rows))
    return rows


def collect_bigcode():
    """Collect scores from bigcode-evaluation-harness metrics.json."""
    rows = []
    bigcode_dir = RESULTS_DIR / "bigcode"
    if not bigcode_dir.exists():
        return rows

    for model_dir in bigcode_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_label = model_dir.name

        run_dirs = sorted(model_dir.iterdir(), reverse=True)
        if not run_dirs:
            continue
        latest = run_dirs[0]

        metrics_file = latest / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            data = json.loads(metrics_file.read_text())
            for task_name, metrics in data.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            rows.append({
                                "model": model_label,
                                "framework": "bigcode",
                                "task": task_name,
                                "category": metric_name,
                                "score": round(value, 4),
                                "score_type": "pass@1",
                                "tok_per_s": 0,
                            })
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("failed to parse %s: %s", metrics_file, e)

    log.info("collected %d bigcode results", len(rows))
    return rows


def collect_deepeval():
    """Collect scores from DeepEval XML results."""
    rows = []
    deepeval_dir = RESULTS_DIR / "deepeval"
    if not deepeval_dir.exists():
        return rows

    for model_dir in deepeval_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_label = model_dir.name

        run_dirs = sorted(model_dir.iterdir(), reverse=True)
        if not run_dirs:
            continue
        latest = run_dirs[0]

        # parse stdout for JSON results
        stdout_file = latest / "stdout.txt"
        if stdout_file.exists():
            text = stdout_file.read_text()
            # extract JSON blocks from pytest output
            import re
            json_blocks = re.findall(r'\{[^{}]*"case_id"[^{}]*\}', text)
            for block in json_blocks:
                try:
                    result = json.loads(block)
                    rows.append({
                        "model": model_label,
                        "framework": "deepeval",
                        "task": result.get("case_id", "unknown"),
                        "category": result.get("metric", ""),
                        "score": float(result.get("score", 0)),
                        "score_type": "judge",
                        "tok_per_s": 0,
                    })
                except (json.JSONDecodeError, ValueError):
                    continue

    log.info("collected %d deepeval results", len(rows))
    return rows


def aggregate():
    """Collect all results and write aggregate CSV."""
    all_rows = []
    all_rows.extend(collect_custom_benchmark())
    all_rows.extend(collect_lm_eval())
    all_rows.extend(collect_bigcode())
    all_rows.extend(collect_deepeval())

    if not all_rows:
        log.warning("no results found to aggregate")
        return

    # write CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = ["model", "framework", "task", "category", "score", "score_type", "tok_per_s"]
    with open(AGGREGATE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    log.info("wrote %d rows to %s", len(all_rows), AGGREGATE_CSV)

    # print summary table
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS SUMMARY")
    print("=" * 80)

    # group by model and framework
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        key = (row["model"], row["framework"])
        summary[key][row["task"]].append(row["score"])

    # print per model
    models_seen = set()
    for (model, framework), tasks in sorted(summary.items()):
        if model not in models_seen:
            print(f"\n--- {model} ---")
            models_seen.add(model)
        avg_scores = {t: sum(s) / len(s) for t, s in tasks.items()}
        overall = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        print(f"  [{framework}] avg={overall:.3f} | ", end="")
        top_tasks = sorted(avg_scores.items(), key=lambda x: -x[1])[:5]
        print(" | ".join(f"{t}={s:.3f}" for t, s in top_tasks))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    aggregate()
