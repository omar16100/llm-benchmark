"""Fast unit tests for scripts/runs_data.py (no network, no model)."""

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import runs_data  # noqa: E402

FIELDS = ["bench_run_id", "run_id", "model_label", "served_model", "quant", "category",
          "prompt_id", "repeat", "seed", "temperature", "top_p", "max_tokens", "ttft_s",
          "gen_s", "total_s", "output_tokens_approx", "tok_per_s", "finish_reason", "valid",
          "score_raw", "score_type"]


def _row(label, category, valid, score, total_s="10.0", served="m", quant="Q"):
    row = dict.fromkeys(FIELDS, "")
    row.update(model_label=label, category=category, valid=valid, score_raw=score,
               total_s=total_s, served_model=served, quant=quant)
    return row


def test_sanitize_reduces_absolute_paths_only():
    assert runs_data.sanitize_served_model("/home/u/models/Foo-4bit") == "Foo-4bit"
    assert runs_data.sanitize_served_model("/home/u/models/Foo-4bit/") == "Foo-4bit"
    assert runs_data.sanitize_served_model("~/models/Bar") == "Bar"
    assert runs_data.sanitize_served_model("minimax-m2.7") == "minimax-m2.7"
    assert runs_data.sanitize_served_model("org/model") == "org/model"


def test_scored_value_skips_invalid_and_unscored_rows():
    assert runs_data.scored_value(_row("a", "math", "True", "5.0")) == 5.0
    assert runs_data.scored_value(_row("a", "math", "False", "5.0")) is None
    assert runs_data.scored_value(_row("a", "math", "True", "")) is None  # needs_judge
    assert runs_data.scored_value(_row("a", "math", "True", "None")) is None


def test_summarise_means_counts_and_order():
    rows = [
        _row("low", "math", "True", "1.0", total_s="4"),
        _row("low", "math", "True", "", total_s="6"),       # valid, not scored
        _row("low", "coding", "False", "", total_s="100"),  # invalid: excluded everywhere
        _row("high", "coding", "True", "5.0", total_s="2"),
        _row("high", "coding", "True", "4.0", total_s="3"),
    ]
    out = runs_data.summarise(rows)
    assert [s["model_label"] for s in out] == ["high", "low"]
    low = out[1]
    assert (low["n_rows"], low["n_valid"], low["n_scored"]) == (3, 2, 1)
    assert low["mean_score"] == 1.0
    assert low["per_category"]["math"] == 1.0
    assert low["per_category"]["coding"] is None
    assert low["median_total_s"] == 5.0
    assert out[0]["mean_score"] == 4.5


def test_markdown_marks_missing_categories():
    md = runs_data.to_markdown(runs_data.summarise([_row("m", "math", "True", "3.0")]))
    body = md.splitlines()[2]
    assert body.startswith("| m | Q | 1/1 | 1 | 3.00 |")
    assert "n/a" in body


def test_export_only_changes_served_model(tmp_path):
    src, dst = tmp_path / "runs.csv", tmp_path / "out.csv"
    rows = [_row("a", "math", "True", "5.0", served="/Users/someone/models/X-8bit"),
            _row("b", "math", "True", "5.0", served="plain-id")]
    with open(src, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    assert runs_data.export_runs(str(src), str(dst)) == 2
    got = list(csv.DictReader(open(dst, newline="")))
    assert [r["served_model"] for r in got] == ["X-8bit", "plain-id"]
    for before, after in zip(rows, got):
        assert {k: v for k, v in before.items() if k != "served_model"} == \
               {k: v for k, v in after.items() if k != "served_model"}


def test_published_data_has_no_local_paths():
    text = (REPO / "data" / "runs_20apr2026.csv").read_text()
    assert "/Users/" not in text and "/home/" not in text


def test_readme_table_is_generated_from_published_data():
    """Every number in the README sample results table must come from data/."""
    rows = runs_data.load_rows(str(REPO / "data" / "runs_20apr2026.csv"))
    table = runs_data.to_markdown(runs_data.summarise(rows))
    assert table in (REPO / "README.md").read_text()
