"""Tests for scripts/recovered_run_summary.py and for the committed recovered data.

The committed-data tests pin every number that docs/27092026_recovered_gemma4_qwen35_run.md
and the corrected blog post quote, and re-derive the bench.log based fields of the
generations CSV from the committed bench.log excerpt.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import recover_lmstudio_log as rl  # noqa: E402
import recovered_run_summary as rs  # noqa: E402

DATA = REPO / "data"
GENS_CSV = DATA / "recovered_05apr2026_gemma4_qwen35_generations.csv"
CALLS_CSV = DATA / "recovered_05apr2026_gemma4_qwen35.csv"
EXCERPT = DATA / "bench_log_05apr2026.txt"
CASES = json.loads((REPO / "cases.json").read_text())
G, Q = rs.MODEL_ORDER


def _gen(run, model, case_id, category, repeat, score="", total="10", complete="True",
         prefill="0.5", gen_s="9", decode="10", comp="90", trunc="0", method="unrecoverable",
         scoring="exact"):
    return {"run": run, "run_status": "complete", "model_label": model, "case_id": case_id,
            "category": category, "scoring": scoring, "repeat": str(repeat), "complete": complete,
            "total_s": total, "server_prefill_s_turn1": prefill, "server_gen_s": gen_s,
            "decode_tok_s": decode, "completion_tokens": comp, "length_truncated_turns": trunc,
            "overlapped": "False", "score": score, "score_method": method, "score_evidence": "e",
            "bench_t_start": "2026-04-05T10:00:00.000", "bench_t_end": "2026-04-05T10:00:10.000"}


def test_performance_medians_and_best_of_3():
    gens = [_gen("A", G, "C1", "coding", r, score="5.000", total=str(t), method="m")
            for r, t in ((1, 4), (2, 8), (3, 30))]
    gens += [_gen("A", Q, "C1", "coding", r, score=s, total="60", method="m")
             for r, s in ((1, "1.000"), (2, "5.000"), (3, "1.000"))]
    gens += [_gen("A", G, "R1", "reasoning", 1), _gen("A", Q, "R1", "reasoning", 1)]
    gens.append(_gen("A", Q, "C2", "coding", 1, complete="False", total=""))
    perf = rs.performance(gens, "A")
    assert perf[G]["median_total_s"] == 9.0 and perf[G]["n"] == 4  # 4, 8, 10, 30
    assert perf[Q]["n"] == 4  # the incomplete generation is excluded
    scores = rs.recovered_scores(gens, "A")
    assert [s["case_id"] for s in scores] == ["C1"]  # R1 has no recovered score
    assert scores[0]["best_of_3"] == "tie"
    assert scores[0][Q] == [(1.0, 1.0, False), (5.0, 5.0, False), (1.0, 1.0, False)]


def test_compare_uses_bounds_and_marks_assumed_verdicts():
    assert rs.compare({G: (5.0, 5.0), Q: (0.0, 0.0)}) == "Gemma"
    assert rs.compare({G: (2.5, 5.0), Q: (2.5, 2.5)}) == "tie or Gemma"
    assert rs.compare({G: (2.5, 5.0), Q: (2.5, 5.0)}) == "unknown"
    assert rs.compare({G: (None, None), Q: (1.0, 1.0)}) == "unknown"
    gens = [_gen("A", G, "T2", "tool_use", r, score="", method="unrecoverable") for r in (1, 2, 3)]
    for g in gens:
        g.update(score_low="2.500", score_high="5.000")
    gens += [_gen("A", Q, "T2", "tool_use", r, score="2.500", method="m") for r in (1, 2, 3)]
    for g in gens[3:]:
        g.update(score_low="2.500", score_high="5.000", score_assumption="assumed")
    row = rs.recovered_scores(gens, "A")[0]
    assert row["best_of_3"] == "tie or Gemma*"  # without the assumption it is unknown


def test_category_mean_needs_every_programmatic_prompt():
    cases = [{"id": "C1", "category": "coding", "scoring": "unit_tests"},
             {"id": "C2", "category": "coding", "scoring": "unit_tests"},
             {"id": "C3", "category": "coding", "scoring": "judge"}]
    gens = [_gen("A", m, "C1", "coding", 1, score="5.000") for m in (G, Q)]
    gens += [_gen("A", G, "C2", "coding", 1, score="3.000"), _gen("A", Q, "C2", "coding", 1)]
    row = next(c for c in rs.category_means(gens, "A", cases) if c["category"] == "coding")
    assert row["programmatic_prompts"] == ["C1", "C2"]  # judge-only C3 is not required
    assert row[G] == 4.0 and row[Q] is None


# ---------------------------------------------------------------------------
# committed data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def committed():
    return rs.load_csv(GENS_CSV), rs.load_csv(CALLS_CSV)


def test_committed_run_inventory(committed):
    gens, calls = committed
    inv = {r["run"]: r for r in rs.run_inventory(gens, calls)}
    assert sorted(inv) == ["A", "B", "C", "D"]
    for run in ("A", "D"):
        assert inv[run]["status"] == "complete"
        assert inv[run]["generations"] == {G: 78, Q: 78}
        assert inv[run]["complete_generations"] == {G: 78, Q: 78}
        assert inv[run]["warmups"] == 104
    assert inv["A"]["qwen_scored_max_tokens"] == [256, 512, 768]
    assert inv["D"]["qwen_scored_max_tokens"] == [2048, 4096, 6144]
    assert inv["A"]["length_truncated"] == {G: 3, Q: 62}
    assert inv["D"]["length_truncated"] == {G: 3, Q: 6}
    assert inv["B"]["status"] == inv["C"]["status"] == "aborted"


def test_committed_run_d_performance_numbers(committed):
    """The numbers the doc and the corrected post quote for the published run (D)."""
    gens, _ = committed
    d = rs.performance(gens, "D")
    assert round(d[G]["median_total_s"], 2) == 8.00
    assert round(d[Q]["median_total_s"], 2) == 65.63
    assert round(d[G]["mean_total_s"], 1) == 15.8
    assert round(d[Q]["mean_total_s"], 1) == 87.1
    assert round(d[Q]["median_total_s"] / d[G]["median_total_s"], 1) == 8.2
    assert round(d[Q]["sum_total_s"] / d[G]["sum_total_s"], 1) == 5.5
    assert round(d[G]["median_decode_tok_s"], 2) == 7.85
    assert round(d[Q]["median_decode_tok_s"], 2) == 10.60
    assert round(d[G]["median_prefill_s"], 2) == 1.24
    assert round(d[Q]["median_prefill_s"], 2) == 0.27
    assert d[G]["median_completion_tokens"] == 50.5
    assert d[Q]["median_completion_tokens"] == 764
    assert round(d[G]["median_gen_s"], 2) == 7.00 and round(d[Q]["median_gen_s"], 2) == 63.51
    assert d[G]["excluded_overlapped"] == d[Q]["excluded_overlapped"] == 0
    assert rs.performance(gens, "A")[Q]["excluded_overlapped"] == 1


def test_committed_run_d_recovered_scores(committed):
    gens, _ = committed
    got = {s["case_id"]: s for s in rs.recovered_scores(gens, "D")}
    assert sorted(got) == ["C1", "C2", "C4", "C5", "I4", "T1", "T2"]
    five = [(5.0, 5.0, False)] * 3
    for cid in ("C1", "C2", "C4", "C5"):
        assert got[cid][G] == got[cid][Q] == five
        assert got[cid]["best_of_3"] == "tie"
    assert got["I4"][G] == five and got["I4"][Q] == [(2.0, 2.0, False)] * 3
    assert got["I4"]["best_of_3"] == "Gemma"
    # T1: Gemma's 5.0 rests on the truncated-arguments assumption, the win does not
    assert got["T1"][G] == [(5.0, 5.0, True)] * 3 and got["T1"][Q] == [(0.0, 0.0, False)] * 3
    assert got["T1"]["best_of_3"] == "Gemma"
    assert got["T2"][G] == [(2.5, 5.0, False)] * 3 and got["T2"][Q] == [(2.5, 2.5, True)] * 3
    assert got["T2"]["best_of_3"] == "tie or Gemma*"
    coding = next(c for c in rs.category_means(gens, "D", CASES) if c["category"] == "coding")
    assert coding[G] == coding[Q] == 5.0


def test_committed_csv_matches_committed_bench_log_excerpt(committed):
    """total_s and the bench.log based scores are re-derived from the committed excerpt."""
    gens, _ = committed
    runs = rl.parse_bench_log(EXCERPT.read_text())
    assert [r.label for r in runs] == ["A", "B", "C", "D"]
    by_key = {(g["run"], g["model_label"], g["case_id"], int(g["repeat"])): g for g in gens}
    cases = {c["id"]: c for c in CASES}
    checked = 0
    for run in runs:
        for call in run.calls:
            if call.kind != "scored":
                continue
            g = by_key[(run.label, call.label, call.case_id, call.repeat)]
            if call.t_end is None:
                assert g["total_s"] == ""
                continue
            assert float(g["total_s"]) == pytest.approx((call.t_end - call.t_start).total_seconds())
            if g["score_method"] in ("bench_log_unit_test", "bench_log_constraint"):
                score = rl.score_generation(cases[call.case_id], call.scoring, []).score
                assert float(g["score"]) == score
            checked += 1
    assert checked == 156 * 2 + 11 + 4  # A and D, plus the finished calls of B and C


def test_doc_tables_are_the_script_output(committed):
    gens, calls = committed
    doc = (REPO / "docs" / "27092026_recovered_gemma4_qwen35_run.md").read_text()
    block = doc.split(rs.DOC_START, 1)[1].split(rs.DOC_END, 1)[0]
    assert block.strip() == rs.to_markdown(gens, calls, CASES).strip()


def test_committed_csvs_hold_no_local_paths(committed):
    text = GENS_CSV.read_text() + CALLS_CSV.read_text() + EXCERPT.read_text()
    for needle in ("/Users/", "/home/", "\\Users\\"):
        assert needle not in text
