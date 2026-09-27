#!/usr/bin/env python3
"""Unit tests for the LiveCodeBench guards. No GPU, no network, no venv.

Run: uv run pytest tests/test_lcb_guards.py

Convention copied from test_niah_guards.py: test docstrings name the failure they prevent.
"""
import sys
import unittest
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lcb_guards import (answer_inconclusive, filter_by_date, is_refusal, no_extractable_answer, is_truncated,
                        pass_at_1, strip_think, summarise_run)


class Problem:
    def __init__(self, d):
        self.contest_date = d


class TestPassAt1(unittest.TestCase):
    def test_uses_only_the_first_sample(self):
        """pass@1 must not become best-of-k by accident. A later passing sample cannot rescue a
        failed first one, because these runs are greedy and one-shot."""
        self.assertEqual(pass_at_1({"a": [False, True, True]}), 0.0)
        self.assertEqual(pass_at_1({"a": [True, False]}), 1.0)

    def test_all_pass_and_all_fail(self):
        self.assertEqual(pass_at_1({"a": [True], "b": [True]}), 1.0)
        self.assertEqual(pass_at_1({"a": [False], "b": [False]}), 0.0)

    def test_empty_returns_zero_rather_than_raising(self):
        """An empty result set must report zero, not crash a campaign midway through a GPU window."""
        self.assertEqual(pass_at_1({}), 0.0)
        self.assertEqual(pass_at_1({"a": []}), 0.0)

    def test_fraction_is_exact(self):
        self.assertAlmostEqual(pass_at_1({"a": [True], "b": [False], "c": [True], "d": [False]}), 0.5)


class TestTruncation(unittest.TestCase):
    """THE FAILURE MODE THIS FILE EXISTS FOR: a generation budget too small for a reasoning
    model, where a truncated answer would be scored as inability."""

    def test_finish_reason_length_is_truncation(self):
        self.assertTrue(is_truncated(finish_reason="length"))
        self.assertTrue(is_truncated(finish_reason="LENGTH"))

    def test_stop_is_not_truncation(self):
        self.assertFalse(is_truncated(finish_reason="stop"))

    def test_token_count_reaching_the_cap_is_truncation(self):
        """Servers disagree on finish_reason, so the token count is an independent signal."""
        self.assertTrue(is_truncated(generated_tokens=4096, max_tokens=4096))
        self.assertFalse(is_truncated(generated_tokens=4095, max_tokens=4096))

    def test_absent_signals_do_not_assert_truncation(self):
        self.assertFalse(is_truncated())


class TestInconclusive(unittest.TestCase):
    def test_open_think_at_cap_is_inconclusive(self):
        self.assertTrue(answer_inconclusive("<think>reasoning that never ends", hit_cap=True))

    def test_closed_think_is_conclusive_even_at_cap(self):
        self.assertFalse(answer_inconclusive("<think>a</think>def f(): pass", hit_cap=True))

    def test_open_think_without_cap_is_not_inconclusive(self):
        """Not hitting the cap means the model chose to stop, which is a real answer."""
        self.assertFalse(answer_inconclusive("<think>a", hit_cap=False))

    def test_rule_is_output_keyed_not_prompt_keyed(self):
        """THE REGRESSION THIS TEST PREVENTS. A model fed a raw file with NO chat template can emit
        <think> spontaneously and still be inside it at the cap, so a prompt-keyed rule misses it.
        The guard must key on the generation, not the prompt."""
        self.assertTrue(answer_inconclusive("<think>spontaneous, no template involved", hit_cap=True))


class TestRefusalVsFailure(unittest.TestCase):
    """Not answering is not the same as answering wrongly, and the two must not share a number."""

    def test_leading_refusal_is_detected(self):
        self.assertTrue(is_refusal("I'm sorry, I cannot help with that."))
        self.assertTrue(is_refusal("I cannot write this program."))

    def test_code_is_not_a_refusal(self):
        self.assertFalse(is_refusal("def merge(a, b):\n    return sorted(a + b)"))

    def test_refusal_phrase_inside_reasoning_is_not_a_refusal(self):
        """'I cannot assume the list is sorted' is reasoning. Matching it would over-count refusals
        badly, which is why the check runs on the ANSWER segment only."""
        text = "<think>I cannot assume the list is sorted</think>def f(a):\n    return sorted(a)"
        self.assertFalse(is_refusal(text))

    def test_empty_answer_is_not_counted_as_refusal(self):
        """An unterminated think block has no answer segment. That is inconclusive, not refusal, and
        conflating them would double-count the same failure."""
        self.assertFalse(is_refusal("<think>still thinking"))


class TestStripThink(unittest.TestCase):
    def test_takes_text_after_the_last_close(self):
        self.assertEqual(strip_think("<think>a</think>X</think>Y"), "Y")

    def test_unterminated_think_yields_no_answer(self):
        self.assertEqual(strip_think("<think>never closed"), "")

    def test_plain_text_passes_through(self):
        self.assertEqual(strip_think("def f(): pass"), "def f(): pass")


class TestDateWindow(unittest.TestCase):
    """The window is the contamination control, and it is also what keeps a run affordable: the full
    release_v6 set is 1055 problems, which is hours of GPU time per model on local hardware."""

    def setUp(self):
        self.ps = [Problem(datetime(2024, 1, 1)), Problem(datetime(2025, 1, 1)),
                   Problem(datetime(2025, 4, 1))]

    def test_start_is_inclusive(self):
        self.assertEqual(len(filter_by_date(self.ps, start=datetime(2025, 1, 1))), 2)

    def test_end_is_inclusive(self):
        self.assertEqual(len(filter_by_date(self.ps, end=datetime(2025, 1, 1))), 2)

    def test_both_bounds(self):
        got = filter_by_date(self.ps, start=datetime(2024, 6, 1), end=datetime(2025, 2, 1))
        self.assertEqual(len(got), 1)

    def test_no_bounds_keeps_everything(self):
        self.assertEqual(len(filter_by_date(self.ps)), 3)


class TestSummarise(unittest.TestCase):
    def test_truncated_and_passing_are_reported_separately(self):
        """The two pass@1 figures answer different questions: what the model delivers at this budget,
        and what it can do when the budget is not binding. A large gap indicts the budget."""
        recs = [
            {"passed": True,  "text": "def f(): pass", "finish_reason": "stop"},
            {"passed": False, "text": "<think>ran out", "finish_reason": "length"},
            {"passed": True,  "text": "def g(): pass", "finish_reason": "stop"},
        ]
        s = summarise_run(recs)
        self.assertEqual(s["n_problems"], 3)
        self.assertEqual(s["n_truncated"], 1)
        self.assertEqual(s["n_inconclusive"], 1)
        self.assertAlmostEqual(s["pass_at_1"], 2 / 3)
        self.assertAlmostEqual(s["pass_at_1_excluding_truncated"], 1.0)

    def test_empty_run_reports_zero_problems(self):
        self.assertEqual(summarise_run([]), {"n_problems": 0})

    def test_all_truncated_leaves_excluding_figure_undefined(self):
        """With every sample truncated there is no usable subset, so the excluding figure must be
        None rather than a fabricated 0.0 that would read as total incapability."""
        s = summarise_run([{"passed": False, "text": "x", "finish_reason": "length"}])
        self.assertIsNone(s["pass_at_1_excluding_truncated"])


class TestSourceDrift(unittest.TestCase):
    """Convention from test_niah_guards.py (TestRuleMatchesClientSource). The guard rule is
    duplicated in prose here, so if someone weakens the shipped rule this test fails rather than
    silently passing."""

    def test_shipped_guard_is_still_output_keyed(self):
        src = (REPO / "lcb_guards.py").read_text()
        self.assertIn("think_open and not think_closed and hit_cap", src)

    def test_refusal_still_checks_the_answer_segment(self):
        src = (REPO / "lcb_guards.py").read_text()
        self.assertIn("answer = strip_think(", src)

    def test_no_extractable_answer_is_not_keyed_on_a_marker(self):
        """Drift check, and the reason this guard exists at all.

        The inconclusive rule keys on <think>, so a model that never emits one can report
        n_inconclusive 0 while many completions never reached a code block. If someone
        re-narrows this guard to require a tag or to require hitting the cap, it goes blind the
        same way.
        """
        body = ((REPO / "lcb_guards.py").read_text()
                .split("def no_extractable_answer", 1)[1].split("\ndef ", 1)[0])
        # Strip the docstring: it legitimately NAMES the other guards while explaining why this
        # one does not use them. The check is about executable code.
        code = body.split('"""')[2] if body.count('"""') >= 2 else body
        for marker in ("THINK_OPEN", "THINK_CLOSE", "hit_cap", "is_truncated"):
            self.assertNotIn(marker, code, f"guard re-narrowed to {marker}")


class TestNoExtractableAnswer(unittest.TestCase):
    """The artefact the tag-keyed guard misses: long prose, no extractable answer."""

    PROSE = ("I'll solve this step by step.\n\n## Problem Understanding\n\n1. We have an array "
             "and an integer k\n2. For each subarray we may perform at most k operations\n"
             "3. We need to count how many subarrays can become non-decreasing after at most k")

    def test_long_prose_with_no_code_is_flagged(self):
        self.assertTrue(no_extractable_answer(self.PROSE, ""))

    def test_a_real_answer_is_not_flagged(self):
        self.assertFalse(no_extractable_answer(self.PROSE, "def solve():\n    return 1"))

    def test_empty_output_is_NOT_this_artefact(self):
        """A null completion is a different failure and is counted elsewhere."""
        self.assertFalse(no_extractable_answer("", ""))
        self.assertFalse(no_extractable_answer(None, None))

    def test_whitespace_only_extraction_counts_as_no_answer(self):
        self.assertTrue(no_extractable_answer(self.PROSE, "   \n\t "))

    def test_flagged_without_any_think_tag_present(self):
        """The whole point: no marker anywhere, still caught."""
        self.assertNotIn("think", self.PROSE.lower())
        self.assertTrue(no_extractable_answer(self.PROSE, ""))

    def test_flagged_even_when_generation_stopped_early(self):
        """Not conditioned on the cap: a short answerless completion is still unusable."""
        self.assertTrue(no_extractable_answer("Let me think about this.", ""))


class TestSummariseWithNoAnswer(unittest.TestCase):
    def _rec(self, passed, text, extracted):
        return {"passed": passed, "text": text, "extracted": extracted,
                "finish_reason": "stop", "generated_tokens": 10, "max_tokens": 4096}

    def test_counts_and_conditional_rate(self):
        recs = [self._rec(True, "x", "code"), self._rec(False, "x", "code"),
                self._rec(False, "long prose", "")]
        out = summarise_run(recs)
        self.assertEqual(out["n_no_answer"], 1)
        self.assertAlmostEqual(out["pass_at_1"], 1 / 3)
        self.assertAlmostEqual(out["pass_at_1_excluding_no_answer"], 1 / 2)

    def test_omitting_extracted_reports_None_not_zero(self):
        """A caller that never populated `extracted` has measured nothing, so the count is None.

        Reporting 0 there would be false reassurance, and reporting len(records) would flag every
        row from an absence. None is the only honest value.
        """
        recs = [{"passed": False, "text": "long prose"}]
        out = summarise_run(recs)
        self.assertIsNone(out["n_no_answer"])
        self.assertIsNone(out["pass_at_1_excluding_no_answer"])

    def test_unknown_extraction_is_not_an_artefact(self):
        self.assertFalse(no_extractable_answer("long prose", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
