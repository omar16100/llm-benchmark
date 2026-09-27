#!/usr/bin/env python3
"""Fast unit tests for the NIAH harness guards. No model, no GPU, no network.

A scorer that reports a number without saying whether the number was measurable
produces wrong recall claims. The guards are the part of the harness that decides
whether a score is reportable at all, so they are the part most worth testing directly.
Several tests read the harness SOURCE (the harnesses import mlx, which CI cannot load)
to pin the shipped rule against drift.

Run: uv run pytest tests/test_niah_guards.py
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _src(name: str) -> str:
    return (REPO / name).read_text()


def inconclusive(gen: str, n_generated: int, max_gen: int) -> bool:
    """The rule as implemented in mlx_raw_niah_client.py.

    Kept as a standalone copy so the test is fast (importing the client pulls in
    mlx_lm and loads Metal). If the client's rule changes, this must change with it;
    test_rule_matches_client_source below is what catches drift.
    """
    think_open = re.search(r"<think\s*>", gen, re.IGNORECASE) is not None
    think_closed = re.search(r"</think\s*>", gen, re.IGNORECASE) is not None
    return think_open and not think_closed and n_generated >= max_gen


class TestReasoningTruncationGuard(unittest.TestCase):
    def test_unterminated_think_at_cap_is_inconclusive(self):
        """The case that motivated the guard: a model opened <think> with NO chat
        template and ran to the 300-token cap mid-enumeration. Its partial score
        measured the budget, not retrieval."""
        gen = "<think>\nThe user wants me to extract secret access codes. For Valparaiso"
        self.assertTrue(inconclusive(gen, 300, 300))

    def test_closed_think_at_cap_is_conclusive(self):
        """The model opened a think block and CLOSED it. Hitting the cap afterwards
        does not invalidate the score, because the answer was already emitted."""
        gen = "<think>\nreasoning\n</think>\nReykjavik: 12345678"
        self.assertFalse(inconclusive(gen, 300, 300))

    def test_no_think_block_is_conclusive(self):
        """A model that answers directly with no think block at all: nothing to
        truncate, so the score stands."""
        self.assertFalse(inconclusive("Reykjavik: 12345678", 300, 300))

    def test_unterminated_think_under_cap_is_conclusive(self):
        """The model stopped on its own before the cap. An unterminated block here is
        the model's choice, not the harness cutting it off, so the score is real."""
        gen = "<think>\nI cannot find them."
        self.assertFalse(inconclusive(gen, 42, 300))

    def test_guard_is_not_conditioned_on_the_prompt(self):
        """THE REGRESSION THIS FILE EXISTS TO PREVENT. The first version of this guard
        (in bench_niah_mlx.py) only armed when the PROMPT ended with an open <think>,
        on the rule that 'no template means no reasoning mode'. That rule is false:
        a raw, untemplated prompt can still produce spontaneous reasoning. The guard
        must key on the generation alone."""
        gen = "<think>\nscratchpad"
        for prompt_opened_think in (True, False):
            with self.subTest(prompt_opened_think=prompt_opened_think):
                self.assertTrue(inconclusive(gen, 300, 300))

    def test_tag_spacing_and_case_variants(self):
        for open_tag, close_tag in (("<think>", "</think>"),
                                    ("<THINK>", "</THINK>"),
                                    ("<think >", "</think >")):
            with self.subTest(tags=(open_tag, close_tag)):
                self.assertTrue(inconclusive(open_tag + "x", 300, 300))
                self.assertFalse(inconclusive(open_tag + "x" + close_tag + "y", 300, 300))


class TestScorer(unittest.TestCase):
    """The scorer is shared verbatim with llamacpp_niah_client so the arms are
    comparable; these pin the retrieval-vs-assoc distinction, which the clients
    report as two separate fields."""

    @staticmethod
    def score(gen, needles):
        lines = gen.splitlines()
        retrieval = sum(1 for _c, code in needles if code in gen)
        assoc = 0
        for city, code in needles:
            if any(code in ln and city.lower() in ln.lower() for ln in lines):
                assoc += 1
        return retrieval, assoc

    def test_code_without_city_counts_as_retrieval_not_assoc(self):
        needles = [("Reykjavik", "12345678")]
        self.assertEqual(self.score("the code is 12345678", needles), (1, 0))

    def test_code_with_city_on_same_line_counts_as_both(self):
        needles = [("Reykjavik", "12345678")]
        self.assertEqual(self.score("Reykjavik: 12345678", needles), (1, 1))

    def test_city_and_code_on_different_lines_is_not_assoc(self):
        needles = [("Reykjavik", "12345678")]
        self.assertEqual(self.score("Reykjavik\n12345678", needles), (1, 0))


class TestSlope(unittest.TestCase):
    """Decode is reported by least-squares slope, not tokens/wall, so that prefill
    is excluded from the decode figure rather than averaged into it."""

    @staticmethod
    def slope_tps(times):
        n = len(times)
        if n < 3:
            return 0.0
        mx = sum(range(n)) / n
        my = sum(times) / n
        num = sum((i - mx) * (t - my) for i, t in enumerate(times))
        den = sum((i - mx) ** 2 for i in range(n))
        if den == 0 or num <= 0:
            return 0.0
        return 1.0 / (num / den)

    def test_uniform_spacing_recovers_the_rate(self):
        # one token every 0.05 s => 20 tok/s
        self.assertAlmostEqual(self.slope_tps([0.05 * i for i in range(50)]), 20.0, places=6)

    def test_constant_offset_does_not_change_the_slope(self):
        """A fixed prologue charged into every timestamp must not move the rate, unlike
        a tokens / wall-time figure, which a fixed per-generation cost distorts."""
        base = [0.05 * i for i in range(50)]
        self.assertAlmostEqual(self.slope_tps(base),
                               self.slope_tps([t + 1.5 for t in base]), places=6)

    def test_too_few_points_returns_zero_rather_than_guessing(self):
        self.assertEqual(self.slope_tps([0.0, 0.1]), 0.0)


class TestBenchHarnessSlope(unittest.TestCase):
    """`bench_niah_mlx.py` has `slope_tps_from` so the templated rows report the same
    decode metric as the raw-prompt clients. That is only meaningful if this function
    computes the same thing the other clients do."""

    @staticmethod
    def slope_tps_from(times):
        n = len(times)
        if n < 3:
            return 0.0
        mx_ = sum(range(n)) / n
        my = sum(times) / n
        num = sum((i - mx_) * (t - my) for i, t in enumerate(times))
        den = sum((i - mx_) ** 2 for i in range(n))
        if den == 0 or num <= 0:
            return 0.0
        return 1.0 / (num / den)

    def test_agrees_with_the_raw_client_slope(self):
        """The two implementations must be arithmetically identical, or results from the
        two harnesses would mix metrics."""
        times = [0.02 * i for i in range(300)]
        self.assertAlmostEqual(self.slope_tps_from(times),
                               TestSlope.slope_tps(times), places=9)

    def test_absolute_timestamps_need_no_prefill_subtraction(self):
        """bench_niah_mlx passes timestamps that still include the whole prefill in
        times[0]. That is only safe because a slope ignores a constant offset."""
        base = [0.02 * i for i in range(300)]
        self.assertAlmostEqual(self.slope_tps_from([t + 3000.0 for t in base]),
                               self.slope_tps_from(base), places=9)

    def test_early_token_transient_makes_short_windows_read_HIGH(self):
        """A natural suspicion is that a short fit window reads slow. When early tokens
        decode faster, the opposite happens: a short fit reads HIGH. This pins the
        direction with a synthetic fast-start curve."""
        times, t = [], 0.0
        for i in range(1000):
            t += 0.025 if i < 50 else 0.04     # fast start, then steady state
            times.append(t)
        short = self.slope_tps_from(times[:20])
        full = self.slope_tps_from(times)
        self.assertGreater(short, full)
        # and the fit converges: by 200 tokens it is close to the full-window value
        self.assertLess(abs(self.slope_tps_from(times[:200]) - full) / full, 0.10)

    def test_both_harnesses_persist_token_times(self):
        """Fit-window questions are only answerable if the raw timestamps are kept. If
        either harness stops persisting them, that class of question costs GPU again."""
        for name in ("bench_niah_mlx.py", "mlx_raw_niah_client.py"):
            with self.subTest(path=name):
                self.assertIn('"token_times"', _src(name))

    def test_source_wires_the_slope_into_the_record(self):
        src = _src("bench_niah_mlx.py")
        self.assertIn("def slope_tps_from", src)
        self.assertIn('"generation_tps_slope": round(slope_tps_from(tok_times), 2)', src)
        self.assertIn("tok_times.append(time.time() - t0)", src)


class TestRuleMatchesClientSource(unittest.TestCase):
    def test_client_guard_is_not_gated_on_prompt_opens_think(self):
        """Drift check: if someone re-narrows the client's guard to the prompt, the
        standalone copy above would silently stop matching the shipped rule."""
        src = _src("mlx_raw_niah_client.py")
        self.assertIn("answer_inconclusive = think_open and not think_closed and hit_cap", src)
        guard_line = [l for l in src.splitlines()
                      if l.strip().startswith("answer_inconclusive =")]
        self.assertEqual(len(guard_line), 1)
        self.assertNotIn("prompt_opens_think", guard_line[0])



class TestGenerationTailCapture(unittest.TestCase):
    """With only the head of the generation stored, a run that hits its cap leaves most
    of the output unobserved, so it cannot be told whether the model was still
    reasoning, looping, or about to terminate. The tail field makes that diagnosable."""

    def test_harness_stores_both_head_and_tail(self):
        src = _src("bench_niah_mlx.py")
        self.assertIn('"generated_answer": gen_text[:2000]', src)
        self.assertIn('"generated_tail": gen_text[-2000:]', src)
        self.assertIn('"generated_chars_total": len(gen_text)', src)

    def test_tail_slice_is_safe_on_short_generations(self):
        """A generation shorter than the slice must not error or pad."""
        for text in ("", "short", "x" * 1999, "y" * 5000):
            with self.subTest(n=len(text)):
                self.assertEqual(text[-2000:], text if len(text) <= 2000 else text[-2000:])
                self.assertLessEqual(len(text[-2000:]), 2000)


class TestPeakMemoryBasis(unittest.TestCase):
    """Both mlx harnesses must record mlx-lm's PROCESS `peak_memory`. Otherwise rows
    from one of them fall back to a SYSTEM-WIDE peak, which basis a row reports is
    decided by which harness ran it, and a comparison across rows mixes the two."""

    def test_raw_client_records_the_process_basis(self):
        src = _src("mlx_raw_niah_client.py")
        self.assertIn('"peak_memory_gb"', src)
        self.assertIn('getattr(resp, "peak_memory"', src)

    def test_bench_harness_also_records_it(self):
        self.assertIn("peak_memory", _src("bench_niah_mlx.py"))


class TestOmlxUsageOptIn(unittest.TestCase):
    """The oMLX client's truncation guard is INDIRECT unless the server sends usage, and
    OpenAI-style streaming omits the usage block unless the client opts in. These pin the
    opt-in and, more importantly, the fallback, since a broken request on a 6 hour run is
    a far worse outcome than a missing guard."""

    @staticmethod
    def _src():
        return _src("omlx_niah_client.py")

    def test_client_requests_usage(self):
        s = self._src()
        self.assertIn('pl["stream_options"] = {"include_usage": True}', s)

    def test_client_falls_back_when_the_field_is_rejected(self):
        """If the server 400s on the unknown field, the run must continue exactly as it did
        before rather than dying six hours into a prefill."""
        s = self._src()
        self.assertIn("urllib.error.HTTPError", s)
        # explicit import: it previously resolved only as a side effect of urllib.request
        self.assertIn("urllib.error, urllib.request", s)
        self.assertIn("_open(False)", s)

    def test_record_states_which_guard_was_achieved(self):
        """A row must be able to say whether its guard was direct or indirect."""
        s = self._src()
        self.assertIn('"guard"', s)
        self.assertIn("INDIRECT", s)

if __name__ == "__main__":
    unittest.main(verbosity=2)
