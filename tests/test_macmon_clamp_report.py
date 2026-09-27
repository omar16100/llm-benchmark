#!/usr/bin/env python3
"""Unit tests for the clamp reporter. No macmon, no GPU, no files beyond fixtures.

Run: uv run pytest tests/test_macmon_clamp_report.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from macmon_clamp_report import format_line, percentile, summarise

try:
    import numpy
except ImportError:          # the cross-check below is optional; hand values cover the rest
    numpy = None


def sample(freq, ram_gib=100.0, power=50.0):
    return {"gpu_freq_mhz": freq, "gpu_power": power,
            "memory": {"ram_usage": int(ram_gib * 2 ** 30)}}


class TestSummarise(unittest.TestCase):
    def test_empty_trace_returns_empty_not_zeros(self):
        """A cell that produced no samples must report NO SAMPLES, not 0 MHz. Reporting
        zeros would let an empty trace masquerade as a catastrophically clamped run."""
        self.assertEqual(summarise([]), {})
        self.assertIn("NO SAMPLES", format_line({}, "X"))

    def test_p10_exposes_a_tail_that_the_clamp_percentage_misses(self):
        """THE CASE THIS FILE EXISTS FOR: 0 percent clamped, yet a tenth of the run sits far
        below the typical clock without ever crossing 400 MHz. Only p10 shows it."""
        rows = [sample(800)] * 11 + [sample(1300)] * 89    # 11 low samples in 100
        s = summarise(rows)
        self.assertEqual(s["pct_at_or_under_400"], 0.0)   # the old metric sees nothing
        self.assertEqual(s["clock_p10_mhz"], 800)          # p10 sees it
        self.assertEqual(s["clock_median_mhz"], 1300)

    def test_p10_of_one_low_sample_in_ten_interpolates(self):
        """With linear interpolation, one low sample in ten pulls p10 below the median but
        not all the way down: h = 9 * 0.10 = 0.9, so 800 + 0.9 * (1300 - 800) = 1250."""
        s = summarise([sample(800)] + [sample(1300)] * 9)
        self.assertAlmostEqual(s["clock_p10_mhz"], 1250.0)
        self.assertEqual(s["clock_median_mhz"], 1300)

    def test_338_floor_is_flagged(self):
        """Runs can share the 338 MHz clock floor and differ only in dwell. Touching it at
        all is worth flagging."""
        self.assertTrue(summarise([sample(338), sample(1300)])["touched_338_floor"])
        self.assertFalse(summarise([sample(510), sample(1300)])["touched_338_floor"])
        self.assertIn("TOUCHED 338 MHz", format_line(summarise([sample(338)]), "s"))

    def test_ram_delta_is_reported_not_just_peak(self):
        """A raw peak is the model's footprint ON TOP OF what was already resident, so the
        delta is what compares across runs."""
        s = summarise([sample(1000, ram_gib=60.0), sample(1000, ram_gib=90.0)])
        self.assertAlmostEqual(s["ram_floor_gib"], 60.0, places=3)
        self.assertAlmostEqual(s["ram_peak_gib"], 90.0, places=3)
        self.assertAlmostEqual(s["ram_delta_gib"], 30.0, places=3)

    def test_clamp_percentage_counts_at_or_under_400(self):
        s = summarise([sample(400), sample(401), sample(399), sample(1300)])
        self.assertAlmostEqual(s["pct_at_or_under_400"], 50.0)   # 400 and 399 count

    def test_single_sample_does_not_crash(self):
        s = summarise([sample(700)])
        self.assertEqual(s["clock_median_mhz"], 700)
        self.assertEqual(s["clock_p10_mhz"], 700)


class TestMedianAndPercentile(unittest.TestCase):
    """Regression tests for the median and percentile definitions. The reporter used to
    take the upper-middle sample as the median (median of [800, 1300] came out as 1300)
    and a rounded nearest rank for p10. Expected values below are numpy.percentile's
    default 'linear' method, worked by hand: h = (n - 1) * p / 100, then interpolate
    between the samples at floor(h) and floor(h) + 1."""

    def test_even_sized_median_averages_the_two_middle_samples(self):
        s = summarise([sample(800), sample(1300)])
        self.assertEqual(s["clock_median_mhz"], 1050)
        s = summarise([sample(338), sample(700), sample(900), sample(1300)])
        self.assertEqual(s["clock_median_mhz"], 800)

    def test_odd_sized_median_is_the_middle_sample(self):
        s = summarise([sample(1300), sample(338), sample(900)])
        self.assertEqual(s["clock_median_mhz"], 900)

    def test_power_median_uses_the_same_definition(self):
        s = summarise([sample(1000, power=40.0), sample(1000, power=60.0)])
        self.assertAlmostEqual(s["power_median_w"], 50.0)
        s = summarise([sample(1000, power=p) for p in (70.0, 40.0, 55.0)])
        self.assertAlmostEqual(s["power_median_w"], 55.0)

    def test_p10_even_and_odd_sizes(self):
        # n=2: h = 0.1 -> 800 + 0.1 * 500 = 850
        self.assertAlmostEqual(summarise([sample(800), sample(1300)])["clock_p10_mhz"], 850.0)
        # n=5: h = 0.4 -> 338 + 0.4 * (500 - 338) = 402.8
        rows = [sample(f) for f in (1300, 338, 900, 500, 700)]
        self.assertAlmostEqual(summarise(rows)["clock_p10_mhz"], 402.8)

    def test_percentile_even_size_p10_p50_p90(self):
        vals = [100, 200, 300, 400]                     # n=4, h = 3 * p / 100
        self.assertAlmostEqual(percentile(vals, 10), 130.0)    # h=0.3
        self.assertAlmostEqual(percentile(vals, 50), 250.0)    # h=1.5
        self.assertAlmostEqual(percentile(vals, 90), 370.0)    # h=2.7

    def test_percentile_odd_size_p10_p50_p90(self):
        vals = [10, 20, 30, 40, 50]                     # n=5, h = 4 * p / 100
        self.assertAlmostEqual(percentile(vals, 10), 14.0)     # h=0.4
        self.assertAlmostEqual(percentile(vals, 50), 30.0)     # h=2.0, exact rank
        self.assertAlmostEqual(percentile(vals, 90), 46.0)     # h=3.6

    def test_percentile_ends_and_single_value(self):
        vals = [338, 700, 1300]
        self.assertEqual(percentile(vals, 0), 338)
        self.assertEqual(percentile(vals, 100), 1300)
        self.assertEqual(percentile([700], 10), 700)
        self.assertEqual(percentile([700], 90), 700)

    def test_percentile_interpolates_from_the_upper_sample_past_halfway(self):
        """numpy computes a fraction t >= 0.5 as b - (b - a) * (1 - t). With samples many
        orders of magnitude apart, a + (b - a) * t cancels badly: here it gave -2.0, while
        numpy gives 1 - 1e16 * 2**-53 = -0.1102230246251565."""
        self.assertAlmostEqual(percentile([-1e16, 1.0], 99.99999999999999),
                               1.0 - 1e16 * 2.0 ** -53, places=9)

    def test_percentile_rejects_empty_and_out_of_range(self):
        with self.assertRaises(ValueError):
            percentile([], 10)
        with self.assertRaises(ValueError):
            percentile([1, 2], 101)
        with self.assertRaises(ValueError):
            percentile([1, 2], -1)

    @unittest.skipIf(numpy is None, "numpy not installed")
    def test_matches_numpy_default_method(self):
        traces = [
            [800, 1300],
            [338, 700, 900],
            [1300, 338, 900, 500, 700, 650, 1200, 400, 399, 1000, 338],
            [338] * 12 + [700] * 88,
            list(range(338, 1400, 7)),
            [-1e16, 1.0],
            [0.1, 0.2, 0.3, 1e9],
        ]
        for vals in traces:
            ordered = sorted(vals)
            for p in (0, 10, 25, 50, 75, 90, 99.99999999999999, 100):
                self.assertAlmostEqual(percentile(ordered, p),
                                       float(numpy.percentile(vals, p)), places=9,
                                       msg="p%r of %d samples" % (p, len(vals)))
            s = summarise([sample(f) for f in vals])
            self.assertAlmostEqual(s["clock_median_mhz"], float(numpy.median(vals)), places=9)
            self.assertAlmostEqual(s["clock_p10_mhz"], float(numpy.percentile(vals, 10)),
                                   places=9)


class TestRealTraceShapes(unittest.TestCase):
    """Synthetic traces that share the 338 MHz floor, to pin the distinction the reporter
    is meant to surface."""

    def test_paced_run_reads_cleaner_than_unpaced_at_the_same_floor(self):
        unpaced = [sample(338)] * 12 + [sample(700)] * 88     # 12 percent dwell
        paced = [sample(338)] * 3 + [sample(650)] * 97        # 3 percent dwell
        u, p = summarise(unpaced), summarise(paced)
        self.assertTrue(u["touched_338_floor"] and p["touched_338_floor"])
        self.assertGreater(u["pct_at_or_under_400"], p["pct_at_or_under_400"])
        self.assertGreater(p["clock_p10_mhz"], u["clock_p10_mhz"])



class TestDeltaValidity(unittest.TestCase):
    """The delta is only a model footprint if the trace CAPTURED the load."""

    def test_trace_that_began_after_load_is_flagged(self):
        """A trace that opens close to its peak (monitoring started after the weights were
        resident) understates the footprint, so its delta must be flagged."""
        rows = [sample(1200, ram_gib=90.0), sample(1200, ram_gib=95.0)]
        s = summarise(rows)
        self.assertFalse(s["delta_trustworthy"])
        self.assertIn("DELTA NOT MEASURABLE", format_line(s, "late_start"))

    def test_trace_that_captured_the_load_is_trusted(self):
        """A trace that opens well below its peak captured the load, so the delta is real."""
        rows = [sample(1200, ram_gib=50.0), sample(1200, ram_gib=80.0)]
        s = summarise(rows)
        self.assertTrue(s["delta_trustworthy"])
        self.assertNotIn("DELTA NOT MEASURABLE", format_line(s, "captured"))

    def test_boundary_is_90_percent_of_peak(self):
        self.assertFalse(summarise([sample(700, ram_gib=91.0), sample(700, ram_gib=100.0)])["delta_trustworthy"])
        self.assertTrue(summarise([sample(700, ram_gib=89.0), sample(700, ram_gib=100.0)])["delta_trustworthy"])


class TestFloorMustBeABaseline(unittest.TestCase):
    """Second flaw in the same metric: the first sample can be well below the peak while
    the MINIMUM still occurs late, so the floor is not a baseline and peak-minus-floor is a
    range rather than a footprint."""

    def test_late_minimum_is_rejected_even_when_first_sample_is_low(self):
        """Opens at 110 GiB, climbs to 140, then drops to 90 late in the run when prefill
        scratch is released. It passes the first-sample check and must still be rejected."""
        rows = ([sample(700, ram_gib=110.0)] + [sample(700, ram_gib=120.0)] * 7
                + [sample(700, ram_gib=140.0)] + [sample(700, ram_gib=90.0)])
        s = summarise(rows)
        self.assertLess(rows[0]["memory"]["ram_usage"] / 2 ** 30 / s["ram_peak_gib"], 0.90)
        self.assertFalse(s["delta_trustworthy"])
        self.assertIn("DELTA NOT MEASURABLE", format_line(s, "late_minimum"))

    def test_early_minimum_with_low_first_sample_is_accepted(self):
        """Minimum at the start, first sample well below peak."""
        rows = [sample(1200, ram_gib=50.0)] + [sample(1200, ram_gib=80.0)] * 9
        self.assertTrue(summarise(rows)["delta_trustworthy"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
