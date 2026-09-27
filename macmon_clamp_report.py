#!/usr/bin/env python3
"""Canonical clamp/clock reporter for a macmon trace.

WHY THIS EXISTS. Reporting GPU state as "median MHz and percent of samples at or under
400 MHz" hides tails: a run can report 0 percent clamped while its p10 sits far below its
median, i.e. it spends a tenth of the run well under its typical clock without ever
touching the 400 MHz threshold. A run with a long low tail therefore reads as clean, so this
reporter adds p10 and the minimum.

It also standardises two things that are easy to recompute inconsistently per run:
  - the RAM FLOOR as well as the peak. The peak alone is not a model measurement, it is the
    model's footprint on top of whatever else was resident, so the delta is what compares
    across runs.
  - the MINIMUM clock, which reveals whether a run touched the 338 MHz clock floor seen in
    macmon traces at all. Runs can share that floor and differ only in how long they dwell
    there, which p10 and the at-or-under-400 percentage show.

STATISTICS. Medians use statistics.median, so an even-sized trace reports the mean of its
two middle samples. p10 uses linear interpolation between the closest ranks, the same as
numpy.percentile's default 'linear' method (Hyndman and Fan type 7); see percentile().

Usage:
    python3 macmon_clamp_report.py <macmon.jsonl> [label]
"""
import json
import math
import statistics
import sys


def percentile(sorted_vals: list, p: float) -> float:
    """The p-th percentile (0 to 100) of an ascending list, by linear interpolation.

    Matches numpy.percentile(vals, p) with its default method='linear' (Hyndman and Fan
    type 7): the position is h = (n - 1) * (p / 100) on the 0-based sorted list, and the
    result is x[floor(h)] + (h - floor(h)) * (x[floor(h) + 1] - x[floor(h)]). So p0 is the
    minimum, p100 the maximum, p50 equals statistics.median, and p10 of [800, 1300] is 850.
    Like numpy, a fraction of 0.5 or more is interpolated back from the upper sample, which
    keeps the result exact at the ends and avoids cancellation when the samples differ by
    many orders of magnitude.
    """
    if not sorted_vals:
        raise ValueError("percentile of an empty list")
    if not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100, got %r" % p)
    h = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_vals) - 1)
    t = h - lo
    below, above = sorted_vals[lo], sorted_vals[hi]
    if t >= 0.5:
        return above - (above - below) * (1 - t)
    return below + (above - below) * t


def summarise(rows: list) -> dict:
    """Reduce macmon samples to the fields a cell row should quote.

    Returns an empty dict for an empty trace rather than inventing zeros, because a cell
    that produced no samples must be reported as having none, not as having run at 0 MHz.
    """
    if not rows:
        return {}
    freq = sorted(r["gpu_freq_mhz"] for r in rows)
    power = sorted(r["gpu_power"] for r in rows)
    ram = sorted(r["memory"]["ram_usage"] / 2 ** 30 for r in rows)
    n = len(freq)

    return {
        "n_samples": n,
        "clock_median_mhz": statistics.median(freq),
        "clock_p10_mhz": percentile(freq, 10),
        "clock_min_mhz": freq[0],
        "clock_max_mhz": freq[-1],
        "pct_at_or_under_400": 100.0 * sum(1 for f in freq if f <= 400) / n,
        "power_median_w": statistics.median(power),
        "ram_floor_gib": ram[0],
        "ram_peak_gib": ram[-1],
        "ram_delta_gib": ram[-1] - ram[0],
        "touched_338_floor": freq[0] <= 340,
        # DELTA VALIDITY. The delta is only a model footprint if the trace CAPTURED the
        # load. If macmon started after the weights were resident, the trace opens close to
        # its peak and the delta badly understates the footprint. The discriminator is how
        # close the FIRST sample already sits to the peak.
        "delta_trustworthy": _delta_reason(rows, ram) is None,
        "delta_untrustworthy_reason": _delta_reason(rows, ram),
    }


def _delta_reason(rows: list, ram_sorted: list):
    """Returns None if the delta is a usable footprint, else a reason string.

    The delta is a model footprint only if BOTH hold.

    1. The first sample is well below the peak, so the load happened inside the trace.
    2. The minimum occurs near the START of the trace, so the "floor" is a pre-run baseline
       rather than something reached later. A run whose memory climbs for most of the run
       and then drops when prefill scratch is released passes check 1 but has its minimum
       late; peak minus that minimum is not a footprint, it is the range of a run whose
       shape changed halfway.
    """
    peak = ram_sorted[-1]
    if peak <= 0 or not rows:
        return "empty or degenerate trace"
    first = rows[0]["memory"]["ram_usage"] / 2 ** 30
    if (first / peak) >= 0.90:
        return "trace began after the model was resident"
    mem = [r["memory"]["ram_usage"] for r in rows]
    i = mem.index(min(mem))
    if i >= max(1, len(mem) // 10):
        return ("the minimum occurs %.0f percent through the run, so the floor is not a "
                "baseline" % (100.0 * i / len(mem)))
    return None


def format_line(s: dict, label: str = "") -> str:
    if not s:
        return "macmon %s: NO SAMPLES" % label
    return (
        "macmon %s: %d samples, median %.0f MHz, p10 %.0f MHz, min %.0f MHz, "
        "%.0f%% at or under 400 MHz, median %.1f W, RAM floor %.1f GiB peak %.1f GiB "
        "delta %.1f GiB%s"
        % (label, s["n_samples"], s["clock_median_mhz"], s["clock_p10_mhz"],
           s["clock_min_mhz"], s["pct_at_or_under_400"], s["power_median_w"],
           s["ram_floor_gib"], s["ram_peak_gib"],
           s["ram_delta_gib"] if s["delta_trustworthy"] else float("nan"),
           ("" if s["delta_trustworthy"]
            else " [DELTA NOT MEASURABLE: %s]" % s["delta_untrustworthy_reason"])
           + (" [TOUCHED 338 MHz FIRMWARE FLOOR]" if s["touched_338_floor"] else ""))
    )


def load(path: str) -> list:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue          # a truncated final line is normal if macmon was killed
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    print(format_line(summarise(load(sys.argv[1])), label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
