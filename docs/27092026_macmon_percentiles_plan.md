# macmon reporter: standard median and percentile (27 Sep 2026)

## Context

`macmon_clamp_report.py` summarises a macmon JSONL trace into clock, power and RAM figures.
It took `sorted[n // 2]` as the median, which for an even number of samples is the upper of
the two middle samples: the median of `[800, 1300]` came out as 1300 instead of 1050. The
same applied to the power median. p10 used a nearest rank with Python's `round`, which rounds
half to even, so it was neither the textbook nearest-rank (`ceil`) nor an interpolated
percentile. The earlier plan (`27092026_readme_results_backing_plan.md`) had left this as is.

## Changes

- `percentile(sorted_vals, p)`: linear interpolation between the closest ranks, identical to
  `numpy.percentile` with its default `method='linear'` (Hyndman and Fan type 7). Position
  `h = (n - 1) * (p / 100)`, result `x[floor(h)] + (h - floor(h)) * (x[floor(h) + 1] - x[floor(h)])`,
  computed from the upper sample when the fraction is 0.5 or more, as numpy does.
  Raises `ValueError` on an empty list or `p` outside 0 to 100. Still stdlib only.
- `clock_median_mhz` and `power_median_w` use `statistics.median`; `clock_p10_mhz` uses
  `percentile(freq, 10)`. The method is documented in the module and function docstrings.
- Tests (`tests/test_macmon_clamp_report.py`): even and odd medians, p10 for 2 and 5 samples,
  p10/p50/p90 on even and odd lists against values worked by hand, ends and single values,
  input validation, and a cross-check against `numpy.percentile` and `numpy.median` (numpy
  is in the lock through scipy; the test skips if it is missing). The tail test now uses 11
  low samples in 100, because with interpolation one low sample in ten gives p10 = 1250, not
  800; a separate test pins that value.

## Impact

- Reported medians change only for traces with an even number of samples whose two middle
  samples differ. p10 changes for most traces, since it now interpolates instead of picking
  a rounded rank: for 25 evenly spaced samples 0 to 24 it moves from 1 to 2.4.
- No committed README or doc quotes a figure produced by this script (searched for `p10`,
  `MHz`, `macmon` and `clamp`), so no published number changes.
- `uv run --locked pytest tests/`: 142 passed, 12 skipped (131 before, plus 11 new tests).
