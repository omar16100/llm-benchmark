"""Tests for bench_niah_mlx.py helpers that need the module itself imported.

bench_niah_mlx imports mlx and mlx-lm at module level, which only install on Apple
silicon and are not project dependencies, so this module is marked `mlx` and skips
unless they are present (e.g. `uv run --with mlx-lm pytest tests/test_bench_niah_mlx.py`
on a Mac). The guard logic itself is also pinned platform-independently, from source,
in test_niah_guards.py.
"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench_niah_mlx as b  # noqa: E402


def test_split_think_returns_text_after_last_close():
    assert b.split_think("<think>a</think>b</think>Answer") == ("Answer", "<think>a</think>b</think>")


def test_split_think_unterminated_has_no_answer():
    assert b.split_think("<think>still going") == ("", "<think>still going")


def test_split_think_plain_text_is_all_answer():
    assert b.split_think("Reykjavik: 12345678") == ("Reykjavik: 12345678", "")


def test_slope_tps_from_ignores_prefill_offset():
    times = [100.0 + 0.05 * i for i in range(50)]
    assert b.slope_tps_from(times) == pytest.approx(20.0)
    assert b.slope_tps_from([0.0, 1.0]) == 0.0
