"""Import-level tests for the NIAH clients (no model, no server, no GPU).

The raw-prompt clients keep their own copies of the scorer and the decode slope so
each file runs standalone. These tests import every client (mlx and mlx_vlm are only
imported inside main(), so this works on any platform) and pin that the copies agree,
so an edit to one arm cannot silently change how it is scored.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import glm53_niah_client  # noqa: E402
import llamacpp_niah_client  # noqa: E402
import mlx_raw_niah_client  # noqa: E402
import omlx_niah_client  # noqa: E402
import dspark_niah_client  # noqa: E402

SCORERS = [llamacpp_niah_client.score, mlx_raw_niah_client.score,
           glm53_niah_client.score, omlx_niah_client.score]
NEEDLES = [("Reykjavik", "12345678"), ("Ushuaia", "87654321"), ("Timbuktu", "11112222")]
PROMPT = "".join(f"filler. IMPORTANT RECORD: the secret access code for {c} is {k}. "
                 for c, k in NEEDLES)


@pytest.mark.parametrize("gen, expected", [
    ("Reykjavik: 12345678\nUshuaia: 87654321\nTimbuktu: 11112222", (3, 3)),
    ("Reykjavik\n12345678\nushuaia 87654321", (2, 1)),   # city on another line, case-insensitive
    ("nothing here", (0, 0)),
])
def test_all_client_scorers_agree(gen, expected):
    for score in SCORERS:
        assert score(gen, NEEDLES) == expected, score.__module__


def test_needle_regex_is_shared_and_extracts_the_needles():
    for mod in (llamacpp_niah_client, mlx_raw_niah_client, glm53_niah_client,
                omlx_niah_client):
        assert mod.NEEDLE_RE.findall(PROMPT) == NEEDLES, mod.__name__
    assert dspark_niah_client.NEEDLE_RE is omlx_niah_client.NEEDLE_RE


def test_slopes_agree_on_uniform_arrivals():
    times = [2.5 + 0.04 * i for i in range(100)]   # 25 tok/s after a fixed prefill
    for value in (mlx_raw_niah_client.slope_tps(times), glm53_niah_client.slope_tps(times),
                  omlx_niah_client.slope_tps(times, warmup=0)):
        assert value == pytest.approx(25.0)


def test_omlx_slope_refuses_non_monotonic_arrivals():
    assert omlx_niah_client.slope_tps([3.0, 2.0, 1.0, 0.5], warmup=0) == 0.0


def test_lcb_dir_defaults_to_repo_and_honours_env(monkeypatch):
    monkeypatch.delenv("LCB_DIR", raising=False)
    mod = importlib.reload(importlib.import_module("lcb_local_runner"))
    assert mod.LCB_DIR == REPO / "eval_frameworks" / "LiveCodeBench"
    monkeypatch.setenv("LCB_DIR", "/tmp/elsewhere/LiveCodeBench")
    mod = importlib.reload(mod)
    assert mod.LCB_DIR == Path("/tmp/elsewhere/LiveCodeBench")


def test_no_client_hardcodes_a_home_directory():
    for name in ("bench_niah_mlx.py", "dspark_niah_client.py", "glm53_niah_client.py",
                 "lcb_local_runner.py", "llamacpp_niah_client.py", "make_1m_prompt.py",
                 "mlx_raw_niah_client.py", "omlx_niah_client.py",
                 "scripts/run_nemotron_mlx8_token_test.sh"):
        src = (REPO / name).read_text()
        assert "/Users/" not in src and "/home/" not in src, name


def test_default_endpoints_are_loopback_only():
    for name in ("dspark_niah_client.py", "llamacpp_niah_client.py", "omlx_niah_client.py",
                 "lcb_local_runner.py"):
        src = (REPO / name).read_text()
        assert "http://127.0.0.1:" in src, name
        assert "0.0.0.0" not in src, name


def test_lcb_limit_is_rejected_rather_than_ignored(monkeypatch):
    """LiveCodeBench has no problem-count cap, so --limit must fail loudly instead of
    silently launching the full problem set."""
    import lcb_local_runner
    monkeypatch.setattr(sys, "argv", ["lcb_local_runner.py", "--served-model", "m",
                                      "--repr", "r", "--limit", "2"])
    with pytest.raises(SystemExit) as exc:
        lcb_local_runner.main()
    assert exc.value.code == 2
