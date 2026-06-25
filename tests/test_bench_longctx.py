"""Fast unit tests for bench_longctx.py and bench_common.py (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import bench_common
import bench_longctx


def test_build_prompt_inserts_needle_and_is_recallable():
    code = "QX2K-50-7000"
    prompt = bench_longctx.build_prompt(2000, 50, code, tokens_per_word=1.4)
    assert code in prompt
    assert "authorization code for vault seven" in prompt


def test_build_prompt_depth_controls_position():
    code = "ABC"
    early = bench_longctx.build_prompt(4000, 5, code, 1.4)
    late = bench_longctx.build_prompt(4000, 95, code, 1.4)
    # needle near the front in early, near the back in late
    assert early.index(code) < len(early) * 0.30
    assert late.index(code) > len(late) * 0.70


def test_build_prompt_size_scales_with_tokens_per_word():
    # words = target_tokens / tokens_per_word, so a smaller ratio yields more words
    fewer = bench_longctx.build_prompt(1000, 50, "X", tokens_per_word=2.0)
    more = bench_longctx.build_prompt(1000, 50, "X", tokens_per_word=1.0)
    assert len(more.split()) > len(fewer.split())


def test_prompt_tokens_prefers_usage_total():
    # usage.prompt_tokens is the full context; prefer it over timings.prompt_n (processed only)
    assert bench_common.prompt_tokens({"usage": {"prompt_tokens": 1234}, "timings": {"prompt_n": 600}}) == 1234
    assert bench_common.prompt_tokens({"usage": {}, "timings": {"prompt_n": 600}}) == 600
    assert bench_common.prompt_tokens({}) == 0


def test_prefill_tps_from_timings_else_none():
    assert bench_common.prefill_tps({"timings": {"prompt_per_second": 31.4}}) == 31.4
    # derive from prompt_n / prompt_ms when per_second absent
    assert bench_common.prefill_tps({"timings": {"prompt_n": 100, "prompt_ms": 1000}}) == 100.0
    # no timings => None (never derived from wall clock)
    assert bench_common.prefill_tps({"timings": {}, "wall_s": 5.0}) is None


def test_decode_tps_from_timings_else_none():
    assert bench_common.decode_tps({"timings": {"predicted_per_second": 15.7}}) == 15.7
    assert bench_common.decode_tps({"timings": {}}) is None


class _Args:
    base_url = "http://x"
    model = "m"
    api_key = None
    max_tokens = 40
    tokens_per_word = 1.4
    no_thinking = True
    stream = False
    cache_prompt = False
    timeout = 10
    server_label = "unit-test"


def test_run_cell_recall_pass_and_fail(monkeypatch):
    captured = {}

    def fake_chat(base_url, model, messages, **kw):
        captured["kw"] = kw
        prompt = messages[0]["content"]
        code = prompt.split("vault seven is ")[1].split(".")[0]
        return {"content": code, "usage": {"prompt_tokens": 42},
                "timings": {"prompt_per_second": 30.0, "predicted_per_second": 16.0},
                "wall_s": 1.5, "ttft_s": None}

    monkeypatch.setattr(bench_common, "chat_completion", fake_chat)
    row = bench_longctx.run_cell(_Args, 2000, 50, 0)
    assert row["recall"] == "PASS"
    assert row["prompt_tokens"] == 42
    assert row["prefill_tps"] == 30.0
    assert row["server"] == "unit-test"
    assert captured["kw"]["chat_template_kwargs"] == {"enable_thinking": False}
    # caching disabled by default -> cache_prompt:false sent as extra_body
    assert captured["kw"]["extra_body"] == {"cache_prompt": False}

    # a wrong answer fails recall
    monkeypatch.setattr(bench_common, "chat_completion",
                        lambda *a, **k: {"content": "WRONG", "usage": {}, "timings": {},
                                         "wall_s": 1.0, "ttft_s": None})
    row2 = bench_longctx.run_cell(_Args, 2000, 50, 1)
    assert row2["recall"] == "FAIL"
    assert row2["prefill_tps"] is None


def test_recall_rejects_substring_superset(monkeypatch):
    # the real code is QX2K-50-7000; a longer token that merely contains it must NOT pass
    def fake_chat(base_url, model, messages, **kw):
        return {"content": "the code is QX2K-50-700012", "usage": {}, "timings": {},
                "wall_s": 1.0, "ttft_s": None}

    monkeypatch.setattr(bench_common, "chat_completion", fake_chat)
    row = bench_longctx.run_cell(_Args, 2000, 50, 0)  # code == QX2K-50-7000
    assert row["recall"] == "FAIL"


def test_no_thinking_false_omits_template_kwargs(monkeypatch):
    captured = {}

    def fake_chat(base_url, model, messages, **kw):
        captured["kw"] = kw
        return {"content": "x", "usage": {}, "timings": {}, "wall_s": 1.0, "ttft_s": None}

    monkeypatch.setattr(bench_common, "chat_completion", fake_chat)

    class Args(_Args):
        no_thinking = False
        server_label = ""

    bench_longctx.run_cell(Args, 1000, 50, 0)
    assert captured["kw"]["chat_template_kwargs"] is None


def test_parse_args_validates_depths_and_tokens_per_word():
    import pytest
    with pytest.raises(SystemExit):
        bench_longctx.parse_args(["--target-tokens", "1000", "--depths", "150"])
    with pytest.raises(SystemExit):
        bench_longctx.parse_args(["--target-tokens", "1000", "--tokens-per-word", "0"])
    # valid args parse fine
    a = bench_longctx.parse_args(["--target-tokens", "1000", "2000", "--depths", "0", "50", "100"])
    assert a.target_tokens == [1000, 2000]
    assert a.depths == [0, 50, 100]
