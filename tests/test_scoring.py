"""Unit tests for scoring functions in run_bench.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_bench import (
    score_exact,
    score_keywords,
    score_constraint_check,
    _merge_tool_call_deltas,
    is_invalid_result,
    score_case,
    _scoreable_text,
)


def test_score_exact_perfect():
    assert score_exact("5,12", "5,12") == 5.0


def test_score_exact_contained():
    assert score_exact("The answer is 5,12.", "5,12") == 3.0


def test_score_exact_miss():
    assert score_exact("wrong answer", "5,12") == 0.0


def test_score_exact_whitespace():
    assert score_exact("  5,12  ", "5,12") == 5.0


def test_score_exact_backticks():
    assert score_exact("```5,12```", "5,12") == 5.0


def test_score_exact_case_insensitive_contains():
    assert score_exact("BEST=Expired Credentials; RULED_OUT=Disk Full",
                       "best=expired credentials; ruled_out=disk full") == 3.0


def test_score_keywords_all():
    text = "The cache warm-up job was disabled due to a config typo."
    assert score_keywords(text, ["cache", "warm-up", "config", "typo"]) == 5.0


def test_score_keywords_partial():
    text = "The cache was restarted."
    assert score_keywords(text, ["cache", "warm-up", "config", "typo"]) == 1.2


def test_score_keywords_none():
    text = "Something unrelated."
    assert score_keywords(text, ["cache", "warm-up"]) == 0.0


def test_score_keywords_empty():
    assert score_keywords("anything", []) == 0.0


def test_constraint_check_word_count():
    text = "one two three four five"
    assert score_constraint_check(text, {"word_count": 5}) == 5.0
    assert score_constraint_check(text, {"word_count": 3}) == 3.0


def test_constraint_check_required_words():
    text = "index helps reduce latency and ensure consistency"
    result = score_constraint_check(text, {"required_words": ["index", "latency", "consistency"]})
    assert result == 5.0


def test_constraint_check_required_words_missing():
    text = "index helps reduce latency"
    result = score_constraint_check(text, {"required_words": ["index", "latency", "consistency"]})
    assert result == 4.0


def test_constraint_check_forbidden_chars():
    text = "no commas or semicolons here"
    assert score_constraint_check(text, {"forbidden_chars": [",", ";"]}) == 5.0

    text_bad = "has, commas; and semicolons"
    assert score_constraint_check(text_bad, {"forbidden_chars": [",", ";"]}) == 3.0


def test_constraint_check_line_count():
    text = "line1\nline2\nline3"
    assert score_constraint_check(text, {"line_count": 3}) == 5.0
    assert score_constraint_check(text, {"line_count": 5}) == 3.0


def test_constraint_check_word_range():
    text = " ".join(["word"] * 130)
    assert score_constraint_check(text, {"word_range": [120, 140]}) == 5.0
    assert score_constraint_check(text, {"word_range": [150, 200]}) == 3.0


def test_constraint_check_forbidden_words():
    text = "this is a revolutionary product"
    assert score_constraint_check(text, {"forbidden_words": ["revolutionary"]}) == 4.0


def test_merge_tool_call_deltas():
    deltas = [
        {"index": 0, "id": "call_1", "function_name": "lookup", "function_args": ""},
        {"index": 0, "id": None, "function_name": "", "function_args": '{"order'},
        {"index": 0, "id": None, "function_name": "", "function_args": '_id":"A1009"}'},
    ]
    result = _merge_tool_call_deltas(deltas)
    assert len(result) == 1
    assert result[0]["id"] == "call_1"
    assert result[0]["function_name"] == "lookup"
    assert result[0]["function_args"] == '{"order_id":"A1009"}'


def test_merge_tool_call_deltas_multiple():
    deltas = [
        {"index": 0, "id": "c1", "function_name": "fn1", "function_args": "{}"},
        {"index": 1, "id": "c2", "function_name": "fn2", "function_args": "{}"},
    ]
    result = _merge_tool_call_deltas(deltas)
    assert len(result) == 2


def _empty_result(finish_reason="stop"):
    return {"response_text": "", "reasoning_text": "", "finish_reason": finish_reason}


def _result(text, reasoning="", finish_reason="stop"):
    return {"response_text": text, "reasoning_text": reasoning, "finish_reason": finish_reason}


def test_invalid_when_both_text_empty():
    case = {"category": "instruction", "scoring": "constraint_check",
            "constraints": {"word_count": 10}}
    assert is_invalid_result(_empty_result(), case) is True


def test_invalid_when_finish_reason_error():
    case = {"category": "math", "scoring": "exact", "expected": "42"}
    assert is_invalid_result(_result("42", finish_reason="error"), case) is True


def test_invalid_when_finish_reason_length_non_creative():
    case = {"category": "instruction", "scoring": "exact", "expected": "abc"}
    assert is_invalid_result(_result("abc", finish_reason="length"), case) is True


def test_valid_when_finish_reason_length_creative():
    case = {"category": "creative", "scoring": "judge"}
    assert is_invalid_result(_result("a story...", finish_reason="length"), case) is False


def test_score_case_invalid_returns_none():
    case = {"category": "instruction", "scoring": "constraint_check",
            "constraints": {"word_count": 5, "required_words": ["foo"]}}
    score, stype = score_case(_empty_result(), case)
    assert score is None
    assert stype == "invalid"


def test_score_case_thinking_model_uses_reasoning_when_response_empty():
    # thinking model emitted answer in reasoning_content with empty content
    case = {"category": "math", "scoring": "exact", "expected": "42"}
    result = _result("", reasoning="The answer is 42.", finish_reason="stop")
    score, stype = score_case(result, case)
    assert score == 3.0  # contained match
    assert stype == "exact"


def test_scoreable_text_prefers_response():
    assert _scoreable_text(_result("hello", reasoning="world")) == "hello"


def test_scoreable_text_falls_back_to_reasoning():
    assert _scoreable_text(_result("", reasoning="world")) == "world"


def test_constraint_check_never_scores_empty_string_positive():
    """Even when called with empty text, constraint scoring should not award
    spurious positives. (score_case wraps this with is_invalid_result, but the
    primitive must also be safe.)"""
    constraints = {"word_count": 10, "required_words": ["alpha", "beta"]}
    score = score_constraint_check("", constraints)
    # 5.0 - 2 (word_count) - 2 (missing words) = 1.0; that's fine for the
    # primitive, what matters is score_case wraps and rejects.
    case = {"category": "instruction", "scoring": "constraint_check",
            "constraints": constraints}
    s, st = score_case(_empty_result(), case)
    assert s is None and st == "invalid"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
