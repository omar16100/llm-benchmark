"""Fast fixture-based tests for scripts/recover_lmstudio_log.py.

Every log line here is synthetic: request bodies are built from cases.json prompts, and the
one non-benchmark request carries a made-up marker string that must never reach the outputs.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import recover_lmstudio_log as rl  # noqa: E402

CASES = json.loads((REPO / "cases.json").read_text())
CASE = {c["id"]: c for c in CASES}
T0 = datetime(2026, 4, 5, 10, 0, 0)
PRIVATE = "PRIVATE-MARKER-do-not-publish-7f3a"
LABELS = rl.DEFAULT_MODEL_LABELS
G, Q = "gemma-4-31b", "qwen3.5-27b"


# ---------------------------------------------------------------------------
# synthetic log builders
# ---------------------------------------------------------------------------

def _trunc(value):
    """Shorten long strings exactly the way LM Studio logs them."""
    if isinstance(value, str):
        return rl.lm_logged_form(value)
    if isinstance(value, list):
        return [_trunc(v) for v in value]
    if isinstance(value, dict):
        return {k: _trunc(v) for k, v in value.items()}
    return value


def _lm_ts(t):
    return f"[{(T0 + timedelta(seconds=t)).strftime('%Y-%m-%d %H:%M:%S')}]"


def lm_request(t, body):
    lines = [f"{_lm_ts(t)}[DEBUG] Received request: POST to /v1/chat/completions with body {{"]
    lines += json.dumps(_trunc(body), indent=2).split("\n")[1:]
    lines.append(f"{_lm_ts(t)}[INFO][{body['model']}] Running chat completion on conversation "
                 f"with {len(body['messages'])} messages.")
    lines.append(f"{_lm_ts(t)}[INFO][{body['model']}] Streaming response...")
    return lines


def lm_launch(t, task, prompt_tokens):
    return [f"{_lm_ts(t)}[DEBUG] LlamaV4::predict slot selection: session_id=<empty> server-selected",
            f"slot launch_slot_: id  2 | task {task} | processing task, is_child = 0",
            f"slot update_slots: id  2 | task {task} | new prompt, n_ctx_slot = 32768, "
            f"n_keep = {prompt_tokens}, task.n_tokens = {prompt_tokens}"]


def lm_finish(t, model, task, prompt_ms, prompt_tokens, eval_ms, completion, assigned=True):
    out = [f"{_lm_ts(t)}[DEBUG] slot print_timing: id  2 | task {task} | ",
           f"prompt eval time =     {prompt_ms:.2f} ms /   {prompt_tokens} tokens (    1.00 ms "
           f"per token,   100.00 tokens per second)",
           f"       eval time =    {eval_ms:.2f} ms /    {completion} tokens (   10.00 ms per "
           f"token,    10.00 tokens per second)",
           f"      total time =    {prompt_ms + eval_ms:.2f} ms /   {prompt_tokens + completion} tokens",
           f"slot      release: id  2 | task {task} | stop processing: n_tokens = "
           f"{prompt_tokens + completion}, truncated = 0"]
    if assigned:
        out.append(f"{_lm_ts(t)}[DEBUG] LlamaV4: server assigned slot 2 to task {task}")
    out.append(f"{_lm_ts(t)}[INFO][{model}] Finished streaming response")
    return out


def lm_call(t_start, t_end, body, task, completion=20, prompt_tokens=50):
    return (lm_request(t_start, body) + lm_launch(t_start, task, prompt_tokens)
            + lm_finish(t_end, body["model"], task, 500.0, prompt_tokens,
                        1000.0 * (t_end - t_start) - 600, completion))


def _bl_ts(t):
    ts = T0 + timedelta(seconds=t)
    return ts.strftime("%Y-%m-%d %H:%M:%S") + f",{ts.microsecond // 1000:03d}"


def bl(t, msg, level="INFO"):
    return f"{_bl_ts(t)} {level} {msg}"


def bl_http(t):
    return bl(t, 'HTTP Request: POST http://localhost:1234/v1/chat/completions "HTTP/1.1 200 OK"')


def bl_start(t=0.0):
    return [bl(t, 'HTTP Request: GET http://localhost:1234/v1/models "HTTP/1.1 200 OK"'),
            bl(t + 0.001, "available models: ['gemma-4-31b', 'qwen3.5-27b']")]


def body_for(model, case_id, max_tokens, messages=None, seed=42):
    case = CASE[case_id]
    if messages is None:
        messages = [{"role": "user", "content": case["prompt"]}]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": case["temperature"], "top_p": 1, "seed": seed, "stream": True}
    if case.get("tools") and max_tokens != 64:
        body["tools"] = case["tools"]
    return body


REFUND_ARGS = ('{"amount_usd": 84.0, "order_id": "A1009", "reason": "'
               + "late delivery beyond the seven day policy window " * 3 + '"}')


def t1_messages(with_calls, refund_args=REFUND_ARGS):
    case = CASE["T1"]
    msgs = [{"role": "system", "content": case["system"]},
            {"role": "user", "content": case["prompt"]}]
    calls = [("lookup_order", '{"order_id": "A1009"}', case["tool_responses"]["lookup_order"]),
             ("create_refund", refund_args, "{}")]
    for n, (name, args, resp) in enumerate(calls[:with_calls]):
        msgs.append({"role": "assistant", "tool_calls": [
            {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": args}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{n}", "content": resp})
    return msgs


def build_fixture():
    """One complete run: gemma C1 (warmups + 1 scored), gemma I4, gemma T1 (3 turns),
    qwen T1 (1 turn, tool not called), gemma C2 (unit test failure); one private request
    overlapping gemma C1; one ad hoc C1 request after the run."""
    lm, blog = [], bl_start()
    # C1 warmups
    blog += [bl(0.002, "warmup: gemma4_31b_bf16 / C1"), bl_http(0.010), bl_http(2.010)]
    lm += lm_call(0, 2, body_for(G, "C1", 64), task=1, completion=64)
    lm += lm_call(2, 4, body_for(G, "C1", 64), task=2, completion=64)
    # C1 scored repeat 1, overlapped by a private request to the other model
    blog += [bl(4.000, "[1/5] C1 | gemma4_31b_bf16 | repeat 1"), bl_http(4.010)]
    lm += lm_request(4, body_for(G, "C1", 512)) + lm_launch(4, 3, 60)
    private = {"model": Q, "stream": True,
               "messages": [{"role": "user", "content": f"remember {PRIVATE} please"}]}
    lm += lm_request(5, private) + lm_launch(5, 90, 12)
    lm += lm_finish(6, Q, 90, 100.0, 12, 900.0, 7)
    lm += lm_finish(9, G, 3, 800.0, 60, 4000.0, 200)
    # I4 scored repeat 1: whole budget used, two logged constraint failures
    blog += [bl(9.500, "[2/5] I4 | gemma4_31b_bf16 | repeat 1"), bl_http(9.510),
             bl(12.250, "constraint fail: word_count 30 != 17"),
             bl(12.251, "constraint fail: forbidden char ','")]
    lm += lm_call(9, 12, body_for(G, "I4", 256), task=4, completion=256)
    # T1 gemma: three turns, create_refund arguments truncated in the log
    blog += [bl(12.300, "[3/5] T1 | gemma4_31b_bf16 | repeat 1"), bl_http(12.310),
             bl_http(14.310), bl_http(16.310)]
    lm += lm_call(12, 14, body_for(G, "T1", 512, t1_messages(0)), task=5)
    lm += lm_call(14, 16, body_for(G, "T1", 512, t1_messages(1)), task=6)
    lm += lm_call(16, 18, body_for(G, "T1", 512, t1_messages(2)), task=7)
    # T1 qwen: one turn, no tool call
    blog += [bl(18.500, "[4/5] T1 | qwen35_27b_q8 | repeat 1"), bl_http(18.510),
             bl(21.000, "expected tool 'create_refund' not called, got: []")]
    lm += lm_call(18, 21, body_for(Q, "T1", 4096, t1_messages(0)), task=91, completion=300)
    # C2 scored with a unit test failure (multi-line warning)
    blog += [bl(21.100, "[5/5] C2 | gemma4_31b_bf16 | repeat 1"), bl_http(21.110),
             bl(24.000, 'unit test failed:   File "<string>", line 3', level="WARNING"),
             "SyntaxError: unterminated string",
             bl(24.100, "benchmark complete. results: results/runs.csv")]
    lm += lm_call(21, 24, body_for(G, "C2", 512), task=8)
    # ad hoc C1 request after the run, logged through the harness logger too
    blog.append(bl_http(30.010))
    lm += lm_call(30, 33, body_for(G, "C1", 4096), task=9)
    return "\n".join(lm) + "\n", "\n".join(blog) + "\n"


def run_recover(tmp_path, lm_text, bench_text):
    lm_path, bench_path = tmp_path / "2026-04-05.1.log", tmp_path / "bench.log"
    lm_path.write_text(lm_text)
    bench_path.write_text(bench_text)
    return rl.recover([lm_path], bench_path, CASES, LABELS)


@pytest.fixture()
def recovered(tmp_path):
    calls, gens, stats = run_recover(tmp_path, *build_fixture())
    calls_csv, gens_csv = tmp_path / "calls.csv", tmp_path / "gens.csv"
    rl.write_csv(calls_csv, rl.CALL_COLUMNS, calls)
    rl.write_csv(gens_csv, rl.GENERATION_COLUMNS, gens)
    return calls, gens, stats, calls_csv.read_text() + gens_csv.read_text()


# ---------------------------------------------------------------------------
# classification and argument parsing
# ---------------------------------------------------------------------------

def test_logged_form_is_canonical_50_plus_50_above_120_units():
    short = "x" * 120
    assert rl.lm_logged_form(short) == short
    full = "x" * 60 + "middle" + "y" * 60
    assert rl.lm_logged_form(full) == "x" * 50 + rl.TRUNC_MARK + "y" * 50
    assert rl.logged_matches(rl.lm_logged_form(full), full)
    assert not rl.logged_matches(full, full)  # a long string is never logged verbatim
    assert not rl.logged_matches("x" * 49 + rl.TRUNC_MARK + "y" * 50, full)
    assert not rl.logged_matches(None, full)
    # UTF-16 units, as in JavaScript: an emoji counts twice
    emoji = "\U0001F600" * 30 + "z" * 70
    head = rl.lm_logged_form(emoji).split(rl.TRUNC_MARK)[0]
    assert head == "\U0001F600" * 25


def test_classify_accepts_benchmark_bodies_only():
    body = _trunc(body_for(G, "R1", 512))
    assert rl.classify_request(body, CASES)["id"] == "R1"
    t1 = _trunc(body_for(G, "T1", 512, t1_messages(2)))
    assert rl.classify_request(t1, CASES)["id"] == "T1"
    assert rl.classify_request({"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                               CASES) is None
    bad_sys = json.loads(json.dumps(t1))
    bad_sys["messages"][0]["content"] = "You are a helpful assistant."
    assert rl.classify_request(bad_sys, CASES) is None
    tools_on_r1 = dict(body, tools=CASE["T1"]["tools"])
    assert rl.classify_request(tools_on_r1, CASES) is None
    foreign = json.loads(json.dumps(t1))
    foreign["messages"][3]["content"] = "some other tool output"
    assert rl.classify_request(foreign, CASES) is None


def test_classify_rejects_non_canonical_truncation():
    """Only the exact 50 + 50 form LM Studio writes matches; other heads and tails do not."""
    prompt = CASE["R1"]["prompt"]
    loose = prompt[:40] + rl.TRUNC_MARK + prompt[-40:]  # would pass a prefix/suffix test
    assert rl.classify_request({"model": G, "messages": [{"role": "user", "content": loose}]},
                               CASES) is None


def test_parse_visible_args():
    head = '{"amount_usd": 84.0, "order_id": "A1009", "reason"'
    assert rl.parse_visible_args(head + rl.TRUNC_MARK + 'end."}') == {"amount_usd": 84.0,
                                                                      "order_id": "A1009"}
    cut_number = rl.parse_visible_args('{"order_id": "A1009", "amount_usd": 84.0' + rl.TRUNC_MARK + '"}')
    assert cut_number["order_id"] == "A1009" and cut_number["amount_usd"] is rl.CUT
    prefix = rl.parse_visible_args('{"title": "Walk", "start_iso": "2026-05-1' + rl.TRUNC_MARK + '"}')
    assert isinstance(prefix["start_iso"], rl.ArgPrefix) and prefix["start_iso"] == "2026-05-1"
    escaped = rl.parse_visible_args('{"start_iso": "\\u0031' + rl.TRUNC_MARK + '"}')
    assert escaped["start_iso"] is rl.CUT


# ---------------------------------------------------------------------------
# scoring rules against the harness's own functions
# ---------------------------------------------------------------------------

def test_logged_constraint_penalties_match_harness(caplog):
    """Feed the harness's own log output back in and compare with its return value."""
    import run_bench

    case = CASE["I4"]
    base = ("The index keeps latency low and consistency high for every database query "
            "we run today and tonight")  # 17 words, all required words, no , or ;
    assert run_bench.score_constraint_check(base, case["constraints"]) == 5.0
    texts = [base, base + " now", base.replace("index", "table"), base.replace("low", "low,"),
             "", base.replace("low", "low;").replace("index", "table") + " ok"]
    for text in texts:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="run_bench"):
            want = run_bench.score_constraint_check(text, case["constraints"])
        msgs = [r.getMessage() for r in caplog.records if r.name == "run_bench"]
        got = rl.score_generation(case, msgs, [])
        assert got.method == "bench_log_constraint"
        assert got.score == want, (text, msgs)


def test_unit_test_log_messages_match_harness(caplog):
    import run_bench

    case = CASE["C2"]
    for code in ("def rotate_right(a, k):\n    return a\n", "not python at all ("):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="run_bench"):
            want = run_bench.score_unit_tests(code, case["test_code"])
        msgs = [r.getMessage() for r in caplog.records if r.name == "run_bench"]
        got = rl.score_generation(case, msgs, [])
        assert (got.score, got.method) == (want, "bench_log_unit_test")
    assert rl.score_generation(case, [], []).score == 5.0


def test_unrecoverable_cases_have_no_score():
    for cid in ("R1", "M1", "I2", "W1", "W2", "C3", "T3", "T4"):
        got = rl.score_generation(CASE[cid], [], [])
        assert got.score is None and got.method == "unrecoverable", cid


def _t2_body(args):
    case = CASE["T2"]
    return {"model": Q, "messages": [
        {"role": "user", "content": case["prompt"]},
        {"role": "assistant", "tool_calls": [{"id": "c0", "function": {
            "name": "create_calendar_event", "arguments": args}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "{}"}]}


def test_tool_scores_follow_harness_semantics():
    import run_bench

    case = CASE["T2"]
    # complete arguments go to the harness verbatim: missing key and malformed JSON score 2.5
    for args, want in (("{}", 2.5), ("not JSON", 2.5), ('{"start_iso": "15:00"}', 5.0)):
        got = rl.score_generation(case, [], [_t2_body(args)])
        trace = [[{"function_name": "create_calendar_event", "function_args": args}]]
        assert got.score == want == run_bench.score_tool_trace(
            {"tool_calls_trace": trace, "response_text": ""}, case)
        assert got.assumption == "" and got.low == got.high == want
    # a visible prefix that already differs decides the argument, under a stated assumption
    got = rl.score_generation(case, [], [_t2_body('{"title": "Walk", "start_iso": "2026-05-1'
                                                  + rl.TRUNC_MARK + '"}')])
    assert got.score == 2.5 and got.assumption == rl.HIDDEN_ARGS_ASSUMPTION
    assert (got.low, got.high) == (2.5, 5.0)
    # a hidden or escaped value leaves only the bounds
    for args in ('{"end_iso": "x", "loc' + rl.TRUNC_MARK + '"}',
                 '{"start_iso": "\\u0031' + rl.TRUNC_MARK + '"}'):
        got = rl.score_generation(case, [], [_t2_body(args)])
        assert got.score is None and (got.low, got.high) == (2.5, 5.0)
        assert "between 2.5 and 5.0" in got.evidence
    # a number cut at the marker is not taken as complete
    long_tail = '{"order_id": "A1009", "amount_usd": 84.0' + "1" * 90 + '}'
    t1 = rl.score_generation(CASE["T1"], [], [_trunc(body_for(G, "T1", 512,
                                                            t1_messages(2, refund_args=long_tail)))])
    assert t1.score is None


# ---------------------------------------------------------------------------
# end to end on the synthetic logs
# ---------------------------------------------------------------------------

def test_private_request_is_counted_but_never_written(recovered):
    calls, gens, stats, text = recovered
    assert stats["other"] == 1 and stats["requests"] == stats["benchmark"] + 1
    assert PRIVATE not in text
    assert all(c["case_id"] in CASE for c in calls)


def test_alignment_kinds_and_adhoc(recovered):
    calls, gens, stats, _ = recovered
    kinds = [(c["case_id"], c["call_kind"], c["model_label"]) for c in calls]
    assert kinds.count(("C1", "warmup", "gemma4_31b_bf16")) == 2
    warm = [c for c in calls if c["call_kind"] == "warmup"]
    assert [c["finish_reason_inferred"] for c in warm] == ["length", "length"]  # 64 of 64
    assert [c["repeat"] for c in warm] == [1, 2]
    assert ("C1", "adhoc", "gemma4_31b_bf16") in kinds
    assert stats["adhoc_benchmark"] == 1 and stats["runs"] == 1
    assert stats["dropped_benchmark"] == 0 and stats["unexplained_http"] == 0
    t1 = [c for c in calls if c["case_id"] == "T1" and c["model"] == G]
    assert [c["turn"] for c in t1] == [1, 2, 3]
    assert [c["finish_reason_inferred"] for c in t1] == ["tool_calls", "tool_calls", "stop"]


def test_overlap_flag_and_pairing_by_task(recovered):
    calls, _, stats, _ = recovered
    c1 = next(c for c in calls if c["case_id"] == "C1" and c["call_kind"] == "scored")
    assert c1["overlapped"] is True and c1["pairing"] == "unique"
    assert (c1["completion_tokens"], c1["server_eval_ms"]) == (200, 4000.0)
    assert c1["t_response"] == "2026-04-05T10:00:09"
    assert stats["overlapped_benchmark"] == 1


def test_generation_scores_and_timing(recovered):
    _, gens, _, _ = recovered
    by = {(g["case_id"], g["model_label"]): g for g in gens}
    assert len(gens) == 5 and all(g["complete"] for g in gens)
    c1 = by[("C1", "gemma4_31b_bf16")]
    assert (c1["score"], c1["score_method"]) == (5.0, "bench_log_unit_test")
    assert c1["total_s"] == pytest.approx(5.5)  # 10:00:04.000 -> 10:00:09.500
    i4 = by[("I4", "gemma4_31b_bf16")]
    assert i4["score"] == 2.0 and i4["finish_reason_inferred"] == "length"
    assert i4["length_truncated_turns"] == 1
    t1g = by[("T1", "gemma4_31b_bf16")]
    assert t1g["score"] == 5.0 and t1g["turns"] == 3
    assert t1g["score_assumption"] == rl.HIDDEN_ARGS_ASSUMPTION
    t1q = by[("T1", "qwen35_27b_q8")]
    assert (t1q["score"], t1q["turns"], t1q["score_assumption"]) == (0.0, 1, "")
    c2 = by[("C2", "gemma4_31b_bf16")]
    assert c2["score"] == 1.0  # multi-line "unit test failed" warning attributed to C2
    assert c2["total_s"] == pytest.approx(2.9)


def test_log_cut_inside_scoring_messages_is_not_complete(tmp_path):
    lm_text, bench_text = build_fixture()
    cut = bench_text.split("constraint fail: forbidden char")[0].rsplit("\n", 1)[0] + "\n"
    _, gens, _ = run_recover(tmp_path, lm_text, cut)
    i4 = next(g for g in gens if g["case_id"] == "I4")
    assert i4["complete"] is False and i4["score"] is None


def test_missing_middle_turn_is_not_complete(tmp_path):
    lm_text, bench_text = build_fixture()
    lines = lm_text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("[2026-04-05 10:00:14]")
                 and "Received request" in ln)
    end = next(i for i, ln in enumerate(lines) if ln.startswith("[2026-04-05 10:00:16]")
               and "Received request" in ln)
    _, gens, stats = run_recover(tmp_path, "\n".join(lines[:start] + lines[end:]), bench_text)
    t1g = next(g for g in gens if g["case_id"] == "T1" and g["model"] == G)
    assert t1g["complete"] is False and t1g["score"] is None


def test_same_second_request_with_wrong_seed_does_not_take_the_slot(tmp_path):
    lm, blog = [], bl_start()
    blog += [bl(4.000, "[1/1] C1 | gemma4_31b_bf16 | repeat 1"), bl_http(4.010), bl_http(4.020),
             bl(9.0, "benchmark complete")]
    lm += lm_call(4, 6, body_for(G, "C1", 512, seed=99), task=1, completion=11)
    lm += lm_call(4, 9, body_for(G, "C1", 512), task=2, completion=22)
    calls, gens, stats = run_recover(tmp_path, "\n".join(lm) + "\n", "\n".join(blog) + "\n")
    scored = next(c for c in calls if c["call_kind"] == "scored")
    assert (scored["seed"], scored["completion_tokens"]) == (42, 22)
    assert [c["seed"] for c in calls if c["call_kind"] == "adhoc"] == [99]


def test_http_line_before_the_request_second_is_not_paired():
    call = rl.BenchCall("scored", "gemma4_31b_bf16", "C1", 1, T0 + timedelta(seconds=9.8),
                        http=[T0 + timedelta(seconds=9.9), T0 + timedelta(seconds=10.01)])
    req = rl.LmRequest("x", 1, T0 + timedelta(seconds=10), G, _trunc(body_for(G, "C1", 512)),
                       case_id="C1", case=CASE["C1"])
    pairs, ambiguous = rl._match_call(call, [req])
    assert not ambiguous and [h for h, _ in pairs] == [1]


def test_fourth_tool_turn_finish_is_not_assumed_stop():
    req = rl.LmRequest("x", 1, T0, G, body_for(G, "T3", 512), completion_tokens=40,
                       case_id="T3", case=CASE["T3"], t_response=T0 + timedelta(seconds=4))
    assert rl._finish_reason(req, "scored", True, True, rl.MAX_TOOL_TURNS) == "stop_or_tool_calls"
    assert rl._finish_reason(req, "scored", True, False, 2) == ""


def test_task_ids_and_finish_lines_are_scoped():
    """Each model has its own llama.cpp task counter, and finish lines follow the task."""
    text = "\n".join(
        lm_call(0, 2, body_for(G, "C1", 512), task=7, completion=11)
        # same id 7, other model, after the first finished: must not touch the first request
        + lm_call(3, 5, body_for(Q, "C1", 4096), task=7, completion=22)
        # two live requests that share id 9: both ambiguous
        + lm_request(6, body_for(G, "C2", 512)) + lm_launch(6, 9, 30)
        + lm_request(6, body_for(Q, "C2", 4096)) + lm_launch(6, 9, 30)
        + lm_finish(8, G, 9, 100.0, 30, 900.0, 5)
        # same model, finishing out of order: "server assigned" decides
        + lm_request(10, body_for(G, "M1", 256)) + lm_launch(10, 20, 30)
        + lm_request(11, body_for(G, "M2", 256)) + lm_launch(11, 21, 30)
        + lm_finish(12, G, 21, 100.0, 30, 900.0, 6)
        + lm_finish(15, G, 20, 100.0, 30, 3900.0, 9)
    ) + "\n"
    reqs = rl.parse_lmstudio_log(text, "x.log")
    assert [r.completion_tokens for r in reqs[:2]] == [11, 22]
    assert not reqs[0].ambiguous and not reqs[1].ambiguous
    assert reqs[2].ambiguous and reqs[3].ambiguous
    m1, m2 = reqs[4], reqs[5]
    assert (m1.t_response, m2.t_response) == (T0 + timedelta(seconds=15), T0 + timedelta(seconds=12))
    assert not m1.ambiguous and not m2.ambiguous


def test_candidates_of_an_ambiguous_call_are_never_exported(tmp_path):
    """Two continuations fit one turn: the call is ambiguous and none of its requests leave."""
    lm, blog = [], bl_start()
    blog += [bl(12.300, "[1/1] T1 | gemma4_31b_bf16 | repeat 1"), bl_http(12.310), bl_http(14.310),
             bl(18.0, "benchmark complete")]
    private_args = '{"amount_usd": 84.0, "order_id": "A1009", "note": "' + PRIVATE + '"}'
    lm += lm_call(12, 14, body_for(G, "T1", 512, t1_messages(0)), task=5)
    lm += lm_call(14, 16, body_for(G, "T1", 512, t1_messages(2)), task=6)
    lm += lm_call(14, 17, body_for(G, "T1", 512, t1_messages(2, refund_args=private_args)), task=7)
    calls, gens, stats = run_recover(tmp_path, "\n".join(lm) + "\n", "\n".join(blog) + "\n")
    assert PRIVATE not in json.dumps(calls) + json.dumps(gens, default=str)
    assert stats["ambiguous_calls"] == 1 and stats["dropped_benchmark"] == 3
    assert not [c for c in calls if c["case_id"] == "T1"]


def test_two_bodies_before_their_start_lines_are_ambiguous():
    lines = []
    for body in (body_for(G, "C1", 512), body_for(G, "C2", 512)):
        req = lm_request(0, body)
        lines += [ln for ln in req if "Running chat completion" not in ln and "Streaming" not in ln]
    lines += [f"{_lm_ts(0)}[INFO][{G}] Running chat completion on conversation with 1 messages."]
    lines += lm_launch(0, 1, 30) + lm_finish(1, G, 1, 100.0, 30, 900.0, 11)
    lines += [f"{_lm_ts(1)}[INFO][{G}] Running chat completion on conversation with 1 messages."]
    lines += lm_launch(1, 2, 30) + lm_finish(2, G, 2, 100.0, 30, 900.0, 22)
    reqs = rl.parse_lmstudio_log("\n".join(lines) + "\n", "x.log")
    assert all(r.ambiguous for r in reqs)


def test_scored_turn_needs_the_case_system_prompt_and_tool_schema(tmp_path):
    base = _trunc(body_for(G, "T1", 512, t1_messages(0)))
    req = rl.LmRequest("x", 1, T0, G, base, case_id="T1", case=CASE["T1"])
    call = rl.BenchCall("scored", "gemma4_31b_bf16", "T1", 1, T0)
    assert rl._config_fits(req, call)
    no_system = dict(base, messages=base["messages"][1:])
    assert not rl._config_fits(rl.LmRequest("x", 1, T0, G, no_system, case_id="T1",
                                            case=CASE["T1"]), call)
    other_schema = json.loads(json.dumps(base))
    other_schema["tools"][0]["function"]["parameters"] = {"type": "object", "properties": {}}
    assert rl.classify_request(other_schema, CASES) is None


def test_missing_server_timing_lines_make_the_generation_incomplete(tmp_path):
    lm_text, bench_text = build_fixture()
    lines = lm_text.split("\n")
    # drop the "total time" and release lines of the C2 request (task 8)
    i = next(k for k, ln in enumerate(lines) if "print_timing" in ln and "task 8 " in ln)
    lines = lines[:i + 3] + lines[i + 5:]
    _, gens, _ = run_recover(tmp_path, "\n".join(lines), bench_text)
    c2 = next(g for g in gens if g["case_id"] == "C2")
    assert c2["complete"] is False and c2["score"] is None and c2["server_total_s"] is None


def _t2_messages(with_calls, calendar_args='{"title": "Walk", "start_iso": "2026-05-12T15:00:00"}'):
    case = CASE["T2"]
    msgs = [{"role": "user", "content": case["prompt"]}]
    calls = [("get_weather", '{"location": "San Francisco", "date": "2026-05-12"}',
              case["tool_responses"]["get_weather"]),
             ("create_calendar_event", calendar_args, "{}")]
    for n, (name, args, resp) in enumerate(calls[:with_calls]):
        msgs.append({"role": "assistant", "tool_calls": [
            {"id": f"w{n}", "type": "function", "function": {"name": name, "arguments": args}}]})
        msgs.append({"role": "tool", "tool_call_id": f"w{n}", "content": resp})
    return msgs


def test_taint_of_an_ambiguous_call_holds_for_later_calls(tmp_path):
    lm, blog = [], bl_start()
    blog += [bl(4.000, "[1/2] T2 | gemma4_31b_bf16 | repeat 1"), bl_http(4.010), bl_http(5.200),
             bl(5.500, "[2/2] T2 | gemma4_31b_bf16 | repeat 2"), bl_http(5.510), bl_http(5.910),
             bl(7.100, "benchmark complete")]
    private_args = '{"title": "' + PRIVATE + '", "start_iso": "15:00"}'
    lm += lm_call(4, 5, body_for(G, "T2", 512, _t2_messages(0)), task=1)
    lm += lm_call(5, 6, body_for(G, "T2", 512, _t2_messages(1)), task=2)
    lm += lm_call(5, 6, body_for(G, "T2", 512, _t2_messages(0)), task=3)
    # a second continuation that also extends repeat 1's first turn, carrying private text
    lm += lm_call(5, 6, body_for(G, "T2", 512, _t2_messages(2, private_args)), task=4)
    calls, gens, stats = run_recover(tmp_path, "\n".join(lm) + "\n", "\n".join(blog) + "\n")
    assert PRIVATE not in json.dumps(calls) + json.dumps(gens, default=str)
    assert not [g for g in gens if g["complete"]]


def test_scored_request_needs_harness_top_p_and_streaming(tmp_path):
    for change in ({"top_p": 0.25}, {"stream": False}):
        lm, blog = [], bl_start()
        blog += [bl(4.000, "[1/1] C1 | gemma4_31b_bf16 | repeat 1"), bl_http(4.010),
                 bl(9.0, "benchmark complete")]
        lm += lm_call(4, 9, dict(body_for(G, "C1", 512), **change), task=1)
        calls, gens, stats = run_recover(tmp_path, "\n".join(lm) + "\n", "\n".join(blog) + "\n")
        assert not gens and not [c for c in calls if c["call_kind"] == "scored"], change


def test_finish_reason_unknown_without_a_finish_line():
    req = rl.LmRequest("x", 1, T0, G, body_for(G, "C1", 4096), completion_tokens=40,
                       case_id="C1", case=CASE["C1"])
    assert rl._finish_reason(req, "adhoc", True, True, 1) == ""
    req.t_response = T0 + timedelta(seconds=3)
    assert rl._finish_reason(req, "adhoc", True, True, 1) == "stop"
