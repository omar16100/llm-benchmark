#!/usr/bin/env python3
"""Recover a run_bench.py run from LM Studio server logs plus the harness bench.log.

Built to recover the 5 Apr 2026 Gemma 4 31B (bf16) vs Qwen 3.5 27B (Q8_0) run, whose
results/runs.csv and transcripts.jsonl were never committed and were later overwritten.
See docs/27092026_recovered_gemma4_qwen35_run.md for provenance and findings.

Inputs (local only, never committed):
  * LM Studio server logs (for example ~/.lmstudio/server-logs/2026-04/2026-04-05.1.log).
    They hold every request body and llama.cpp slot timings, but NOT the text of streamed
    responses. Strings longer than 120 UTF-16 units are logged as their first 50 and last 50
    units around a "... <Truncated in logs> ..." marker. The logs also hold unrelated traffic
    from other clients. A request is treated as a benchmark request only if its messages are
    exactly what LM Studio would log for a cases.json case (classify_request); every other
    request keeps nothing but its model and time (to explain harness log lines) and no field of
    it reaches the outputs.
  * the harness's own bench.log (gitignored): millisecond timestamps for each warmup group and
    scored generation, one httpx line per HTTP request, and the scoring messages run_bench.py
    logged while it ran.

A benchmark request is exported only when it is tied to the harness: either paired with an
HTTP line of the harness call that sent it (matching model, case, seed, temperature, budget,
turn order and time), or, for a request made outside the harness loop, paired with a stray
harness HTTP line that no other request (benchmark or not) could have produced ("adhoc").
Anything else, including every candidate of a call whose pairing is ambiguous, is dropped
and counted.

Outputs:
  * calls CSV: one row per exported benchmark HTTP request (warmups and every tool-use turn).
  * generations CSV: one row per scored generation (run x model x case x repeat).

Scores are filled only where the logs determine them; every other score is left empty with
score_method "unrecoverable". See score_generation for the rules.

Usage:
  uv run python scripts/recover_lmstudio_log.py \\
      --lmstudio-log ~/.lmstudio/server-logs/2026-04/2026-04-05.1.log \\
      --bench-log /path/to/llm-benchmark/bench.log \\
      --calls-csv data/recovered_05apr2026_gemma4_qwen35.csv \\
      --generations-csv data/recovered_05apr2026_gemma4_qwen35_generations.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("recover_lmstudio_log")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRUNC_MARK = "... <Truncated in logs> ..."
LM_TRUNC_ABOVE = 120  # LM Studio logs strings longer than this many UTF-16 units shortened
LM_TRUNC_KEEP = 50  # ... to this many units at each end

# served model id (as sent in the request body) -> run_bench.py MODELS label, as
# configured in the initial commit ca4a63a that ran on 5 Apr 2026.
DEFAULT_MODEL_LABELS = {
    "gemma-4-31b": "gemma4_31b_bf16",
    "qwen3.5-27b": "qwen35_27b_q8",
}

# run_bench.py at ca4a63a
WARMUP_RUNS = 2
WARMUP_MAX_TOKENS_CAP = 64  # warmup max_tokens = min(case max_tokens, 64)
WARMUP_SEED = 42  # stream_completion default; warmups pass no seed
HARNESS_TOP_P = 1.0  # stream_completion default; run_benchmark never passes top_p
DETERMINISTIC_SEEDS = [42, 42, 42]
CREATIVE_SEEDS = [41, 42, 43]
MAX_TOOL_TURNS = 4

# An LM Studio request line precedes the harness's httpx line for the same request by a few
# milliseconds, and LM Studio logs whole seconds: 0 <= harness time - floor(LM time) < 1 s
# plus that delay.
PAIR_WINDOW_S = 1.5
# run_bench.py logs "[i/N]" or "warmup:" and sends the first request at once (about 15 ms
# later in the 5 Apr log), so a call's first request is the one right after its log line.
FIRST_REQUEST_WINDOW_S = 1.0

# ---------------------------------------------------------------------------
# LM Studio server log parsing
# ---------------------------------------------------------------------------

LM_TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
LM_REQ_MARK = "Received request: POST to /v1/chat/completions with body {"
LM_MODEL_INFO_RE = re.compile(r"^\[[^\]]+\]\[INFO\]\[([^\]]+)\] (.*)$")
LM_LAUNCH_RE = re.compile(r"slot launch_slot_: id +(\d+) \| task (\d+) \| processing task")
LM_NEW_PROMPT_RE = re.compile(
    r"slot update_slots: id +\d+ \| task (\d+) \| new prompt, n_ctx_slot = (\d+), "
    r"n_keep = \d+, task\.n_tokens = (\d+)")
LM_TIMING_HDR_RE = re.compile(r"slot print_timing: id +\d+ \| task (\d+) \|")
LM_PROMPT_EVAL_RE = re.compile(r"^\s*prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
LM_EVAL_RE = re.compile(r"^\s*eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
LM_TOTAL_RE = re.compile(r"^\s*total time =\s*([\d.]+) ms /\s*(\d+) tokens")
LM_RELEASE_RE = re.compile(
    r"slot +release: id +\d+ \| task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d+)")
LM_ASSIGNED_RE = re.compile(r"LlamaV4: server assigned slot \d+ to task (\d+)")


@dataclass
class LmRequest:
    """One POST /v1/chat/completions seen in an LM Studio server log."""
    source: str  # log file name (no directory)
    line_no: int  # 1-based line of the "Received request" line
    t_request: datetime
    model: str
    body: Optional[dict]  # set to None for non-benchmark requests
    stream: bool = False
    parse_error: bool = False
    started: bool = False  # "Running chat completion" seen
    task_id: Optional[int] = None
    n_ctx_slot: Optional[int] = None
    prompt_tokens: Optional[int] = None  # task.n_tokens: full prompt size
    prompt_eval_tokens: Optional[int] = None  # tokens actually evaluated (cache hits excluded)
    prompt_eval_ms: Optional[float] = None
    completion_tokens: Optional[int] = None
    eval_ms: Optional[float] = None
    server_total_ms: Optional[float] = None
    slot_truncated: Optional[bool] = None
    t_response: Optional[datetime] = None
    overlapped: bool = False  # another request was in flight at the same time
    ambiguous: bool = False  # pairing with its llama.cpp task or finish line is not unique
    case_id: Optional[str] = None  # set by classification
    case: Optional[dict] = None


def _parse_body(lines: list, start: int) -> tuple:
    """Collect the pretty-printed JSON body that starts after lines[start].

    LM Studio prints the body with the closing brace alone on a line; JSON strings cannot
    contain a raw newline, so the first line equal to "}" ends the body.
    Returns (body or None, index of the closing line).
    """
    buf = ["{"]
    j = start + 1
    while j < len(lines) and lines[j] != "}":
        buf.append(lines[j])
        j += 1
    buf.append("}")
    try:
        body = json.loads("\n".join(buf))
        return (body if isinstance(body, dict) else None), j
    except json.JSONDecodeError:
        return None, j


def parse_lmstudio_log(text: str, source: str) -> list:
    """Parse one LM Studio server log into LmRequest records with llama.cpp timings.

    Pairing, in log order:
      * "Received request" creates the request; its model's next "Running chat completion"
        line marks it started.
      * the next "slot launch_slot_ ... processing task" binds the oldest started request to
        that task. If more than one request was waiting, the binding is flagged ambiguous.
      * prompt size, timings and release lines are read by task id while the task is live
        (launch to release). Each model has its own llama.cpp server and task counter, so a
        task id live for two requests at once is flagged ambiguous.
      * "server assigned slot S to task T" names the request that the model's next
        "Finished streaming response" (or "Generated prediction") belongs to. Without it, the
        oldest in-flight request of that model and streaming mode is used, and flagged
        ambiguous if there was more than one.
    Requests that were in flight at the same time as another request are flagged overlapped.
    """
    lines = text.split("\n")
    requests: list = []
    started_q: list = []
    live: dict = {}
    released: dict = {}
    last_assigned: Optional[LmRequest] = None
    inflight: list = []
    timing_task: Optional[int] = None
    current_ts: Optional[datetime] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m_ts = LM_TS_RE.match(line)
        if m_ts:
            current_ts = datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S")
        if LM_REQ_MARK in line:
            body, end = _parse_body(lines, i)
            req = LmRequest(source=source, line_no=i + 1, t_request=current_ts,
                            model=(body or {}).get("model") or "", body=body,
                            stream=(body or {}).get("stream") is True, parse_error=body is None)
            if inflight:
                req.overlapped = True
                for other in inflight:
                    other.overlapped = True
            requests.append(req)
            inflight.append(req)
            i = end + 1
            continue
        m_info = LM_MODEL_INFO_RE.match(line)
        if m_info:
            model, msg = m_info.group(1), m_info.group(2)
            if msg.startswith("Running chat completion"):
                waiting = [r for r in requests
                           if r.model == model and not r.started and r.t_response is None]
                if len(waiting) > 1:  # which body this start line belongs to is not unique
                    for r in waiting:
                        r.ambiguous = True
                if waiting:
                    waiting[0].started = True
                    started_q.append(waiting[0])
            elif msg.startswith(("Finished streaming response", "Generated prediction")):
                streamed = msg.startswith("Finished streaming")
                cand = last_assigned
                if not (cand is not None and cand.model == model and cand.t_response is None
                        and cand.stream == streamed):
                    pool = [r for r in inflight if r.model == model and r.stream == streamed]
                    cand = pool[0] if pool else None
                    if len(pool) > 1:
                        for r in pool:
                            r.ambiguous = True
                if cand is not None:
                    cand.t_response = current_ts
                    inflight.remove(cand)
                    if cand in started_q:  # finished without ever launching (cancelled)
                        started_q.remove(cand)
                last_assigned = None
            i += 1
            continue
        m = LM_LAUNCH_RE.search(line)
        if m:
            task = int(m.group(2))
            if not started_q:  # a task from another endpoint (for example /v1/completions)
                log.debug("%s:%d task %d launched with no waiting chat request", source, i + 1, task)
            else:
                if len(started_q) > 1:
                    for r in started_q:
                        r.ambiguous = True
                req = started_q.pop(0)
                clash = live.get(task)
                if clash is not None:
                    log.warning("%s:%d task id %d is live for two requests (lines %d, %d)",
                                source, i + 1, task, clash.line_no, req.line_no)
                    clash.ambiguous = req.ambiguous = True
                req.task_id = task
                live[task] = req
            i += 1
            continue
        m = LM_NEW_PROMPT_RE.search(line)
        if m and int(m.group(1)) in live:
            req = live[int(m.group(1))]
            req.n_ctx_slot, req.prompt_tokens = int(m.group(2)), int(m.group(3))
            i += 1
            continue
        m = LM_TIMING_HDR_RE.search(line)
        if m:
            timing_task = int(m.group(1))
            i += 1
            continue
        if timing_task is not None and timing_task in live:
            req = live[timing_task]
            m = LM_PROMPT_EVAL_RE.match(line)
            if m:
                req.prompt_eval_ms, req.prompt_eval_tokens = float(m.group(1)), int(m.group(2))
                i += 1
                continue
            m = LM_EVAL_RE.match(line)
            if m:
                req.eval_ms, req.completion_tokens = float(m.group(1)), int(m.group(2))
                i += 1
                continue
            m = LM_TOTAL_RE.match(line)
            if m:
                req.server_total_ms, timing_task = float(m.group(1)), None
                i += 1
                continue
        m = LM_RELEASE_RE.search(line)
        if m and int(m.group(1)) in live:
            req = live.pop(int(m.group(1)))
            req.slot_truncated = m.group(3) != "0"
            released[int(m.group(1))] = req
            i += 1
            continue
        m = LM_ASSIGNED_RE.search(line)
        if m:
            task = int(m.group(1))
            last_assigned = released.pop(task, None) or live.get(task)
        i += 1
    log.info("%s: %d chat completion requests parsed (%d body parse errors, %d without timings, "
             "%d ambiguous)", source, len(requests), sum(r.parse_error for r in requests),
             sum(r.eval_ms is None for r in requests), sum(r.ambiguous for r in requests))
    return requests


# ---------------------------------------------------------------------------
# Benchmark classification
# ---------------------------------------------------------------------------

def _utf16(s: str) -> bytes:
    return s.encode("utf-16-le", "surrogatepass")


def lm_logged_form(full: str) -> str:
    """The string exactly as LM Studio logs it: verbatim, or 50 + marker + 50 UTF-16 units."""
    raw = _utf16(full)
    if len(raw) // 2 <= LM_TRUNC_ABOVE:
        return full
    keep = 2 * LM_TRUNC_KEEP
    head = raw[:keep].decode("utf-16-le", "surrogatepass")
    tail = raw[-keep:].decode("utf-16-le", "surrogatepass")
    return head + TRUNC_MARK + tail


def lm_logged_value(value):
    """A JSON value with every string in the form LM Studio logs it."""
    if isinstance(value, str):
        return lm_logged_form(value)
    if isinstance(value, list):
        return [lm_logged_value(v) for v in value]
    if isinstance(value, dict):
        return {k: lm_logged_value(v) for k, v in value.items()}
    return value


def logged_matches(logged: object, full: str) -> bool:
    """True if `logged` is exactly what LM Studio writes for the string `full`."""
    return isinstance(logged, str) and logged == lm_logged_form(full)


def _tool_names(tools) -> list:
    return sorted((t.get("function") or {}).get("name") or "" for t in tools or [])


def classify_request(body: Optional[dict], cases: list) -> Optional[dict]:
    """Return the cases.json case this request body was sent for, or None.

    A body is a benchmark request only if ALL of these hold (every string compared in the
    exact form LM Studio logs it, see lm_logged_form):
      * the first non-system message is a user message matching exactly one case prompt;
      * a system message, if present, matches that case's system prompt;
      * tools, if sent, are exactly the case's tool definitions (tool_use cases only);
      * every later message is an assistant tool call to one of the case's tools followed by
        a tool message whose content is one of the case's canned tool_responses or "{}"
        (run_bench.py's default).
    Two different prompts that share their first and last 50 characters would still match
    the same logged form, which is why the caller also requires harness provenance.
    """
    if not isinstance(body, dict):
        return None
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
        return None
    idx = 0
    system = None
    if msgs[0].get("role") == "system":
        system, idx = msgs[0], 1
    if idx >= len(msgs) or msgs[idx].get("role") != "user":
        return None
    hits = [c for c in cases if logged_matches(msgs[idx].get("content"), c["prompt"])]
    if len(hits) != 1:
        return None
    case = hits[0]
    if system is not None and not (case.get("system")
                                   and logged_matches(system.get("content"), case["system"])):
        return None
    rest = msgs[idx + 1:]
    tools = body.get("tools") or []
    if (rest or tools) and case.get("category") != "tool_use":
        return None
    if tools and tools != lm_logged_value(case.get("tools")):
        return None
    if len(rest) % 2:
        return None
    allowed = list((case.get("tool_responses") or {}).values()) + ["{}"]
    case_tools = set(_tool_names(case.get("tools")))
    for asst, tool in zip(rest[::2], rest[1::2]):
        calls = asst.get("tool_calls") or []
        if asst.get("role") != "assistant" or len(calls) != 1 or tool.get("role") != "tool":
            return None  # run_bench.py replays one assistant message per tool call
        if (calls[0].get("function") or {}).get("name") not in case_tools:
            return None
        if tool.get("tool_call_id") != calls[0].get("id"):
            return None
        if not any(logged_matches(tool.get("content"), v) for v in allowed):
            return None
    return case


def visible_tool_calls(body: dict) -> list:
    """Tool calls (name, logged arguments) replayed in a continuation request, in order."""
    out = []
    for msg in body.get("messages") or []:
        if msg.get("role") == "assistant":
            for c in msg.get("tool_calls") or []:
                fn = c.get("function") or {}
                out.append({"function_name": fn.get("name") or "",
                            "function_args": fn.get("arguments") or ""})
    return out


class ArgPrefix(str):
    """A string argument value of which only a (JSON-unescaped) prefix is visible."""


CUT = object()  # a value whose visible part does not determine it (for example 84.0 of 84.01)


def parse_visible_args(args: str) -> dict:
    """Key/value pairs readable from the head of a truncated tool-call argument string.

    Values are decoded from the head up to the truncation marker. A string value whose
    closing quote is visible is complete. A number or literal is complete only if a ","
    or "}" follows it inside the head. A string cut by the marker is kept as an ArgPrefix
    if it has no escape sequence (so the prefix is the decoded text), else as CUT.
    The hidden middle may still hold anything, including a repeated key; callers state that
    assumption when they use these values.
    """
    head = args.split(TRUNC_MARK, 1)[0]
    dec = json.JSONDecoder()
    out: dict = {}
    pos = head.find("{")
    if pos < 0:
        return out
    pos += 1
    while True:
        while pos < len(head) and head[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(head) or head[pos] != '"':
            break
        try:
            key, pos = dec.raw_decode(head, pos)
        except json.JSONDecodeError:
            break
        while pos < len(head) and head[pos] in " \t\r\n":
            pos += 1
        if pos >= len(head) or head[pos] != ":":
            break
        pos += 1
        while pos < len(head) and head[pos] in " \t\r\n":
            pos += 1
        try:
            value, end = dec.raw_decode(head, pos)
        except json.JSONDecodeError:
            if pos < len(head) and head[pos] == '"':
                rest = head[pos + 1:]
                out[key] = CUT if "\\" in rest else ArgPrefix(rest)
            else:
                out[key] = CUT
            break
        nxt = end
        while nxt < len(head) and head[nxt] in " \t\r\n":
            nxt += 1
        delimited = nxt < len(head) and head[nxt] in ",}"
        if isinstance(value, str) or delimited:
            out[key] = value
        else:
            out[key] = CUT
            break
        pos = nxt
    return out


# ---------------------------------------------------------------------------
# bench.log parsing
# ---------------------------------------------------------------------------

BL_LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) (\w+) (.*)$")
BL_SCORED_RE = re.compile(r"^\[(\d+)/(\d+)\] (\S+) \| (\S+) \| repeat (\d+)$")
BL_WARMUP_RE = re.compile(r"^warmup: (\S+) / (\S+)$")
BL_HTTP_CHAT_RE = re.compile(r"^HTTP Request: POST \S+/chat/completions ")
SCORING_PREFIXES = ("unit test failed", "unit test timeout", "unit test error",
                    "constraint fail:", "expected tool ")


@dataclass
class BenchCall:
    kind: str  # "warmup" or "scored"
    label: str
    case_id: str
    repeat: int  # scored repeat (1-3); for warmups, 0
    t_start: datetime
    http: list = field(default_factory=list)  # datetimes of chat HTTP lines inside the call
    scoring: list = field(default_factory=list)  # scoring messages (first line of each)
    t_end: Optional[datetime] = None  # first harness line after the call's requests
    closed: bool = False  # the next call (or "benchmark complete") was logged
    ambiguous: bool = False  # more than one way to pair its HTTP lines with requests
    unexplained_http: int = 0  # HTTP lines matched by no LM Studio request at all


@dataclass
class BenchRun:
    label: str
    t_start: datetime
    complete: bool = False
    t_end: Optional[datetime] = None
    calls: list = field(default_factory=list)
    stray_http: list = field(default_factory=list)  # chat HTTP lines outside any call


def parse_bench_log(text: str, since: Optional[datetime] = None,
                    until: Optional[datetime] = None) -> list:
    """Split a run_bench.py bench.log into runs of warmup and scored calls.

    Only lines with since <= timestamp < until are read. A run starts at an "available
    models" line and is complete when "benchmark complete" is logged before the next start.
    A call ends (t_end) at the first harness line after its HTTP lines (a scoring message or
    the next call), and is closed only once the next call or "benchmark complete" is logged,
    so a log cut in the middle of a call's scoring messages leaves it open.
    """
    runs: list = []
    current: Optional[BenchCall] = None

    def finish(call: Optional[BenchCall], ts: datetime, close: bool) -> None:
        if call is None:
            return
        if call.t_end is None:
            call.t_end = ts
        if close:
            call.closed = True

    for raw in text.split("\n"):
        m = BL_LINE_RE.match(raw)
        if not m:
            continue  # continuation of a multi-line message (for example a traceback)
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") + timedelta(
            milliseconds=int(m.group(2)))
        if (since and ts < since) or (until and ts >= until):
            continue
        msg = m.group(4)
        if msg.startswith("available models:"):
            runs.append(BenchRun(label=chr(ord("A") + len(runs)), t_start=ts))
            current = None  # a previous run without "benchmark complete" was aborted
            continue
        if not runs:
            continue
        run = runs[-1]
        if BL_HTTP_CHAT_RE.match(msg):
            if current is not None and current.t_end is None:
                current.http.append(ts)
            else:
                run.stray_http.append(ts)
            continue
        if msg.startswith(SCORING_PREFIXES):
            finish(current, ts, close=False)
            if current is not None:
                current.scoring.append(msg)
            continue
        if msg.startswith("benchmark complete"):
            finish(current, ts, close=True)
            run.complete, run.t_end, current = True, ts, None
            continue
        m_s = BL_SCORED_RE.match(msg)
        m_w = BL_WARMUP_RE.match(msg)
        if m_s or m_w:
            finish(current, ts, close=True)
            if m_s:
                current = BenchCall("scored", m_s.group(4), m_s.group(3), int(m_s.group(5)), ts)
            else:
                current = BenchCall("warmup", m_w.group(1), m_w.group(2), 0, ts)
            run.calls.append(current)
    for run in runs:
        log.info("bench.log run %s: start %s, %s, %d warmup groups, %d scored calls",
                 run.label, run.t_start, "complete" if run.complete else "aborted",
                 sum(c.kind == "warmup" for c in run.calls),
                 sum(c.kind == "scored" for c in run.calls))
    return runs


# ---------------------------------------------------------------------------
# Alignment of LM Studio requests with harness calls
# ---------------------------------------------------------------------------

def _floor_s(ts: datetime) -> datetime:
    return ts.replace(microsecond=0)


def _lag(req: LmRequest, http_ts: datetime) -> float:
    """Harness HTTP line time minus the LM Studio request's (whole-second) time."""
    return (http_ts - req.t_request).total_seconds()


def _pairable(req: LmRequest, http_ts: datetime) -> bool:
    return 0.0 <= _lag(req, http_ts) <= PAIR_WINDOW_S


def request_shape(body: dict, case: dict) -> str:
    """"warmup", "turn1" (first request of a scored generation) or "continuation"."""
    msgs = body.get("messages") or []
    if any(m.get("role") == "assistant" for m in msgs):
        return "continuation"
    warm_cap = min(case.get("max_tokens", 512), WARMUP_MAX_TOKENS_CAP)
    if not body.get("tools") and len(msgs) == 1 and body.get("max_tokens") == warm_cap:
        return "warmup"
    return "turn1"


def expected_seed(case: dict, repeat: int) -> int:
    seeds = CREATIVE_SEEDS if case.get("temperature", 0.0) > 0 else DETERMINISTIC_SEEDS
    return seeds[repeat - 1]


def _config_fits(req: LmRequest, call: BenchCall) -> bool:
    """Does the request carry the configuration run_bench.py sends for this call?"""
    body, case = req.body or {}, req.case or {}
    if body.get("temperature") != case.get("temperature", 0.0):
        return False
    if body.get("top_p") != HARNESS_TOP_P or body.get("stream") is not True:
        return False  # stream_completion always streams with its default top_p
    shape = request_shape(body, case)
    if call.kind == "warmup":
        return shape == "warmup" and body.get("seed") == WARMUP_SEED
    if shape == "warmup" or body.get("seed") != expected_seed(case, call.repeat):
        return False
    budget, base = body.get("max_tokens"), case.get("max_tokens", 512)
    if not isinstance(budget, int) or budget <= 0 or budget % base:
        return False  # scored budget is the case budget times the model's thinking multiplier
    has_system = ((body.get("messages") or [{}])[0].get("role") == "system")
    if has_system != bool(case.get("system")):
        return False  # run_tool_use_case sends the case's system prompt on every scored turn
    return bool(body.get("tools")) == (case.get("category") == "tool_use")


def _msg_key(msg: dict) -> tuple:
    calls = tuple((c.get("id"), (c.get("function") or {}).get("name"),
                   (c.get("function") or {}).get("arguments")) for c in msg.get("tool_calls") or [])
    return msg.get("role"), msg.get("content"), msg.get("tool_call_id"), calls


def _extends(prev: LmRequest, nxt: LmRequest) -> bool:
    """Is nxt a later turn of the same tool-use generation as prev?"""
    a = [_msg_key(m) for m in (prev.body or {}).get("messages") or []]
    b = [_msg_key(m) for m in (nxt.body or {}).get("messages") or []]
    return (len(b) > len(a) and b[:len(a)] == a
            and (prev.body or {}).get("max_tokens") == (nxt.body or {}).get("max_tokens"))


def _sequence_ok(call: BenchCall, reqs: list) -> bool:
    if call.kind == "warmup":
        return len(reqs) <= WARMUP_RUNS
    if not reqs or request_shape(reqs[0].body, reqs[0].case) != "turn1":
        return False
    if len(reqs) > MAX_TOOL_TURNS:
        return False
    return all(_extends(a, b) for a, b in zip(reqs, reqs[1:]))


def _match_call(call: BenchCall, cands: list) -> tuple:
    """Best order-preserving pairing of a call's HTTP lines with candidate requests.

    Returns ([(index into call.http, request)], ambiguous). The first paired HTTP line must
    follow the call's own log line within FIRST_REQUEST_WINDOW_S. The best pairing pairs the
    most HTTP lines; if best pairings with different requests exist, the call is ambiguous
    and nothing is paired.
    """
    best, best_n, n_best = [], 0, 0
    for k in range(min(len(call.http), len(cands)), 0, -1):
        found = []
        for h_idx in itertools.combinations(range(len(call.http)), k):
            if (call.http[h_idx[0]] - call.t_start).total_seconds() > FIRST_REQUEST_WINDOW_S:
                continue
            for r_idx in itertools.combinations(range(len(cands)), k):
                reqs = [cands[j] for j in r_idx]
                if not all(_pairable(cands[j], call.http[h]) for h, j in zip(h_idx, r_idx)):
                    continue
                if not _sequence_ok(call, reqs):
                    continue
                found.append(list(zip(h_idx, r_idx)))
        if found:
            # the same requests paired with different HTTP lines of the call (for example an
            # ad hoc line in the same second) is not ambiguous about the requests; the
            # earliest lines win, because the harness sends right after its log line
            best, best_n = found[0], k
            n_best = len({tuple(j for _, j in pairing) for pairing in found})
            break
    if n_best > 1:
        return [], True
    return [(h, cands[j]) for h, j in best], False


def align(runs: list, bench_reqs: list, other_reqs: list, model_labels: dict) -> tuple:
    """Assign each LM Studio benchmark request to the harness call that sent it.

    For every harness call, the candidates are unassigned benchmark requests of the same model
    and case, inside the call's time span, whose configuration fits (seed, temperature, budget,
    tools, warmup shape). _match_call pairs them with the call's HTTP lines in order, and
    tool-use turns must extend the previous turn's messages. A call with two equally good
    pairings is marked ambiguous and pairs nothing.

    Harness HTTP lines left unpaired (inside a call or between calls) are then explained by
    an unassigned benchmark request of the same second (exported as ad hoc) or by a
    non-benchmark request (not exported). A call with an HTTP line that nothing explains is
    counted in `unexplained_http`, which makes its generation incomplete.

    Returns (aligned rows, ad hoc requests, number of benchmark requests dropped).
    """
    label_to_model = {v: k for k, v in model_labels.items()}
    assigned: set = set()
    tainted: set = set()  # candidates of ambiguous calls
    paired_http: dict = {}  # id(call) -> indices of its HTTP lines that were paired
    rows = []
    for n, run in enumerate(runs):
        run_hi = runs[n + 1].t_start if n + 1 < len(runs) else None
        for c_idx, call in enumerate(run.calls):
            nxt = run.calls[c_idx + 1].t_start if c_idx + 1 < len(run.calls) else run_hi
            hi = call.t_end or nxt
            model = label_to_model.get(call.label)
            cands = [r for r in bench_reqs if id(r) not in assigned and id(r) not in tainted
                     and r.model == model
                     and r.case_id == call.case_id and r.t_request >= _floor_s(call.t_start)
                     and (hi is None or r.t_request <= hi) and _config_fits(r, call)]
            cands.sort(key=lambda r: (r.source, r.line_no))
            pairs, ambiguous = _match_call(call, cands)
            if ambiguous:
                call.ambiguous = True
                tainted.update(id(r) for r in cands)  # never exported, not even as ad hoc
                log.warning("run %s %s %s %s repeat %d: ambiguous pairing, left unpaired",
                            run.label, call.kind, call.label, call.case_id, call.repeat)
            paired_http[id(call)] = {h for h, _ in pairs}
            for k, (h, req) in enumerate(pairs):
                assigned.add(id(req))
                t_next = call.http[pairs[k + 1][0]] if k + 1 < len(pairs) else call.t_end
                rows.append({"run": run, "call": call, "req": req, "index": k + 1,
                             "n_http": len(pairs), "bench_t_request": call.http[h],
                             "bench_t_next": t_next})
    # explain every unpaired harness HTTP line
    adhoc, used_other = [], set()
    for run in runs:
        loose = [(h, call) for call in run.calls for i, h in enumerate(call.http)
                 if i not in paired_http.get(id(call), set())]
        loose += [(h, None) for h in run.stray_http]
        for h, call in sorted(loose, key=lambda x: x[0]):
            bench = [r for r in bench_reqs if id(r) not in assigned and _pairable(r, h)]
            other = [r for r in other_reqs if id(r) not in used_other and _pairable(r, h)]
            if len(bench) + len(other) > 1 or any(id(r) in tainted for r in bench):
                # more than one request could have produced this line: export none of them
                tainted.update(id(r) for r in bench)
                if call is not None:
                    call.unexplained_http += 1
                log.warning("harness HTTP line %s matches %d requests; none exported", h,
                            len(bench) + len(other))
                continue
            if bench:
                req = bench[0]
                assigned.add(id(req))
                adhoc.append((req, h))
                log.info("ad hoc benchmark request %s:%d (%s %s, max_tokens %s) at harness line %s",
                         req.source, req.line_no, req.model, req.case_id,
                         req.body.get("max_tokens"), h)
                if (call is not None and req.case_id == call.case_id
                        and req.model == label_to_model.get(call.label)):
                    # same model and case inside the call, yet not a valid turn of it: the
                    # call's own pairing cannot be trusted to be whole
                    call.unexplained_http += 1
            elif other:
                used_other.add(id(other[0]))
                log.info("harness HTTP line %s explained by a non-benchmark request", h)
            elif call is not None:
                call.unexplained_http += 1
                log.warning("run %s %s %s %s: harness HTTP line %s has no LM Studio request",
                            run.label, call.kind, call.label, call.case_id, h)
    dropped = [r for r in bench_reqs if id(r) not in assigned]
    for r in dropped:
        log.warning("benchmark-looking request %s:%d has no harness provenance; dropped",
                    r.source, r.line_no)
    log.info("aligned %d LM Studio benchmark requests with harness calls; %d ad hoc; %d dropped",
             len(rows), len(adhoc), len(dropped))
    return rows, adhoc, len(dropped)


# ---------------------------------------------------------------------------
# Scoring from log evidence
# ---------------------------------------------------------------------------

# score_constraint_check penalties for the constraint types that run_bench.py logs a
# "constraint fail" line for. tests/test_recover_lmstudio_log.py checks these weights
# against run_bench.score_constraint_check so the two cannot drift apart.
LOGGED_CONSTRAINT_PENALTY = {
    "word_count": ("constraint fail: word_count", 2.0),
    "required_words": ("constraint fail: missing word", 1.0),
    "forbidden_chars": ("constraint fail: forbidden char", 1.0),
}
NOT_CALLED_RE = re.compile(r"^expected tool '([^']+)' not called, got: \[(.*)\]$")
HIDDEN_ARGS_ASSUMPTION = ("the part of the tool-call arguments cut from the log is well-formed "
                          "JSON that does not repeat a compared key")


@dataclass
class Score:
    score: Optional[float]
    method: str
    evidence: str
    low: Optional[float] = None
    high: Optional[float] = None
    assumption: str = ""

    def __post_init__(self):
        if self.score is not None and self.low is None and self.high is None:
            self.low = self.high = self.score


def score_generation(case: dict, scoring_msgs: list, turn_bodies: list) -> Score:
    """Score one complete scored generation from log evidence.

    Scores follow run_bench.py as it ran on 5 Apr 2026 (commit ca4a63a): its score_case
    scored every row, including empty and length-truncated ones, which main's router now
    marks invalid. The scorers themselves are unchanged since then.
    * unit_tests: score_unit_tests logs a warning on every failure ("unit test failed" -> 1.0,
      "timeout" / "error" -> 0.0) and nothing on a pass (5.0).
    * constraint_check whose constraint types are all logged ones (word_count,
      required_words, forbidden_chars): 5 minus the logged penalties.
    * tool_trace_exact with expected_tool_call and no expected_keywords:
      run_bench.score_tool_trace on the tool calls replayed in the last turn's request, or
      the harness's "expected tool ... not called" line.
    * anything else needs the response text, which LM Studio does not log for streamed
      responses.
    """
    scoring = case.get("scoring")
    if scoring == "unit_tests":
        if any(m.startswith(("unit test timeout", "unit test error")) for m in scoring_msgs):
            return Score(0.0, "bench_log_unit_test", "harness logged a unit test timeout or error")
        if any(m.startswith("unit test failed") for m in scoring_msgs):
            return Score(1.0, "bench_log_unit_test", "harness logged 'unit test failed'")
        return Score(5.0, "bench_log_unit_test", "no unit test failure logged (pass)")
    constraints = case.get("constraints") or {}
    if (scoring == "constraint_check" and constraints
            and set(constraints) <= set(LOGGED_CONSTRAINT_PENALTY)):
        penalty = 0.0
        for key in constraints:
            prefix, weight = LOGGED_CONSTRAINT_PENALTY[key]
            penalty += weight * sum(m.startswith(prefix) for m in scoring_msgs)
        fails = [m for m in scoring_msgs if m.startswith("constraint fail")]
        return Score(max(0.0, 5.0 - penalty), "bench_log_constraint",
                     "; ".join(fails) or "no constraint failure logged")
    if (scoring == "tool_trace_exact" and case.get("expected_tool_call")
            and not case.get("expected_keywords")):
        return _score_tool_case(case, scoring_msgs, turn_bodies)
    return Score(None, "unrecoverable",
                 "needs the response text, which LM Studio does not log for streamed responses")


def _score_tool_case(case: dict, scoring_msgs: list, turn_bodies: list) -> Score:
    import run_bench  # the harness's own scorer; imported lazily (it pulls in openai)

    expected = case["expected_tool_call"]
    want_args = case.get("expected_args_contain") or {}
    not_called = [m for m in (NOT_CALLED_RE.match(s) for s in scoring_msgs) if m]
    # tool calls of turn t are replayed in the request of turn t + 1; calls of the last turn
    # are only visible through the harness's "not called, got: [...]" line
    calls = visible_tool_calls(turn_bodies[-1]) if turn_bodies else []
    first = next((c for c in calls if c["function_name"] == expected), None)
    if not_called:
        # score_tool_trace logs this whenever the expected tool is absent from all turns
        got = not_called[0].group(2)
        score = run_bench.score_tool_trace({"tool_calls_trace": [[]], "response_text": ""}, case)
        return Score(score, "harness_score_tool_trace",
                     f"harness logged '{expected}' not called, got: [{got}]")
    if first is None:
        return Score(None, "unrecoverable",
                     f"'{expected}' was called (no 'not called' line) but only in the final "
                     "turn, whose arguments are not logged; score between 2.5 and 5.0",
                     low=2.5, high=5.0)
    args = first["function_args"]
    if TRUNC_MARK not in args:
        trace = [[{"function_name": expected, "function_args": args}]]
        score = run_bench.score_tool_trace({"tool_calls_trace": trace, "response_text": ""}, case)
        return Score(score, "harness_score_tool_trace",
                     f"{expected} with complete logged arguments")
    visible = parse_visible_args(args)
    rebuilt, unknown = {}, []
    for key, want in want_args.items():
        value = visible.get(key, CUT)
        if value is CUT or (isinstance(value, ArgPrefix) and str(want).startswith(value)):
            unknown.append(key)
            continue
        rebuilt[key] = f"{value}..." if isinstance(value, ArgPrefix) else value
    if unknown:
        return Score(None, "unrecoverable",
                     f"'{expected}' called; argument(s) {', '.join(unknown)} cut from the "
                     "log, so the harness score is between 2.5 and 5.0", low=2.5, high=5.0)
    trace = [[{"function_name": expected, "function_args": json.dumps(rebuilt)}]]
    score = run_bench.score_tool_trace({"tool_calls_trace": trace, "response_text": ""}, case)
    shown = ", ".join(f"{k}={rebuilt[k]!r}" for k in want_args)
    return Score(score, "harness_score_tool_trace",
                 f"{expected}({shown}) visible in the truncated arguments", low=2.5, high=5.0,
                 assumption=HIDDEN_ARGS_ASSUMPTION)


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

CALL_COLUMNS = [
    "run", "run_status", "model", "model_label", "case_id", "category", "call_kind",
    "repeat", "turn", "seed", "temperature", "top_p", "max_tokens", "tools_sent", "n_messages",
    "finish_reason_inferred", "prompt_tokens", "prompt_eval_tokens", "completion_tokens",
    "server_prompt_eval_ms", "server_eval_ms", "server_total_ms", "decode_tok_s", "n_ctx_slot",
    "t_request", "t_response", "bench_t_request", "total_s", "overlapped", "pairing",
    "visible_tool_calls", "lmstudio_log", "lmstudio_line",
]

GENERATION_COLUMNS = [
    "run", "run_status", "model", "model_label", "case_id", "category", "scoring", "repeat",
    "seed", "temperature", "max_tokens", "turns", "complete", "finish_reason_inferred",
    "length_truncated_turns", "prompt_tokens_turn1", "completion_tokens",
    "server_prefill_s_turn1", "server_gen_s", "server_total_s", "decode_tok_s",
    "bench_t_start", "bench_t_end", "total_s", "overlapped", "score", "score_low",
    "score_high", "score_method", "score_evidence", "score_assumption",
]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _iso(ts: Optional[datetime], ms: bool = False) -> str:
    if ts is None:
        return ""
    return ts.isoformat(timespec="milliseconds" if ms else "seconds")


def _finish_reason(req: LmRequest, kind: str, is_last: bool, gen_complete: bool,
                   turns: int) -> str:
    """finish_reason is not logged for streamed responses; infer it from token counts.

    llama.cpp stops at exactly max_tokens when the budget runs out, so a completion that used
    the whole budget is "length". A scored turn followed by another turn of the same
    generation ended in tool calls. The last turn stopped on its own, except after the
    harness's fourth tool-use turn (it stops asking either way) and in an interrupted
    generation, where it is unknown.
    """
    max_tokens = (req.body or {}).get("max_tokens")
    if req.completion_tokens is None:
        return ""
    if max_tokens is not None and req.completion_tokens >= max_tokens:
        return "length"
    if req.t_response is None:
        return ""  # no finish line: it may have been cancelled
    if kind in ("warmup", "adhoc"):
        return "stop"
    if not is_last:
        return "tool_calls"
    if not gen_complete:
        return ""
    if (req.case or {}).get("category") == "tool_use" and turns >= MAX_TOOL_TURNS:
        return "stop_or_tool_calls"
    return "stop"


def _call_row(req: LmRequest, run_label: str, run_status: str, label: str, kind: str,
              repeat, turn, finish: str, bench_t_request, total_s) -> dict:
    body = req.body or {}
    timings_ok = not req.ambiguous  # an ambiguous task binding makes the timings unknown
    comp = req.completion_tokens if timings_ok else None
    eval_ms = req.eval_ms if timings_ok else None
    decode = comp / (eval_ms / 1000.0) if eval_ms and comp else None
    visible = visible_tool_calls(body)
    return {
        "run": run_label, "run_status": run_status, "model": req.model, "model_label": label,
        "case_id": req.case_id, "category": (req.case or {}).get("category"), "call_kind": kind,
        "repeat": repeat, "turn": turn, "seed": body.get("seed"),
        "temperature": body.get("temperature"), "top_p": body.get("top_p"),
        "max_tokens": body.get("max_tokens"), "tools_sent": ",".join(_tool_names(body.get("tools"))),
        "n_messages": len(body.get("messages") or []),
        "finish_reason_inferred": finish if timings_ok else "",
        "prompt_tokens": req.prompt_tokens if timings_ok else None,
        "prompt_eval_tokens": req.prompt_eval_tokens if timings_ok else None,
        "completion_tokens": comp,
        "server_prompt_eval_ms": req.prompt_eval_ms if timings_ok else None,
        "server_eval_ms": eval_ms,
        "server_total_ms": req.server_total_ms if timings_ok else None,
        "decode_tok_s": decode, "n_ctx_slot": req.n_ctx_slot if timings_ok else None,
        "t_request": _iso(req.t_request), "t_response": _iso(req.t_response) if timings_ok else "",
        "bench_t_request": _iso(bench_t_request, ms=True), "total_s": total_s,
        "overlapped": req.overlapped, "pairing": "ambiguous" if req.ambiguous else "unique",
        "visible_tool_calls": json.dumps(visible, ensure_ascii=False) if visible else "",
        "lmstudio_log": req.source, "lmstudio_line": req.line_no,
    }


def _timings_whole(req: LmRequest) -> bool:
    """Every server measurement of the request was logged and its binding is unique."""
    return (not req.ambiguous and req.t_response is not None
            and None not in (req.prompt_eval_ms, req.eval_ms, req.server_total_ms,
                             req.completion_tokens, req.prompt_tokens))


def build_rows(aligned: list, adhoc: list, model_labels: dict) -> tuple:
    """Turn aligned calls into (call rows, generation rows). Ad hoc requests get call rows."""
    by_call: dict = {}
    for a in aligned:
        by_call.setdefault(id(a["call"]), []).append(a)
    call_rows, gen_rows = [], []
    for items in by_call.values():
        run, call = items[0]["run"], items[0]["call"]
        reqs = [a["req"] for a in items]
        case = reqs[0].case
        complete = (call.kind == "scored" and call.closed and call.t_end is not None
                    and not call.ambiguous and call.unexplained_http == 0
                    and all(_timings_whole(r) for r in reqs))
        status = "complete" if run.complete else "aborted"
        for a in items:
            req = a["req"]
            is_last = a["index"] == a["n_http"]
            total_s = ((a["bench_t_next"] - a["bench_t_request"]).total_seconds()
                       if a["bench_t_next"] and (call.kind == "warmup" or not is_last or complete)
                       else None)
            if call.kind == "warmup":
                repeat, turn = a["index"], 1
            else:
                repeat, turn = call.repeat, a["index"]
            finish = _finish_reason(req, call.kind, is_last, complete, len(items))
            call_rows.append(_call_row(req, run.label, status, call.label, call.kind, repeat,
                                       turn, finish, a["bench_t_request"], total_s))
        if call.kind != "scored":
            continue
        rows = call_rows[-len(items):]
        if complete:
            sc = score_generation(case, call.scoring, [r.body for r in reqs])
        else:
            sc = Score(None, "incomplete", "generation interrupted or not fully paired")
        comp = sum(r.completion_tokens or 0 for r in reqs)
        gen_ms = sum(r.eval_ms or 0.0 for r in reqs)
        first = rows[0]
        gen_rows.append({
            "run": run.label, "run_status": status, "model": first["model"],
            "model_label": call.label, "case_id": call.case_id, "category": case["category"],
            "scoring": case.get("scoring"), "repeat": call.repeat, "seed": first["seed"],
            "temperature": first["temperature"], "max_tokens": first["max_tokens"],
            "turns": len(rows), "complete": complete,
            "finish_reason_inferred": rows[-1]["finish_reason_inferred"],
            "length_truncated_turns": sum(r["finish_reason_inferred"] == "length" for r in rows),
            "prompt_tokens_turn1": first["prompt_tokens"],
            "completion_tokens": comp if complete else None,
            "server_prefill_s_turn1": (reqs[0].prompt_eval_ms / 1000.0
                                       if complete and reqs[0].prompt_eval_ms is not None else None),
            "server_gen_s": gen_ms / 1000.0 if complete else None,
            "server_total_s": (sum(r.server_total_ms or 0.0 for r in reqs) / 1000.0
                               if complete else None),
            "decode_tok_s": comp / (gen_ms / 1000.0) if complete and gen_ms else None,
            "bench_t_start": _iso(call.t_start, ms=True), "bench_t_end": _iso(call.t_end, ms=True),
            "total_s": (call.t_end - call.t_start).total_seconds() if complete else None,
            "overlapped": any(r.overlapped for r in reqs),
            "score": sc.score, "score_low": sc.low, "score_high": sc.high,
            "score_method": sc.method, "score_evidence": sc.evidence,
            "score_assumption": sc.assumption,
        })
    for req, http_ts in adhoc:
        call_rows.append(_call_row(req, "adhoc", "", model_labels.get(req.model, ""), "adhoc",
                                   "", 1, _finish_reason(req, "adhoc", True, True, 1), http_ts,
                                   None))
    call_rows.sort(key=lambda r: (r["lmstudio_log"], r["lmstudio_line"]))
    return call_rows, gen_rows


def write_csv(path: Path, columns: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in columns})
    log.info("wrote %d rows to %s", len(rows), path.name)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recover(lm_logs: list, bench_log: Path, cases: list, model_labels: dict) -> tuple:
    """Parse, classify and align. Returns (call rows, generation rows, stats dict)."""
    bench_reqs, other_reqs = [], []
    stats = {"requests": 0, "benchmark": 0, "other": 0, "benchmark_other_model": 0,
             "overlapped_benchmark": 0, "ambiguous_benchmark": 0}
    for path in lm_logs:
        log.info("reading %s (sha256 %s)", path.name, sha256_of(path))
        reqs = parse_lmstudio_log(path.read_text(encoding="utf-8", errors="replace"), path.name)
        for req in reqs:
            stats["requests"] += 1
            case = classify_request(req.body, cases)
            if case is None or req.model not in model_labels:
                key = "other" if case is None else "benchmark_other_model"
                stats[key] += 1
                req.body = None  # never keep non-benchmark content
                other_reqs.append(req)
                continue
            req.case_id, req.case = case["id"], case
            stats["benchmark"] += 1
            stats["overlapped_benchmark"] += req.overlapped
            stats["ambiguous_benchmark"] += req.ambiguous
            bench_reqs.append(req)
    log.info("classified %d requests: %d benchmark, %d other, %d benchmark prompts to other models",
             stats["requests"], stats["benchmark"], stats["other"], stats["benchmark_other_model"])
    if stats["overlapped_benchmark"]:
        log.warning("%d benchmark requests overlapped another request in flight",
                    stats["overlapped_benchmark"])
    if not bench_reqs:
        return [], [], stats
    first_day = min(r.t_request for r in bench_reqs).replace(hour=0, minute=0, second=0)
    last_ts = max(r.t_response or r.t_request for r in bench_reqs)
    log.info("reading %s (sha256 %s)", bench_log.name, sha256_of(bench_log))
    runs = parse_bench_log(bench_log.read_text(encoding="utf-8", errors="replace"),
                           since=first_day, until=last_ts + timedelta(hours=1))
    runs = [r for r in runs if any(c.label in model_labels.values() for c in r.calls)]
    for n, run in enumerate(runs):
        run.label = chr(ord("A") + n)
    aligned, adhoc, dropped = align(runs, bench_reqs, other_reqs, model_labels)
    call_rows, gen_rows = build_rows(aligned, adhoc, model_labels)
    stats.update(runs=len(runs), adhoc_benchmark=len(adhoc), dropped_benchmark=dropped,
                 unexplained_http=sum(c.unexplained_http for r in runs for c in r.calls),
                 ambiguous_calls=sum(c.ambiguous for r in runs for c in r.calls))
    return call_rows, gen_rows, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lmstudio-log", action="append", required=True, type=Path,
                    help="LM Studio server log (repeatable)")
    ap.add_argument("--bench-log", required=True, type=Path, help="run_bench.py bench.log")
    ap.add_argument("--cases", type=Path, default=REPO_ROOT / "cases.json")
    ap.add_argument("--calls-csv", required=True, type=Path)
    ap.add_argument("--generations-csv", required=True, type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    sys.path.insert(0, str(REPO_ROOT))
    cases = json.loads(a.cases.read_text())
    call_rows, gen_rows, stats = recover(a.lmstudio_log, a.bench_log, cases, DEFAULT_MODEL_LABELS)
    write_csv(a.calls_csv, CALL_COLUMNS, call_rows)
    write_csv(a.generations_csv, GENERATION_COLUMNS, gen_rows)
    scored = [g for g in gen_rows if g["score"] is not None]
    log.info("stats: %s; %d scored generations, %d with a recovered score",
             json.dumps(stats, sort_keys=True), len(gen_rows), len(scored))
    complete = [g["total_s"] for g in gen_rows if g["complete"]]
    if complete:
        log.info("median total_s over complete generations: %.1f", statistics.median(complete))
    return 0


if __name__ == "__main__":
    sys.exit(main())
