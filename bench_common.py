#!/usr/bin/env python3
"""Shared helpers for the llm-benchmark scripts.

One OpenAI-compatible chat call plus server-side timing extraction, used by
bench_long_prompt.py and bench_longctx.py. Targets any OpenAI-compatible
/v1/chat/completions endpoint (llama-server, LM Studio, vLLM, and similar).

llama.cpp and LM Studio return a non-standard `timings` block with
prompt_per_second and predicted_per_second. We use it when present (true prefill
and decode throughput) and fall back to usage plus wall clock otherwise. Client
wall time is never reported as prefill, because it includes generation.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import requests


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 256,
    temperature: Optional[float] = 0.0,
    timeout: float = 600.0,
    api_key: Optional[str] = None,
    chat_template_kwargs: Optional[dict] = None,
    extra_body: Optional[dict] = None,
    stream: bool = False,
) -> dict[str, Any]:
    """POST one chat completion.

    Returns a dict with content, usage, timings, wall_s, and (when stream=True)
    ttft_s, the client-observed time to the first non-empty content token.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:  # None means omit the field (server default)
        payload["temperature"] = temperature
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if extra_body:  # server-specific fields, e.g. {"cache_prompt": False} for llama.cpp
        payload.update(extra_body)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        return _chat_stream(url, payload, headers, timeout)

    payload["stream"] = False
    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    wall_s = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    content = (msg.get("content") or "").strip()
    return {
        "content": content,
        "usage": data.get("usage", {}) or {},
        "timings": data.get("timings", {}) or {},
        "wall_s": wall_s,
        "ttft_s": None,
    }


def _chat_stream(url, payload, headers, timeout) -> dict[str, Any]:
    """Streaming variant. Captures TTFT = time to first non-empty content token.

    Proper SSE framing: an event may span multiple `data:` lines and is terminated
    by a blank line. We accumulate `data:` payload lines per event, then parse the
    joined JSON. Note: llama-server may not emit the `timings` block in streaming
    mode, in which case prefill_tps() returns None for this row (acceptable; rerun
    without --stream for guaranteed server-side prefill numbers).
    """
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft_s: Optional[float] = None
    chunks: list[str] = []
    usage: dict = {}
    timings: dict = {}

    def handle(event_data: str) -> bool:
        """Process one SSE event payload. Returns False to stop (on [DONE])."""
        nonlocal ttft_s, usage, timings
        if event_data == "[DONE]":
            return False
        try:
            obj = json.loads(event_data)
        except json.JSONDecodeError:
            return True
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("timings"):
            timings = obj["timings"]
        for ch in obj.get("choices", []):
            piece = (ch.get("delta", {}) or {}).get("content") or ""
            if piece:
                if ttft_s is None:
                    ttft_s = time.time() - t0
                chunks.append(piece)
        return True

    with requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        data_buf: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data:"):
                    data_buf.append(line[len("data:"):].lstrip())
                continue
            # blank line: event boundary
            if data_buf:
                cont = handle("\n".join(data_buf))
                data_buf = []
                if not cont:
                    break
        if data_buf:  # flush a final event with no trailing blank line
            handle("\n".join(data_buf))

    return {
        "content": "".join(chunks).strip(),
        "usage": usage,
        "timings": timings,
        "wall_s": time.time() - t0,
        "ttft_s": ttft_s,
    }


def prompt_tokens(result: dict[str, Any]) -> int:
    """Total prompt (context) size.

    Prefers usage.prompt_tokens (the full context) over timings.prompt_n (only the
    tokens the server actually evaluated, which is smaller when prompt caching reuses
    a prefix). Falls back to timings.prompt_n when usage is absent.
    """
    usage = result.get("usage") or {}
    if usage.get("prompt_tokens"):
        return int(usage["prompt_tokens"])
    return int((result.get("timings") or {}).get("prompt_n", 0) or 0)


def prefill_tps(result: dict[str, Any]) -> Optional[float]:
    """True prefill throughput (tok/s) from the server `timings` block.

    Returns None when the server does not expose timings. We deliberately do not
    derive prefill from client wall clock, because wall clock includes generation.
    """
    t = result.get("timings") or {}
    if t.get("prompt_per_second"):
        return round(float(t["prompt_per_second"]), 2)
    if t.get("prompt_n") and t.get("prompt_ms"):
        return round(t["prompt_n"] / (t["prompt_ms"] / 1000.0), 2)
    return None


def decode_tps(result: dict[str, Any]) -> Optional[float]:
    """Decode (generation) throughput (tok/s) from server timings, else None."""
    t = result.get("timings") or {}
    if t.get("predicted_per_second"):
        return round(float(t["predicted_per_second"]), 2)
    if t.get("predicted_n") and t.get("predicted_ms"):
        return round(t["predicted_n"] / (t["predicted_ms"] / 1000.0), 2)
    return None
