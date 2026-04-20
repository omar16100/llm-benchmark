"""
Benchmark runner: Qwen3.5-397B-A17B MLX (current primary)
Sends prompts to models via LM Studio OpenAI-compatible API,
captures timing metrics and responses, outputs CSV + JSONL.
"""

import csv
import json
import logging
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("bench.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

import os
BASE_URL = os.environ.get("BENCH_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("BENCH_API_KEY", "lm-studio")

MODELS = {
    "glm_51_mlx_36bit": {
        "served_model": "/Users/macmini/models/GLM-5.1-RAM-420GB-MLX",
        "quant": "MLX-3.6bit",
        "thinking": True,
    },
}

# Qwen 3.5 uses reasoning_content (thinking mode) which shares max_tokens budget.
# We multiply max_tokens for thinking models so the actual answer fits.

WARMUP_RUNS = 2
SCORED_REPEATS = 3
DETERMINISTIC_SEEDS = [42, 42, 42]
CREATIVE_SEEDS = [41, 42, 43]

# Always give models a generous token budget — length-truncation produces
# invalid rows with no scoring signal. Per user policy 2026-04-14.
# See memory/feedback_always_max_tokens.md.
MAX_RESPONSE_TOKENS = 32768

OUTPUT_DIR = Path("results")
CSV_PATH = OUTPUT_DIR / "runs.csv"
JSONL_PATH = OUTPUT_DIR / "transcripts.jsonl"

CSV_COLUMNS = [
    "bench_run_id", "run_id", "model_label", "served_model", "quant", "category",
    "prompt_id", "repeat", "seed", "temperature", "top_p", "max_tokens",
    "ttft_s", "gen_s", "total_s", "output_tokens_approx", "tok_per_s",
    "finish_reason", "valid", "score_raw", "score_type",
]


class TransportError(RuntimeError):
    """Raised when the inference server is unreachable after retries."""


MAX_TRANSPORT_RETRIES = 3
RETRY_BACKOFF_S = 2.0
SANITY_CHECK_AFTER = 3  # abort if first N scored runs all have empty content


def create_client():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def stream_completion(client, model_cfg, messages, tools=None, max_tokens=512,
                      temperature=0.0, top_p=1.0, seed=42):
    """Send a streaming chat completion and capture timing metrics.

    Uses raw httpx streaming to capture reasoning_content (Qwen thinking mode)
    alongside regular content.
    """
    import httpx

    t0 = time.perf_counter()
    first_tok_t = None
    last_tok_t = None
    text_parts = []
    reasoning_parts = []
    tool_calls_raw = []
    finish_reason = None

    body = {
        "model": model_cfg["served_model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": True,
    }
    if tools:
        body["tools"] = tools

    try:
        with httpx.Client(timeout=300) as http:
            with http.stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            ) as resp:
                buffer = ""
                for raw_chunk in resp.iter_text():
                    buffer += raw_chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        now = time.perf_counter()
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # capture content (final answer)
                        content = delta.get("content")
                        if content:
                            if first_tok_t is None:
                                first_tok_t = now
                            text_parts.append(content)
                            last_tok_t = now

                        # capture reasoning (thinking) — LM Studio uses
                        # "reasoning_content", mlx_lm.server uses "reasoning"
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            if first_tok_t is None:
                                first_tok_t = now
                            reasoning_parts.append(reasoning)
                            last_tok_t = now

                        # capture tool calls
                        tc_deltas = delta.get("tool_calls")
                        if tc_deltas:
                            if first_tok_t is None:
                                first_tok_t = now
                            last_tok_t = now
                            for tc in tc_deltas:
                                fn = tc.get("function", {})
                                tool_calls_raw.append({
                                    "index": tc.get("index", 0),
                                    "id": tc.get("id"),
                                    "function_name": fn.get("name"),
                                    "function_args": fn.get("arguments"),
                                })

                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr

    except Exception as e:
        # propagate so caller can retry; do NOT synthesize a result that
        # downstream scoring would silently accept as valid data
        log.warning("stream error (will retry if attempts remain): %s", e)
        raise TransportError(str(e)) from e

    text = "".join(text_parts).strip()
    reasoning_text = "".join(reasoning_parts)
    # approximate token count: ~4 chars per token across BOTH channels.
    # Thinking models emit most tokens in reasoning_content; counting only
    # `text` under-reports throughput by 5-10x.
    all_chars = len(text) + len(reasoning_text)
    output_tokens_approx = max(all_chars // 4, 1) if all_chars else 0
    total_s = time.perf_counter() - t0
    gen_s = max((last_tok_t or t0) - (first_tok_t or t0), 1e-9)

    # merge tool call deltas
    merged_tool_calls = _merge_tool_call_deltas(tool_calls_raw)

    return {
        "response_text": text,
        "reasoning_text": reasoning_text,
        "tool_calls": merged_tool_calls,
        "ttft_s": (first_tok_t - t0) if first_tok_t else None,
        "gen_s": gen_s,
        "total_s": total_s,
        "output_tokens_approx": output_tokens_approx,
        "tok_per_s": output_tokens_approx / gen_s if output_tokens_approx else 0,
        "finish_reason": finish_reason,
    }


def stream_completion_retried(client, model_cfg, messages, **kwargs):
    """Wrap stream_completion with retry/backoff. Re-raises TransportError on
    persistent failure so the caller can abort cleanly instead of treating
    transport noise as benchmark data."""
    last_err = None
    for attempt in range(1, MAX_TRANSPORT_RETRIES + 1):
        try:
            return stream_completion(client, model_cfg, messages, **kwargs)
        except TransportError as e:
            last_err = e
            if attempt < MAX_TRANSPORT_RETRIES:
                wait = RETRY_BACKOFF_S * (2 ** (attempt - 1))
                log.warning(
                    "transport retry %d/%d in %.1fs after: %s",
                    attempt, MAX_TRANSPORT_RETRIES, wait, e,
                )
                time.sleep(wait)
    raise last_err


def _merge_tool_call_deltas(raw_deltas):
    """Merge streaming tool call deltas into complete calls."""
    calls = {}
    for d in raw_deltas:
        idx = d["index"]
        if idx not in calls:
            calls[idx] = {"id": None, "function_name": "", "function_args": ""}
        if d["id"]:
            calls[idx]["id"] = d["id"]
        if d["function_name"]:
            calls[idx]["function_name"] += d["function_name"]
        if d["function_args"]:
            calls[idx]["function_args"] += d["function_args"]
    return list(calls.values())


def run_tool_use_case(client, model_cfg, case, seed):
    """Handle multi-turn tool use cases."""
    messages = []
    if case.get("system"):
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["prompt"]})

    tools = case.get("tools", [])
    tool_responses = case.get("tool_responses", {})
    all_results = []
    max_turns = 4

    # Always use the generous ceiling — no artificial truncation.
    effective_max_tokens = MAX_RESPONSE_TOKENS

    for turn in range(max_turns):
        result = stream_completion_retried(
            client, model_cfg, messages, tools=tools,
            max_tokens=effective_max_tokens,
            temperature=case.get("temperature", 0.0),
            seed=seed,
        )
        all_results.append(result)

        if not result["tool_calls"] or result["finish_reason"] != "tool_calls":
            break

        # process tool calls and add responses
        for tc in result["tool_calls"]:
            fn_name = tc["function_name"]
            fn_args = tc.get("function_args", "{}")

            # build lookup key (some tools need args for key)
            response_key = fn_name
            try:
                args = json.loads(fn_args)
                if fn_name == "read_file" and "path" in args:
                    response_key = f"{fn_name}:{args['path']}"
            except json.JSONDecodeError:
                pass

            tool_result = tool_responses.get(response_key, "{}")

            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": tc.get("id", f"call_{turn}"),
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{turn}"),
                "content": tool_result,
            })

    # combine metrics from all turns
    combined = {
        "response_text": all_results[-1]["response_text"],
        "reasoning_text": all_results[-1].get("reasoning_text", ""),
        "tool_calls_trace": [r["tool_calls"] for r in all_results],
        "ttft_s": all_results[0]["ttft_s"],
        "gen_s": sum(r["gen_s"] for r in all_results),
        "total_s": sum(r["total_s"] for r in all_results),
        "output_tokens_approx": sum(r["output_tokens_approx"] for r in all_results),
        "finish_reason": all_results[-1]["finish_reason"],
        "turns": len(all_results),
    }
    combined["tok_per_s"] = (
        combined["output_tokens_approx"] / combined["gen_s"]
        if combined["gen_s"] > 0 else 0
    )
    return combined


def score_exact(response_text, expected):
    """Check if response contains the exact expected answer."""
    clean = response_text.strip().strip("`").strip()
    expected_clean = expected.strip()
    if clean == expected_clean:
        return 5.0
    if expected_clean.lower() in clean.lower():
        return 3.0
    # try normalizing whitespace
    norm_clean = re.sub(r"\s+", " ", clean).strip()
    norm_expected = re.sub(r"\s+", " ", expected_clean).strip()
    if norm_expected.lower() in norm_clean.lower():
        return 3.0
    return 0.0


def score_keywords(response_text, keywords):
    """Check how many expected keywords appear in response."""
    text_lower = response_text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(5.0 * found / len(keywords), 1) if keywords else 0.0


def score_unit_tests(response_text, test_code):
    """Extract code from response and run unit tests."""
    # extract code block if present
    code = response_text
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # try to find function/class definition
        lines = response_text.strip().split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith(("def ", "class ")):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            code = "\n".join(code_lines)

    full_code = code + "\n\n" + test_code
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return 5.0
        log.warning("unit test failed: %s", result.stderr[:200])
        return 1.0
    except subprocess.TimeoutExpired:
        log.warning("unit test timeout")
        return 0.0
    except Exception as e:
        log.warning("unit test error: %s", e)
        return 0.0


def score_constraint_check(response_text, constraints):
    """Check structural constraints on response."""
    score = 5.0
    penalties = 0

    if "word_count" in constraints:
        words = len(response_text.split())
        if words != constraints["word_count"]:
            penalties += 2
            log.info("constraint fail: word_count %d != %d", words, constraints["word_count"])

    if "required_words" in constraints:
        text_lower = response_text.lower()
        for w in constraints["required_words"]:
            if w.lower() not in text_lower:
                penalties += 1
                log.info("constraint fail: missing word '%s'", w)

    if "forbidden_chars" in constraints:
        for ch in constraints["forbidden_chars"]:
            if ch in response_text:
                penalties += 1
                log.info("constraint fail: forbidden char '%s'", ch)

    if "bullet_count" in constraints:
        bullets = [l for l in response_text.strip().split("\n") if l.strip().startswith(("-", "*", "•"))]
        if len(bullets) != constraints["bullet_count"]:
            penalties += 1

    if "words_per_bullet" in constraints:
        bullets = [l for l in response_text.strip().split("\n") if l.strip().startswith(("-", "*", "•"))]
        for b in bullets:
            words = len(b.strip().lstrip("-*• ").split())
            if words != constraints["words_per_bullet"]:
                penalties += 0.5

    if "line_count" in constraints:
        lines = [l for l in response_text.strip().split("\n") if l.strip()]
        if len(lines) != constraints["line_count"]:
            penalties += 2

    if "word_range" in constraints:
        words = len(response_text.split())
        lo, hi = constraints["word_range"]
        if words < lo or words > hi:
            penalties += 2

    if "required_elements" in constraints:
        text_lower = response_text.lower()
        for elem in constraints["required_elements"]:
            if elem.lower() not in text_lower:
                penalties += 1

    if "forbidden_words" in constraints:
        text_lower = response_text.lower()
        for fw in constraints["forbidden_words"]:
            if fw.lower() in text_lower:
                penalties += 1

    if "forbidden_elements" in constraints:
        text_lower = response_text.lower()
        for fe in constraints["forbidden_elements"]:
            if fe.lower() in text_lower:
                penalties += 1

    return max(0.0, score - penalties)


def score_tool_trace(result, case):
    """Score tool use cases by checking tool call correctness."""
    score = 0.0
    tool_calls_trace = result.get("tool_calls_trace", [])

    # check if expected tool was called
    expected_tool = case.get("expected_tool_call")
    if expected_tool:
        all_calls = [tc for turn in tool_calls_trace for tc in turn]
        called_names = [tc["function_name"] for tc in all_calls]
        if expected_tool in called_names:
            score += 2.5
            # check expected args
            expected_args = case.get("expected_args_contain", {})
            if expected_args:
                for tc in all_calls:
                    if tc["function_name"] == expected_tool:
                        try:
                            actual_args = json.loads(tc["function_args"])
                            matches = sum(
                                1 for k, v in expected_args.items()
                                if str(actual_args.get(k)) == str(v)
                            )
                            score += 2.5 * matches / len(expected_args)
                        except json.JSONDecodeError:
                            pass
                        break
        else:
            log.info("expected tool '%s' not called, got: %s", expected_tool, called_names)

    # check expected keywords in final response
    expected_kw = case.get("expected_keywords", [])
    if expected_kw:
        kw_score = score_keywords(result["response_text"], expected_kw)
        score = (score + kw_score) / 2 if score > 0 else kw_score

    return min(score, 5.0)


def _scoreable_text(result):
    """Extract the model's actual answer for scoring. Thinking models often
    emit the answer in reasoning_content with empty content; in that case
    the reasoning IS the answer."""
    response = (result.get("response_text") or "").strip()
    if response:
        return response
    reasoning = (result.get("reasoning_text") or "").strip()
    return reasoning


def is_invalid_result(result, case):
    """A run is invalid (don't score) if the model produced no usable text or
    finished due to error/length truncation on a non-creative case."""
    fr = result.get("finish_reason")
    if fr in ("error",):
        return True
    text = _scoreable_text(result)
    if not text:
        return True
    # length-truncated answers on tasks with short max_tokens are noise
    if fr == "length" and case.get("category") not in ("creative",):
        return True
    return False


def score_case(result, case):
    """Route to appropriate scoring function. Returns (score, score_type) or
    (None, 'invalid') if the result cannot be fairly scored."""
    if is_invalid_result(result, case):
        return None, "invalid"

    scoring = case.get("scoring", "judge")
    response = _scoreable_text(result)

    if scoring == "exact" and "expected" in case:
        return score_exact(response, case["expected"]), "exact"

    if scoring == "judge_keyword" and "expected_keywords" in case:
        return score_keywords(response, case["expected_keywords"]), "keyword"

    if scoring == "unit_tests" and "test_code" in case:
        return score_unit_tests(response, case["test_code"]), "unit_test"

    if scoring == "constraint_check" and "constraints" in case:
        return score_constraint_check(response, case["constraints"]), "constraint"

    if scoring in ("tool_trace_exact", "tool_trace_judge"):
        return score_tool_trace(result, case), "tool_trace"

    if scoring == "judge_constraint" and "constraints" in case:
        return score_constraint_check(response, case["constraints"]), "constraint"

    # judge-only cases get scored later by judge_claude.py
    return None, "needs_judge"


def run_benchmark():
    """Main benchmark loop."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open("cases.json") as f:
        cases = json.load(f)

    client = create_client()

    # verify models are loaded
    models_resp = client.models.list()
    available = [m.id for m in models_resp.data]
    log.info("available models: %s", available)
    for label, cfg in MODELS.items():
        if cfg["served_model"] not in available:
            log.error("model %s not loaded!", cfg["served_model"])
            return

    csv_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if not csv_exists:
        writer.writeheader()

    jsonl_file = open(JSONL_PATH, "a")

    bench_run_id = str(uuid.uuid4())
    log.info("bench_run_id=%s models=%s", bench_run_id, list(MODELS.keys()))

    total_cases = len(cases) * len(MODELS) * SCORED_REPEATS
    completed = 0
    sanity_empty = 0  # count of consecutive empty responses for early abort
    sanity_total = 0

    try:
        for case in cases:
            is_creative = case.get("temperature", 0.0) > 0
            seeds = CREATIVE_SEEDS if is_creative else DETERMINISTIC_SEEDS

            for model_label, model_cfg in MODELS.items():
                # warmup
                log.info("warmup: %s / %s", model_label, case["id"])
                for _ in range(WARMUP_RUNS):
                    messages = [{"role": "user", "content": case["prompt"]}]
                    stream_completion_retried(
                        client, model_cfg, messages,
                        max_tokens=min(case.get("max_tokens", 512), 64),
                        temperature=case.get("temperature", 0.0),
                    )

                # scored runs
                for repeat_idx in range(SCORED_REPEATS):
                    seed = seeds[repeat_idx]
                    run_id = str(uuid.uuid4())[:8]

                    log.info(
                        "[%d/%d] %s | %s | repeat %d",
                        completed + 1, total_cases, case["id"], model_label, repeat_idx + 1,
                    )

                    is_tool_case = case["category"] == "tool_use"

                    # Always use generous ceiling; no artificial truncation.
                    effective_max_tokens = MAX_RESPONSE_TOKENS

                    if is_tool_case:
                        result = run_tool_use_case(client, model_cfg, case, seed)
                    else:
                        messages = [{"role": "user", "content": case["prompt"]}]
                        result = stream_completion_retried(
                            client, model_cfg, messages,
                            tools=case.get("tools"),
                            max_tokens=effective_max_tokens,
                            temperature=case.get("temperature", 0.0),
                            seed=seed,
                        )

                    score_raw, score_type = score_case(result, case)
                    valid = score_type != "invalid"

                    # write JSONL first (fsync) so on crash transcripts >= csv
                    transcript = {
                        "bench_run_id": bench_run_id,
                        "run_id": run_id,
                        "model_label": model_label,
                        "prompt_id": case["id"],
                        "category": case["category"],
                        "prompt": case["prompt"],
                        "response": result.get("response_text", ""),
                        "reasoning": result.get("reasoning_text", ""),
                        "tool_calls_trace": result.get("tool_calls_trace"),
                        "valid": valid,
                        "score_raw": score_raw,
                        "score_type": score_type,
                        "metrics": {
                            "ttft_s": result.get("ttft_s"),
                            "gen_s": result.get("gen_s"),
                            "total_s": result.get("total_s"),
                            "tok_per_s": result.get("tok_per_s"),
                        },
                    }
                    jsonl_file.write(json.dumps(transcript) + "\n")
                    jsonl_file.flush()

                    row = {
                        "bench_run_id": bench_run_id,
                        "run_id": run_id,
                        "model_label": model_label,
                        "served_model": model_cfg["served_model"],
                        "quant": model_cfg["quant"],
                        "category": case["category"],
                        "prompt_id": case["id"],
                        "repeat": repeat_idx + 1,
                        "seed": seed,
                        "temperature": case.get("temperature", 0.0),
                        "top_p": 1.0 if not is_creative else 0.95,
                        "max_tokens": case.get("max_tokens", 512),
                        "ttft_s": round(result.get("ttft_s", 0) or 0, 4),
                        "gen_s": round(result.get("gen_s", 0), 4),
                        "total_s": round(result.get("total_s", 0), 4),
                        "output_tokens_approx": result.get("output_tokens_approx", 0),
                        "tok_per_s": round(result.get("tok_per_s", 0), 2),
                        "finish_reason": result.get("finish_reason"),
                        "valid": valid,
                        "score_raw": score_raw,
                        "score_type": score_type,
                    }
                    writer.writerow(row)
                    csv_file.flush()

                    completed += 1

                    # early abort: if the first SANITY_CHECK_AFTER scored runs
                    # all produced empty/invalid output, the model is misconfigured
                    sanity_total += 1
                    if not _scoreable_text(result):
                        sanity_empty += 1
                    if sanity_total == SANITY_CHECK_AFTER and sanity_empty == SANITY_CHECK_AFTER:
                        raise RuntimeError(
                            f"sanity check failed: first {SANITY_CHECK_AFTER} runs all "
                            f"produced empty text. Model misconfigured (thinking flag, "
                            f"token budget, or chat template). Aborting bench_run_id={bench_run_id}."
                        )
    except TransportError as e:
        log.error("transport error after retries — aborting bench_run_id=%s: %s", bench_run_id, e)
        raise
    finally:
        csv_file.close()
        jsonl_file.close()

    log.info("benchmark complete. bench_run_id=%s rows=%d results=%s transcripts=%s",
             bench_run_id, completed, CSV_PATH, JSONL_PATH)


if __name__ == "__main__":
    run_benchmark()
