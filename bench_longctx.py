#!/usr/bin/env python3
"""Long-context needle-in-haystack plus prefill-throughput eval.

Works against any OpenAI-compatible /v1/chat/completions endpoint (llama-server,
LM Studio, vLLM, and similar). Builds a haystack of roughly target tokens, inserts
a unique needle at each requested depth (one request per length x depth cell), asks
for it back, checks exact recall, and reports prompt tokens, server-side prefill
tok/s (when the server exposes `timings`), end-to-end wall seconds, and optional
client-observed TTFT (with --stream).

Example:
  bench_longctx.py --base-url http://127.0.0.1:8081 --model glm-5.2 \
      --target-tokens 2000 8000 32000 --depths 25 50 90 --no-thinking --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import bench_common

# Neutral filler. One unique needle is inserted per cell; recall is exact-substring.
FILLER = (
    "The quarterly logistics review covered warehouse throughput, fleet routing, "
    "cold-chain compliance, and vendor lead times across the regional network. "
)


def build_prompt(target_tokens: int, depth_pct: int, needle_code: str, tokens_per_word: float) -> str:
    """Build a haystack sized to ~target_tokens and insert the needle at depth_pct."""
    words_total = max(1, int(target_tokens / tokens_per_word))
    unit = FILLER.split()
    body: list[str] = []
    while len(body) < words_total:
        body.extend(unit)
    body = body[:words_total]
    needle = f"IMPORTANT FACT: the authorization code for vault seven is {needle_code}. Remember it."
    pos = int(len(body) * depth_pct / 100)
    words = body[:pos] + needle.split() + body[pos:]
    return " ".join(words)


def run_cell(args, target: int, depth: int, idx: int) -> dict:
    """Run one (length, depth) cell and return a result row."""
    code = f"QX{target // 1000}K-{depth}-{7000 + idx}"
    prompt = build_prompt(target, depth, code, args.tokens_per_word)
    messages = [{
        "role": "user",
        "content": prompt + (
            "\n\nQuestion: What is the authorization code for vault seven? "
            "Answer with only the code."
        ),
    }]
    ctk = {"enable_thinking": False} if args.no_thinking else None
    # Disable server prompt caching by default so each cell is a clean cold prefill
    # (cross-cell prefix caching otherwise skews prefill_tps and processed-token counts).
    extra = None if args.cache_prompt else {"cache_prompt": False}
    res = bench_common.chat_completion(
        args.base_url, args.model, messages,
        max_tokens=args.max_tokens, temperature=0.0, timeout=args.timeout,
        api_key=args.api_key, chat_template_kwargs=ctk, extra_body=extra, stream=args.stream,
    )
    # exact code match with non-code boundaries (avoid substring false positives)
    recalled = re.search(rf"(?<![A-Z0-9-]){re.escape(code)}(?![A-Z0-9-])", res["content"]) is not None
    return {
        "target_tokens": target,
        "depth_pct": depth,
        "prompt_tokens": bench_common.prompt_tokens(res),
        "prefill_tps": bench_common.prefill_tps(res),
        "decode_tps": bench_common.decode_tps(res),
        "ttft_s": round(res["ttft_s"], 3) if res.get("ttft_s") is not None else None,
        "end_to_end_s": round(res["wall_s"], 2),
        "recall": "PASS" if recalled else "FAIL",
        "got": res["content"][:48],
        "server": args.server_label,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Long-context needle + prefill eval (OpenAI-compatible).")
    ap.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8081"),
                    help="OpenAI-compatible base URL, with or without a /v1 suffix (env: BENCH_BASE_URL)")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY"),
                    help="bearer token if the server requires one (env: BENCH_API_KEY)")
    ap.add_argument("--target-tokens", type=int, nargs="+", required=True,
                    help="one or more context sizes; a grid runs over target x depth")
    ap.add_argument("--depths", type=int, nargs="+", default=[25, 50, 90],
                    help="needle insertion depths in percent")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--tokens-per-word", type=float, default=1.4,
                    help="approx tokens per word for sizing the haystack (model-dependent)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false (GLM and Qwen style); "
                         "omit for servers that reject unknown template kwargs")
    ap.add_argument("--stream", action="store_true",
                    help="stream the response to capture client-observed TTFT")
    ap.add_argument("--cache-prompt", action="store_true",
                    help="keep server prompt caching on (llama.cpp); default disables it so each "
                         "cell measures a clean cold prefill")
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--server-label", default="",
                    help="free-text server implementation and version, recorded in every row")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--csv", dest="csv_out", default=None)
    args = ap.parse_args(argv)
    if args.tokens_per_word <= 0:
        ap.error("--tokens-per-word must be > 0")
    if any(not 0 <= d <= 100 for d in args.depths):
        ap.error("--depths must be in 0..100")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"# longctx grid: targets={args.target_tokens} depths={args.depths} "
          f"model={args.model} url={args.base_url}", file=sys.stderr)
    rows: list[dict] = []
    idx = 0
    for target in args.target_tokens:
        for depth in args.depths:
            try:
                row = run_cell(args, target, depth, idx)
            except Exception as e:  # noqa: BLE001 (report and continue the grid)
                row = {
                    "target_tokens": target, "depth_pct": depth, "prompt_tokens": 0,
                    "prefill_tps": None, "decode_tps": None, "ttft_s": None,
                    "end_to_end_s": 0, "recall": "ERROR", "got": str(e)[:48],
                    "server": args.server_label,
                }
            rows.append(row)
            idx += 1
            pf = f"{row['prefill_tps']:.1f}" if row["prefill_tps"] else "n/a"
            print(f"tgt~{target:>7} depth {depth:>3}% | tok={row['prompt_tokens']:>7} | "
                  f"prefill={pf:>8} tok/s | wall={row['end_to_end_s']:>7}s | "
                  f"{row['recall']:>5} | got={row['got']!r}")
    npass = sum(1 for r in rows if r["recall"] == "PASS")
    print(f"# RECALL {npass}/{len(rows)} cells passed")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"# wrote {args.json_out}", file=sys.stderr)
    if args.csv_out and rows:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {args.csv_out}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
