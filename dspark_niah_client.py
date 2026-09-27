#!/usr/bin/env python3
"""NIAH client for mlx-dspark's CHAT endpoint, with the direct truncation guard.

WHY THIS EXISTS RATHER THAN A FLAG ON omlx_niah_client.py. That client posts to /v1/completions
with the prompt RAW, which is right for raw-prompt comparisons. It is the wrong form for a model
whose reference result was measured on the CHAT-TEMPLATED prompt: prompt form can matter per
model, and a reasoning model driven raw can stop after a couple of tokens with an empty answer,
leaving no decode slope to fit.

omlx_niah_client.py is deliberately not modified, so results already produced with it stay
reproducible with the same code.

WHAT THIS ADDS BEYOND THE SIBLING:
  * /v1/chat/completions, so the SERVER applies the model's own chat template
  * reads delta.content AND the reasoning deltas. Some models route text into the reasoning
    channel when the budget is short, so a client that reads only `content` sees an empty answer
    and scores it a failure.
  * captures dspark's x_mlx_dspark block, which carries accept_len and target_forwards.
    accept_len (accepted tokens per speculation round) is the column that makes a speculative
    decoding speed result interpretable.

The truncation guard, the needle regex and the least-squares slope are IMPORTED from
omlx_niah_client so the two clients cannot drift apart.
"""
import argparse, json, sys, time, urllib.error, urllib.request
from pathlib import Path

from omlx_niah_client import NEEDLE_RE, slope_tps, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8091")
    ap.add_argument("--model", required=True, help="model id exactly as /v1/models reports it")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=6,
                    help="tokens skipped before the slope fit, so the first-token cost is excluded")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompt_text = Path(a.prompt_file).read_text(encoding="utf-8")
    needles = NEEDLE_RE.findall(prompt_text)
    print(f"[dspark] prompt {len(prompt_text)} bytes, {len(needles)} needles", flush=True)
    if len(needles) != 8:
        print(f"[dspark] WARNING: expected 8 needles, found {len(needles)}", flush=True)

    base_payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": a.max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    def _open(with_usage):
        pl = dict(base_payload)
        if with_usage:
            pl["stream_options"] = {"include_usage": True}
        rq = urllib.request.Request(
            a.base + "/v1/chat/completions", data=json.dumps(pl).encode(),
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(rq, timeout=21600)

    # Ask for usage, and fall back without it if the server rejects the field, so the worst case is
    # the old indirect guard rather than a broken client on a long run (same as omlx_niah_client).
    usage_requested = True
    try:
        stream = _open(True)
    except urllib.error.HTTPError:
        usage_requested = False
        stream = _open(False)

    t0 = time.time()
    t_arrive, chunks, reasoning_chunks = [], [], []
    usage, dspark_meta, finish = {}, {}, None

    for raw in stream:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            ev = json.loads(body)
        except json.JSONDecodeError:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("x_mlx_dspark"):
            dspark_meta = ev["x_mlx_dspark"]
        for ch in ev.get("choices") or []:
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            piece = delta.get("content")
            if piece:
                chunks.append(piece)
                t_arrive.append(time.time())
            # dspark's chat stream carries thinking text under `reasoning_content`, NOT `reasoning`
            # and NOT `content`. Checking only `content` and `reasoning` would report 0 characters
            # for a healthy all-reasoning generation. Both spellings are accepted so this client
            # does not break if a server uses the other one.
            for key in ("reasoning_content", "reasoning"):
                reason = delta.get(key)
                if reason:
                    reasoning_chunks.append(reason)
                    t_arrive.append(time.time())
                    break

    wall = time.time() - t0
    gen = "".join(chunks)
    reasoning = "".join(reasoning_chunks)
    served = int(usage.get("prompt_tokens", -1))
    completed = int(usage.get("completion_tokens", len(t_arrive)))

    # The prompt is tokenized by the SERVER (chat template applied server-side), so there is no
    # client-side token count to compare against: the guard is direct when the server reports
    # usage.prompt_tokens, and indirect otherwise.
    guard = "direct" if served > 0 else "indirect"

    # Score the whole output. Some models put the answer in the reasoning channel when the budget
    # is tight, so scoring only `content` would understate recall for a reason that is not the
    # model's.
    whole = gen + ("\n" + reasoning if reasoning else "")
    retrieval, assoc = score(whole, needles)

    # per_city is built here because score() returns only the two counts. Recording it per city
    # keeps the evidence: a built haystack's needles differ from the source file's, so a run must
    # always be scored against ITS OWN needles.
    lines = whole.splitlines()
    per_city = {
        city: {
            "code": code,
            "present": code in whole,
            "associated": any(code in ln and city.lower() in ln.lower() for ln in lines),
        }
        for city, code in needles
    }

    slope = slope_tps(t_arrive, a.warmup)
    prefill_tps = (served / dspark_meta["prefill_seconds"]
                   if dspark_meta.get("prefill_seconds") else 0.0)

    rec = {
        "engine": "mlx-dspark serve (chat)",
        "model": a.model,
        "mode": dspark_meta.get("mode"),
        "n_prompt_tokens_served": served,
        "truncated": False if served > 0 else None,
        "guard": guard,
        "usage_requested": usage_requested,
        "n_generated": completed,
        "finish_reason": finish,
        "prefill_tps": round(prefill_tps, 2),
        "decode_tps_slope": round(slope, 3),
        "accept_len": dspark_meta.get("accept_len"),
        "target_forwards": dspark_meta.get("target_forwards"),
        "decode_tps_server": dspark_meta.get("decode_tokens_per_sec"),
        "ttft_s": dspark_meta.get("ttft_seconds"),
        "roofline_ratio": dspark_meta.get("roofline_ratio"),
        "retrieval": f"{retrieval}/8",
        "assoc": f"{assoc}/8",
        "per_city": per_city,
        "content_chars": len(gen),
        "reasoning_chars": len(reasoning),
        "generated_tail": whole[-1200:],
        "wall_s": round(wall, 1),
    }
    Path(a.out).write_text(json.dumps(rec, indent=2))

    print(f"[dspark] mode={rec['mode']} served={served} guard={guard} "
          f"gen={completed} finish={finish}", flush=True)
    print(f"[dspark] prefill {rec['prefill_tps']} tok/s | decode {rec['decode_tps_slope']} slope "
          f"({rec['decode_tps_server']} server) | accept_len {rec['accept_len']} | "
          f"forwards {rec['target_forwards']}", flush=True)
    print(f"[dspark] retrieval {retrieval}/8 assoc {assoc}/8 | "
          f"content {len(gen)} chars, reasoning {len(reasoning)} chars", flush=True)


if __name__ == "__main__":
    main()
