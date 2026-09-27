#!/usr/bin/env python3
"""Drive a running llama-server through one 256K NIAH cell and score it.

Why this exists: driving a 256K prompt through `llama-cli` (interactive path) or
`llama-bench` under a fixed timeout can fail for harness reasons rather than model
reasons: an interactive session that hangs with the GPU idle, or a guard shorter than
a long prefill. This client talks to `llama-server` over HTTP instead: no interactive
UI, no fixed guard shorter than the work, and it records the same fields as the other
NIAH clients.

Apples-to-apples notes:
  - the prompt is sent RAW (/completion "prompt"), with no chat template, which
    matches mlx_raw_niah_client.py and other raw-prompt engines reading the same
    file. bench_niah_mlx.py applies the chat template, so its rows are only
    class-level comparable with this client; raw-vs-raw is the clean comparison.
  - temperature 0 (greedy) and cache_prompt false, so the prefill is really paid.

Stdlib only, so it runs under any python on the box.
"""
import argparse, json, re, sys, time, urllib.request, urllib.error

NEEDLE_RE = re.compile(
    r"IMPORTANT RECORD: the secret access code for ([A-Z][A-Za-z'\- ]+) is (\d{8,})\.")


def post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_healthy(base, timeout_s):
    """llama-server returns 503 while the model loads; 200 once ready."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit(f"llama-server not healthy after {timeout_s}s")


def score(gen, needles):
    """Same two metrics the mlx harness reports: retrieval = the code appears
    anywhere; assoc = the code and its city appear on the SAME line."""
    lines = gen.splitlines()
    retrieval = sum(1 for _c, code in needles if code in gen)
    assoc = 0
    for city, code in needles:
        if any(code in ln and city.lower() in ln.lower() for ln in lines):
            assoc += 1
    return retrieval, assoc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--n-predict", type=int, default=300)
    ap.add_argument("--load-timeout", type=int, default=1800)
    ap.add_argument("--run-timeout", type=int, default=14400)  # 4h, > any prefill seen
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompt = open(a.prompt_file, encoding="utf-8").read()
    needles = NEEDLE_RE.findall(prompt)
    print(f"prompt {len(prompt)} bytes, {len(needles)} needles extracted", flush=True)
    if len(needles) != 8:
        print(f"WARNING: expected 8 needles, got {len(needles)}", flush=True)

    load_s = wait_healthy(a.base, a.load_timeout)
    print(f"server healthy after {load_s:.1f}s", flush=True)

    # Truncation guard: tokenize first so we know what the model SHOULD see.
    n_tok = None
    try:
        tk = post(a.base + "/tokenize", {"content": prompt}, timeout=600)
        n_tok = len(tk.get("tokens", []))
        print(f"tokenized: {n_tok} tokens", flush=True)
    except Exception as e:
        print(f"tokenize failed ({e}); guard will rely on timings only", flush=True)

    t0 = time.time()
    r = post(a.base + "/completion",
             {"prompt": prompt, "n_predict": a.n_predict, "temperature": 0.0,
              "cache_prompt": False, "stream": False},
             timeout=a.run_timeout)
    wall = time.time() - t0

    gen = r.get("content", "")
    tim = r.get("timings", {}) or {}
    prompt_n = int(tim.get("prompt_n", -1))
    pred_n = int(tim.get("predicted_n", -1))
    prompt_ms = float(tim.get("prompt_ms", 0.0))
    pred_ms = float(tim.get("predicted_ms", 0.0))
    prefill_tps = (prompt_n / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0
    decode_tps = (pred_n / (pred_ms / 1000.0)) if pred_ms > 0 else 0.0

    # Guard: the server must have processed every prompt token. cache_prompt is
    # off, so a short prompt_n means real truncation, not a cache hit.
    truncated = (n_tok is not None and prompt_n != n_tok)
    retrieval, assoc = score(gen, needles)

    rec = {"engine": "llama.cpp b10200 (llama-server)", "raw_prompt": True,
           "n_prompt_tokens_tokenizer": n_tok, "n_prompt_tokens_served": prompt_n,
           "truncated": truncated, "n_generated": pred_n,
           "prefill_tps": round(prefill_tps, 2), "decode_tps": round(decode_tps, 2),
           "prefill_s": round(prompt_ms / 1000.0, 1), "decode_s": round(pred_ms / 1000.0, 1),
           "wall_s": round(wall, 1), "retrieval": f"{retrieval}/8", "assoc": f"{assoc}/8",
           "load_s": round(load_s, 1), "generation": gen[:2000]}
    json.dump(rec, open(a.out, "w"), indent=2)

    print(f"[256k] prefill {prefill_tps:.1f} tok/s (~{prompt_ms/1000:.1f}s) | "
          f"decode {decode_tps:.2f} tok/s | assoc {assoc}/8 retrieval {retrieval}/8 | "
          f"served {prompt_n} of {n_tok} tokens | truncated={truncated}", flush=True)
    print(f"answer: {gen[:400]!r}", flush=True)
    return 0 if not truncated else 3


if __name__ == "__main__":
    sys.exit(main())
