#!/usr/bin/env python3
"""Drive a running oMLX server through one 256K NIAH cell and score it.

Compares oMLX with mlx-lm on a SHARED model. oMLX 0.5.5 bundles the same mlx-lm,
so what such a comparison actually measures is server orchestration (paged
batching, scheduler, KV management), not different compute kernels. That only means anything if the
prompt is identical to the mlx-lm arm's, so this client builds the prompt with
the SAME code path bench_niah_mlx.py uses (niah_haystack.build_haystack at the
same seed, then the model's own chat template) and posts the result RAW to
/v1/completions. Same tokens in, so the delta is the server.

Timing: the OpenAI-compatible response carries no prefill/decode split, so this
streams and measures arrival times directly, which is what the project's
"decode by slope" protocol wants anyway:
  - prefill  = time to the FIRST token (server queue + prompt ingestion)
  - decode   = least-squares slope over subsequent token arrivals, warmup
               tokens dropped so the first-token transient is excluded

Stdlib + the repo's own niah_haystack; the tokenizer loads CPU-only.
"""
import argparse, json, re, sys, time, urllib.error, urllib.request
from pathlib import Path

# the repo root, so `import niah_haystack` works from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

NEEDLE_RE = re.compile(
    r"IMPORTANT RECORD: the secret access code for ([A-Z][A-Za-z'\- ]+) is (\d{8,})\.")


def slope_tps(t_arrive, warmup):
    """tokens/sec by mean-centered least squares over cumulative arrival times.
    Mean-centered because a naive one-pass fit catastrophically cancels at
    epoch-scale timestamps."""
    xs = list(range(warmup, len(t_arrive)))
    if len(xs) < 3:
        return 0.0
    ys = [t_arrive[i] for i in xs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))   # cov
    den = sum((x - mx) ** 2 for x in xs)                     # var
    if den == 0.0:
        return 0.0
    sec_per_tok = num / den
    if sec_per_tok <= 0.0:      # non-monotonic arrivals: refuse to report a rate
        return 0.0
    return 1.0 / sec_per_tok


def score(gen, needles):
    lines = gen.splitlines()
    retrieval = sum(1 for _c, code in needles if code in gen)
    assoc = sum(1 for city, code in needles
                if any(code in ln and city.lower() in ln.lower() for ln in lines))
    return retrieval, assoc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--model", required=True, help="model_id as oMLX discovered it")
    ap.add_argument("--model-dir", required=True, help="local dir, for the tokenizer")
    ap.add_argument("--target-ctx", type=int, default=262144)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--prompt-file", default=None,
                    help="RAW prompt file, no chat template. Use this for the clean "
                         "cross-engine A/B, built on the identical raw file that "
                         "llamacpp_niah_client.py and mlx_raw_niah_client.py read. Omit "
                         "for the chat-templated haystack that matches bench_niah_mlx.py.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(Path(a.model_dir), {"trust_remote_code": True})

    if a.prompt_file:
        # RAW path. No haystack build and no chat template, so this arm is
        # token-identical to the llama.cpp and mlx-lm raw arms and the only
        # variable left across them is the engine.
        prompt_text = open(a.prompt_file, encoding="utf-8").read()
        n_built = len(tok.encode(prompt_text))
        needles = NEEDLE_RE.findall(prompt_text)
        print(f"RAW prompt (no chat template): {n_built} tokens, "
              f"{len(needles)} needles", flush=True)
    else:
        # Identical prompt construction to bench_niah_mlx.py, so the only variable
        # between this cell and the mlx-lm cell is the engine.
        import niah_haystack as cb
        h = cb.build_haystack(tok, a.target_ctx, seed=a.seed)
        ids = tok.apply_chat_template([{"role": "user", "content": h.prompt_text}],
                                      tokenize=True, add_generation_prompt=True)
        n_built = len(ids)
        prompt_text = tok.decode(ids)
        needles = NEEDLE_RE.findall(h.prompt_text)
        print(f"built prompt: {n_built} tokens, {len(needles)} needles", flush=True)

    # ASK THE SERVER FOR USAGE. Without a usage block the record carries "served = -1" and
    # the truncation guard is INDIRECT: the direct check, built == served, cannot run.
    # OpenAI-style streaming OMITS the usage block unless the client opts in with
    # stream_options.include_usage.
    #
    # This is written to be strictly safe. If the server rejects the unknown field, the
    # request is retried WITHOUT it and behaviour is exactly what it was before, so the worst
    # case is the status quo rather than a broken client on a 6 hour run.
    base_payload = {"model": a.model, "prompt": prompt_text, "max_tokens": a.max_tokens,
                    "temperature": 0.0, "stream": True}

    def _open(with_usage):
        pl = dict(base_payload)
        if with_usage:
            pl["stream_options"] = {"include_usage": True}
        rq = urllib.request.Request(
            a.base + "/v1/completions", data=json.dumps(pl).encode(),
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(rq, timeout=21600)

    try:
        stream = _open(True)
        usage_requested = True
    except urllib.error.HTTPError as e:
        print(f"[256k] server rejected stream_options ({e.code}); retrying without it, "
              f"so the truncation guard stays INDIRECT for this run", flush=True)
        stream = _open(False)
        usage_requested = False

    t0 = time.time()
    t_arrive, gen, usage = [], "", {}
    with stream as r:  # 6h timeout set at open(), > any 256K prefill here
        for raw in r:
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
            ch = (ev.get("choices") or [{}])[0]
            piece = ch.get("text") or ""
            if piece:
                t_arrive.append(time.time() - t0)
                gen += piece
            if ev.get("usage"):
                usage = ev["usage"]
    wall = time.time() - t0

    ttft = t_arrive[0] if t_arrive else 0.0
    n_gen = len(t_arrive)
    served = int(usage.get("prompt_tokens", -1))
    truncated = served != n_built if served > 0 else None
    prefill_tps = (n_built / ttft) if ttft > 0 else 0.0
    decode_tps = slope_tps(t_arrive, min(a.warmup, max(0, n_gen - 3)))
    retrieval, assoc = score(gen, needles)

    rec = {"engine": "oMLX 0.5.5 native", "model": a.model,
           "usage_requested": usage_requested,
           "guard": "direct" if served > 0 else "INDIRECT (server sent no usage)",
           "n_prompt_tokens_built": n_built, "n_prompt_tokens_served": served,
           "truncated": truncated, "n_generated": n_gen,
           "prefill_tps_ttft": round(prefill_tps, 2), "ttft_s": round(ttft, 1),
           "decode_tps_slope": round(decode_tps, 2), "wall_s": round(wall, 1),
           "retrieval": f"{retrieval}/8", "assoc": f"{assoc}/8",
           "generation": gen[:2000]}
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"[256k] prefill {prefill_tps:.1f} tok/s (ttft {ttft:.1f}s) | "
          f"decode {decode_tps:.2f} tok/s (slope) | assoc {assoc}/8 retrieval {retrieval}/8 | "
          f"served {served} of {n_built} | truncated={truncated}", flush=True)
    print(f"answer: {gen[:400]!r}", flush=True)
    return 0 if truncated is not True else 3


if __name__ == "__main__":
    sys.exit(main())
