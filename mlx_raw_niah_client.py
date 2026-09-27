#!/usr/bin/env python3
"""Run mlx-lm on the RAW 256K NIAH prompt file, with no chat template.

WHY THIS EXISTS. In a cross-engine 256K comparison the engine is not the only
difference: bench_niah_mlx.py builds a haystack dynamically and applies the model's chat
template, while llamacpp_niah_client.py (and other raw-prompt engines) read one raw prompt
file. Length, template and needle placement then all differ at once, so a recall gap
between engines cannot be attributed. This client removes every difference except the
engine by feeding mlx-lm the identical file the other engines read, tokenized the same way.

It deliberately mirrors llamacpp_niah_client.py: same needle regex, same scorer, same
reported fields. Differences from that client are only what the runtime forces
(in-process mlx-lm instead of HTTP).

Decode is measured BY SLOPE, per the project protocol: a least-squares fit of token
index against wall time, which excludes the prefill from the decode figure rather than
dividing total tokens by total time.
"""
import argparse, json, re, sys, time

NEEDLE_RE = re.compile(
    r"IMPORTANT RECORD: the secret access code for ([A-Z][A-Za-z'\- ]+) is (\d{8,})\.")


def score(gen, needles):
    """Identical to llamacpp_niah_client.score, so the two arms are scored the same.
    retrieval = the code appears anywhere; assoc = code and city on the SAME line."""
    lines = gen.splitlines()
    retrieval = sum(1 for _c, code in needles if code in gen)
    assoc = 0
    for city, code in needles:
        if any(code in ln and city.lower() in ln.lower() for ln in lines):
            assoc += 1
    return retrieval, assoc


def slope_tps(times):
    """Least-squares slope of index vs elapsed, mean-centered. times[i] is the wall
    clock at which token i was produced. Returns tokens/sec."""
    n = len(times)
    if n < 3:
        return 0.0
    mx = sum(range(n)) / n
    my = sum(times) / n
    num = sum((i - mx) * (t - my) for i, t in enumerate(times))
    den = sum((i - mx) ** 2 for i in range(n))
    if den == 0 or num <= 0:
        return 0.0
    return 1.0 / (num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-gen", type=int, default=300)
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--chat-template", action="store_true",
                    help="Serve the SAME raw file through the model's chat template. "
                         "Isolates the template from the haystack construction: the raw "
                         "arm and this arm differ ONLY by the template wrapper.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = open(a.prompt_file, encoding="utf-8").read()
    needles = NEEDLE_RE.findall(prompt)
    print(f"prompt {len(prompt)} bytes, {len(needles)} needles extracted", flush=True)
    if len(needles) != 8:
        print(f"WARNING: expected 8 needles, got {len(needles)}", flush=True)

    t_load = time.time()
    model, tok = load(a.model_dir)
    load_s = time.time() - t_load
    print(f"model loaded in {load_s:.1f}s", flush=True)

    # RAW tokenization: no apply_chat_template. This is the entire point of the run.
    if a.chat_template:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=True, add_generation_prompt=True)
        n_tok = len(ids)
        print(f"tokenized THROUGH CHAT TEMPLATE: {n_tok} tokens", flush=True)
    else:
        ids = tok.encode(prompt)
        n_tok = len(ids)
        print(f"tokenized RAW (no chat template): {n_tok} tokens", flush=True)

    sampler = make_sampler(temp=0.0)   # greedy, matching every other cell
    t0 = time.time()
    first_t = None
    times, chunks = [], []
    for resp in stream_generate(model, tok, prompt=ids, max_tokens=a.max_gen,
                                sampler=sampler, prefill_step_size=a.prefill_step):
        now = time.time()
        if first_t is None:
            first_t = now          # prefill ends when the first token appears
        times.append(now - t0)
        chunks.append(resp.text)
    wall = time.time() - t0

    gen = "".join(chunks)
    prefill_s = (first_t - t0) if first_t else 0.0
    prefill_tps = n_tok / prefill_s if prefill_s > 0 else 0.0
    decode_tps = slope_tps([t - prefill_s for t in times])

    # Truncation guard. mlx-lm does not silently drop tokens the way a server cache
    # can, but the guard is part of the protocol and a mismatch here would mean the
    # tokenizer disagreed with what was fed in.
    served = getattr(resp, "prompt_tokens", n_tok) if times else 0
    truncated = (served != n_tok)
    retrieval, assoc = score(gen, needles)

    # REASONING-TRUNCATION GUARD. A recall score is only meaningful if the model actually
    # finished answering. A reasoning model that is still inside an unterminated <think>
    # block when the budget runs out has been scored on a partial scratchpad, and the
    # number measures the generation budget rather than what the model retrieved.
    #
    # THIS GUARD IS DELIBERATELY NOT CONDITIONED ON THE CHAT TEMPLATE. The earlier version
    # of this check (in bench_niah_mlx.py) only armed when the PROMPT ended with an open
    # <think>, on the rule that "no template means no reasoning mode". That rule is false:
    # a model fed the raw file with no template at all can emit "<think>" as its first
    # token and still be inside that block when the cap arrives, so its partial score is an
    # artifact. Spontaneous reasoning is a property of the model, so the guard keys on what
    # the GENERATION did, which is observable in every case, rather than on what the prompt
    # did, which is only a predictor.
    think_open = re.search(r"<think\s*>", gen, re.IGNORECASE) is not None
    think_closed = re.search(r"</think\s*>", gen, re.IGNORECASE) is not None
    hit_cap = len(times) >= a.max_gen
    answer_inconclusive = think_open and not think_closed and hit_cap

    # PROCESS-LEVEL PEAK MEMORY. bench_niah_mlx.py records mlx-lm's process `peak_memory`.
    # Without the same field here, rows from this client would have to fall back to a
    # SYSTEM-WIDE peak (e.g. from macmon), and a table mixing the two bases can rank models
    # differently depending on which harness produced each row.
    # `resp` not `last`: this client has no `last` variable, it reads the leaked loop
    # variable the same way the truncation guard above does at `served`.
    peak_gb = round(float(getattr(resp, "peak_memory", 0.0)), 2) if times else 0.0

    rec = {"engine": ("mlx-lm (raw prompt file, chat template applied)" if a.chat_template
                      else "mlx-lm (raw prompt, no chat template)"),
           "raw_prompt": not a.chat_template,
           "model_dir": a.model_dir, "peak_memory_gb": peak_gb,
           "n_prompt_tokens_tokenizer": n_tok, "n_prompt_tokens_served": served,
           "truncated": truncated, "n_generated": len(times),
           "prefill_tps": round(prefill_tps, 2), "decode_tps": round(decode_tps, 2),
           "prefill_s": round(prefill_s, 1), "wall_s": round(wall, 1),
           "retrieval": f"{retrieval}/8", "assoc": f"{assoc}/8",
           "think_open": think_open, "think_closed": think_closed,
           "hit_gen_cap": hit_cap, "answer_inconclusive": answer_inconclusive,
           # Per-token timestamps relative to the END of prefill, so a fit window can be
           # varied after the run. See bench_niah_mlx.py for why this exists.
           "token_times": [round(t - prefill_s, 6) for t in times],
           "load_s": round(load_s, 1), "generation": gen[:2000]}
    json.dump(rec, open(a.out, "w"), indent=2)

    print(f"[256k-raw] prefill {prefill_tps:.1f} tok/s (~{prefill_s:.1f}s) | "
          f"decode {decode_tps:.2f} tok/s (slope) | assoc {assoc}/8 retrieval {retrieval}/8 | "
          f"served {served} of {n_tok} tokens | truncated={truncated}", flush=True)
    if answer_inconclusive:
        print(f"[256k-raw] RECALL INCONCLUSIVE: generation opened <think> and never closed it, "
              f"and hit the {a.max_gen}-token cap. The {retrieval}/8 above measures the "
              f"generation budget, NOT retrieval. Re-run with a larger --max-gen before "
              f"reporting this as a model result.", flush=True)
    print(f"answer: {gen[:400]!r}", flush=True)
    return 0 if not truncated else 3


if __name__ == "__main__":
    sys.exit(main())
