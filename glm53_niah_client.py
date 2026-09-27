#!/usr/bin/env python3
"""NIAH at 262k for GLM-5.3-Flash through the PipeNetwork patched MLX runtime.

WHY A SEPARATE CLIENT. The other mlx clients go through `mlx_lm`. This model is `glm5_next`, which
mlx_lm does not implement; it is an mlx-vlm architecture, and this client uses a patched fork of
that port (`glm53_flash_mlx`, located with --fork-dir or $GLM53_FLASH_MLX_DIR). It must load
through `glm53_flash_mlx.load.load`, because `mlx_vlm.load` resolves `mlx_vlm.models.glm5_next`
from the installed package and would silently pick the upstream module instead of the fork's.
So this client mirrors `mlx_raw_niah_client.py` exactly in protocol and reporting, and differs
only in the two lines that load and generate.

PARITY IS THE FORK AUTHOR'S CLAIM, NOT A MEASUREMENT MADE HERE. The fork states numerical parity
against transformers via its own `tests/test_parity.py`. Any result written from this client should
say the runtime is a patched third-party fork whose correctness rests on the author's test, not on
a measurement made here.

DECODE BY SLOPE, per the project protocol: a least-squares fit of token index against wall time,
which excludes prefill from the decode figure rather than dividing total tokens by total time.
"""
import argparse
import json
import os
import re
import sys
import time

NEEDLE_RE = re.compile(
    r"IMPORTANT RECORD: the secret access code for ([A-Z][A-Za-z'\- ]+) is (\d{8,})\.")


def score(gen, needles):
    """Identical to llamacpp_niah_client.score and mlx_raw_niah_client.score, so every arm is
    scored the same way. retrieval = the code appears anywhere; assoc = code and city on one line."""
    lines = gen.splitlines()
    retrieval = sum(1 for _c, code in needles if code in gen)
    assoc = 0
    for city, code in needles:
        if any(code in ln and city.lower() in ln.lower() for ln in lines):
            assoc += 1
    return retrieval, assoc


def slope_tps(times):
    """Least-squares slope of index vs elapsed, mean-centered. Invariant to a constant offset, so
    absolute timestamps need no prefill subtraction."""
    n = len(times)
    if n < 3:
        return 0.0
    mx_ = sum(range(n)) / n
    my = sum(times) / n
    num = sum((i - mx_) * (t - my) for i, t in enumerate(times))
    den = sum((i - mx_) ** 2 for i in range(n))
    if den == 0 or num <= 0:
        return 0.0
    return 1.0 / (num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork-dir", default=os.environ.get("GLM53_FLASH_MLX_DIR"),
                    help="checkout of the glm53_flash_mlx fork (default: $GLM53_FLASH_MLX_DIR)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-gen", type=int, default=6000,
                    help="generous on purpose. This model reasons out loud before answering, "
                         "which is the shape that produces false recall scores when the budget "
                         "is too small.")
    ap.add_argument("--wired-limit-gb", type=float, default=440.0,
                    help="Metal wired memory limit in GB (the default suits a 512 GB machine)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.fork_dir:
        ap.error("--fork-dir is required (or set GLM53_FLASH_MLX_DIR)")

    sys.path.insert(0, a.fork_dir)
    import mlx.core as mx
    try:
        mx.set_wired_limit(int(a.wired_limit_gb * 1e9))
    except Exception as e:
        print(f"[glm53] wired limit not set: {e}", flush=True)

    from glm53_flash_mlx.load import load
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    raw = open(a.prompt_file, encoding="utf-8").read()
    needles = NEEDLE_RE.findall(raw)
    print(f"[glm53] prompt {len(raw)} bytes, {len(needles)} needles", flush=True)
    if len(needles) != 8:
        print(f"[glm53] WARNING expected 8 needles, got {len(needles)}", flush=True)

    t_load = time.time()
    model, processor = load(a.model_dir, lazy=True)
    load_s = time.time() - t_load
    print(f"[glm53] loaded in {load_s:.0f}s", flush=True)

    prompt = apply_chat_template(processor, model.config, raw, num_images=0)
    tok = getattr(processor, "tokenizer", processor)
    try:
        n_tok = len(tok.encode(prompt))
    except Exception:
        n_tok = -1
    print(f"[glm53] tokenized: {n_tok} tokens", flush=True)

    t0 = time.time()
    first_t = None
    times, chunks = [], []
    for resp in stream_generate(model, processor, prompt, max_tokens=a.max_gen, temperature=0.0):
        now = time.time()
        if first_t is None:
            first_t = now                       # prefill ends at the first token
        times.append(now - t0)
        chunks.append(getattr(resp, "text", "") or "")
    wall = time.time() - t0

    gen = "".join(chunks)
    prefill_s = (first_t - t0) if first_t else 0.0
    prefill_tps = n_tok / prefill_s if (prefill_s > 0 and n_tok > 0) else 0.0
    decode_tps = slope_tps([t - prefill_s for t in times])
    retrieval, assoc = score(gen, needles)

    # Reasoning-truncation guard, output-keyed as in mlx_raw_niah_client.py. Keys on what the
    # GENERATION did, not on what the prompt did, because a model can open <think> spontaneously.
    think_open = re.search(r"<think\s*>", gen, re.IGNORECASE) is not None
    think_closed = re.search(r"</think\s*>", gen, re.IGNORECASE) is not None
    hit_cap = len(times) >= a.max_gen
    answer_inconclusive = think_open and not think_closed and hit_cap

    words = gen.split()
    uniq = (len(set(words)) / len(words)) if words else 0.0

    rec = {"engine": "glm53_flash_mlx (PipeNetwork patched mlx-vlm)", "model_dir": a.model_dir,
           "n_prompt_tokens": n_tok, "n_generated": len(times),
           "prefill_tps": round(prefill_tps, 2), "prefill_s": round(prefill_s, 1),
           "decode_tps_slope": round(decode_tps, 2), "wall_s": round(wall, 1),
           "peak_memory_gb": round(float(mx.get_peak_memory()) / 1e9, 2),
           "retrieval": f"{retrieval}/8", "assoc": f"{assoc}/8",
           "unique_word_ratio": round(uniq, 3),
           "think_open": think_open, "think_closed": think_closed,
           "hit_gen_cap": hit_cap, "answer_inconclusive": answer_inconclusive,
           "token_times": [round(t - prefill_s, 6) for t in times],
           "load_s": round(load_s, 1), "generation": gen[:3000]}
    json.dump(rec, open(a.out, "w"), indent=2)

    print(f"[glm53] prefill {prefill_tps:.1f} tok/s ({prefill_s:.0f}s) | decode {decode_tps:.2f} "
          f"tok/s slope | retrieval {retrieval}/8 assoc {assoc}/8 | peak "
          f"{rec['peak_memory_gb']:.0f} GB | gen {len(times)} tok", flush=True)
    if answer_inconclusive:
        print("[glm53] RECALL INCONCLUSIVE: opened <think>, never closed it, and hit the cap. "
              "The score measures the generation budget, NOT retrieval. Raise --max-gen.", flush=True)
    print(f"[glm53] answer head: {gen[:300]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
