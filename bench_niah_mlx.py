"""Needle-in-a-haystack verification for Kimi-Linear-48B-A3B (native 1M claim).

Runs a staged length ladder, and at every length ENFORCES the guards the
advisor flagged:
  (1) no silent context truncation  -> assert prompt_tokens == tokens we built
  (2) short-context control          -> a 2k run with identical needle format
  (3) staged ladder + metrics        -> prefill/decode tok/s + peak RAM each step
  (4) coherence signal               -> flag garbage/looping output, not just misses
Results are checkpointed to JSON after EACH length so a late OOM/crash never
loses earlier data.

Usage:
  .venv/bin/python bench_niah_mlx.py --lengths control_2k,128k,256k
  .venv/bin/python bench_niah_mlx.py --lengths 512k,1m
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

import niah_haystack as cb

LADDER = {
    "control_2k": 2_000,
    "8k": 8_000,
    "32k": 32_000,
    "128k": 131_072,
    "256k": 262_144,
    "512k": 524_288,
    "1m": 1_048_576,
}

log = logging.getLogger("niah")


def setup_logging(logfile: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(logfile)],
    )


def coherence_flags(text: str) -> dict:
    """Cheap heuristics to catch gibberish / degenerate decoding."""
    t = text.strip()
    words = t.split()
    if not words:
        return {"empty": True, "looping": False, "unique_word_ratio": 0.0}
    uniq = len(set(w.lower() for w in words)) / len(words)
    # crude loop detector: any 6-gram repeated 4+ times
    looping = False
    toks = words
    from collections import Counter
    if len(toks) >= 24:
        grams = Counter(tuple(toks[i:i+6]) for i in range(len(toks) - 5))
        looping = any(c >= 4 for c in grams.values())
    return {"empty": False, "looping": looping,
            "unique_word_ratio": round(uniq, 3)}


def reset_peak():
    for fn in ("reset_peak_memory", "clear_cache"):
        f = getattr(mx, fn, None)
        if f:
            try:
                f()
            except Exception:
                pass
    m = getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    if m:
        try:
            m()
        except Exception:
            pass


def run_length(model, tok, name: str, target: int, max_gen: int,
               prefill_step: int) -> dict:
    log.info("=== LENGTH %s (target ctx %d tokens) ===", name, target)
    h = cb.build_haystack(tok, target, seed=20260706)
    msgs = [{"role": "user", "content": h.prompt_text}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    n_built = len(ids)
    log.info("[%s] built prompt: %d tokens (ctx body ~%d, +template/question)",
             name, n_built, h.n_context_tokens_est)

    reset_peak()
    sampler = make_sampler(temp=0.0)  # greedy, deterministic
    t0 = time.time()
    gen_text, last = "", None
    for resp in stream_generate(model, tok, prompt=ids, max_tokens=max_gen,
                                sampler=sampler, prefill_step_size=prefill_step):
        gen_text += resp.text
        last = resp
    wall = time.time() - t0

    prompt_tokens = int(getattr(last, "prompt_tokens", -1))
    truncated = prompt_tokens != n_built
    if truncated:
        # The one guard the whole test rests on: refuse to score, and never let a
        # truncated run be recorded as a pass. The caller turns this into an error
        # record for this length and moves on.
        raise RuntimeError(
            f"[{name}] CONTEXT TRUNCATION: built={n_built} but model saw="
            f"{prompt_tokens} (delta={n_built - prompt_tokens}). Guard failed; "
            f"any 'pass' at this length would be INVALID."
        )
    log.info("[%s] truncation guard PASSED: model saw all %d tokens",
             name, prompt_tokens)

    score = cb.score_answer(gen_text, h.needles)
    coh = coherence_flags(gen_text)
    peak_gb = round(float(getattr(last, "peak_memory", 0.0)), 2)
    rec = {
        "name": name, "target_ctx_tokens": target,
        "built_prompt_tokens": n_built,
        "model_saw_prompt_tokens": prompt_tokens,
        "truncated": truncated,
        "prompt_tps": round(float(getattr(last, "prompt_tps", 0.0)), 1),
        "generation_tps": round(float(getattr(last, "generation_tps", 0.0)), 2),
        "generation_tokens": int(getattr(last, "generation_tokens", 0)),
        "peak_memory_gb": peak_gb,
        "wall_seconds": round(wall, 1),
        "prefill_est_seconds": (round(n_built / float(last.prompt_tps), 1)
                                if getattr(last, "prompt_tps", 0) else None),
        "retrieval_rate": score["retrieval_rate"],
        "association_rate": score["association_rate"],
        "n_needles": score["n_needles"],
        "association_correct": score["association_correct"],
        "coherence": coh,
        "per_city": score["per_city"],
        "generated_answer": gen_text[:2000],
    }
    log.info("[%s] prefill %.0f tok/s (~%ss) | decode %.2f tok/s | peakRAM %.1fGB "
             "| assoc %d/%d retrieval %d/%d | coherent=%s",
             name, rec["prompt_tps"], rec["prefill_est_seconds"],
             rec["generation_tps"], peak_gb, score["association_correct"],
             score["n_needles"], score["retrieval_present"], score["n_needles"],
             (not coh["looping"] and not coh["empty"]))
    log.info("[%s] answer: %s", name, gen_text.strip()[:400].replace("\n", " | "))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default="/Users/macmini/models/Kimi-Linear-48B-A3B-Instruct-8bit")
    ap.add_argument("--lengths", default="control_2k,8k,32k,128k,256k")
    ap.add_argument("--max-gen", type=int, default=300)
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--out", default="results/niah_results.json")
    args = ap.parse_args()

    proj = Path(__file__).parent
    setup_logging(proj / "logs" / "niah_run.log")
    out = proj / args.out
    out.parent.mkdir(exist_ok=True)

    lengths = [x.strip() for x in args.lengths.split(",") if x.strip()]
    for x in lengths:
        if x not in LADDER:
            log.error("unknown length %s (valid: %s)", x, list(LADDER)); sys.exit(2)

    log.info("loading model from %s ...", args.model_dir)
    t0 = time.time()
    model, tok = load(args.model_dir,
                      tokenizer_config={"trust_remote_code": True})
    log.info("model loaded in %.1fs", time.time() - t0)

    # merge with any existing checkpoint so incremental runs accumulate
    results = {}
    if out.exists():
        try:
            results = {r["name"]: r for r in json.load(open(out)).get("runs", [])}
        except Exception:
            pass

    for name in lengths:
        try:
            rec = run_length(model, tok, name, LADDER[name], args.max_gen,
                             args.prefill_step)
        except Exception as e:
            import traceback
            log.error("[%s] RUN FAILED: %s", name, e)
            log.error(traceback.format_exc())
            rec = {"name": name, "target_ctx_tokens": LADDER[name],
                   "error": f"{type(e).__name__}: {e}"}
        results[name] = rec
        payload = {"model": args.model_dir,
                   "runs": [results[k] for k in LADDER if k in results]}
        json.dump(payload, open(out, "w"), indent=2)
        log.info("checkpointed %d results -> %s", len(results), out)

    log.info("DONE. summary:")
    for k in LADDER:
        if k in results:
            r = results[k]
            if "error" in r:
                log.info("  %-10s ERROR %s", k, r["error"])
            else:
                log.info("  %-10s built=%d saw=%d trunc=%s assoc=%d/%d decode=%.1ft/s "
                         "prefill=%.0ft/s peak=%.0fGB",
                         k, r["built_prompt_tokens"], r["model_saw_prompt_tokens"],
                         r["truncated"], r["association_correct"], r["n_needles"],
                         r["generation_tps"], r["prompt_tps"], r["peak_memory_gb"])


if __name__ == "__main__":
    main()
