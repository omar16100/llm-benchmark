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
import argparse, json, logging, os, re, sys, time
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


def slope_tps_from(times: list) -> float:
    """Least-squares slope of token index against wall time, in tokens/sec.

    Identical arithmetic to mlx_raw_niah_client.slope_tps and omlx_niah_client's, so the
    in-process harness and the raw-prompt clients report decode with the same metric.

    `times` are absolute seconds since generation start, so times[0] still carries the
    whole prefill. THAT IS FINE AND DELIBERATE: a least-squares slope is invariant under a
    constant offset applied to every point, so subtracting the prefill first would change
    nothing. The other clients subtract it only to make their intermediate values readable.
    Returns 0.0 rather than a guess when there are too few points to fit.
    """
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


def split_think(text: str) -> tuple[str, str]:
    """Split a reasoning model's output into (answer, thinking).

    Returns the text AFTER the last </think> as the answer, and the reasoning as
    the second element. If there is no closing tag the whole output is treated as
    the answer, which keeps every non-reasoning model's score byte-identical to
    what it was before this function existed.

    An unterminated <think> (the model ran out of max_tokens mid-reasoning) is the
    one case that needs care: there is no answer at all, so returning "" is
    correct and will score 0, which is the honest result rather than crediting the
    scratchpad.
    """
    import re
    close = list(re.finditer(r"</think\s*>", text, re.IGNORECASE))
    if close:
        end = close[-1].end()
        return text[end:], text[:end]
    open_tag = re.search(r"<think\s*>", text, re.IGNORECASE)
    if open_tag:
        return "", text          # unterminated: reasoning only, no answer
    return text, ""


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
               prefill_step: int, kv_bits=None, kv_group_size: int = 64,
               quantized_kv_start: int = 0) -> dict:
    log.info("=== LENGTH %s (target ctx %d tokens) ===", name, target)
    h = cb.build_haystack(tok, target, seed=20260706)
    msgs = [{"role": "user", "content": h.prompt_text}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    n_built = len(ids)
    log.info("[%s] built prompt: %d tokens (ctx body ~%d, +template/question)",
             name, n_built, h.n_context_tokens_est)

    reset_peak()
    sampler = make_sampler(temp=0.0)  # greedy, deterministic
    # KV-cache quantization is OFF unless --kv-bits is passed, so the default
    # path stays byte-identical to runs made without it. When it is
    # set, stream_generate forwards these through **kwargs to generate_step,
    # which swaps in mlx_lm's QuantizedKVCache (group_size/bits) from step
    # `quantized_kv_start` onward.
    kv_kw = {}
    if kv_bits is not None:
        kv_kw = {"kv_bits": kv_bits, "kv_group_size": kv_group_size,
                 "quantized_kv_start": quantized_kv_start}
        log.info("[%s] KV cache QUANTIZED: %d-bit, group %d, from step %d",
                 name, kv_bits, kv_group_size, quantized_kv_start)
    t0 = time.time()
    gen_text, last = "", None
    # PER-TOKEN TIMESTAMPS. mlx-lm's internal `generation_tps` is a MEAN, while the other
    # harnesses report a least-squares SLOPE. The two can differ in either direction
    # depending on the model, so no single factor converts one to the other. Recording
    # the timestamps lets BOTH figures come from the SAME generation, without a second
    # run's thermal state confounding the comparison.
    tok_times = []
    for resp in stream_generate(model, tok, prompt=ids, max_tokens=max_gen,
                                sampler=sampler, prefill_step_size=prefill_step,
                                **kv_kw):
        tok_times.append(time.time() - t0)
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

    # Reasoning models emit a <think> block before the answer. Scoring the whole
    # output would let a model that merely RECITES codes while thinking, then
    # answers wrongly or not at all, score full retrieval off its scratchpad.
    # That is a false positive, so the headline score is the FINAL answer only.
    # The full-output score is kept alongside it, because the gap between them is
    # itself the interesting signal: needles found but not reported.
    # A reasoning chat template typically opens <think> in the PROMPT, so the
    # generation carries only the CLOSING tag. If the model is cut off by max_gen
    # before reaching </think>, the output looks tag-free and would be scored as
    # though it were the answer, which is the exact false positive this guard
    # exists to stop: a reasoning model cut off by a small --max-gen can "score" full
    # recall off its scratchpad, and only a larger budget shows its real answer.
    # Scope this strictly to templates that actually open a think block. A plain
    # model that simply used its whole budget answers from token one, so treating
    # "hit the cap" as "no answer" would wrongly zero every existing row that ran
    # with a tight --max-gen.
    prompt_tail = tok.decode(ids[-256:]) if len(ids) > 256 else tok.decode(ids)
    last_open = list(re.finditer(r"<think\s*>", prompt_tail, re.IGNORECASE))
    prompt_opens_think = bool(last_open) and not re.search(
        r"</think\s*>", prompt_tail[last_open[-1].end():], re.IGNORECASE)

    n_gen_tokens = int(getattr(last, "generation_tokens", 0))
    hit_cap = n_gen_tokens >= max_gen
    closed_think = re.search(r"</think\s*>", gen_text, re.IGNORECASE) is not None
    answer_inconclusive = prompt_opens_think and hit_cap and not closed_think

    answer_text, think_text = split_think(gen_text)
    if answer_inconclusive:
        answer_text = ""       # no answer was ever emitted; refuse to credit the scratchpad
    score = cb.score_answer(answer_text, h.needles)
    score_full = cb.score_answer(gen_text, h.needles)
    coh = coherence_flags(answer_text)
    if answer_inconclusive:
        log.warning("[%s] ANSWER INCONCLUSIVE: generation hit the max_gen cap (%d "
                    "tokens) without emitting </think>, so no final answer exists. "
                    "Scoring it 0; the whole-output score %d/%d is the scratchpad, "
                    "not a result. Raise --max-gen and re-run.",
                    name, n_gen_tokens, score_full["association_correct"],
                    score_full["n_needles"])
    peak_gb = round(float(getattr(last, "peak_memory", 0.0)), 2)
    rec = {
        "name": name, "target_ctx_tokens": target,
        "built_prompt_tokens": n_built,
        "model_saw_prompt_tokens": prompt_tokens,
        "truncated": truncated,
        "prompt_tps": round(float(getattr(last, "prompt_tps", 0.0)), 1),
        "generation_tps": round(float(getattr(last, "generation_tps", 0.0)), 2),
        "generation_tps_slope": round(slope_tps_from(tok_times), 2),
        # Raw per-token timestamps, persisted so a fit window can be varied AFTER the run
        # without paying for the GPU again. When two runs with different generation
        # lengths disagree on decode speed, a short fit window biased by an early-token
        # transient is one explanation, and only these timestamps can check it.
        "token_times": [round(t, 6) for t in tok_times],
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
        # TAIL. With only the head stored, a run that hits its cap leaves most of the
        # generation unobserved, so whether the model was still reasoning, looping, or
        # about to terminate cannot be told. The tail answers that, and it costs 2 KB.
        "generated_tail": gen_text[-2000:],
        "generated_chars_total": len(gen_text),
        # Reasoning-model diagnostics. think_chars == 0 means there was no <think>
        # block and the two scores are the same object by construction.
        "think_chars": len(think_text),
        "answer_inconclusive": answer_inconclusive,
        "retrieval_full_output": score_full["retrieval_present"],
        "association_full_output": score_full["association_correct"],
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
    ap.add_argument("--model-dir", default=os.environ.get("NIAH_MODEL_DIR"),
                    help="local model directory (default: $NIAH_MODEL_DIR)")
    ap.add_argument("--lengths", default="control_2k,8k,32k,128k,256k")
    ap.add_argument("--max-gen", type=int, default=300)
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--out", default="results/niah_results.json")
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="quantize the KV cache to N bits (mlx_lm QuantizedKVCache). "
                         "Default: off, i.e. the fp16 cache")
    ap.add_argument("--kv-group-size", type=int, default=64,
                    help="group size for --kv-bits (mlx_lm default 64)")
    ap.add_argument("--quantized-kv-start", type=int, default=0,
                    help="step at which to begin quantizing the KV cache")
    ap.add_argument("--loader", default="mlx-lm",
                    choices=["mlx-lm", "bonsai"],
                    help="bonsai: load a Prism ternary pack (model_type "
                         "prism_hadamard_qwen35) via the loader bundled in the "
                         "pack's runtime/ dir. Stock mlx-lm loaders skip the "
                         "Hadamard activation transform and return WRONG output "
                         "rather than erroring, so this is not optional for those "
                         "packs. Everything downstream is unchanged: the model is "
                         "an mlx_lm qwen3_5 TextModel whose projections are "
                         "swapped for Packed modules.")
    ap.add_argument("--bonsai-fast", dest="bonsai_fast", action="store_true",
                    default=False,
                    help="apply the fp16-Hadamard + mx.compile patch to the Bonsai "
                         "projections (bonsai_fast.py, found via $BONSAI_FAST_DIR). "
                         "OFF by default because a paired interleaved A/B (one model "
                         "load, alternating in place) measured it SLOWER than stock "
                         "for both decode and prefill. Output is byte-identical either "
                         "way, so this flag exists only to reproduce that negative "
                         "result.")
    ap.add_argument("--nonstrict", action="store_true",
                    help="load base weights non-strict (ignore extra checkpoint "
                         "tensors, e.g. DeepSeek DSpark MTP heads not modeled by "
                         "stock mlx-lm); needed for the Vontra DeepSeek-V4 MXFP4 dir")
    args = ap.parse_args()
    if not args.model_dir:
        ap.error("--model-dir is required (or set NIAH_MODEL_DIR)")

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
    if args.nonstrict:
        from mlx_lm.utils import load_model, load_tokenizer
        mp = Path(args.model_dir)
        model, cfg = load_model(mp, strict=False)
        tok = load_tokenizer(mp, {"trust_remote_code": True},
                             eos_token_ids=cfg.get("eos_token_id", None))
        log.info("model loaded NON-STRICT (extra checkpoint tensors ignored)")
        if getattr(tok, "chat_template", None) is None:
            # DeepSeek-V3/V4 template; the MXFP4 conversion dropped it. Single
            # user turn renders: {bos}<|User|>{content}<|Assistant|>.
            ds_tmpl = (
                "{% if not add_generation_prompt is defined %}"
                "{% set add_generation_prompt = false %}{% endif %}"
                "{{ bos_token }}"
                "{%- for message in messages %}"
                "{%- if message['role'] == 'user' %}"
                "{{ '<｜User｜>' + message['content'] }}"
                "{%- elif message['role'] == 'assistant' %}"
                "{{ '<｜Assistant｜>' + message['content'] "
                "+ '<｜end▁of▁sentence｜>' }}"
                "{%- endif %}{%- endfor %}"
                "{% if add_generation_prompt %}{{ '<｜Assistant｜>' }}"
                "{% endif %}"
            )
            for obj in (tok, getattr(tok, "_tokenizer", None)):
                if obj is not None:
                    try:
                        obj.chat_template = ds_tmpl
                    except Exception:
                        pass
            log.info("applied fallback DeepSeek chat template")
    elif args.loader == "bonsai":
        # The pack ships its own loader because the ternary weights are stored in a
        # blockwise-Hadamard-rotated basis; the matching transform has to be applied
        # to activations at runtime. runtime/artifact.py imports its siblings by bare
        # name ("from runtime import Packed"), so the pack's runtime/ dir must go on
        # sys.path, not just the pack root.
        mp = Path(args.model_dir)
        rt = mp / "runtime"
        if not (rt / "vision_artifact.py").exists():
            log.error("--loader bonsai: no runtime/vision_artifact.py under %s", mp)
            sys.exit(2)
        sys.path.insert(0, str(rt))
        # NOT runtime/artifact.py: that one hard-rejects anything but
        # schema_version 1 and this pack is schema_version 2, so it raises
        # "Unsupported packed model schema". vision_artifact.load_vl_model is the
        # schema-2 entry point (it is what the model card's Quickstart uses) and
        # it installs the Packed modules into the vlm's language_model only,
        # leaving the FP16 vision tower as passthrough.
        from vision_artifact import load_vl_model
        from mlx_lm.utils import load_tokenizer

        # RETRACTED OPTIMIZATION, kept switchable so the negative result is
        # reproducible. A sequential A/B first suggested a speedup; that was a
        # run-ORDER artifact (its baseline ran first and cold). A paired
        # interleaved re-test showed the patch is SLOWER. Off by default.
        if args.bonsai_fast:
            fast_dir = os.environ.get("BONSAI_FAST_DIR")
            if not fast_dir:
                log.error("--bonsai-fast needs BONSAI_FAST_DIR (dir containing "
                          "bonsai_fast.py)")
                sys.exit(2)
            sys.path.insert(0, fast_dir)
            import bonsai_fast
            log.info(bonsai_fast.apply())
        else:
            log.info("bonsai_fast DISABLED, using the pack's stock Packed.__call__")

        vlm, _, cfg = load_vl_model(str(mp), load_processor=False)

        class _TextOnly:
            """Adapt mlx-vlm's LanguageModel to the interface mlx-lm's
            generate_step expects. Its __call__ returns a LanguageModelOutput
            dataclass, which generate_step indexes as `logits[:, -1, :]` and so
            fails with "'LanguageModelOutput' object is not subscriptable".
            Everything else (make_cache, layers) delegates untouched, so the 48
            ArraysCache / 16 KVCache split is preserved and --kv-bits still
            quantizes exactly the 16 full-attention layers."""

            def __init__(self, lm):
                self._lm = lm

            def __call__(self, *a, **kw):
                out = self._lm(*a, **kw)
                return getattr(out, "logits", out)

            def __getattr__(self, n):
                return getattr(self._lm, n)

        model = _TextOnly(vlm.language_model)
        tok = load_tokenizer(mp, {"trust_remote_code": True},
                             eos_token_ids=cfg.get("eos_token_id", None))
        n_packed = len(cfg.get("modules", []))
        tcfg = cfg.get("text_config", {})
        lt = tcfg.get("layer_types") or []
        log.info("model loaded via BONSAI runtime: schema %s, %d packed modules, "
                 "%d layers (%d linear / %d full attention), rotation=%s, "
                 "components=%s",
                 cfg.get("schema_version"), n_packed,
                 tcfg.get("num_hidden_layers", -1),
                 sum(1 for x in lt if x == "linear_attention"),
                 sum(1 for x in lt if x == "full_attention"),
                 bool(cfg.get("hadamard_config")), cfg.get("components"))
        from collections import Counter as _C
        log.info("cache composition: %s (only KVCache entries are --kv-bits "
                 "quantizable; ArraysCache has no to_quantized)",
                 dict(_C(type(c).__name__ for c in model.make_cache())))
    else:
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
                             args.prefill_step, kv_bits=args.kv_bits,
                             kv_group_size=args.kv_group_size,
                             quantized_kv_start=args.quantized_kv_start)
        except Exception as e:
            import traceback
            log.error("[%s] RUN FAILED: %s", name, e)
            log.error(traceback.format_exc())
            rec = {"name": name, "target_ctx_tokens": LADDER[name],
                   "error": f"{type(e).__name__}: {e}"}
        results[name] = rec
        payload = {"model": args.model_dir,
                   "loader": args.loader,
                   "bonsai_fast": args.bonsai_fast if args.loader == "bonsai" else None,
                   "kv_bits": args.kv_bits,
                   "kv_group_size": args.kv_group_size if args.kv_bits else None,
                   "quantized_kv_start": args.quantized_kv_start if args.kv_bits else None,
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
