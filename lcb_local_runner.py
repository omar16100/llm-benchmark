#!/usr/bin/env python3
"""Run LiveCodeBench against a LOCAL mlx_lm.server, without editing the vendored tree.

WHY THIS FILE EXISTS. LiveCodeBench lives at
`eval_frameworks/LiveCodeBench` (override with $LCB_DIR), which is in `.gitignore` with zero tracked
files: it is a local clone only. Any edit made inside it is lost the moment it is re-cloned, and is invisible to code
review. This wrapper therefore performs the two pieces of wiring LiveCodeBench needs WITHOUT
touching it:

  1. Registers local models in `LanguageModelStore`, the dict `lcb_runner.runner.main` looks up at
     `main.py:21`. LiveCodeBench refuses any `--model` that is not registered.
  2. Points the OpenAI SDK at `mlx_lm.server`. `OpenAIRunner` builds its client at CLASS-DEFINITION
     time with no `base_url` (`oai_runner.py:14-16`), relying entirely on the SDK reading
     `OPENAI_BASE_URL` from the environment. That variable must therefore be set BEFORE the runner
     module is imported, which is why it is set at the top of main() here and why the lcb imports
     are deferred.

DEPENDENCY NOTE. LiveCodeBench declares `vllm` and `torch`, neither of which installs on Apple
Silicon. They are never reached on the OpenAI path because `build_runner` imports runners lazily
(`runner_utils.py:5-8`), so a separate venv for this runner needs only datasets, openai, pebble,
numpy and tqdm. `datasets` is pinned BELOW 4.0 on purpose: 4.x removed script-based datasets, and
`code_generation_lite` is script-based, so 4.x fails with "Dataset scripts are no longer supported".

CONTAMINATION WARNING, and the reason `--start-date` matters less than hoped. LiveCodeBench can
filter problems by release date, which would let a model be scored only on problems published
after its training cutoff. For recent models that property has lapsed: `release_v6` holds 1055
problems with contest dates from 2023-05-07 to **2025-04-06**, and the Hugging Face dataset
(`livecodebench/code_generation_lite`) was last modified 2025-06-05 (both checked 27 Sep 2026).
A model with a 2026 training cutoff has no post-cutoff problems to be scored on. What survives is
RELATIVE comparison: models exposed to broadly the same public corpus can still be ranked against
each other, even though the absolute pass@1 is inflated. Any result written from this harness
should say so.
"""
import argparse
import os
import sys
from pathlib import Path

# absolute, because main() puts it on sys.path and then chdirs into it
LCB_DIR = Path(os.environ.get(
    "LCB_DIR", Path(__file__).resolve().parent / "eval_frameworks" / "LiveCodeBench")
).expanduser().absolute()


def _shim_anthropic_legacy_constants():
    """Restore `HUMAN_PROMPT` / `AI_PROMPT`, which modern anthropic SDKs no longer export.

    `lcb_runner/prompts/test_output_prediction.py:3` imports both at module level, and
    `lcb_runner/prompts/__init__.py` imports that module unconditionally, so the import fires even
    for the codegeneration scenario which never uses them. The vendored rev (28fef95) is internally
    inconsistent here: its own pyproject pins `anthropic>=0.42.0`, a version that dropped these
    constants when the legacy text-completions API was retired.

    The values are the documented legacy turn separators. Nothing on the codegeneration path reads
    them, so they only need to exist. Shimming here keeps the untracked vendored tree unedited.
    """
    try:
        import anthropic
    except ImportError:
        return
    if not hasattr(anthropic, "HUMAN_PROMPT"):
        anthropic.HUMAN_PROMPT = "\n\nHuman:"
    if not hasattr(anthropic, "AI_PROMPT"):
        anthropic.AI_PROMPT = "\n\nAssistant:"


#: incremented by the extract_code shim; reported at the end of a run.
NULL_COMPLETIONS = {"n": 0}
# Counts completions that returned TEXT but nothing extractable. Distinct from NULL_COMPLETIONS,
# which counts a null response: this is the case that LOOKS like success everywhere except the
# extraction, so without this count such a run reports as clean.
EMPTY_EXTRACTIONS = {"n": 0}


def _shim_none_completions():
    """Make a `None` completion a scored zero instead of an AttributeError crash.

    WHAT HAPPENS WITHOUT THIS. `oai_runner._run_single` returns
    `[c.message.content for c in response.choices]`, and `mlx_lm.server` returns `content = None`
    when a reasoning model spends its entire budget inside the think block and emits no answer
    segment. `extraction_utils.extract_code` then does `model_output.split("\\n")` and the whole run
    dies AFTER every generation has been paid for.

    This is the same failure shape the NIAH reasoning-truncation guard exists for: the model was
    still thinking when the cap arrived, so the scored answer measures the budget rather than the
    model. The response is the same too: do not silently treat it as a wrong answer, COUNT it, and
    report it beside the score so a low pass@1 caused by a small budget is visible rather than
    mistaken for incapability. `lcb_guards.summarise_run` reports the same quantity as
    `n_inconclusive`.
    """
    from lcb_runner.utils import extraction_utils

    original = extraction_utils.extract_code

    def safe_extract_code(model_output, lmstyle):
        if model_output is None:
            NULL_COMPLETIONS["n"] += 1
            return ""
        code = original(model_output, lmstyle)
        # A None completion is caught above, but a completion that returns long prose and
        # extracts to "" would otherwise increment nothing, and the run would look clean. These
        # are budget truncations: the model spends the whole budget on analysis and stops
        # mid-sentence, with no <think> tag for the inconclusive guard to key on.
        if str(model_output).strip() and not (code or "").strip():
            EMPTY_EXTRACTIONS["n"] += 1
        return code

    extraction_utils.extract_code = safe_extract_code
    # scenario_router imported the symbol by value, so rebind there too or the patch is a no-op
    from lcb_runner.runner import scenario_router
    if hasattr(scenario_router, "extract_code"):
        scenario_router.extract_code = safe_extract_code


#: set by _shim_request_metadata; every completion appends one JSON line here.
METADATA_PATH = {"path": None}


def _shim_request_metadata(out_path):
    """Log per-request metadata for every generation, as JSONL.

    THIS IS THE PRECONDITION FOR A REPAIRABLE RUN. LiveCodeBench's saved records carry only
    `code_list`, `output_list`, `question_id` and problem metadata: **no finish_reason and no token
    counts**. So when a run turns out to be budget-truncated, there is no way to tell a completion
    that STOPPED ON ITS OWN from one that HIT THE CAP, which is exactly the distinction any repair
    depends on: a cap-hit completion with partial code would otherwise be kept as finished.

    `lcb_guards.is_truncated` has always accepted `finish_reason` and token counts and has never
    been given real data. This closes that gap, making `n_truncated` a MEASURED quantity rather than
    an inference from output length.

    Also records one line per attempt (including exceptions), because `OpenAIRunner` catches an
    exception such as an OOM, sleeps 30 s and retries, so a dead generation can look like slow
    progress. A growing run of exception records is the signal that distinguishes the two.
    """
    import json as _json
    import time as _time
    from lcb_runner.runner import oai_runner

    METADATA_PATH["path"] = out_path
    original_run_single = oai_runner.OpenAIRunner._run_single

    def logged_run_single(self, prompt, n=10):
        t0 = _time.time()
        retries_before = n
        rec = {"n_budget": n, "finish_reason": None, "prompt_tokens": None,
               "completion_tokens": None, "raw_len": 0, "exception": None}
        try:
            # Call the raw API here rather than delegating, so the response object is visible.
            response = oai_runner.OpenAIRunner.client.chat.completions.create(
                messages=prompt, **self.client_kwargs)
            ch = response.choices[0] if response.choices else None
            rec["finish_reason"] = getattr(ch, "finish_reason", None) if ch else None
            usage = getattr(response, "usage", None)
            if usage is not None:
                rec["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                rec["completion_tokens"] = getattr(usage, "completion_tokens", None)
            outs = [c.message.content for c in response.choices]
            rec["raw_len"] = max((len(o or "") for o in outs), default=0)
            rec["wall_ms"] = int((_time.time() - t0) * 1000)
            _append_metadata(rec)
            return outs
        except Exception as e:
            rec["exception"] = repr(e)[:300]
            rec["wall_ms"] = int((_time.time() - t0) * 1000)
            _append_metadata(rec)
            # Fall back to the original, which owns the retry/sleep policy. Its retries are visible
            # as additional records with the same n_budget decrementing.
            return original_run_single(self, prompt, n=retries_before)

    oai_runner.OpenAIRunner._run_single = logged_run_single


def _append_metadata(rec):
    import json as _json
    p = METADATA_PATH.get("path")
    if not p:
        return
    try:
        with open(p, "a") as fh:
            fh.write(_json.dumps(rec) + "\n")
    except Exception:
        pass          # metadata must never take the run down


def register_local_model(model_name: str, repr_name: str, release_date):
    """Insert a local model into LiveCodeBench's registry as an OpenAI-chat target.

    `main.py:21` does `LanguageModelStore[args.model]`, so registration is the whole requirement.
    `LMStyle.OpenAIChat` selects `OpenAIRunner`, which is the only runner that speaks to an
    OpenAI-compatible endpoint without a vendor SDK.
    """
    from lcb_runner.lm_styles import LanguageModel, LanguageModelStore, LMStyle

    LanguageModelStore[model_name] = LanguageModel(
        model_name=model_name,
        model_repr=repr_name,
        model_style=LMStyle.OpenAIChat,
        release_date=release_date,
        link=None,
    )
    return LanguageModelStore[model_name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--served-model", required=True,
                    help="the id mlx_lm.server reports from /v1/models, which is the full "
                         "filesystem path of the model directory")
    ap.add_argument("--repr", dest="repr_name", required=True,
                    help="short label used for output filenames")
    ap.add_argument("--base-url", default="http://127.0.0.1:8090/v1",
                    help="a dedicated port (not 8082, the oMLX client's default), so this run "
                         "never shares a server with other local work")
    ap.add_argument("--release-version", default="release_v6")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--n", type=int, default=1, help="samples per problem; 1 gives pass@1")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="generous on purpose: a budget too small for a reasoning model truncates "
                         "the program, and a truncated program does not compile, so it would be "
                         "scored as a capability failure.")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, matching every other cell")
    ap.add_argument("--num-process-evaluate", type=int, default=4)
    ap.add_argument("--openai-timeout", type=int, default=1800,
                    help="LiveCodeBench defaults this to 90 s (parser.py:95), which is far too "
                         "short for a local reasoning model: a hard problem at a 3000-token budget "
                         "routinely exceeds it. On timeout OpenAIRunner sleeps 30 s and retries "
                         "(oai_runner.py:70-76), so an under-set timeout does not fail loudly, it "
                         "looks like a hang while the server logs BrokenPipeError from the "
                         "abandoned connection. Observed exactly that on the first smoke run.")
    # NOTE: LiveCodeBench has NO --output_dir argument. It derives its own path from
    # `get_output_path(model.model_repr, args)` (`main.py:27`) and writes under the repo's
    # `output/<model_repr>/`. An earlier version of this wrapper forwarded a --output_dir and the
    # run died with "unrecognized arguments" AFTER the server was up and the fans were pinned.
    # Results are collected from LCB's own path by the caller instead.
    ap.add_argument("--limit", type=int, default=None,
                    help="NOT SUPPORTED: LiveCodeBench has no problem-count cap (its own --debug "
                         "mode runs the first 15 problems). Passing this is an error, so a smoke "
                         "run never silently runs the full set; narrow with --start-date/--end-date.")
    ap.add_argument("--metadata-out", default=None,
                    help="JSONL path for per-request metadata (finish_reason, token counts, "
                         "exceptions, wall_ms). A run without this is NOT repairable if it turns "
                         "out truncated.")
    ap.add_argument("--continue-existing", action="store_true",
                    help="reuse saved generations and regenerate only the missing ones. LCB keeps "
                         "every instance whose output_list is non-empty, so BLANK the records you "
                         "want redone before using this. Sound at temperature 0, where a completion "
                         "that stopped on its own is identical at a larger budget.")
    a = ap.parse_args()
    if a.limit is not None:
        ap.error("--limit is not supported (LiveCodeBench has no problem-count cap); "
                 "use --start-date/--end-date to narrow the problem set")

    # MUST precede any lcb_runner.runner import: OpenAIRunner builds its client at import time.
    os.environ["OPENAI_BASE_URL"] = a.base_url
    os.environ.setdefault("OPENAI_KEY", "not-used")
    os.environ.setdefault("OPENAI_API_KEY", "not-used")

    sys.path.insert(0, str(LCB_DIR))
    os.chdir(LCB_DIR)
    _shim_anthropic_legacy_constants()

    from datetime import datetime
    register_local_model(a.served_model, a.repr_name, datetime(2026, 1, 1))

    argv = [
        "lcb_runner.runner.main",
        "--model", a.served_model,
        "--scenario", "codegeneration",
        "--evaluate",
        "--release_version", a.release_version,
        "--n", str(a.n),
        "--temperature", str(a.temperature),
        "--max_tokens", str(a.max_tokens),
        "--num_process_evaluate", str(a.num_process_evaluate),
        "--openai_timeout", str(a.openai_timeout),
    ]
    if a.continue_existing:
        argv += ["--continue_existing"]
    if a.start_date:
        argv += ["--start_date", a.start_date]
    if a.end_date:
        argv += ["--end_date", a.end_date]

    sys.argv = argv
    print("[lcb] base_url=%s" % a.base_url, flush=True)
    print("[lcb] argv=%s" % " ".join(argv[1:]), flush=True)

    _shim_none_completions()
    if a.metadata_out:
        _shim_request_metadata(a.metadata_out)
        print('[lcb] per-request metadata -> %s' % a.metadata_out, flush=True)

    from lcb_runner.runner.main import main as lcb_main
    lcb_main()
    if EMPTY_EXTRACTIONS["n"]:
        print("[lcb] WARNING: %d completion(s) produced TEXT but no extractable code. These are "
              "budget-truncation artifacts, not wrong answers, and they carry no <think> tag for "
              "the inconclusive guard to catch. Report this count beside pass@1, and quote "
              "pass@1 over the completions that produced code as well."
              % EMPTY_EXTRACTIONS["n"], flush=True)
    if NULL_COMPLETIONS["n"]:
        print("[lcb] WARNING: %d completion(s) had NO answer segment (content=None). These are "
              "generation-budget artifacts, not wrong answers: the model was still inside its "
              "reasoning block when max_tokens arrived. Raise --max-tokens and re-run before "
              "reporting this pass@1 as a capability result."
              % NULL_COMPLETIONS["n"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
