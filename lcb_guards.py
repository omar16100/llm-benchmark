#!/usr/bin/env python3
"""Guards for a LiveCodeBench pass@1 run, held to the same standard as the speed rows.

A pass@1 is only meaningful if the model actually produced a finished program. Three failure modes
produce a WRONG capability verdict rather than an honest low score, and each has a guard here:

  1. TRUNCATION. A generation budget too small for a reasoning model cuts the program off. A
     truncated program does not compile, so it scores zero and reads as inability. Counted
     separately, never silently folded into the failure rate.
  2. UNTERMINATED REASONING. The output-keyed rule from `mlx_raw_niah_client.py`, NOT the older
     prompt-keyed variant in `bench_niah_mlx.py`: a model with no template at all can open <think>
     spontaneously, so the prompt is only a predictor.
  3. REFUSAL. Not answering is not the same as answering wrongly, so refusals are counted apart.

Pure functions, no imports beyond the stdlib, so the tests need no GPU, no network and no venv.
"""
import re

THINK_OPEN = re.compile(r"<think\s*>", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)

# Deliberately narrow. A model explaining that a problem is ambiguous is not refusing, so matching
# loose phrases like "I cannot" anywhere in a long chain of thought would over-count badly.
REFUSAL_PATTERNS = [
    re.compile(r"^\s*(i'm sorry|i am sorry|i cannot|i can't|as an ai)", re.IGNORECASE),
    re.compile(r"^\s*(sorry,? (?:but )?i (?:cannot|can't))", re.IGNORECASE),
]


def pass_at_1(results):
    """Fraction of problems whose FIRST sample passed.

    `results` maps problem id to a list of per-sample booleans. pass@1 with n=1 is just the mean of
    the first sample, which is what this project wants: greedy, one shot, no best-of-k inflation.
    Returns 0.0 for an empty set rather than raising, because an empty run must report zero rather
    than crash a campaign midway.
    """
    if not results:
        return 0.0
    firsts = [bool(v[0]) for v in results.values() if v]
    if not firsts:
        return 0.0
    return sum(firsts) / len(firsts)


def is_truncated(finish_reason=None, generated_tokens=None, max_tokens=None):
    """True when generation stopped because it ran out of budget, not because it finished.

    Two independent signals, because servers disagree: an explicit `finish_reason == "length"`, and
    the token count reaching the cap. Either is sufficient. `mlx_lm.server` reports the former.
    """
    if finish_reason is not None and str(finish_reason).lower() == "length":
        return True
    if generated_tokens is not None and max_tokens is not None:
        return generated_tokens >= max_tokens
    return False


def answer_inconclusive(text, hit_cap):
    """Output-keyed unterminated-reasoning guard.

    Keys on what the GENERATION did, which is observable in every case, rather than on what the
    prompt did, which is only a predictor. See the guard comment in `mlx_raw_niah_client.py`.
    """
    think_open = THINK_OPEN.search(text or "") is not None
    think_closed = THINK_CLOSE.search(text or "") is not None
    return bool(think_open and not think_closed and hit_cap)


def no_extractable_answer(raw_output, extracted):
    """True when the model produced text but nothing usable could be extracted from it.

    KEYED ON THE ARTEFACT, NOT ON A MARKER, and that distinction is why this exists.
    `answer_inconclusive` above keys on `<think>`, which is correct for the models that use it and
    blind to the ones that do not. A model that spends the whole budget on prose analysis and
    stops mid-sentence, with no tag anywhere, reports n_inconclusive 0 under that rule even when
    many completions never reached a code block.

    A completion that produced NOTHING is a different case and is not counted here: an empty
    `raw_output` means the request failed or returned null, which `_shim_none_completions` in
    lcb_local_runner.py already counts separately. This function is specifically for the case that
    looks like success from every angle except the one that matters.

    Deliberately NOT conditioned on hitting the cap. A completion that stops early without an answer
    is also unusable, and requiring the cap would reintroduce exactly the marker-dependence this
    replaces. Callers wanting the truncated subset have their own predicate for that.

    `extracted is None` means UNKNOWN, not empty, and returns False. The two must not be conflated:
    a caller that never populated the field has told us nothing, and asserting an artefact from an
    absence would flag every record. `summarise_run` reports None rather than 0 in that case, so a
    missing field cannot be misread as a clean run.
    """
    if extracted is None:
        return False
    if not (raw_output or "").strip():
        return False
    return not extracted.strip()


def is_refusal(text):
    """True when the answer segment is a refusal rather than an attempt.

    Checked against the ANSWER segment only. A refusal phrase inside a chain of thought ("I cannot
    assume the list is sorted") is reasoning, not refusal.
    """
    answer = strip_think(text or "")
    if not answer.strip():
        return False
    return any(p.search(answer) for p in REFUSAL_PATTERNS)


def strip_think(text):
    """Return the answer segment: everything after the LAST closing think tag.

    An unterminated <think> means there is no answer segment at all, so this returns "". That is the
    correct reading and is what makes `is_refusal` and emptiness checks meaningful.
    """
    if not text:
        return ""
    if THINK_OPEN.search(text) and not THINK_CLOSE.search(text):
        return ""
    parts = THINK_CLOSE.split(text)
    return parts[-1] if len(parts) > 1 else text


def filter_by_date(problems, start=None, end=None):
    """Keep problems whose contest_date falls inside [start, end], both inclusive.

    Mirrors `code_generation.py:126-133` so the harness and the benchmark agree on the window.
    `problems` is any iterable of objects exposing `contest_date` as a datetime.
    """
    out = list(problems)
    if start is not None:
        out = [p for p in out if p.contest_date >= start]
    if end is not None:
        out = [p for p in out if p.contest_date <= end]
    return out


def summarise_run(records):
    """Reduce per-problem records to the fields a matrix cell must quote.

    Each record: {"passed": bool, "text": str, "finish_reason": str|None,
                  "generated_tokens": int|None, "max_tokens": int|None,
                  "extracted": str|None}

    `pass_at_1_excluding_truncated` exists because the two numbers answer different questions. The
    headline pass@1 is what the model delivers in practice at this budget. The excluding-truncated
    figure isolates capability from budget, and a large gap between them means the budget is the
    binding constraint rather than the model.

    `n_no_answer` and `pass_at_1_excluding_no_answer` do the same job for models that leave no
    marker. `extracted` is optional so existing callers keep working, but a caller that omits it
    gets None for both, NOT 0: an unmeasured quantity is not a clean run, and reporting 0 there
    would be false reassurance. Pass it.
    """
    n = len(records)
    if n == 0:
        return {"n_problems": 0}
    truncated = sum(1 for r in records if is_truncated(
        r.get("finish_reason"), r.get("generated_tokens"), r.get("max_tokens")))
    inconclusive = sum(1 for r in records if answer_inconclusive(
        r.get("text", ""), is_truncated(r.get("finish_reason"), r.get("generated_tokens"), r.get("max_tokens"))))
    refusals = sum(1 for r in records if is_refusal(r.get("text", "")))
    passed = sum(1 for r in records if r.get("passed"))
    # None, not 0, when NO record carries `extracted`. A caller who never populated the field has
    # measured nothing, and reporting 0 would be false reassurance.
    extraction_known = any(r.get("extracted") is not None for r in records)
    no_answer = (sum(1 for r in records
                     if no_extractable_answer(r.get("text", ""), r.get("extracted")))
                 if extraction_known else None)
    usable = [r for r in records if not is_truncated(
        r.get("finish_reason"), r.get("generated_tokens"), r.get("max_tokens"))]
    answered = [r for r in records
                if not no_extractable_answer(r.get("text", ""), r.get("extracted"))]
    return {
        "n_problems": n,
        "pass_at_1": passed / n,
        "n_passed": passed,
        "n_truncated": truncated,
        "n_inconclusive": inconclusive,
        "n_no_answer": no_answer,
        "n_refusals": refusals,
        "pass_at_1_excluding_truncated": (
            sum(1 for r in usable if r.get("passed")) / len(usable) if usable else None),
        "pass_at_1_excluding_no_answer": (
            sum(1 for r in answered if r.get("passed")) / len(answered)
            if (extraction_known and answered) else None),
    }
