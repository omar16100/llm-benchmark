"""Token-accurate multi-needle haystack builder for NIAH.

No model dependency here (only the tokenizer) so it is unit-testable fast.
Detailed logging + exact token-count control are the whole point: a "1M pass"
that is secretly a truncated 4K pass is the failure mode we must rule out.
"""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field

log = logging.getLogger("niah.ctx")

# Deterministic filler vocabulary -> varied, natural-ish English (not pure
# repetition, which can make NIAH degenerate or cause decode loops).
_ADJ = ["quiet", "ancient", "restless", "amber", "hollow", "distant", "brittle",
        "luminous", "weathered", "sombre", "gilded", "narrow", "vast", "tepid"]
_NOUN = ["harbour", "orchard", "lantern", "corridor", "meadow", "furnace",
         "archive", "glacier", "trellis", "cistern", "belfry", "estuary"]
_VERB = ["drifted past", "settled over", "coiled around", "receded from",
         "pressed against", "wandered through", "loomed beyond", "faded near"]
_PLACE = ["the old mill", "a copper sky", "the tram depot", "the salt flats",
          "a shuttered market", "the river bend", "the north pier", "a dry canal"]

# Needle cities: distinctive, single-token-ish, unlikely to collide with filler.
NEEDLE_CITIES = ["Reykjavik", "Ouagadougou", "Valparaiso", "Nakhodka",
                 "Timbuktu", "Kirkwall", "Ushuaia", "Yakutsk",
                 "Paramaribo", "Trondheim", "Zanzibar", "Murmansk"]

DEFAULT_DEPTHS = [0.03, 0.13, 0.26, 0.39, 0.51, 0.64, 0.77, 0.90]


@dataclass
class Haystack:
    prompt_text: str                 # full user-turn text (context + question)
    needles: dict                    # city -> code (ground truth)
    depths: list                     # requested depth fractions
    n_context_tokens_est: int        # token estimate of context body (pre-template)
    cities: list = field(default_factory=list)


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _filler_sentence(r: random.Random) -> str:
    return (f"The {r.choice(_ADJ)} {r.choice(_NOUN)} {r.choice(_VERB)} "
            f"{r.choice(_PLACE)} as the {r.choice(_ADJ)} {r.choice(_NOUN)} "
            f"held its ground.")


def _needle_sentence(city: str, code: str) -> str:
    return f"IMPORTANT RECORD: the secret access code for {city} is {code}."


def _count(tok, text: str) -> int:
    return len(tok.encode(text))


def build_haystack(tokenizer, target_context_tokens: int, *,
                   depths=None, seed: int = 1234,
                   n_needles=None) -> Haystack:
    """Build a haystack whose *context body* is ~target_context_tokens tokens,
    with needles inserted at the given depth fractions. The final prompt adds a
    retrieval question. Token count is measured with the real tokenizer and
    filler is grown/trimmed to land within a small tolerance of the target.
    """
    depths = list(depths or DEFAULT_DEPTHS)
    if n_needles is not None:
        depths = depths[:n_needles]
    r = _rng(seed)
    cities = NEEDLE_CITIES[:len(depths)]
    # Codes must be unique: scoring keys retrieval/association on the code string,
    # so a collision would make two cities indistinguishable. Redraw on the rare
    # collision (for a seed with no collision the draw order is unchanged).
    needles, _seen = {}, set()
    for c in cities:
        code = r.randint(10_000_000, 99_999_999)
        while code in _seen:
            code = r.randint(10_000_000, 99_999_999)
        _seen.add(code)
        needles[c] = f"{code}"

    # Cost of one needle sentence (so we can budget filler precisely).
    needle_texts = {c: _needle_sentence(c, needles[c]) for c in cities}
    needle_tok = sum(_count(tokenizer, t + " ") for t in needle_texts.values())
    filler_budget = max(0, target_context_tokens - needle_tok)

    # Grow filler in blocks, measuring real tokens, until within tolerance.
    tol = max(64, int(target_context_tokens * 0.002))
    sentences = []
    cur = 0
    # Estimate per-sentence tokens once to size the loop cheaply, then verify.
    sample = _filler_sentence(r)
    per = max(1, _count(tokenizer, sample + " "))
    while cur < filler_budget - tol:
        need = filler_budget - cur
        block_n = max(1, min(4000, need // per))
        block = " ".join(_filler_sentence(r) for _ in range(block_n))
        sentences.append(block)
        cur += _count(tokenizer, block + " ")
    # Trim last block token-wise if we overshot.
    body = " ".join(sentences)
    body_ids = tokenizer.encode(body)
    if len(body_ids) > filler_budget:
        body_ids = body_ids[:filler_budget]
        body = tokenizer.decode(body_ids)

    # Split filler into len(depths)+1 segments and interleave needles by depth.
    words = body.split(" ")
    n = len(words)
    segments = []
    prev = 0
    for d in depths:
        idx = min(n, max(prev, int(d * n)))
        segments.append(" ".join(words[prev:idx]))
        prev = idx
    segments.append(" ".join(words[prev:]))

    parts = []
    for i, c in enumerate(cities):
        parts.append(segments[i])
        parts.append(needle_texts[c])
    parts.append(segments[-1])
    context_body = " ".join(p for p in parts if p)

    ctx_tokens = _count(tokenizer, context_body)
    question = (
        "\n\n---\nYou just read a long document. Somewhere inside it, several "
        "sentences each stated a 'secret access code' for a specific city. "
        "For EACH city listed below, report its secret access code exactly as "
        "written in the document. Do not guess; use only codes from the "
        "document.\nCities: " + ", ".join(cities) + "\n"
        "Answer with one line per city in the exact format 'CITY: CODE'."
    )
    prompt_text = context_body + question
    log.info("built haystack: target_ctx=%d actual_ctx_tokens=%d needles=%d "
             "depths=%s", target_context_tokens, ctx_tokens, len(cities), depths)
    return Haystack(prompt_text=prompt_text, needles=needles, depths=depths,
                    n_context_tokens_est=ctx_tokens, cities=cities)


def score_answer(generated: str, needles: dict) -> dict:
    """Return retrieval (code present anywhere) and association (right code for
    right city) scores, plus per-city detail. Codes are unique 8-digit strings.
    """
    import re
    gen = generated
    per_city = {}
    assoc_hits = 0
    present_hits = 0
    for city, code in needles.items():
        present = code in gen
        # association: find code appearing on/after the city mention on its line
        assoc = False
        for line in gen.splitlines():
            if city.lower() in line.lower():
                nums = re.findall(r"\d{6,}", line)
                if code in nums:
                    assoc = True
                break
        if not assoc and present:
            # fallback: city and code within 40 chars of each other anywhere
            for m in re.finditer(re.escape(city), gen, re.IGNORECASE):
                window = gen[m.start():m.start() + 60]
                if code in window:
                    assoc = True
                    break
        per_city[city] = {"code": code, "present": present, "associated": assoc}
        present_hits += int(present)
        assoc_hits += int(assoc)
    n = len(needles)
    return {
        "n_needles": n,
        "retrieval_present": present_hits,
        "association_correct": assoc_hits,
        "retrieval_rate": present_hits / n if n else 0.0,
        "association_rate": assoc_hits / n if n else 0.0,
        "per_city": per_city,
    }
