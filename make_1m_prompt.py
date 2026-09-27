#!/usr/bin/env python3
"""Build a ~1,000,000-token NIAH prompt from the needles of a 256K raw prompt.

WHY A NEW FILE RATHER THAN SCALING THE OLD ONE. Reaching 1M means more filler, and the needles
must be placed by DEPTH (fraction of the prompt), not by absolute position: keeping absolute
positions would put every needle in the first quarter and make the 1M test easier, not harder.
This script uses its own evenly spaced depths (see the note below), so a 1M prompt built here
matches a 256K prompt's layout only if that prompt also used evenly spaced depths.

TARGET. Qwen2.5-7B-Instruct-1M's config.json sets max_position_embeddings = 1,010,000, NOT
1,048,576. So a "1M" test here means one million tokens, and the honest ceiling is 1,010,000. This
script targets a default of 995,000 raw tokens, which leaves room for a chat template wrapper
without exceeding the model's trained context.

The needle wording and codes are copied from the source prompt, so the scorer and needle regex
work unchanged. The filler sentence, the trailing question (fixed to the first 8 cities of
niah_haystack.NEEDLE_CITIES) and the needle depths are this script's own: the 8 needles go at
evenly spaced depths (1/9 to 8/9 of the filler). A source prompt with other depths (one built by
niah_haystack.py uses 0.03 to 0.90) therefore does not keep its depths at 1M.
"""
import argparse
import os

FILLER = ("The quarterly logistics review covered warehouse throughput, fleet routing, "
          "cold-chain compliance, and vendor lead times across the regional network. ")

# Needle wording is read from the 256K prompt at runtime, never hardcoded here, so the
# two tests cannot drift apart.

QUESTION = ("\n\nExtract the secret access code for each of these cities from the text above: "
            "Reykjavik, Ouagadougou, Valparaiso, Nakhodka, Timbuktu, Kirkwall, Ushuaia, Yakutsk. "
            "List each city and its code.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="used only for its tokenizer")
    ap.add_argument("--target-tokens", type=int, default=995_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-prompt", default=os.environ.get("NIAH_256K_PROMPT"),
                    help="the 256K prompt, read only to copy its exact needle wording "
                         "(default: $NIAH_256K_PROMPT)")
    a = ap.parse_args()
    if not a.source_prompt:
        ap.error("--source-prompt is required (or set NIAH_256K_PROMPT)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)

    # Copy the needle sentences verbatim from the existing prompt so wording cannot drift.
    import re
    src = open(a.source_prompt, encoding="utf-8").read()
    found = re.findall(r"IMPORTANT RECORD: the secret access code for ([A-Z][A-Za-z'\- ]+) is (\d{8,})\.", src)
    if len(found) != 8:
        raise SystemExit(f"expected 8 needles in the source prompt, found {len(found)}")
    needle_sentences = [f"IMPORTANT RECORD: the secret access code for {c} is {code}. "
                        for c, code in found]

    filler_tok = len(tok.encode(FILLER))
    q_tok = len(tok.encode(QUESTION))
    needle_tok = sum(len(tok.encode(s)) for s in needle_sentences)
    budget = a.target_tokens - q_tok - needle_tok
    n_filler = max(1, budget // filler_tok)
    print(f"  filler {filler_tok} tok each, question {q_tok}, needles {needle_tok} total")
    print(f"  using {n_filler} filler sentences")

    # Place the 8 needles at evenly spaced DEPTHS (1/9 to 8/9 of the filler).
    positions = {int(n_filler * (i + 1) / (len(needle_sentences) + 1)): s
                 for i, s in enumerate(needle_sentences)}

    parts = []
    for i in range(n_filler):
        parts.append(FILLER)
        if i in positions:
            parts.append(positions[i])
    parts.append(QUESTION)
    text = "".join(parts)

    n = len(tok.encode(text))
    print(f"  built {len(text)} bytes, {n} tokens (target {a.target_tokens})")
    if n > 1_010_000:
        raise SystemExit(f"ABORT: {n} tokens exceeds the model's trained 1,010,000 context")
    open(a.out, "w", encoding="utf-8").write(text)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
