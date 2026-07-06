"""Fast unit tests for the NIAH context builder (no model load).

Run: .venv/bin/python -m pytest tests/test_niah_haystack.py -q
Needs transformers + tiktoken + blobfile and the model dir's tokenizer.
Uses the REAL Kimi tiktoken tokenizer so token-count guarantees are genuine.
"""
import logging
import os
import pytest

# The real Kimi tiktoken tokenizer needs these; skip cleanly (rather than error)
# when running in an env that does not have the NIAH stack or the local weights,
# so `pytest tests/` in the base benchmark venv stays green.
pytest.importorskip("tiktoken")
pytest.importorskip("blobfile")
from transformers import AutoTokenizer
import niah_haystack as cb

TOKDIR = "/Users/macmini/models/Kimi-Linear-48B-A3B-Instruct-8bit"
if not os.path.isdir(TOKDIR):
    pytest.skip("Kimi model dir not present; NIAH tokenizer tests need it",
                allow_module_level=True)
logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)


@pytest.mark.parametrize("target", [2000, 8000, 32000])
def test_context_length_within_tolerance(tok, target):
    h = cb.build_haystack(tok, target, seed=7)
    # context body (pre-question) must be within 1% of requested target
    assert abs(h.n_context_tokens_est - target) <= max(128, target * 0.01), \
        f"target={target} got={h.n_context_tokens_est}"


def test_all_needles_present_in_prompt(tok):
    h = cb.build_haystack(tok, 8000, seed=11)
    for city, code in h.needles.items():
        assert f"secret access code for {city} is {code}" in h.prompt_text
    # question lists every city
    for city in h.needles:
        assert city in h.prompt_text.split("---")[-1]


def test_needles_are_depth_ordered(tok):
    h = cb.build_haystack(tok, 16000, seed=3)
    cities = list(h.needles)
    positions = [h.prompt_text.index(c) for c in cities]
    assert positions == sorted(positions), "needles not in ascending depth order"
    # first needle should be near the start, last well before the question
    assert positions[0] < len(h.prompt_text) * 0.2


def test_n_needles_override(tok):
    h = cb.build_haystack(tok, 4000, seed=5, n_needles=3)
    assert len(h.needles) == 3


def test_score_perfect(tok):
    h = cb.build_haystack(tok, 4000, seed=9)
    answer = "\n".join(f"{c}: {code}" for c, code in h.needles.items())
    s = cb.score_answer(answer, h.needles)
    assert s["retrieval_rate"] == 1.0 and s["association_rate"] == 1.0


def test_score_partial_and_wrong(tok):
    h = cb.build_haystack(tok, 4000, seed=9, n_needles=4)
    cities = list(h.needles)
    lines = []
    lines.append(f"{cities[0]}: {h.needles[cities[0]]}")      # correct
    lines.append(f"{cities[1]}: 00000000")                     # wrong code
    lines.append(f"{cities[2]}: {h.needles[cities[3]]}")       # swapped -> present but misassociated
    # cities[3] omitted entirely
    s = cb.score_answer("\n".join(lines), h.needles)
    assert s["association_correct"] == 1, s
    # code for cities[3] is present (on cities[2]'s line) but not associated
    assert s["per_city"][cities[3]]["present"] is True
    assert s["per_city"][cities[3]]["associated"] is False


def test_chat_template_exists(tok):
    assert getattr(tok, "chat_template", None), "tokenizer has no chat_template"
    msgs = [{"role": "user", "content": "hi"}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    assert isinstance(ids, list) and len(ids) > 0
