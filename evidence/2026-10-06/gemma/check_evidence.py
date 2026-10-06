"""Check the saved token lists and CLI answer hashes without a model runtime."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main():
    evidence = json.loads((ROOT / "fresh-evidence.json").read_text(encoding="utf-8"))
    prompt = json.loads((ROOT / "prompt.ids.json").read_text(encoding="utf-8"))
    total = 0
    for group, baseline_name in [("api", "stock-serial-1"),
                                  ("diagnostics", "stock-serial-1"),
                                  ("boundary", "boundary-serial")]:
        baseline = json.loads((ROOT / "fresh-runs" / baseline_name / "tokens.json").read_text(encoding="utf-8"))
        for record in evidence[group]:
            case = record["case"]
            tokens = json.loads((ROOT / "fresh-runs" / case / "tokens.json").read_text(encoding="utf-8"))
            receipt = json.loads((ROOT / "fresh-runs" / case / "receipt.json").read_text(encoding="utf-8"))
            assert all(type(t) is int for t in tokens), case
            assert len(tokens) == record["output_tokens"] == receipt["output_tokens"], case
            assert digest(tokens) == record["token_sha256"] == receipt["token_sha256"], case
            assert (tokens == baseline) == record["exact_tokens"], case
            assert first_difference(baseline, tokens) == record["first_difference"], case
            if "prompt_sha256" in receipt:
                assert digest(prompt) == receipt["prompt_sha256"], case
            assert len(prompt) == receipt["input_tokens"], case
            total += 1
    baseline = (ROOT / "fresh-runs/cli-serial-1/answer.txt").read_text(encoding="utf-8")
    for record in evidence["cli"]:
        case = record["case"]
        answer = (ROOT / "fresh-runs" / case / "answer.txt").read_text(encoding="utf-8")
        assert hashlib.sha256(answer.encode("utf-8")).hexdigest() == record["answer_sha256"], case
        assert (answer == baseline) == record["exact_answer"], case
        assert first_difference(baseline, answer) == record["first_answer_character_difference"], case
        total += 1
    print(json.dumps({"saved_cases_checked": total, "all_recorded_comparisons_match": True}))


if __name__ == "__main__":
    main()
