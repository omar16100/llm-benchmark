"""
Claude-as-judge: blind pairwise scoring of benchmark transcripts.
Reads transcripts.jsonl, scores judge-only cases via Claude API,
outputs judged_results.csv.
"""

import csv
import json
import logging
import os
import random
from pathlib import Path

import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("judge.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
TRANSCRIPTS_PATH = RESULTS_DIR / "transcripts.jsonl"
JUDGED_CSV_PATH = RESULTS_DIR / "judged_results.csv"

JUDGE_MODEL = "claude-sonnet-4-6"
TIEBREAK_MODEL = "claude-opus-4-6"

JUDGE_SYSTEM = """You are a strict, blind evaluator comparing two AI model outputs.
You do not know which model produced which output. Evaluate purely on quality.

Scoring rubric:
- correctness: 0-5 (factual accuracy, logical soundness)
- instruction_adherence: 0-3 (followed all constraints and format requirements)
- completeness: 0-1 (addressed all parts of the prompt)
- style: 0-1 (clarity, readability, usefulness)

Total max score: 10 per candidate.

You MUST respond with ONLY valid JSON in this exact schema:
{"winner":"A"|"B"|"tie","a_score":0,"b_score":0,"confidence":1-5,"rationale":"...","violations":{"A":[],"B":[]}}
"""

CSV_COLUMNS = [
    "prompt_id", "category", "model_a", "model_b",
    "a_score", "b_score", "winner", "confidence",
    "rationale", "a_violations", "b_violations",
    "order", "judge_model",
]


def load_transcripts():
    """Load transcripts and group by prompt_id."""
    transcripts = {}
    with open(TRANSCRIPTS_PATH) as f:
        for line in f:
            t = json.loads(line)
            pid = t["prompt_id"]
            model = t["model_label"]
            if pid not in transcripts:
                transcripts[pid] = {}
            # keep best repeat (highest score or first if no score)
            if model not in transcripts[pid]:
                transcripts[pid][model] = t
            else:
                existing_score = transcripts[pid][model].get("score_raw")
                new_score = t.get("score_raw")
                if new_score is not None and (existing_score is None or new_score > existing_score):
                    transcripts[pid][model] = t
    return transcripts


def needs_judge(transcript):
    """Check if this transcript needs Claude judging."""
    return transcript.get("score_type") in ("needs_judge", None)


def build_judge_prompt(case_a, case_b, prompt_text):
    """Build the judge prompt with randomized A/B order."""
    return f"""## Task
{prompt_text}

## Candidate A
{case_a['response']}

## Candidate B
{case_b['response']}

Evaluate both candidates against the task. Score each using the rubric.
Respond with ONLY the JSON object."""


def call_judge(client, prompt, model=JUDGE_MODEL):
    """Call Claude via CLI (non-interactive) to judge a pair of outputs.
    Uses `claude -p` which inherits the session's authentication."""
    import subprocess
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model,
             "--system-prompt", JUDGE_SYSTEM,
             "--output-format", "text",
             "--max-turns", "1",
             prompt],
            capture_output=True, text=True, timeout=120,
        )
        text = result.stdout.strip()
        if not text:
            log.error("judge CLI returned empty output, stderr: %s", result.stderr[:200])
            return None
        # parse JSON from response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        # find JSON object in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("judge JSON parse error: %s | response: %s", e, text[:200])
        return None
    except subprocess.TimeoutExpired:
        log.error("judge CLI timeout (120s)")
        return None
    except Exception as e:
        log.error("judge CLI error: %s", e)
        return None


def run_judge():
    """Main judging loop."""
    client = None  # using claude CLI, no SDK client needed
    transcripts = load_transcripts()
    models = list({t["model_label"] for pid in transcripts for t in transcripts[pid].values()})

    if len(models) < 2:
        log.error("need at least 2 models in transcripts, got: %s", models)
        return

    model_a_label, model_b_label = sorted(models)[:2]
    log.info("judging: %s vs %s", model_a_label, model_b_label)

    csv_file = open(JUDGED_CSV_PATH, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    judged = 0
    skipped = 0

    for pid, model_transcripts in sorted(transcripts.items()):
        if model_a_label not in model_transcripts or model_b_label not in model_transcripts:
            log.warning("skipping %s: missing model output", pid)
            skipped += 1
            continue

        t_a = model_transcripts[model_a_label]
        t_b = model_transcripts[model_b_label]

        # only judge cases that need it, or all cases for comprehensive scoring
        has_programmatic_a = t_a.get("score_raw") is not None
        has_programmatic_b = t_b.get("score_raw") is not None

        if has_programmatic_a and has_programmatic_b:
            # both have programmatic scores, still do pairwise for comparison
            pass

        # randomize order to reduce position bias
        swap = random.random() > 0.5
        if swap:
            first, second = t_b, t_a
            first_label, second_label = model_b_label, model_a_label
            order = "swapped"
        else:
            first, second = t_a, t_b
            first_label, second_label = model_a_label, model_b_label
            order = "original"

        prompt = build_judge_prompt(first, second, t_a["prompt"])
        log.info("judging %s (%s order)", pid, order)

        result = call_judge(client, prompt)
        if not result:
            log.warning("judge failed for %s, trying tiebreak model", pid)
            result = call_judge(client, prompt, model=TIEBREAK_MODEL)

        if not result:
            log.error("both judges failed for %s", pid)
            skipped += 1
            continue

        # unswap scores if order was swapped
        if swap:
            actual_winner = {"A": "B", "B": "A", "tie": "tie"}.get(result.get("winner", "tie"), "tie")
            actual_a_score = result.get("b_score", 0)
            actual_b_score = result.get("a_score", 0)
            actual_a_violations = result.get("violations", {}).get("B", [])
            actual_b_violations = result.get("violations", {}).get("A", [])
        else:
            actual_winner = result.get("winner", "tie")
            actual_a_score = result.get("a_score", 0)
            actual_b_score = result.get("b_score", 0)
            actual_a_violations = result.get("violations", {}).get("A", [])
            actual_b_violations = result.get("violations", {}).get("B", [])

        row = {
            "prompt_id": pid,
            "category": t_a.get("category"),
            "model_a": model_a_label,
            "model_b": model_b_label,
            "a_score": actual_a_score,
            "b_score": actual_b_score,
            "winner": actual_winner,
            "confidence": result.get("confidence", 0),
            "rationale": result.get("rationale", ""),
            "a_violations": json.dumps(actual_a_violations),
            "b_violations": json.dumps(actual_b_violations),
            "order": order,
            "judge_model": JUDGE_MODEL,
        }
        writer.writerow(row)
        csv_file.flush()
        judged += 1

    csv_file.close()
    log.info("judging complete. judged=%d skipped=%d output=%s", judged, skipped, JUDGED_CSV_PATH)

    # print summary
    print_summary()


def print_summary():
    """Print win/tie/loss summary from judged results."""
    if not JUDGED_CSV_PATH.exists():
        return

    wins = {"A": 0, "B": 0, "tie": 0}
    scores = {"A": [], "B": []}
    categories = {}

    with open(JUDGED_CSV_PATH) as f:
        reader = csv.DictReader(f)
        model_a_name = None
        model_b_name = None
        for row in reader:
            if not model_a_name:
                model_a_name = row["model_a"]
                model_b_name = row["model_b"]

            winner = row["winner"]
            wins[winner] = wins.get(winner, 0) + 1
            scores["A"].append(float(row["a_score"]))
            scores["B"].append(float(row["b_score"]))

            cat = row["category"]
            if cat not in categories:
                categories[cat] = {"A": 0, "B": 0, "tie": 0}
            categories[cat][winner] = categories[cat].get(winner, 0) + 1

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{model_a_name} vs {model_b_name}")
    print(f"\nOverall: {model_a_name} wins={wins['A']}  {model_b_name} wins={wins['B']}  ties={wins['tie']}")

    if scores["A"]:
        avg_a = sum(scores["A"]) / len(scores["A"])
        avg_b = sum(scores["B"]) / len(scores["B"])
        print(f"Avg scores: {model_a_name}={avg_a:.1f}  {model_b_name}={avg_b:.1f}")

    print("\nBy category:")
    for cat, counts in sorted(categories.items()):
        print(f"  {cat}: {model_a_name}={counts['A']}  {model_b_name}={counts['B']}  tie={counts['tie']}")
    print("=" * 60)


if __name__ == "__main__":
    run_judge()
