"""
DeepEval LLM-as-judge tests for benchmark models.

Evaluates model responses using standardized judge metrics:
- Answer relevancy
- Hallucination
- Task completion (G-Eval)

Requires env vars: EVAL_SERVED_MODEL, EVAL_BASE_URL, EVAL_API_KEY
Set by run_eval.py or manually:
    EVAL_SERVED_MODEL=qwen3.5-122b-a10b EVAL_BASE_URL=http://localhost:1234/v1 EVAL_API_KEY=lm-studio
"""

import json
import os

import pytest

from openai import OpenAI

# test prompts covering reasoning, coding, instruction following
TEST_CASES = [
    {
        "id": "de_reasoning_1",
        "input": "A farmer has 17 sheep. All but 9 die. How many sheep are alive?",
        "expected_output": "9",
        "context": "Classic reasoning trick question — 'all but 9' means 9 survive.",
    },
    {
        "id": "de_coding_1",
        "input": "Write a Python function that checks if a string is a palindrome. Return code only.",
        "expected_output": "A correct Python function that checks palindromes using string reversal or two-pointer approach.",
        "context": "Basic coding task. Function should handle edge cases like empty string.",
    },
    {
        "id": "de_instruction_1",
        "input": "List exactly 3 benefits of unit testing. Use numbered list. No more than 15 words per item.",
        "expected_output": "A numbered list of exactly 3 items about unit testing benefits, each under 15 words.",
        "context": "Tests instruction following: exact count, format, and word limit constraints.",
    },
    {
        "id": "de_factual_1",
        "input": "What is the capital of Australia? Answer in one word.",
        "expected_output": "Canberra",
        "context": "Common trick question — many incorrectly say Sydney or Melbourne.",
    },
    {
        "id": "de_analysis_1",
        "input": "A server returns 502 errors after deploying v2.1. The load balancer health check passes. Upstream service logs show connection timeouts. CPU is at 30%. What is the most likely cause?",
        "expected_output": "The upstream service is experiencing connection timeouts, likely due to the new deployment causing slow startup or resource contention, not CPU exhaustion.",
        "context": "DevOps/SRE reasoning task requiring elimination of red herrings.",
    },
]


def get_model_response(prompt: str) -> str:
    """Get response from model via LM Studio API."""
    base_url = os.environ.get("EVAL_BASE_URL", "http://localhost:1234/v1")
    api_key = os.environ.get("EVAL_API_KEY", "lm-studio")
    served_model = os.environ.get("EVAL_SERVED_MODEL", "")

    if not served_model:
        pytest.skip("EVAL_SERVED_MODEL not set")

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=served_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _try_import_deepeval():
    """Import deepeval metrics, skip tests if not installed."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric, GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        return AnswerRelevancyMetric, GEval, LLMTestCase, LLMTestCaseParams
    except ImportError:
        pytest.skip("deepeval not installed")


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
def test_answer_relevancy(case):
    """Test that model responses are relevant to the input question."""
    AnswerRelevancyMetric, _, LLMTestCase, _ = _try_import_deepeval()

    actual_output = get_model_response(case["input"])

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=actual_output,
        expected_output=case["expected_output"],
        context=[case["context"]],
    )

    metric = AnswerRelevancyMetric(threshold=0.5)
    metric.measure(test_case)

    result = {
        "case_id": case["id"],
        "metric": "answer_relevancy",
        "score": metric.score,
        "passed": metric.is_successful(),
        "reason": metric.reason,
        "actual_output": actual_output[:200],
    }
    print(f"\n  {json.dumps(result, indent=2)}")
    assert metric.is_successful(), f"Answer relevancy too low: {metric.score} (reason: {metric.reason})"


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
def test_task_completion(case):
    """Test task completion quality using G-Eval."""
    _, GEval, LLMTestCase, LLMTestCaseParams = _try_import_deepeval()

    actual_output = get_model_response(case["input"])

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=actual_output,
        expected_output=case["expected_output"],
    )

    metric = GEval(
        name="task_completion",
        criteria="Evaluate whether the response correctly and completely addresses the task. Consider accuracy, completeness, and adherence to any format constraints.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )
    metric.measure(test_case)

    result = {
        "case_id": case["id"],
        "metric": "task_completion",
        "score": metric.score,
        "passed": metric.is_successful(),
        "reason": metric.reason,
        "actual_output": actual_output[:200],
    }
    print(f"\n  {json.dumps(result, indent=2)}")
    assert metric.is_successful(), f"Task completion too low: {metric.score} (reason: {metric.reason})"
