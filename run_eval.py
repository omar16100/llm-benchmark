"""
Unified evaluation runner: wraps lm-evaluation-harness, LiveCodeBench,
DeepEval, and bigcode-evaluation-harness against LM Studio API.

Usage:
    uv run python run_eval.py --framework lm-eval --model qwen35_122b_a10b_q8 --tasks mmlu,gsm8k
    uv run python run_eval.py --framework deepeval --model qwen35_122b_a10b_q8
    uv run python run_eval.py --framework bigcode --model qwen35_122b_a10b_q8 --tasks humaneval
    uv run python run_eval.py --framework livecodebench --model qwen35_122b_a10b_q8
    uv run python run_eval.py --list-tasks --framework lm-eval
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("eval.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"

# reuse models from run_bench.py
# hf_model: HuggingFace repo name for tokenizer resolution (lm-eval needs this)
MODELS = {
    "glm_5_1_iq3": {
        "served_model": "glm-5.1",
        "hf_model": "zai-org/GLM-5.1",
        "quant": "IQ3_XXS",
        "thinking": True,
    },
    "glm_47_flash_q4": {
        "served_model": "glm-4.7-flash",
        "hf_model": "zai-org/GLM-4.7-Flash",
        "quant": "Q4_K_M",
        "thinking": True,
    },
}

RESULTS_DIR = Path("results")

# default task sets per framework
DEFAULT_TASKS = {
    "lm-eval": "mmlu,gpqa_diamond_zeroshot,gsm8k,ifeval,humaneval",
    "bigcode": "humaneval,humanevalplus,mbpp,mbppplus",
    "livecodebench": "",
    "deepeval": "",
}

LIVECODEBENCH_DIR = Path("eval_frameworks/LiveCodeBench")
BIGCODE_DIR = Path("eval_frameworks/bigcode-evaluation-harness")


def run_lm_eval(model_label: str, model_cfg: dict, tasks: str, output_dir: Path):
    """Run lm-evaluation-harness against LM Studio API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    served_model = model_cfg["served_model"]
    hf_model = model_cfg.get("hf_model", served_model)
    # for thinking models, we need to tell lm-eval to set max_tokens
    # so the model produces actual content, not just reasoning tokens
    model_args = (
        f"model={served_model},"
        f"base_url={BASE_URL}/chat/completions,"
        f"tokenizer_backend=huggingface,"
        f"tokenizer={hf_model},"
        f"num_concurrent=1,"
        f"tokenized_requests=False,"
        f"max_gen_toks=1024"
    )

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--apply_chat_template",
        "--tasks", tasks,
        "--batch_size", "1",
        "--output_path", str(output_dir),
        "--log_samples",
    ]

    log.info("running lm-eval: model=%s tasks=%s", model_label, tasks)
    log.info("command: %s", " ".join(cmd))

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)  # 10 hours
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("lm-eval failed (%.1fs):\nstdout: %s\nstderr: %s",
                  elapsed, result.stdout[-2000:], result.stderr[-2000:])
        return False

    log.info("lm-eval completed in %.1fs for %s", elapsed, model_label)
    log.info("output: %s", result.stdout[-1000:])

    # save metadata
    meta = {
        "model_label": model_label,
        "served_model": served_model,
        "quant": model_cfg["quant"],
        "tasks": tasks,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return True


def run_bigcode(model_label: str, model_cfg: dict, tasks: str, output_dir: Path):
    """Run bigcode-evaluation-harness against LM Studio API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not BIGCODE_DIR.exists():
        log.error("bigcode-evaluation-harness not found at %s", BIGCODE_DIR)
        return False

    served_model = model_cfg["served_model"]

    cmd = [
        sys.executable, str(BIGCODE_DIR / "main.py"),
        "--model", served_model,
        "--tasks", tasks,
        "--max_length_generation", "1024",
        "--temperature", "0.0",
        "--n_samples", "1",
        "--save_generations",
        "--save_generations_path", str(output_dir / "generations.json"),
        "--metric_output_path", str(output_dir / "metrics.json"),
        "--allow_code_execution",
    ]

    log.info("running bigcode-eval: model=%s tasks=%s", model_label, tasks)
    log.info("command: %s", " ".join(cmd))

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("bigcode-eval failed (%.1fs):\nstdout: %s\nstderr: %s",
                  elapsed, result.stdout[-2000:], result.stderr[-2000:])
        return False

    log.info("bigcode-eval completed in %.1fs for %s", elapsed, model_label)

    meta = {
        "model_label": model_label,
        "served_model": served_model,
        "quant": model_cfg["quant"],
        "tasks": tasks,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return True


def run_livecodebench(model_label: str, model_cfg: dict, output_dir: Path):
    """Run LiveCodeBench against LM Studio API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LIVECODEBENCH_DIR.exists():
        log.error("LiveCodeBench not found at %s", LIVECODEBENCH_DIR)
        return False

    served_model = model_cfg["served_model"]

    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", served_model,
        "--scenario", "codegeneration",
        "--evaluate",
        "--output_dir", str(output_dir),
    ]

    log.info("running LiveCodeBench: model=%s", model_label)
    log.info("command: %s", " ".join(cmd))

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=14400,
        cwd=str(LIVECODEBENCH_DIR),
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("LiveCodeBench failed (%.1fs):\nstdout: %s\nstderr: %s",
                  elapsed, result.stdout[-2000:], result.stderr[-2000:])
        return False

    log.info("LiveCodeBench completed in %.1fs for %s", elapsed, model_label)

    meta = {
        "model_label": model_label,
        "served_model": served_model,
        "quant": model_cfg["quant"],
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return True


def run_deepeval(model_label: str, model_cfg: dict, output_dir: Path):
    """Run DeepEval LLM-as-judge tests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    test_file = Path("tests/test_deepeval.py")
    if not test_file.exists():
        log.error("DeepEval test file not found at %s", test_file)
        return False

    env_vars = {
        "EVAL_MODEL_LABEL": model_label,
        "EVAL_SERVED_MODEL": model_cfg["served_model"],
        "EVAL_BASE_URL": BASE_URL,
        "EVAL_API_KEY": API_KEY,
    }

    import os
    env = os.environ.copy()
    env.update(env_vars)

    cmd = [
        sys.executable, "-m", "pytest", str(test_file),
        "-v", "--tb=short",
        f"--junitxml={output_dir / 'deepeval_results.xml'}",
    ]

    log.info("running DeepEval: model=%s", model_label)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    elapsed = time.perf_counter() - t0

    log.info("DeepEval completed in %.1fs (exit=%d)", elapsed, result.returncode)
    log.info("output: %s", result.stdout[-1000:])

    # save output
    (output_dir / "stdout.txt").write_text(result.stdout)
    (output_dir / "stderr.txt").write_text(result.stderr)

    meta = {
        "model_label": model_label,
        "served_model": model_cfg["served_model"],
        "quant": model_cfg["quant"],
        "elapsed_s": round(elapsed, 1),
        "passed": result.returncode == 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return result.returncode == 0


def list_lm_eval_tasks():
    """List available lm-eval tasks."""
    cmd = [sys.executable, "-m", "lm_eval", "--tasks", "list"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Unified LLM evaluation runner")
    parser.add_argument("--framework", choices=["lm-eval", "bigcode", "livecodebench", "deepeval"],
                        help="Evaluation framework to use")
    parser.add_argument("--model", help="Model label from MODELS dict")
    parser.add_argument("--tasks", help="Comma-separated task list (framework-specific)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks for framework")
    args = parser.parse_args()

    if args.list_models:
        for label, cfg in MODELS.items():
            print(f"  {label:40s} served_model={cfg['served_model']}, quant={cfg['quant']}")
        return

    if args.list_tasks:
        if args.framework == "lm-eval":
            list_lm_eval_tasks()
        else:
            print(f"default tasks for {args.framework}: {DEFAULT_TASKS.get(args.framework, 'N/A')}")
        return

    if not args.framework:
        parser.error("--framework is required")
    if not args.model:
        parser.error("--model is required")

    if args.model not in MODELS:
        log.error("unknown model '%s'. available: %s", args.model, list(MODELS.keys()))
        sys.exit(1)

    model_cfg = MODELS[args.model]
    tasks = args.tasks or DEFAULT_TASKS.get(args.framework, "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / args.framework / args.model / timestamp

    log.info("=" * 60)
    log.info("framework: %s | model: %s | tasks: %s", args.framework, args.model, tasks)
    log.info("output: %s", output_dir)
    log.info("=" * 60)

    success = False
    if args.framework == "lm-eval":
        success = run_lm_eval(args.model, model_cfg, tasks, output_dir)
    elif args.framework == "bigcode":
        success = run_bigcode(args.model, model_cfg, tasks, output_dir)
    elif args.framework == "livecodebench":
        success = run_livecodebench(args.model, model_cfg, output_dir)
    elif args.framework == "deepeval":
        success = run_deepeval(args.model, model_cfg, output_dir)

    if success:
        log.info("evaluation completed successfully. results: %s", output_dir)
    else:
        log.error("evaluation failed. check %s for details", output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
