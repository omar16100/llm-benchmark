"""Check portable benchmark receipts without an inference engine."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def rates(rows, warmup=32):
    points = [(r["seconds"], r["index"]) for r in rows if r["index"] > warmup]
    if len(points) < 3 or not all(number(x) and number(y) for x, y in points):
        return None, None
    xs, ys = zip(*points)
    mean_x, mean_y = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
    variance = math.fsum((x - mean_x) ** 2 for x in xs)
    fit = math.fsum((x - mean_x) * (y - mean_y) for x, y in points) / variance if variance > 0 else None
    elapsed = xs[-1] - xs[0]
    endpoint = (ys[-1] - ys[0]) / elapsed if elapsed > 0 else None
    return fit, endpoint


def digest(data):
    return hashlib.sha256(data).hexdigest()


def token_hash(ids):
    return digest(json.dumps(ids).encode())


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_rows(path):
    return [json.loads(line, parse_constant=reject_constant)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def relative_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("A reference must use a relative path.")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("A reference must stay inside the receipt directory.")
    return target


def validate(root, policy=None, warmup=None):
    root = Path(root)
    checks, qualification = {}, {}
    fit = endpoint = None
    result = {}
    requested_warmup, warmup = warmup, 32
    try:
        result = read_json(root / "result.json")
        warmup = result["warmup_tokens"]
        if type(warmup) is not int or warmup < 0:
            raise ValueError("The warmup count must be a nonnegative integer.")
        checks["warmup"] = requested_warmup is None or (type(requested_warmup) is int and requested_warmup == warmup)
        checks["schema"] = type(result["schema_version"]) is int and result["schema_version"] == 1
        checks["status"] = result["status"] in ("completed", "failed", "controlled_stop")
        checks["provenance"] = all(isinstance(result[group][key], str) and bool(result[group][key])
                                   for group, keys in (("engine", ("name", "version")),
                                                       ("model", ("id", "revision"))) for key in keys)
        if result["status"] != "completed":
            checks["failed_request_has_no_rate"] = result["decode_tps_slope"] is None
        else:
            ids = read_json(root / "prompt.ids.json")
            meta = read_json(root / "prompt.meta.json")
            prompt = (root / "prompt.txt").read_bytes()
            rows = read_rows(root / "tokens.jsonl")
            prefill = read_rows(root / "prefill.jsonl")
            tokens = [r["token"] for r in rows]
            times = [r["seconds"] for r in rows]
            done = [r["done"] for r in prefill]
            counts = [result[key] for key in ("target_tokens", "input_tokens_reported", "prefill_tokens_observed")]
            checks["input_count"] = all(type(x) is int and x == len(ids) for x in counts + [meta["prompt_tokens"]])
            checks["token_ids"] = all(type(x) is int and x >= 0 for x in ids + tokens)
            checks["input_ids_hash"] = token_hash(ids) == meta["ids_sha256"]
            checks["prompt_hash"] = digest(prompt) == meta["text_sha256"]
            needles = meta["needle_texts"]
            checks["needles"] = isinstance(needles, list) and all(isinstance(x, str) and x and
                                      prompt.decode().count(x) == 1 for x in needles)
            checks["output_count"] = type(result["generation_tokens"]) is int and len(rows) == result["generation_tokens"]
            checks["output_hash"] = token_hash(tokens) == result["token_sha256"]
            checks["positions"] = all(type(r["index"]) is int for r in rows) and [r["index"] for r in rows] == list(range(1, len(rows)+1))
            checks["token_times"] = all(number(t) and t >= 0 for t in times) and all(a <= b for a, b in zip(times, times[1:]))
            checks["prefill"] = bool(done) and all(type(x) is int and 0 <= x <= len(ids) for x in done) and all(a <= b for a, b in zip(done, done[1:])) and done[-1] == len(ids)
            checks["cache_counts"] = all(type(result[key]) is int and result[key] == 0
                                         for key in ("cached_input_tokens", "cache_initial_tokens"))
            checks["ttft"] = bool(times) and number(result["first_token_seconds"]) and math.isclose(times[0], result["first_token_seconds"], abs_tol=1e-8)
            checks["decode_interval"] = bool(times) and number(result["decode_seconds"]) and math.isclose(times[-1]-times[0], result["decode_seconds"], abs_tol=1e-8)
            if checks["positions"] and checks["token_times"]:
                fit, endpoint = rates(rows, warmup)
            reported = result["decode_tps_slope"]
            checks["decode_slope"] = reported is None if fit is None else number(reported) and math.isclose(reported, fit, rel_tol=1e-10)
            if "comparison" in result:
                comparison = result["comparison"]
                reference = read_json(relative_file(root, comparison["token_ids_file"]))
                checks["comparison_ids"] = isinstance(reference, list) and all(type(x) is int and x >= 0 for x in reference)
                checks["comparison_hash"] = token_hash(reference) == comparison["token_sha256"]
                checks["comparison_result"] = type(comparison["exact_tokens"]) is bool and comparison["exact_tokens"] == (reference == tokens)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        checks["readable_receipt"] = False
        receipt_error = type(error).__name__ + ": " + str(error)
    else:
        receipt_error = None
    integrity = bool(checks) and all(checks.values())
    if policy is not None:
        try:
            valid_policy = type(policy["schema_version"]) is int and policy["schema_version"] == 1 and all(type(policy[key]) is int and policy[key] > 0
                                for key in ("context_tokens", "min_output_tokens")) and number(policy["min_decode_tps"]) and policy["min_decode_tps"] > 0
            qualification["policy"] = valid_policy
            qualification["integrity"] = integrity
            qualification["completed"] = result.get("status") == "completed"
            qualification["context"] = result.get("input_tokens_reported", -1) == policy["context_tokens"]
            qualification["output_length"] = result.get("generation_tokens", 0) >= policy["min_output_tokens"]
            qualification["decode_rate"] = fit is not None and fit >= policy["min_decode_tps"]
            requirements = policy["telemetry"]
            if not isinstance(requirements, dict):
                raise ValueError("Telemetry requirements must be an object.")
            telemetry = read_json(root / "telemetry.json") if requirements else {}
            for key, bounds in requirements.items():
                value = telemetry.get(key)
                valid = key in telemetry and isinstance(bounds, dict) and bool(bounds) and set(bounds) <= {"equals", "min", "max"}
                if "equals" in bounds:
                    valid = valid and type(value) is type(bounds["equals"]) and value == bounds["equals"]
                for bound in ("min", "max"):
                    if bound in bounds:
                        valid = valid and number(value) and number(bounds[bound]) and (value >= bounds[bound] if bound == "min" else value <= bounds[bound])
                qualification["telemetry_" + key] = bool(valid)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            qualification["readable_policy_and_telemetry"] = False
    return {"schema_version": 1, "integrity": {"passed": integrity, "checks": checks, "error": receipt_error},
            "qualification": {"evaluated": policy is not None,
                              "passed": all(qualification.values()) if qualification else None,
                              "checks": qualification},
            "rates": {"slope_tps": fit, "endpoint_tps": endpoint, "warmup_tokens": warmup}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--warmup", type=int, help="Require this recorded warmup count.")
    args = parser.parse_args()
    try:
        policy = read_json(args.policy) if args.policy else None
    except (OSError, ValueError) as error:
        parser.error(f"Cannot read the policy: {error}")
    report = validate(args.receipt, policy, args.warmup)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["integrity"]["passed"]:
        return 1
    return 3 if report["qualification"]["evaluated"] and not report["qualification"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
