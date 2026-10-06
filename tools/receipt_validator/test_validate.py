"""Test receipt integrity and separate qualification decisions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from validate import digest, rates, token_hash, validate


def write_json(root, name, data):
    (root / name).write_text(json.dumps(data))


def fixture(root, count=64, shared_time=False):
    ids = list(range(128))
    prompt = b"Synthetic prompt. Needle: BLUE-123."
    rows = [{"index": i+1, "token": i, "seconds": 1.0 if shared_time else 1.0+i/125}
            for i in range(count)]
    tokens = [r["token"] for r in rows]
    result = {"schema_version": 1, "status": "completed", "warmup_tokens": 32,
              "engine": {"name": "synthetic", "version": "1"},
              "model": {"id": "synthetic", "revision": "fixture-v1"},
              "target_tokens": 128, "input_tokens_reported": 128, "prefill_tokens_observed": 128,
              "generation_tokens": count, "cached_input_tokens": 0, "cache_initial_tokens": 0,
              "first_token_seconds": rows[0]["seconds"],
              "decode_seconds": rows[-1]["seconds"]-rows[0]["seconds"],
              "decode_tps_slope": rates(rows)[0], "token_sha256": token_hash(tokens)}
    write_json(root, "result.json", result)
    write_json(root, "prompt.ids.json", ids)
    write_json(root, "prompt.meta.json", {"prompt_tokens": 128, "ids_sha256": token_hash(ids),
               "text_sha256": digest(prompt), "needle_texts": ["Needle: BLUE-123."]})
    (root / "prompt.txt").write_bytes(prompt)
    (root / "tokens.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    (root / "prefill.jsonl").write_text(json.dumps({"done": 128, "seconds": 0.9})+"\n")
    write_json(root, "telemetry.json", {"gpu_temp_c": 55, "gpu_coverage_complete": True,
                                        "discovery_errors": 0})
    return result


POLICY = {"schema_version": 1, "context_tokens": 128, "min_output_tokens": 64,
          "min_decode_tps": 100, "telemetry": {"gpu_temp_c": {"min": 1, "max": 80},
          "gpu_coverage_complete": {"equals": True}, "discovery_errors": {"equals": 0}}}


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_receipt(self):
        report = validate(self.root, POLICY)
        self.assertTrue(report["integrity"]["passed"], report)
        self.assertTrue(report["qualification"]["passed"], report)
        self.assertAlmostEqual(report["rates"]["slope_tps"], 125)

    def test_integrity_does_not_imply_qualification(self):
        report = validate(self.root)
        self.assertTrue(report["integrity"]["passed"])
        self.assertFalse(report["qualification"]["evaluated"])
        self.assertIsNone(report["qualification"]["passed"])

    def test_result_mutations(self):
        for key, value in (("schema_version", True), ("status", "unknown"),
                           ("target_tokens", 127), ("input_tokens_reported", 127),
                           ("prefill_tokens_observed", 127), ("generation_tokens", 63),
                           ("cached_input_tokens", 1), ("cache_initial_tokens", True),
                           ("token_sha256", "bad"), ("first_token_seconds", 2),
                           ("decode_seconds", 999), ("decode_tps_slope", 999),
                           ("engine", {"name": "synthetic", "version": ""})):
            with self.subTest(key=key):
                result = fixture(self.root)
                result[key] = value
                write_json(self.root, "result.json", result)
                self.assertFalse(validate(self.root)["integrity"]["passed"])

    def test_prompt_and_trace_mutations(self):
        changes = (("prompt.txt", "changed"), ("prompt.ids.json", "[0]"),
                   ("prompt.meta.json", "{}"), ("prefill.jsonl", '{"done":127}\n'),
                   ("tokens.jsonl", '{"index":1,"token":0,"seconds":NaN}\n'))
        for name, text in changes:
            with self.subTest(name=name):
                fixture(self.root)
                (self.root / name).write_text(text)
                self.assertFalse(validate(self.root)["integrity"]["passed"])

    def test_needle_text_is_configurable(self):
        meta = json.loads((self.root / "prompt.meta.json").read_text())
        meta["needle_texts"] = ["Absent needle"]
        write_json(self.root, "prompt.meta.json", meta)
        self.assertFalse(validate(self.root)["integrity"]["checks"]["needles"])

    def test_short_completion(self):
        fixture(self.root, count=20)
        report = validate(self.root, POLICY)
        self.assertTrue(report["integrity"]["passed"], report)
        self.assertIsNone(report["rates"]["slope_tps"])
        self.assertFalse(report["qualification"]["passed"])

    def test_shared_timestamp(self):
        fixture(self.root, shared_time=True)
        report = validate(self.root, POLICY)
        self.assertTrue(report["integrity"]["passed"], report)
        self.assertIsNone(report["rates"]["slope_tps"])
        self.assertIsNone(report["rates"]["endpoint_tps"])
        self.assertFalse(report["qualification"]["passed"])

    def test_missing_telemetry(self):
        for value in (None, True, 0, 151):
            write_json(self.root, "telemetry.json", {"gpu_temp_c": value,
                       "gpu_coverage_complete": True, "discovery_errors": 0})
            report = validate(self.root, POLICY)
            self.assertTrue(report["integrity"]["passed"])
            self.assertFalse(report["qualification"]["passed"])

    def test_controlled_stop_and_failed_rate(self):
        for status in ("controlled_stop", "failed"):
            result = fixture(self.root, count=20)
            result["status"] = status
            write_json(self.root, "result.json", result)
            report = validate(self.root, POLICY)
            self.assertTrue(report["integrity"]["passed"])
            self.assertFalse(report["qualification"]["passed"])
            result["decode_tps_slope"] = 0
            write_json(self.root, "result.json", result)
            self.assertFalse(validate(self.root)["integrity"]["passed"])

    def test_relative_comparison_and_relocation(self):
        result = fixture(self.root)
        tokens = list(range(64))
        write_json(self.root, "reference.ids.json", tokens)
        result["comparison"] = {"token_ids_file": "reference.ids.json",
                                "token_sha256": token_hash(tokens), "exact_tokens": True}
        write_json(self.root, "result.json", result)
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "copy"
            shutil.copytree(self.root, target)
            self.assertTrue(validate(target)["integrity"]["passed"])
        for unsafe in ("/private/reference.json", "../reference.json"):
            result["comparison"]["token_ids_file"] = unsafe
            write_json(self.root, "result.json", result)
            self.assertFalse(validate(self.root)["integrity"]["passed"])

    def test_invalid_policy(self):
        for policy in ({}, {**POLICY, "min_decode_tps": -1}, {**POLICY, "telemetry": []},
                       {**POLICY, "min_decode_tps": 10**400}):
            report = validate(self.root, policy)
            self.assertTrue(report["integrity"]["passed"])
            self.assertFalse(report["qualification"]["passed"])

    def test_unicode_and_required_warmup(self):
        meta = json.loads((self.root / "prompt.meta.json").read_text())
        prompt = "A Unicode needle: 東京.".encode("utf-8")
        (self.root / "prompt.txt").write_bytes(prompt)
        meta.update(text_sha256=digest(prompt), needle_texts=["東京"])
        (self.root / "prompt.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.assertTrue(validate(self.root, warmup=32)["integrity"]["passed"])
        self.assertFalse(validate(self.root, warmup=0)["integrity"]["passed"])

    def test_missing_telemetry_key_and_huge_value(self):
        for telemetry in ({}, {"gpu_temp_c": 10**400, "gpu_coverage_complete": True,
                              "discovery_errors": 0}):
            write_json(self.root, "telemetry.json", telemetry)
            self.assertFalse(validate(self.root, POLICY)["qualification"]["passed"])
        null_policy = {**POLICY, "telemetry": {"absent": {"equals": None}}}
        self.assertFalse(validate(self.root, null_policy)["qualification"]["passed"])

    def test_cli_exit_codes(self):
        command = [sys.executable, str(Path(__file__).with_name("validate.py")), str(self.root)]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        policy = self.root / "policy.json"
        write_json(self.root, "policy.json", {**POLICY, "min_decode_tps": 999})
        self.assertEqual(subprocess.run(command+["--policy", str(policy)], capture_output=True).returncode, 3)
        policy.write_text("{")
        self.assertEqual(subprocess.run(command+["--policy", str(policy)], capture_output=True).returncode, 2)
        (self.root / "result.json").unlink()
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)

    def test_origin_invariance_and_empty_rate(self):
        rows = [{"seconds": 10+i/125, "index": i+1} for i in range(64)]
        shifted = [{**r, "seconds": r["seconds"]+1000} for r in rows]
        self.assertAlmostEqual(rates(rows)[0], rates(shifted)[0], places=8)
        self.assertEqual(rates([]), (None, None))


if __name__ == "__main__":
    unittest.main()
