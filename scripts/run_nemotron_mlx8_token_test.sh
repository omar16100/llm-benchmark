#!/usr/bin/env bash
set -euo pipefail

# repo root = parent of this script's directory; model dir comes from the environment
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${NEMOTRON_MLX8_DIR:?set NEMOTRON_MLX8_DIR to the nemotron-3-nano-30b-a3b-mlx-8bit model directory}"
OUT_DIR="$ROOT/results/1m_candidates"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/nemotron_mlx8_token_test_${STAMP}.log"
CSV="$OUT_DIR/nemotron_mlx8_token_test_${STAMP}.csv"

mkdir -p "$OUT_DIR"

if [[ "${CODEX_SANDBOX:-}" != "" ]]; then
  echo "ERROR: This must be run from a normal macOS Terminal, not from the Codex sandbox."
  echo "CODEX_SANDBOX=${CODEX_SANDBOX}"
  exit 2
fi

if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: model directory not found: $MODEL"
  exit 1
fi
# absolute, so a relative path still means the same directory after `cd "$ROOT"` below
MODEL="$(cd "$MODEL" && pwd)"

cd "$ROOT"

{
  echo "# Nemotron 3 Nano 30B-A3B MLX 8Bit token throughput test"
  echo "# Date: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "# Hardware: $(sysctl -n hw.model 2>/dev/null || uname -m)"
  echo "# Model: $MODEL"
  echo "# MLX command: $ROOT/.venv/bin/python -m mlx_lm.benchmark"
  echo

  echo "## Device check"
  "$ROOT/.venv/bin/python" - <<'PY'
import mlx.core as mx
print("default_device:", mx.default_device())
print("metal_available:", mx.metal.is_available())
print("device_info:", mx.device_info())
PY
  echo

  echo "## Long-context profile: 100K prompt, 100 generated tokens, 1 trial"
  "$ROOT/.venv/bin/python" -m mlx_lm.benchmark \
    --model "$MODEL" \
    --prompt-tokens 100000 \
    --generation-tokens 100 \
    --num-trials 1 \
    --batch-size 1 \
    --prefill-step-size 2048
} 2>&1 | tee "$LOG"

echo
echo "Wrote log: $LOG"

"$ROOT/.venv/bin/python" - "$LOG" "$CSV" <<'PY'
import csv
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
lines = log_path.read_text().splitlines()

profile = None
rows = []
for line in lines:
    if line.startswith("## Long-context profile: 100K"):
        profile = "100k_prompt_100_gen"
        continue
    if line.startswith("## Short decode profile"):
        profile = "4k_prompt_100_gen"
        continue
    if line.startswith("## True long-context profile"):
        profile = "1m_prompt_100_gen"
        continue
    match = re.match(
        r"(Trial \d+|Averages):\s+prompt_tps=([0-9.]+), "
        r"generation_tps=([0-9.]+), peak_memory=([0-9.]+)"
        r"(?:, total_time=([0-9.]+))?",
        line,
    )
    if match and profile:
        label, prompt_tps, generation_tps, peak_memory, total_time = match.groups()
        rows.append({
            "profile": profile,
            "row": label,
            "prompt_tokens": {
                "4k_prompt_100_gen": 4096,
                "100k_prompt_100_gen": 100000,
                "1m_prompt_100_gen": 1000000,
            }[profile],
            "generation_tokens": 100,
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
            "peak_memory_gb": peak_memory,
            "total_time_s": total_time or "",
            "log": str(log_path),
        })

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "profile",
            "row",
            "prompt_tokens",
            "generation_tokens",
            "prompt_tps",
            "generation_tps",
            "peak_memory_gb",
            "total_time_s",
            "log",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote CSV: {csv_path}")
if not rows:
    print("WARNING: no benchmark throughput rows found in log")
PY
