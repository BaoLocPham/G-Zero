#!/usr/bin/env bash
# Ablation: 3 rounds of Challenger ↔ Solver co-evolution.
# Round 1 captures most of the gain on Qwen3-8B-Base; round 2/3 either
# regress or drift. Included so the curve can be reproduced.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to two GPU IDs, in inference,training order." >&2
  exit 1
fi

exec "${PYTHON_BIN:-python3}" -m g_zero.multi_round \
  --tag ablation_multi_round \
  --model_name Qwen/Qwen3-8B-Base \
  --run_phase1 true \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  --num_rounds 3 \
  "$@"
