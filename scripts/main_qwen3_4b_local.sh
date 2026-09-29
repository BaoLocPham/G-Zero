#!/usr/bin/env bash
# Main recipe with the locally cached Qwen3-4B-Base model.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash run.sh \
  --tag main_qwen3_4b_local \
  --model_name /home/locpb/self-evolution-repro/papers/agent0/shared-models/Qwen3-4B-Base \
  --run_phase1 true \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  "$@"
