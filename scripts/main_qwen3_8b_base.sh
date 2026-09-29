#!/usr/bin/env bash
# Paper-main: Qwen3-8B-Base, single round, bot50 δ-cutoff.
# Runs on two visible local GPUs; runtime depends on available hardware.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash run.sh \
  --tag main_qwen3_8b_base \
  --model_name Qwen/Qwen3-8B-Base \
  --run_phase1 true \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  "$@"
