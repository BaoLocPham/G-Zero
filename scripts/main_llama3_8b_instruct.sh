#!/usr/bin/env bash
# Paper-main on Llama-3.1-8B-Instruct.
# Same recipe as Qwen3-8B-Base with the model's chat template.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash run.sh \
  --tag main_llama3_8b_instruct \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --run_phase1 true \
  --num_questions 2000 \
  --pct_low 0 --pct_high 50 \
  "$@"
