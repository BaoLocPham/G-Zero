#!/usr/bin/env bash
# Paper-style AIME mean@32 for local 4B base and a G-Zero adapter.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER=${1:?usage: eval_4b_mean32.sh <adapter_dir>}
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "2,3" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES=2,3 for this two-GPU evaluation." >&2
  exit 1
fi
export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

COMMON=(
  --aime_samples_per_problem 32
  --eval_max_tokens 2048
  --aime_temperature 0.7
  --aime_top_p 0.95
  --max_model_len 4096
  --inference_batch_size 64
  --vllm_tensor_parallel_size 2
  --vllm_gpu_memory_utilization 0.9
  --eval_tasks aime24,aime25
)

bash scripts/eval_only.sh base --tag gzero_4b_base_mean32_20260929 "${COMMON[@]}"
bash scripts/eval_only.sh "$ADAPTER" --tag gzero_4b_bot50_mean32_20260929 "${COMMON[@]}"
