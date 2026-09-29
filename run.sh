#!/usr/bin/env bash
# One-command driver for the G-Zero pipeline.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=2,3               # inference, training
#   bash run.sh                                   # default: paper-main R1
#   bash run.sh --run_phase1 false                # no_phase1 ablation
#   bash run.sh --tag big --num_questions 5000    # custom tag / pool size
#
# All flags after `bash run.sh` are forwarded verbatim to g_zero/main.py.
# See scripts/ for canned single-command experiments.

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to two GPU IDs, in inference,training order." >&2
  echo "Example: export CUDA_VISIBLE_DEVICES=2,3" >&2
  exit 1
fi

cd "$(dirname "$0")"
exec "${PYTHON_BIN:-python3}" -m g_zero.main "$@"
