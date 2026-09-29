#!/usr/bin/env bash
# Run AIME24+AIME25 on a local PEFT adapter without retraining.
#
# Usage:
#   bash scripts/eval_only.sh <adapter_dir> [--eval_tasks aime24,aime25,...]
#
# Pass the adapter's base model with --model_name if it differs from the default.
set -euo pipefail
cd "$(dirname "$0")/.."

SAMPLER=${1:?usage: eval_only.sh <adapter_dir|base> [extra args]}
shift

"${PYTHON_BIN:-python3}" - "$SAMPLER" "$@" <<'PY'
import sys, logging, hashlib
from pathlib import Path
from g_zero.config import parse_cli_overrides
from g_zero.eval import evaluate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

sampler = None if sys.argv[1] == "base" else str(Path(sys.argv[1]).resolve())
config = parse_cli_overrides(sys.argv[2:])
if not getattr(config, "tag", None) or config.tag == "g_zero_demo":
    digest = hashlib.sha256((sampler or "base").encode()).hexdigest()[:10]
    config.tag = f"eval_only_{digest}"
evaluate(config, sampler_path=sampler)
PY
