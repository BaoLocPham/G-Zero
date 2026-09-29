"""G-Zero one-command pipeline.

Usage (one round):
    python3 -m g_zero.main --tag my_run

Steps (in order):
    1. (optional) Phase 1: Challenger GRPO  → trained Challenger checkpoint
    2. Phase 2: build DPO pool              → dpo_data.jsonl
    3. Phase 3: DPO-train the Solver        → solver checkpoint
    4. Eval: AIME24 + AIME25

Toggle Phase 1 with --run_phase1 true. Override any other field via --field.
Caching: completed pools and local adapter checkpoints are reused.
"""
from __future__ import annotations

import logging
import random

import torch

from .config import Config, parse_cli_overrides
from .eval import evaluate
from .local_backend import LocalBackend
from .phase1 import train_challenger
from .phase2 import build_dpo_dataset
from .phase3 import train_solver_dpo

logger = logging.getLogger(__name__)


def _sampler(result) -> str:
    """Extract the local adapter path from a training result."""
    if isinstance(result, dict):
        return result.get("sampler_path") or result.get("state_path")
    return str(result)


def _state(result) -> str:
    if isinstance(result, dict):
        return result.get("state_path") or result.get("path")
    return str(result)


def run(config: Config) -> dict[str, float]:
    config.run_dir().mkdir(parents=True, exist_ok=True)
    config.save_run_config()
    random.seed(config.seed)
    logger.info("Run dir: %s", config.run_dir())
    backend = LocalBackend(config)
    torch.manual_seed(config.seed)

    challenger_sampler: str | None = None

    if config.run_phase1:
        logger.info("=" * 60)
        logger.info("Phase 1: Challenger GRPO")
        logger.info("=" * 60)
        ch_save = str(config.challenger_save_dir())
        ch_result = train_challenger(
            config,
            challenger_init_path=None,    # start from base
            solver_sampling_path=None,    # base solver as δ scorer
            save_path=ch_save,
            backend=backend,
        )
        challenger_sampler = _sampler(ch_result)
        logger.info("Challenger sampler: %s", challenger_sampler)
    else:
        logger.info("Phase 1 skipped (--run_phase1 false): base model used as Challenger")

    logger.info("=" * 60)
    logger.info("Phase 2: build DPO pool")
    logger.info("=" * 60)
    build_dpo_dataset(
        config,
        challenger_sampling_path=challenger_sampler,
        solver_sampling_path=None,
        backend=backend,
    )
    if config.stop_after_phase2:
        logger.info("Stopping after Phase 2 as requested")
        return {}

    logger.info("=" * 60)
    logger.info("Phase 3: Solver DPO")
    logger.info("=" * 60)
    solver_result = train_solver_dpo(
        config,
        solver_init_path=None,
        data_path=config.dpo_data_path(),
        save_path=str(config.solver_save_dir()),
        backend=backend,
    )
    solver_sampler = _sampler(solver_result)
    logger.info("Solver sampler: %s", solver_sampler)

    logger.info("=" * 60)
    logger.info("Eval")
    logger.info("=" * 60)
    return evaluate(config, sampler_path=solver_sampler, backend=backend)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = parse_cli_overrides()
    run(config)


if __name__ == "__main__":
    main()
