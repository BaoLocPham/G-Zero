"""Local AIME24 and AIME25 evaluation; no LLM judge."""
from __future__ import annotations

import json
import logging
import os

from .config import Config
from .eval_aime import run_aime
from .local_backend import LocalBackend

logger = logging.getLogger(__name__)


def evaluate(
    config: Config,
    *,
    sampler_path: str | None = None,
    backend: LocalBackend | None = None,
) -> dict[str, float]:
    """Run all enabled eval tasks. Returns flat metric dict.

    sampler_path: local PEFT adapter directory, or None for the base model.
    """
    out_dir = config.eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Eval output dir: %s", out_dir)

    manifest = {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "sampler_path": str(sampler_path) if sampler_path else None,
        "eval_max_tokens": config.eval_max_tokens,
        "aime_temperature": config.aime_temperature,
        "aime_top_p": config.aime_top_p,
        "aime_samples_per_problem": config.aime_samples_per_problem,
        "aime_seed_2024": config.aime_seed_2024,
        "aime_seed_2025": config.aime_seed_2025,
    }
    manifest_path = out_dir / "eval_config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(
            f"evaluation cache at {out_dir} belongs to different model/settings; use a new tag"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    enabled = {t.strip() for t in config.eval_tasks.split(",") if t.strip()}
    unknown = enabled - {"aime24", "aime25"}
    if unknown:
        raise ValueError(f"unsupported local evaluation tasks: {sorted(unknown)}")
    logger.info("Enabled eval tasks: %s", sorted(enabled))
    backend = (backend or LocalBackend(config)) if enabled else None

    results: dict[str, float] = {}

    for year in ("aime24", "aime25"):
        if year not in enabled:
            continue
        try:
            avg = run_aime(
                backend,
                year=year,
                samples_per_problem=config.aime_samples_per_problem,
                temperature=config.aime_temperature,
                top_p=config.aime_top_p,
                seed=config.aime_seed_2024 if year == "aime24" else config.aime_seed_2025,
                max_tokens=config.eval_max_tokens,
                batch_size=config.inference_batch_size,
                adapter_path=sampler_path,
                out_path=out_dir / f"results_{year}.jsonl",
            )
            results[year] = float(avg)
        except Exception as e:  # noqa: BLE001
            logger.exception("[%s] failed: %s", year, e)
            raise

    summary_path = out_dir / "summary.json"
    summary_tmp = summary_path.with_suffix(".json.tmp")
    with open(summary_tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(summary_tmp, summary_path)
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %-40s  %.4f", k, v)
    logger.info("Summary -> %s", summary_path)
    return results
