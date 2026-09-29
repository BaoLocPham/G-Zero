"""All G-Zero hyperparameters in one place.

Defaults use the locally cached Qwen3-4B-Base model.
Override any field via CLI: `python3 -m g_zero.main --field value`.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ─── Run identity ────────────────────────────────────────────────────────
    tag: str = "g_zero_demo"
    storage_path: str = field(
        default_factory=lambda: os.environ.get(
            "G_ZERO_STORAGE", str(Path(__file__).resolve().parents[1] / "runs")
        )
    )

    # ─── Model ───────────────────────────────────────────────────────────────
    # Local Hugging Face model with a tokenizer chat template.
    model_name: str = "/home/locpb/self-evolution-repro/papers/agent0/shared-models/Qwen3-4B-Base"
    model_revision: str | None = None
    seed: int = 42
    lora_rank: int = 32
    max_model_len: int = 32768
    vllm_gpu_memory_utilization: float = 0.85
    vllm_enforce_eager: bool = True
    vllm_tensor_parallel_size: int = 1
    training_device: str = "cuda:1"
    inference_batch_size: int = 16

    # ─── Phase 1: Challenger GRPO ────────────────────────────────────────────
    # Set run_phase1=False to skip; the base model is then used directly as
    # the Challenger. Default is on — runs 6 GRPO steps before Phase 2.
    run_phase1: bool = True
    stop_after_phase2: bool = False               # generate/filter pool only
    challenger_steps: int = 6                       # GRPO steps
    challenger_batch_size: int = 8                  # # of "problems" per step
    challenger_group_size: int = 16                 # rollouts per problem
    challenger_lr: float = 4e-5
    delta_clip_min: float = -5.0                   # floor for δ in reward
    hint_length_target_chars: int = 200             # length-hinge target
    hint_length_penalty_lambda: float = 0.03        # λ in length penalty
    bleu_distance_threshold: float = 0.5            # for cluster_share()

    # ─── Phase 2: data generation ────────────────────────────────────────────
    num_questions: int = 2000                       # paper-main: 2000
    challenger_max_tokens: int = 8192               # cap on hint generation
    solver_max_tokens: int = 8192                   # cap on a_hard / a_assisted
    solver_sample_temperature: float = 0.7

    # δ percentile filter [pct_low, pct_high]. paper-main = bot50 = [0, 50].
    pct_low: float = 0.0
    pct_high: float = 50.0

    # Quality filters on chosen (= a_assisted)
    chosen_min_chars: int = 100
    chosen_max_chars: int = 10_000
    chosen_rejected_ratio_max: float = 2.5
    chosen_repetition_zlib_threshold: float = 0.15
    chosen_prompt_prefix_overlap: int = 30
    chosen_role_marker_prefixes: tuple[str, ...] = (
        "Assistant:", "AI:", "Bot:", "User:", "Human:"
    )

    # ─── Phase 3: DPO ────────────────────────────────────────────────────────
    dpo_beta: float = 2.0
    dpo_lr: float = 1e-5
    dpo_batch_size: int = 8
    dpo_max_steps: int = 50
    dpo_num_epochs: int = 1

    # ─── Eval ────────────────────────────────────────────────────────────────
    eval_max_tokens: int = 16384
    aime_temperature: float = 0.7                   # mean@k requires > 0
    aime_top_p: float = 0.95
    aime_samples_per_problem: int = 1
    aime_seed_2024: int = 2026092201
    aime_seed_2025: int = 2026092202

    # Eval task selector — comma-separated subset of {aime24,aime25}
    eval_tasks: str = "aime24,aime25"

    # ─── Per-run paths ───────────────────────────────────────────────────────
    def run_dir(self) -> Path:
        return Path(self.storage_path) / self.tag

    def raw_pool_path(self) -> Path:
        return self.run_dir() / "raw_pool.jsonl"

    def dpo_data_path(self) -> Path:
        return self.run_dir() / "dpo_data.jsonl"

    def challenger_save_dir(self) -> Path:
        return self.run_dir() / "challenger"

    def solver_save_dir(self) -> Path:
        return self.run_dir() / "solver"

    def eval_dir(self) -> Path:
        return self.run_dir() / "eval"

    def save_run_config(self, path: Path | None = None) -> None:
        """Refuse to reuse cached data under a changed configuration."""
        path = path or self.run_dir() / "config.json"
        current = json.loads(json.dumps(asdict(self)))
        if path.exists() and json.loads(path.read_text()) != current:
            raise RuntimeError(f"run configuration changed at {path}; use a new tag")
        path.write_text(json.dumps(current, indent=2))


def parse_cli_overrides(argv: list[str] | None = None) -> Config:
    """Build a Config and overlay CLI overrides like `--tag foo --num_questions 100`."""
    cfg = Config()
    parser = argparse.ArgumentParser(description="G-Zero pipeline")
    for f in cfg.__dataclass_fields__.values():
        if f.type in (bool, "bool"):
            parser.add_argument(f"--{f.name}", type=lambda x: str(x).lower() in ("1","true","y"))
        elif f.type in (int, "int"):
            parser.add_argument(f"--{f.name}", type=int)
        elif f.type in (float, "float"):
            parser.add_argument(f"--{f.name}", type=float)
        else:
            parser.add_argument(f"--{f.name}", type=str)
    args = parser.parse_args(argv)
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
