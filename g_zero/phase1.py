"""Optional local Challenger GRPO with G-Zero's delta and diversity reward."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from .bleu_penalty import cluster_share
from .config import Config
from .hint_delta import score_batch
from .local_backend import LocalBackend
from .local_training import (
    append_metrics, load_resume_state, load_trainable, response_logprobs,
    restore_rng_state, save_training_state,
)
from .parse import extract_qh
from .prompts import build_challenger_convo

logger = logging.getLogger(__name__)


def _make_advantages(rewards: list[float]) -> list[float]:
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    return [reward - mean for reward in rewards]


def train_challenger(
    config: Config,
    *,
    challenger_init_path: str | None = None,
    solver_sampling_path: str | None = None,
    save_path: str,
    backend: LocalBackend | None = None,
) -> dict[str, str]:
    save_dir = Path(save_path)
    final_dir = save_dir / "final"
    if (final_dir / "complete").exists():
        logger.info("Phase 1: reusing Challenger adapter %s", final_dir)
        return {"state_path": str(final_dir), "sampler_path": str(final_dir)}
    save_dir.mkdir(parents=True, exist_ok=True)
    backend = backend or LocalBackend(config)
    resume = load_resume_state(save_dir)
    init_path = str(resume[1]) if resume else challenger_init_path
    model = load_trainable(config, init_path)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.challenger_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )
    start_step = 0
    if resume:
        state, _ = resume
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state, model)
        start_step = int(state["step"])
    else:
        save_training_state(model, optimizer, save_dir, 0)

    prompt = backend.prompt(build_challenger_convo())
    for step in range(start_step, config.challenger_steps):
        started = time.time()
        # Every snapshot has a distinct path and LoRA ID; vLLM never caches
        # weights from a previous optimizer step under the current request.
        current_adapter = str(save_dir / f"step_{step:04d}")
        groups = backend.generate_batch(
            [prompt] * config.challenger_batch_size,
            n=config.challenger_group_size,
            max_tokens=config.challenger_max_tokens,
            temperature=1.0,
            adapter_path=current_adapter,
            return_logprobs=True,
        )
        sequences = [sequence for group in groups for sequence in group]
        parsed = [extract_qh(backend.decode(sequence.tokens)) for sequence in sequences]
        valid = [i for i, (q, h) in enumerate(parsed) if q and h]
        deltas = [0.0] * len(sequences)
        if valid:
            scored = score_batch(
                backend, [parsed[i] for i in valid], mode="delta_only",
                adapter_path=solver_sampling_path,
                max_tokens=config.solver_max_tokens,
                temperature=config.solver_sample_temperature,
                batch_size=config.inference_batch_size,
            )
            for i, item in zip(valid, scored):
                deltas[i] = max(item.delta, config.delta_clip_min)

        questions = [q or "__INVALID__" for q, _ in parsed]
        penalties = cluster_share(questions, config.bleu_distance_threshold)
        rewards, length_penalties = [], []
        for i, (q, h) in enumerate(parsed):
            if q and h:
                length_penalty = config.hint_length_penalty_lambda * (
                    max(0, len(h) - config.hint_length_target_chars) / 100.0
                )
                rewards.append(deltas[i] - penalties[i] - length_penalty)
            else:
                length_penalty = 0.0
                rewards.append(-1.0 - penalties[i])
            length_penalties.append(length_penalty)

        advantages = []
        for group_index in range(config.challenger_batch_size):
            left = group_index * config.challenger_group_size
            right = left + config.challenger_group_size
            advantages.extend(_make_advantages(rewards[left:right]))
        usable = [
            (sequence, advantage)
            for sequence, advantage in zip(sequences, advantages)
            if sequence.tokens and len(sequence.tokens) == len(sequence.logprobs)
            and abs(advantage) >= 1e-8
        ]
        if usable:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for sequence, advantage in usable:
                new_logprobs = response_logprobs(model, prompt, sequence.tokens)
                old_logprobs = torch.tensor(
                    sequence.logprobs, device=new_logprobs.device, dtype=torch.float32
                )
                # The policy-gradient importance ratio is evaluated per token.
                # Mean each rollout first so long completions do not dominate.
                ratio = torch.exp(new_logprobs - old_logprobs)
                loss = -(ratio * advantage).mean()
                (loss / len(usable)).backward()
                losses.append(float(loss.detach()))
            optimizer.step()
            policy_loss = sum(losses) / len(losses)
        else:
            logger.warning("Challenger step %d has no usable rollouts", step)
            policy_loss = 0.0
        append_metrics(save_dir / "metrics.jsonl", {
            "challenger/step": step,
            "challenger/reward_mean": sum(rewards) / max(1, len(rewards)),
            "challenger/delta_mean_valid": sum(deltas[i] for i in valid) / max(1, len(valid)),
            "challenger/valid_frac": len(valid) / max(1, len(sequences)),
            "challenger/penalty_mean": sum(penalties) / max(1, len(penalties)),
            "challenger/length_penalty_mean": sum(length_penalties) / max(1, len(length_penalties)),
            "challenger/n_datums": len(usable),
            "challenger/policy_loss": policy_loss,
            "time/step": time.time() - started,
        })
        save_training_state(model, optimizer, save_dir, step + 1)
        logger.info("Challenger step %d/%d reward=%.4f", step + 1,
                    config.challenger_steps, sum(rewards) / max(1, len(rewards)))

    model.save_pretrained(final_dir)
    (final_dir / "complete").write_text("ok\n")
    return {"state_path": str(final_dir), "sampler_path": str(final_dir)}
