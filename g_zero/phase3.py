"""Local LoRA DPO for the Solver, preserving G-Zero's response-mean objective."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config
from .local_backend import LocalBackend
from .local_training import (
    append_metrics, load_resume_state, load_trainable, response_logprobs,
    restore_rng_state, save_training_state,
)

logger = logging.getLogger(__name__)


def _load_pairs(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _dpo_loss(chosen, rejected, chosen_ref, rejected_ref, beta: float):
    margin = beta * ((chosen - chosen_ref) - (rejected - rejected_ref))
    return -F.logsigmoid(margin), margin


def train_solver_dpo(
    config: Config,
    *,
    solver_init_path: str | None = None,
    data_path: Path | None = None,
    save_path: str | None = None,
    backend: LocalBackend | None = None,
) -> dict[str, str]:
    save_dir = Path(save_path or config.solver_save_dir())
    final_dir = save_dir / "final"
    if (final_dir / "complete").exists():
        logger.info("Phase 3: reusing Solver adapter %s", final_dir)
        return {"state_path": str(final_dir), "sampler_path": str(final_dir)}
    rows = _load_pairs(data_path or config.dpo_data_path())
    if not rows:
        raise RuntimeError("DPO dataset is empty")
    save_dir.mkdir(parents=True, exist_ok=True)
    backend = backend or LocalBackend(config)
    resume = load_resume_state(save_dir)
    adapter_path = str(resume[1]) if resume else solver_init_path
    model = load_trainable(config, adapter_path)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.dpo_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )
    start_step = 0
    if resume:
        state, _ = resume
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_state(state, model)
        start_step = int(state["step"])
        logger.info("Phase 3: resuming at step %d", start_step)

    n_batches = len(rows) // config.dpo_batch_size
    if n_batches == 0:
        raise RuntimeError("DPO dataset is smaller than dpo_batch_size")
    total_steps = n_batches * config.dpo_num_epochs
    if config.dpo_max_steps:
        total_steps = min(total_steps, config.dpo_max_steps)
    for step in range(start_step, total_steps):
        batch_index = step % n_batches
        batch = rows[
            batch_index * config.dpo_batch_size :
            (batch_index + 1) * config.dpo_batch_size
        ]
        prompts = [backend.prompt([{"role": "user", "content": row["prompt"]}]) for row in batch]
        chosen = [backend.encode_response(row["chosen"]) for row in batch]
        rejected = [backend.encode_response(row["rejected"]) for row in batch]
        if any(not c or not r for c, r in zip(chosen, rejected)):
            raise RuntimeError("DPO batch contains an empty tokenized response")
        ref_pairs = []
        for prompt, c, r in zip(prompts, chosen, rejected):
            ref_pairs.extend([(prompt, c), (prompt, r)])
        reference = backend.continuation_logprobs_batch(
            ref_pairs, adapter_path=solver_init_path,
        )

        lr = config.dpo_lr * max(0.0, 1.0 - step / max(total_steps, 1))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        losses, margins, accuracies = [], [], []
        # One pair at a time bounds activation memory; gradients accumulate
        # across the configured number of pairs before the optimizer step.
        for i, (prompt, c, r) in enumerate(zip(prompts, chosen, rejected)):
            c_lp = response_logprobs(model, prompt, c).mean()
            r_lp = response_logprobs(model, prompt, r).mean()
            c_ref = torch.tensor(
                sum(reference[2 * i]) / len(reference[2 * i]),
                device=c_lp.device, dtype=torch.float32,
            )
            r_ref = torch.tensor(
                sum(reference[2 * i + 1]) / len(reference[2 * i + 1]),
                device=c_lp.device, dtype=torch.float32,
            )
            loss, margin = _dpo_loss(c_lp, r_lp, c_ref, r_ref, config.dpo_beta)
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
            margins.append(float(margin.detach()))
            accuracies.append(float((margin.detach() > 0).float()))
        optimizer.step()
        append_metrics(save_dir / "metrics.jsonl", {
            "solver_dpo/step": step,
            "solver_dpo/lr": lr,
            "solver_dpo/dpo_loss": sum(losses) / len(losses),
            "solver_dpo/margin": sum(margins) / len(margins),
            "solver_dpo/accuracy": sum(accuracies) / len(accuracies),
        })
        save_training_state(model, optimizer, save_dir, step + 1)
        logger.info("DPO step %d/%d loss=%.4f", step + 1, total_steps, sum(losses) / len(losses))

    model.save_pretrained(final_dir)
    (final_dir / "complete").write_text("ok\n")
    return {"state_path": str(final_dir), "sampler_path": str(final_dir)}
