"""Shared local LoRA training and response-logprob utilities."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def training_device(config) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("local training requires CUDA")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "training needs two visible GPUs: inference on cuda:0 and training on cuda:1"
        )
    return torch.device(config.training_device)


def load_trainable(config, adapter_path: str | None = None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    device = training_device(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=2 * config.lora_rank,
                lora_dropout=0.0,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    model.train()
    return model


def response_logprobs(model, prompt: list[int], response: list[int]) -> torch.Tensor:
    """Differentiable per-token log probabilities, excluding prompt tokens."""
    if not prompt or not response:
        raise ValueError("prompt and response must both contain tokens")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits[:, len(prompt) - 1 : -1, :]
    targets = ids[:, len(prompt) :]
    return F.log_softmax(logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(0).squeeze(-1)


def append_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(metrics) + "\n")


def save_training_state(model, optimizer, save_dir: Path, step: int) -> Path:
    """Write each adapter to a new path so vLLM never serves stale weights."""
    adapter_dir = save_dir / f"step_{step:04d}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    state_tmp = save_dir / "state.pt.tmp"
    device = next(model.parameters()).device
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device),
    }, state_tmp)
    os.replace(state_tmp, save_dir / "state.pt")
    return adapter_dir


def load_resume_state(save_dir: Path):
    state_path = save_dir / "state.pt"
    if not state_path.exists():
        return None
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    adapter_dir = save_dir / f"step_{state['step']:04d}"
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError(f"resume adapter is missing: {adapter_dir}")
    return state, adapter_dir


def restore_rng_state(state: dict, model) -> None:
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "torch_rng" in state:
        torch.set_rng_state(state["torch_rng"])
    if "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"], next(model.parameters()).device)
