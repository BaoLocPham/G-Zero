"""Compare vLLM teacher-forced response logprobs to PyTorch on short text."""

import torch
from transformers import AutoModelForCausalLM

from g_zero.config import Config
from g_zero.local_backend import LocalBackend
from g_zero.local_training import response_logprobs


def main() -> None:
    config = Config(max_model_len=1024, inference_batch_size=2,
                    vllm_gpu_memory_utilization=0.2)
    backend = LocalBackend(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=torch.bfloat16, device_map={"": "cuda:1"}
    ).eval()
    pairs = [
        (backend.prompt([{"role": "user", "content": "What is 6 times 7?"}]),
         backend.encode_response("The answer is 42.")),
        (backend.prompt([{"role": "user", "content": "Say OK."}]),
         backend.encode_response("OK")),
    ]
    reference = backend.continuation_logprobs_batch(pairs)
    with torch.no_grad():
        local = [response_logprobs(model, prompt, answer).float().tolist()
                 for prompt, answer in pairs]
    diffs = [abs(a - b) for x, y in zip(reference, local)
             for a, b in zip(x, y)]
    maximum = max(diffs)
    print(f"compared {len(diffs)} response tokens; max absolute difference {maximum:.5f}",
          flush=True)
    if maximum > 0.1:
        raise RuntimeError("vLLM/PyTorch token logprobs disagree")


if __name__ == "__main__":
    main()
