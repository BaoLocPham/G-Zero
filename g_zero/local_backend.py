"""Token-level local inference for G-Zero. No hosted model service is used."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer


@dataclass
class Generation:
    tokens: list[int]
    logprobs: list[float]
    finish_reason: str | None = None


class LocalBackend:
    """One vLLM engine on the first visible GPU, with optional PEFT adapters."""

    def __init__(self, config):
        from vllm import LLM

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, revision=config.model_revision
        )
        if not self.tokenizer.chat_template:
            raise ValueError(f"{config.model_name} has no tokenizer chat template")
        self.llm = LLM(
            model=config.model_name,
            revision=config.model_revision,
            tokenizer_revision=config.model_revision,
            seed=config.seed,
            dtype="bfloat16",
            tensor_parallel_size=config.vllm_tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            enforce_eager=config.vllm_enforce_eager,
            max_num_seqs=config.inference_batch_size,
            enable_lora=True,
            max_lora_rank=config.lora_rank,
            max_loras=16,
        )
        self._adapter_ids: dict[str, int] = {}

    def prompt(self, conversation: list[dict]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return list(ids)

    def encode_response(self, response: str) -> list[int]:
        return list(self.tokenizer.encode(response, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _lora_request(self, adapter_path: str | None):
        if adapter_path is None:
            return None
        from vllm.lora.request import LoRARequest

        path = str(Path(adapter_path).resolve())
        if not (Path(path) / "adapter_config.json").exists():
            raise FileNotFoundError(f"PEFT adapter not found: {path}")
        if path not in self._adapter_ids:
            self._adapter_ids[path] = len(self._adapter_ids) + 1
        adapter_id = self._adapter_ids[path]
        return LoRARequest(f"g_zero_{adapter_id}", adapter_id, path)

    def _check_length(self, ids: list[int], *, extra_tokens: int = 0) -> None:
        length = len(ids) + extra_tokens
        if length > self.config.max_model_len:
            raise ValueError(
                f"sequence needs {length} tokens; max_model_len={self.config.max_model_len}"
            )

    def generate_batch(
        self,
        prompts: list[list[int]],
        *,
        max_tokens: int,
        temperature: float,
        n: int = 1,
        adapter_path: str | None = None,
        return_logprobs: bool = False,
    ) -> list[list[Generation]]:
        from vllm import SamplingParams

        if not prompts:
            return []
        for prompt in prompts:
            self._check_length(prompt, extra_tokens=max_tokens)
        eos = self.tokenizer.eos_token_id
        stop_ids = [eos] if eos is not None else []
        for special in ("<|im_end|>", "<|eot_id|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(special)
            if isinstance(token_id, int) and token_id != self.tokenizer.unk_token_id:
                stop_ids.append(token_id)
        params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=1 if return_logprobs else None,
            stop_token_ids=list(dict.fromkeys(stop_ids)) or None,
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": p} for p in prompts],
            sampling_params=params,
            lora_request=self._lora_request(adapter_path),
            use_tqdm=False,
        )
        result: list[list[Generation]] = []
        for request in outputs:
            choices: list[Generation] = []
            for item in request.outputs:
                tokens = list(item.token_ids)
                logprobs: list[float] = []
                if return_logprobs:
                    if item.logprobs is None or len(item.logprobs) != len(tokens):
                        raise RuntimeError("vLLM did not return token-aligned generation logprobs")
                    for token, candidates in zip(tokens, item.logprobs):
                        if token not in candidates:
                            raise RuntimeError("sampled token missing from vLLM logprobs")
                        logprobs.append(float(candidates[token].logprob))
                choices.append(Generation(tokens, logprobs, item.finish_reason))
            result.append(choices)
        return result

    def continuation_logprobs_batch(
        self,
        pairs: list[tuple[list[int], list[int]]],
        *,
        adapter_path: str | None = None,
    ) -> list[list[float]]:
        """Teacher-force continuations; return only their token log probabilities."""
        from vllm import SamplingParams

        if not pairs:
            return []
        full = []
        for prompt, continuation in pairs:
            if not prompt or not continuation:
                raise ValueError("logprob scoring requires nonempty prompt and continuation")
            sequence = prompt + continuation
            # vLLM generates one throwaway token to expose prompt logprobs.
            self._check_length(sequence, extra_tokens=1)
            full.append(sequence)
        # prompt_logprobs includes the probability of the actual prompt token
        # at each position. The first prompt token has no predecessor.
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in full],
            sampling_params=SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1),
            lora_request=self._lora_request(adapter_path),
            use_tqdm=False,
        )
        scored: list[list[float]] = []
        for request, (prompt, continuation) in zip(outputs, pairs):
            entries = request.prompt_logprobs
            if entries is None or len(entries) != len(prompt) + len(continuation):
                raise RuntimeError("vLLM returned incomplete prompt logprobs")
            values = []
            for offset, token in enumerate(continuation, start=len(prompt)):
                candidates = entries[offset]
                if candidates is None or token not in candidates:
                    raise RuntimeError("target token missing from vLLM prompt logprobs")
                values.append(float(candidates[token].logprob))
            scored.append(values)
        return scored
