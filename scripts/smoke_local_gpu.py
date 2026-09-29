"""Short local vLLM generation and teacher-forced scoring check."""

from g_zero.config import Config
from g_zero.local_backend import LocalBackend


def main() -> None:
    config = Config(max_model_len=1024, inference_batch_size=2,
                    vllm_gpu_memory_utilization=0.2)
    backend = LocalBackend(config)
    prompt = backend.prompt([{"role": "user", "content": "Reply with OK."}])
    assert prompt and all(isinstance(token, int) for token in prompt)
    generated = backend.generate_batch([prompt], max_tokens=8,
                                       temperature=0, return_logprobs=True)[0][0]
    assert generated.tokens and len(generated.tokens) == len(generated.logprobs)
    continuation = backend.encode_response("OK")
    scored = backend.continuation_logprobs_batch([(prompt, continuation)])[0]
    assert len(scored) == len(continuation)
    print("generation:", repr(backend.decode(generated.tokens)), flush=True)
    print("generated tokens:", len(generated.tokens), flush=True)
    print("scored tokens:", len(scored), flush=True)


if __name__ == "__main__":
    main()
