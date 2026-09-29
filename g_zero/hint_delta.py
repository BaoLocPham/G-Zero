"""Hint-effectiveness delta from a frozen local Solver.

delta(q, h, a_hard) = mean log p(a_hard | q) - mean log p(a_hard | q, h).
The same sampled continuation tokens are scored under both contexts.
"""
from __future__ import annotations

from dataclasses import dataclass

from .prompts import build_solver_convo


@dataclass
class QHScore:
    question: str
    hint: str
    a_hard: str
    a_assisted: str
    a_hard_tokens: list[int]
    logp_q: float
    logp_qh: float
    delta: float


def score_batch(
    backend,
    qh_pairs: list[tuple[str, str]],
    *,
    mode: str = "full",
    adapter_path: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    batch_size: int = 16,
) -> list[QHScore]:
    if mode not in {"full", "delta_only"}:
        raise ValueError(f"unknown scoring mode: {mode}")
    result: list[QHScore] = []
    for start in range(0, len(qh_pairs), batch_size):
        chunk = qh_pairs[start : start + batch_size]
        q_prompts = [backend.prompt(build_solver_convo(q)) for q, _ in chunk]
        qh_prompts = [backend.prompt(build_solver_convo(q, h)) for q, h in chunk]
        hard = backend.generate_batch(
            q_prompts, max_tokens=max_tokens, temperature=temperature,
            adapter_path=adapter_path,
        )
        assisted = (
            backend.generate_batch(
                qh_prompts, max_tokens=max_tokens, temperature=temperature,
                adapter_path=adapter_path,
            ) if mode == "full" else None
        )
        valid = [i for i, group in enumerate(hard) if group[0].tokens]
        logp_q = [[] for _ in chunk]
        logp_qh = [[] for _ in chunk]
        if valid:
            logp_q_valid = backend.continuation_logprobs_batch(
                [(q_prompts[i], hard[i][0].tokens) for i in valid],
                adapter_path=adapter_path,
            )
            logp_qh_valid = backend.continuation_logprobs_batch(
                [(qh_prompts[i], hard[i][0].tokens) for i in valid],
                adapter_path=adapter_path,
            )
            for i, a, b in zip(valid, logp_q_valid, logp_qh_valid):
                logp_q[i], logp_qh[i] = a, b
        for i, (q, h) in enumerate(chunk):
            tokens = hard[i][0].tokens
            a = sum(logp_q[i]) / len(logp_q[i]) if logp_q[i] else 0.0
            b = sum(logp_qh[i]) / len(logp_qh[i]) if logp_qh[i] else 0.0
            result.append(QHScore(
                question=q, hint=h,
                a_hard=backend.decode(tokens),
                a_assisted=backend.decode(assisted[i][0].tokens) if assisted else "",
                a_hard_tokens=tokens,
                logp_q=a, logp_qh=b, delta=a - b,
            ))
    return result
