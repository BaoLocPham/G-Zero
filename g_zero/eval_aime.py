"""Fast, bounded AIME evaluation using Agent-0's saved benchmark protocol."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

from vllm import SamplingParams

logger = logging.getLogger(__name__)

AGENT0_ROOT = Path("/home/locpb/self-evolution-repro")
DATASETS = {
    "aime24": AGENT0_ROOT / "benchmarks/aime_2024.json",
    "aime25": AGENT0_ROOT / "benchmarks/aime_2025.json",
}
NO_TOOL_SYSTEM = (
    "Solve using only reasoning in this response. Do not call, simulate, or rely "
    "on any external tools, code, browser, or calculator."
)
_BOXED = re.compile(
    r"\\boxed\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?([0-9]{1,3})"
    r"\s*(?:\}\s*)?\}"
)
_EXPLICIT = re.compile(
    r"(?:^|\n)[ \t#>*_`]*(?:final[ \t]+)?answer"
    r"[ \t*_`]*[:=][ \t$*_`]*([0-9]{1,3})(?![0-9])",
    re.IGNORECASE,
)


def parse_aime_answer(text: str) -> str | None:
    """Match Agent-0's explicit-answer parser; ignore bare trailing integers."""
    boxed = _BOXED.findall(text)
    if boxed:
        return str(int(boxed[-1]))
    explicit = _EXPLICIT.findall(text)
    return str(int(explicit[-1])) if explicit else None


def _load_aime(year: str) -> tuple[list[dict], str]:
    path = DATASETS[year]
    raw = path.read_bytes()
    rows = json.loads(raw)
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 AIME questions in {path}, got {len(rows)}")
    return rows, hashlib.sha256(raw).hexdigest()


def run_aime(
    backend,
    *,
    year: str,
    samples_per_problem: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 2026092201,
    max_tokens: int = 2048,
    batch_size: int = 8,
    adapter_path: str | None = None,
    out_path: Path | None = None,
) -> float:
    """Generate seeded completions in bounded batches and return mean@k."""
    if samples_per_problem < 1:
        raise ValueError("samples_per_problem must be positive")
    problems, dataset_sha256 = _load_aime(year)
    expected = len(problems) * samples_per_problem
    cached: dict[tuple[int, int], dict] = {}
    if out_path and out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["problem_index"], row["sample_id"])
            if key in cached or row["dataset_sha256"] != dataset_sha256:
                raise RuntimeError(f"incompatible AIME cache at {out_path}")
            cached[key] = row

    pending = []
    for item in problems:
        problem_index = int(item["index"])
        question = item["prompt"][-1]["content"]
        prompt = backend.prompt([
            {"role": "system", "content": NO_TOOL_SYSTEM},
            {"role": "user", "content": question},
        ])
        backend._check_length(prompt, extra_tokens=max_tokens)
        problem_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{problem_index}".encode()).digest()[:4],
            "big",
        )
        for sample_id in range(samples_per_problem):
            if (problem_index, sample_id) not in cached:
                pending.append((item, sample_id, prompt, problem_seed + sample_id))

    logger.info("[%s] %d/%d completions pending", year, len(pending), expected)
    if pending:
        if out_path is None:
            raise ValueError("out_path is required to save AIME completions")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as handle:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                outputs = backend.llm.generate(
                    [{"prompt_token_ids": row[2]} for row in batch],
                    sampling_params=[
                        SamplingParams(n=1, temperature=temperature, top_p=top_p,
                                       max_tokens=max_tokens, seed=row[3])
                        for row in batch
                    ],
                    lora_request=backend._lora_request(adapter_path),
                    use_tqdm=False,
                )
                for (item, sample_id, _, seed), output in zip(batch, outputs):
                    completion = backend.decode(list(output.outputs[0].token_ids))
                    gold = str(int(item["answer"]))
                    parsed = parse_aime_answer(completion)
                    record = {
                        "problem_index": int(item["index"]),
                        "sample_id": sample_id,
                        "seed": seed,
                        "answer": gold,
                        "completion": completion,
                        "parsed_answer": parsed,
                        "correct": parsed == gold,
                        "token_count": len(output.outputs[0].token_ids),
                        "dataset_sha256": dataset_sha256,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cached[(record["problem_index"], sample_id)] = record
                handle.flush()
                os.fsync(handle.fileno())
                logger.info("[%s] saved %d/%d completions", year,
                            min(start + batch_size, len(pending)), len(pending))

    if len(cached) != expected:
        raise RuntimeError(f"AIME cache has {len(cached)} rows, expected {expected}")
    correct = sum(bool(row["correct"]) for row in cached.values())
    accuracy = correct / expected
    logger.info("[%s] mean@%d = %.4f (%d/%d)", year, samples_per_problem,
                accuracy, correct, expected)
    return accuracy
