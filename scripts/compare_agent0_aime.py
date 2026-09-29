"""Compare saved G-Zero AIME completions with completed Agent-0 pools."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g_zero.eval_aime import parse_aime_answer


AGENT0 = Path(
    "/home/locpb/self-evolution-repro/papers/agent0/evaluation/paper-aligned-aime-v1"
)
LABELS = ("base-no-tool", "base-with-tool", "executor-r4-with-tool")
YEARS = (("aime24", "AIME-2024"), ("aime25", "AIME-2025"))


def gzero_scores(run_dir: Path, k: int) -> list[tuple[int, int]]:
    scores = []
    for short, _ in YEARS:
        path = run_dir / "eval" / f"results_{short}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        groups: dict[int, dict[int, dict]] = {}
        for row in rows:
            groups.setdefault(int(row["problem_index"]), {})[int(row["sample_id"])] = row
        if len(groups) != 30 or any(set(samples) < set(range(k)) for samples in groups.values()):
            raise RuntimeError(f"{path} lacks {k} samples for each of 30 questions")
        selected = [groups[index][sample] for index in sorted(groups)
                    for sample in range(k)]
        scores.append((sum(
            parse_aime_answer(row["completion"]) == row["answer"]
            for row in selected
        ), len(selected)))
    return scores


def agent0_scores(label: str, k: int) -> list[float]:
    return [float(json.loads((AGENT0 / "metrics" / f"{label}-{year}.json")
                             .read_text())["scores"][f"mean_at_{k}"])
            for _, year in YEARS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bot50", type=Path, required=True)
    parser.add_argument("--top50", type=Path)
    parser.add_argument("--k", type=int, choices=(1, 4, 32), default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gzero = {
        "G-Zero base, no tool": gzero_scores(args.base, args.k),
        "G-Zero bot50, no tool": gzero_scores(args.bot50, args.k),
    }
    if args.top50:
        gzero["G-Zero top50, no tool"] = gzero_scores(args.top50, args.k)
    agent0 = {
        "Agent-0 base, no tool": agent0_scores("base-no-tool", args.k),
        "Agent-0 base, Python tool": agent0_scores("base-with-tool", args.k),
        "Agent-0 Executor R4, Python tool": agent0_scores("executor-r4-with-tool", args.k),
    }
    lines = [f"# G-Zero 4B and Agent-0 AIME mean@{args.k}", "",
             f"Exact-answer accuracy over the first {args.k} saved completions per question.",
             "Each year has 30 questions. G-Zero and Agent-0 use the same local",
             "Qwen3-4B-Base model, benchmark files, no-tool instruction, answer",
             "parser, temperature 0.7, top-p 0.95, and 2,048-token cap. G-Zero",
             "uses the Agent-0 no-tool per-question seeds; tool rows use other seeds.",
             "Both evaluations used tensor parallelism over two GPUs.",
             "Tool-enabled rows are context, not controlled",
             "algorithm comparisons.", "",
             "| Condition | AIME24 | AIME25 | Combined |", "| --- | ---: | ---: | ---: |"]
    for label, ((correct24, total24), (correct25, total25)) in gzero.items():
        lines.append(f"| {label} | {correct24}/{total24} ({correct24/total24:.2%}) | "
                     f"{correct25}/{total25} ({correct25/total25:.2%}) | "
                     f"{correct24+correct25}/{total24+total25} "
                     f"({(correct24+correct25)/(total24+total25):.2%}) |")
    for label, (a, b) in agent0.items():
        lines.append(f"| {label} | {a:.2%} | {b:.2%} | {(a+b)/2:.2%} |")
    base = sum(c for c, _ in gzero["G-Zero base, no tool"]) / (60 * args.k)
    bot = sum(c for c, _ in gzero["G-Zero bot50, no tool"]) / (60 * args.k)
    lines += ["", f"G-Zero bot50 minus its base: {(bot-base)*100:+.2f} percentage points."]
    if args.top50:
        top = sum(c for c, _ in gzero["G-Zero top50, no tool"]) / (60 * args.k)
        lines.append(f"G-Zero top50 minus its base: {(top-base)*100:+.2f} percentage points.")
    lines += ["", "Source artifacts:", f"- G-Zero base: `{args.base.resolve()}`",
              f"- G-Zero bot50: `{args.bot50.resolve()}`"]
    if args.top50:
        lines.append(f"- G-Zero top50: `{args.top50.resolve()}`")
    lines += [f"- Agent-0: `{AGENT0}`", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
