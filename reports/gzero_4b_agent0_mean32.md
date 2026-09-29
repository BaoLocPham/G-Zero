# G-Zero 4B and Agent-0 AIME mean@32

Exact-answer accuracy over the first 32 saved completions per question.
Each year has 30 questions. G-Zero and Agent-0 use the same local
Qwen3-4B-Base model, benchmark files, no-tool instruction, answer
parser, temperature 0.7, top-p 0.95, and 2,048-token cap. G-Zero
uses the Agent-0 no-tool per-question seeds; tool rows use other seeds.
Both evaluations used tensor parallelism over two GPUs.
Tool-enabled rows are context, not controlled
algorithm comparisons.

| Condition | AIME24 | AIME25 | Combined |
| --- | ---: | ---: | ---: |
| G-Zero base, no tool | 45/960 (4.69%) | 45/960 (4.69%) | 90/1920 (4.69%) |
| G-Zero bot50, no tool | 48/960 (5.00%) | 35/960 (3.65%) | 83/1920 (4.32%) |
| G-Zero top50, no tool | 55/960 (5.73%) | 43/960 (4.48%) | 98/1920 (5.10%) |
| Agent-0 base, no tool | 4.17% | 3.75% | 3.96% |
| Agent-0 base, Python tool | 6.56% | 4.06% | 5.31% |
| Agent-0 Executor R4, Python tool | 9.06% | 5.31% | 7.19% |

G-Zero bot50 minus its base: -0.36 percentage points.
G-Zero top50 minus its base: +0.42 percentage points.

Source artifacts:
- G-Zero base: `/home/locpb/G-Zero/runs/gzero_4b_base_mean32_20260929`
- G-Zero bot50: `/home/locpb/G-Zero/runs/gzero_4b_bot50_mean32_20260929`
- G-Zero top50: `/home/locpb/G-Zero/runs/gzero_4b_top50_native_mean32_20260929`
- Agent-0: `/home/locpb/self-evolution-repro/papers/agent0/evaluation/paper-aligned-aime-v1`

## Run details

- Model: local Qwen3-4B-Base at `/home/locpb/self-evolution-repro/papers/agent0/shared-models/Qwen3-4B-Base`. This is a 4B adaptation; the paper's main Qwen run used 8B.
- Training: one round with Phase 1 disabled. From 256 Challenger attempts, 174 valid question/hint pairs entered the shared raw pool. Bot50 retained 64 DPO pairs and top50 retained 69; each LoRA rank-32 adapter trained for 10 DPO steps. Challenger and Solver generation caps were 768 and 1,024 tokens, respectively. These are compute-limited settings.
- Evaluation: each condition used its base model or **native LoRA adapter**, with 32 independent seeded completions per question, temperature 0.7, top-p 0.95, 2,048 generated-token cap, and two-GPU tensor parallel vLLM. No external API, judge, or tools were used for G-Zero. AIME 2024 and 2025 each contain 30 questions.
- Benchmark SHA-256: AIME 2024 `ae6ee1ae34846c11190ceea5933fbb8d68fe588e86e59277d61e6f91ef666069`; AIME 2025 `0b259288cf20faa73f5e34e00bec2eb5db93bc8193b5421332421be185e9d930`.
- GPU assignment: native top50 AIME 2024 ran on GPUs 0–1; native bot50 and then top50 AIME 2025 ran on GPUs 2–3. The two top50 years were combined from those saved result files. The top50 AIME 2025 file is linked from `gzero_4b_top50_native_aime25_parallel_20260929`.

The bot50 result does not show the expected gain on this 4B, one-round run. The 7/1,920 bot50-base difference is small relative to variation across only 60 AIME questions. The top50 control did better than bot50, so this run does not support a benefit from selecting low-δ examples under these compute limits. The Agent-0 tool-enabled rows change tool access as well as training and should only be read as context. Agent-0's own Executor R4 improvement over its tool-enabled base is 1.88 percentage points.

The 4B adapter and evaluation artifacts are under `runs/` and are excluded from Git by `.gitignore`. The report and comparison script are kept in the repository.
