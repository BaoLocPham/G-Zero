# Local GPU migration plan

**Execution status (2026-09-29):** the local inference, training, evaluation,
checkpoint, and script paths are implemented. The 4B generation and
continuation-log-probability smoke test passed; vLLM and PyTorch logprobs
agreed within 0.1 on the checked tokens. A 32-attempt pilot produced a finite
DPO step. One-round bot50 and top50 adapters finished training. Base, bot50,
and top50 native-adapter mean@32 evaluations are complete. Their combined
AIME24/25 scores are 90/1920 (4.69%), 83/1920 (4.32%), and 98/1920
(5.10%), respectively. The full comparison with Agent-0 is in
[`reports/gzero_4b_agent0_mean32.md`](reports/gzero_4b_agent0_mean32.md).

## Compute-limited Qwen3-4B reproduction plan (current)

This is a **4B adaptation of the method**, not a reproduction of the paper's
8B scores. Use only `/home/locpb/self-evolution-repro/papers/agent0/shared-models/Qwen3-4B-Base`, two
visible GPUs (vLLM on the first, PEFT training on the second), one training
round, and AIME 2024/2025 evaluation. Report model, revision/source path,
seeds, prompt format, token limits, pool sizes, kept-pair counts, training
steps, GPU hours, and raw AIME results. Do not compare 4B scores to the 8B
paper as if model size and compute were controlled.

### Required checks before any long run

1. Finish the one-prompt vLLM generation and continuation-log-probability
   smoke test. Check that a sampled completion has token-aligned logprobs.
2. Compare continuation logprobs from vLLM and the local PyTorch model on two
   short examples, including the first response token. Check the sign of δ.
3. Run 16–32 generated pairs with Phase 1 off and short caps. Confirm valid
   `(q,h)` parsing, nonempty chosen/rejected answers, positive filtered count,
   and one finite DPO step. Reload the adapter and generate once. Measure
   actual wall time and peak memory before setting the larger pool size.

### Experiment matrix and order

| Order | Experiment | Training | Data / evaluation | Purpose |
| --- | --- | --- | --- | --- |
| 1 | Base 4B | None | AIME24+25, 1 sample/problem | Paired baseline for every trained run |
| 2 | Main 4B, bot50 | Phase 1 off; one DPO round | 256 generated `(q,h)` attempts initially; increase to 512 only if the pilot yield and runtime support it | Test whether the central δ-filtered DPO method improves AIME over base |
| 3 | Control, top50 | Reuse the **same** raw pool and train a fresh DPO adapter from base | Same AIME settings | Test whether selecting low δ matters at 4B |
| 4 | Phase 1 on (optional) | Tiny Challenger GRPO then fresh pool and DPO | Only if budget remains after orders 1–3 | Test whether Challenger training helps at 4B |

For order 2, start with `challenger_max_tokens=1024`,
`solver_max_tokens=2048`, `dpo_batch_size=4`, and at most 10 DPO steps.
These are pilot limits, not paper settings; raise the generation caps if
truncation prevents usable answers. Use a fresh tag for each changed config.
If 256 attempts produce fewer than 40 usable bot50 pairs, inspect parse and
filter failures before increasing the pool. Stop an experiment if the pilot
shows empty data, non-finite losses, or severe truncation.

For AIME, use the configurable sample count. Run one sample per problem first
as a quick diagnostic for base, bot50, and top50. The **primary reproduction
endpoint is mean@32 for base and bot50**, as requested, using the two GPUs
together with tensor parallel inference and bounded batches like Agent-0.
That is 1,920 completions per model across both years. Keep decoding settings
identical for base and bot50. Top50 mean@32 is optional if compute remains.
Evaluate on **q-only prompts** without hints. AIME has only 30 questions per
year, so small score changes are noisy; report correct counts and per-problem
outcomes, not only percentages. Do not tune on AIME results and then claim an
unbiased final score.

### Script and code adjustments before the matrix

- Add a configurable AIME sample count and use it in `g_zero/eval.py`.
- Add an eval-only base-model entry point (`sampler_path=None`) so order 1
  uses exactly the same AIME evaluator as the trained adapters.
- Make the no-Phase-1 and cutoff scripts use the cached 4B path, or pass an
  explicit local `--model_name`; their present defaults still name the 8B
  model. The cutoff script also needs its Phase-1 setting made consistent
  between pool creation and replay. For this matrix use Phase 1 off in both.
- Make `PYTHON_BIN=.venv/bin/python` explicit in run commands. Keep the pilot
  and real experiments under distinct tags so cached data cannot mask a
  changed configuration.
- Confirm GPU memory is free immediately before each run. Do not schedule the
  paper's 2,000/3,000-question sweeps, 3-round run, Llama run, or 32-sample
  AIME evaluation for this compute budget.

Success means the local 4B pipeline completes, saves reloadable adapters and
raw artifacts, and permits a paired base/bot50/top50 AIME comparison. A gain
would support a limited 4B replication of the method; no gain would still be
informative if the data, training, and scoring checks pass.

### Comparison with the completed Agent-0 run

Use the read-only artifacts in
`/home/locpb/self-evolution-repro/papers/agent0/evaluation/paper-aligned-aime-v1`.
Its `ACCEPTANCE.json` marks 12 completed AIME pools (30 problems × 64
completions for each of six conditions and two years). The conditions are
base-no-tool, base-with-tool, and Executor rounds 1–4 with the Python tool.
The same Qwen3-4B-Base model directory is available in that repo. Do not run
Agent-0 again merely to build this comparison.

Primary comparison: G-Zero base and G-Zero bot50 versus Agent-0
`base-no-tool` on AIME24/25. Show separate year and combined scores, plus
G-Zero's change from its own base and Agent-0's change from its own base.
The Agent-0 no-tool base is a cross-check on the baseline; differences in
prompting, decoding, token limits, or answer parsing must be disclosed.
Secondary context: show Agent-0 `base-with-tool` and `executor-r4-with-tool`
as tool-enabled results in separate rows. Do not present their absolute score
gap versus G-Zero as a controlled algorithm comparison, since tool access and
training differ. Include rounds 1–3 in an appendix or chart if useful.

To make the comparison meaningful:

1. Use the exact 30-question files in the Agent-0 `benchmarks/aime_2024.json`
   and `benchmarks/aime_2025.json`, or verify problem text and gold answers
   match before using G-Zero's Hugging Face mirrors. Save dataset hashes.
2. Score G-Zero's saved completions with Agent-0's conservative explicit-answer
   parser (`scripts/aime_answer.py`) as a shared comparison metric. G-Zero's
   current fallback to the last bare integer can overcount unfinished answers;
   retain raw completions and report any parser disagreement.
3. For the comparison evaluation, match Agent-0's no-tool decoding where
   practical: temperature 0.7, top-p 0.95, and 2,048 generated tokens. Record
   the prompt and tool policy; Agent-0's no-tool prompt forbids external tools.
   Keep all G-Zero conditions under one fixed evaluator and seed policy.
4. Compare G-Zero base and bot50 `mean@32` with Agent-0's saved `mean@32` as
   the primary table. Show `mean@1` for all three G-Zero conditions as a
   quick diagnostic, with Agent-0's corresponding `mean@1` clearly labeled.
5. Report per-problem correctness and uncertainty or at least exact counts.
   The comparison should identify the artifacts and scoring script version so
   it can be reproduced without another Agent-0 GPU run.

Agent-0's completed report lists combined AIME24/25 `mean@32` of 3.96% for
base-no-tool, 5.31% for base-with-tool, and 7.19% for Executor round 4 with
tool. These are reference results from the saved Agent-0 protocol, not
predictions for the G-Zero run or like-for-like scores at the proposed k=1/4.

## Goal and scope

Run the complete G-Zero pipeline on local NVIDIA GPUs without a Tinker account:
Challenger generation and optional GRPO, Solver generation and hint scoring,
Solver DPO, checkpointing, multi-round runs, and AIME evaluation.
AlpacaEval and all LLM judge calls are out of scope. Keep the existing reward,
filtering, DPO objective, and output formats wherever practical.

At planning time, this host had four H200 GPUs with about 140 GiB each. GPUs 2
and 3 were free; GPUs 0 and 1 were occupied. Recheck availability before runs.
The current local run uses the cached BF16 Qwen3-4B-Base model at
`/home/locpb/self-evolution-repro/papers/agent0/shared-models/Qwen3-4B-Base` with LoRA rank 32. The
original paper-main target is Qwen3-8B-Base. Do not assume the current
Tinker wall-clock estimates apply locally.

## Proposed local stack

| Need | Local implementation |
| --- | --- |
| Batched generation | vLLM offline inference, with base model and PEFT LoRA adapters |
| Teacher-forced token log probabilities | vLLM prompt log probabilities or a batched PyTorch forward pass, selected after token-level parity checks |
| LoRA training | PyTorch + Transformers + PEFT, with gradient accumulation and BF16 |
| Tokenization and chat formatting | Hugging Face tokenizer and model chat template, with explicit checks against the current renderer's prompt and stop-token behavior |
| Checkpoints | Local PEFT adapter directories plus optimizer, scheduler, RNG, configuration, and progress state |

Use a small internal interface such as `generate(prompt_token_ids, params, n)`
and `continuation_logprobs(prompt_token_ids, continuation_token_ids)`. It must
return generated token IDs, token-aligned log probabilities, and stop reasons.
Keep model loading, LoRA selection, batching, and device placement behind this
interface so the phase and evaluation code does not depend on a serving engine.

Official references: [vLLM LoRA](https://docs.vllm.ai/en/latest/features/lora/),
[vLLM log probabilities](https://docs.vllm.ai/en/latest/api/vllm/logprobs/),
[PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), and
[TRL DPO](https://huggingface.co/docs/trl/dpo_trainer). Pin and test compatible
package versions during implementation.

## Implementation sequence

### 1. Establish a reproducible local baseline

- Record the model revision, tokenizer revision, random seeds, generation
  parameters, prompt token IDs, EOS and stop handling, and maximum context
  length. Output caps of 8,192 tokens during data generation and 16,384 during
  AIME evaluation require explicit context and memory limits.
- Keep Phase 1 enabled in the existing main scripts. The no-Phase-1 ablation
  remains available through `--run_phase1 false`; the original release used
  both descriptions of its paper-main setting.
- Set up a virtual environment and a local dependency set. Remove the Tinker
  API-key check from `run.sh`. This host currently has `python3` but no `python`
  executable on `PATH`; use the virtual environment interpreter in scripts.

**Done when:** the base Qwen model loads on an available GPU and a single
Challenger and Solver prompt can be generated with recorded token IDs.

### 2. Replace Phase 2 generation and hint scoring

- Port `g_zero/phase2.py` and `g_zero/hint_delta.py` to the local interface.
  Generate `(q, h)`, `a_hard` under `q`, and `a_assisted` under `(q, h)`.
- Compute the same `a_hard` continuation's mean token log probability under
  both contexts. Preserve `delta = logp_q - logp_qh`, including the response
  token mask and context boundary. Check for off-by-one shifts around the first
  response token and EOS.
- Replace the current unbounded `asyncio.gather` over the entire question pool
  with bounded batches. Append completed records to `raw_pool.jsonl` safely,
  and resume without duplicating records after interruption. Keep the existing
  percentile and structural filters and `dpo_data.jsonl` schema.

**Done when:** a 32-question no-Phase-1 run produces a resumable raw pool and
nonempty filtered DPO data. Independent PyTorch scoring agrees with the chosen
backend on token log probabilities within a documented numeric tolerance.

### 3. Replace Solver DPO training

- Port `g_zero/phase3.py` to a trainable PEFT LoRA adapter. Freeze the Solver
  state at DPO start as `pi_ref`; for later rounds, that state is the previous
  round's Solver, not the original base model.
- Preserve the current objective: question-only prompt; chosen and rejected
  response log probabilities averaged over response tokens; frozen-reference
  log-ratio; `-logsigmoid(beta * (chosen_ratio - rejected_ratio))` with beta 2.0
  by default. A stock trainer is acceptable only after its masking and
  normalization match this implementation.
- Add gradient accumulation and configurable microbatch size without changing
  the effective batch size. Save adapter and training state at intervals and at
  completion. Return a local adapter path rather than a Tinker URI.

**Done when:** one DPO step has finite loss and gradients, the saved adapter
reloads for generation, and an interrupted run resumes at the correct step.

### 4. Replace optional Challenger GRPO training

- Port `g_zero/phase1.py` after the Phase 2 and DPO path works. Keep its reward
  components, invalid-format penalty, BLEU cluster share, and per-group
  mean-centered Dr.GRPO advantages.
- Use a local trainable Challenger LoRA. Collect generation token IDs and old
  policy log probabilities, compute the current policy log probabilities with
  gradients, apply the intended importance-sampling objective, then update the
  adapter. Refresh the generation engine's adapter after each update.
- Keep the Solver used for delta scoring frozen throughout each Challenger
  stage. Check group sizes, response masks, and update metrics on a tiny run.

**Done when:** a two-step GRPO smoke run completes with finite rewards and
losses, valid `(q, h)` output, and a reloadable Challenger adapter.

### 5. Port orchestration, multi-round state, and evaluation

- Update `g_zero/main.py`, `g_zero/multi_round.py`, `g_zero/eval.py`,
  `scripts/eval_only.sh`, and experiment scripts to accept local model and
  adapter paths. Make checkpoint detection reflect actual files; the current
  `main.py` docstring promises solver-checkpoint skipping that its `run()` does
  not implement.
- Run AIME 2024 and 2025 with the local generation interface. Cache the raw
  completions so rescoring does not require sampling again.
- Set `Config.eval_tasks` and local scripts to `aime24,aime25`. Remove IFEval,
  AlpacaEval, and judge code from the local pipeline. It must not download
  judge weights or require a judge API.
- Document model downloads, GPU selection, command examples, resume behavior,
  and local artifact layout. Remove Tinker and tinker-cookbook dependencies
  after their renderer and logging uses are replaced.

**Done when:** one round completes without `TINKER_API_KEY`, writes local
Challenger/Solver checkpoints and evaluation artifacts, and `eval_only.sh`
accepts a local adapter path. A two-round smoke run resumes correctly.

## Validation gates before a full 2,000-question run

1. Compare prompt token IDs and teacher-forced log probabilities for several
   base-model examples, including stop/EOS boundaries and variable lengths.
2. Verify delta sign and response-only averaging with hand-checked examples.
3. Verify DPO masking, frozen-reference behavior, and finite gradients.
4. Run a 32-question pool, one DPO step, and a checkpoint reload/resume test.
5. Run a modest pool with AIME before the 2,000-question configuration.
   Review valid-pair rate, delta distribution, filtered count, throughput,
   GPU memory, DPO metrics, and generation quality.

For the first end-to-end milestone, use `--run_phase1 false`. That exercises
the central G-Zero data and DPO path while Phase 1 is being ported. The
completed migration includes optional Phase 1 and AIME only.
