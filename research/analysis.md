# Critical Analysis of the Four Research Reports

This document reviews the four Deep Research reports in this folder. It decides which findings are worth implementing for Second. The implementation plan is in [TRAINING_PLAN.md](../TRAINING_PLAN.md).

| Report | Topic |
|---|---|
| [01](01-post-training-methods-for-small-models.md) | Post-training methods for small models (LoRA vs full fine-tuning, DAPO, S-GRPO, GR3, MO-GRPO, teacher distillation) |
| [02](02-length-collapse-and-reward-hacking.md) | Length collapse and reward hacking with rule verifiers (CMDP, Lagrange multipliers, length-controlled eval) |
| [03](03-training-data-strategies.md) | SFT data strategies (engine-grounded teacher data, data selection, cognitive templates) |
| [04](04-evaluation-framework.md) | Evaluation framework (judge bias, calibration against humans, release gates) |
| [00](00-research-brief.md) | The project brief that we gave to the research tool |

---

## 1. Summary

1. **The main problem is the SFT data, not the RL algorithm.** Our own numbers show this (section 2). The reports agree in part (reports 01 and 03), but reports 01 and 02 spend most of their length on RL math that solves a smaller problem.
2. **The best idea in the reports:** make new SFT data with a strong teacher model. The teacher gets the engine facts and writes the explanation. Our rule checks then reject bad outputs (reports 01 and 03). We adopt this. We do **not** use Gemini as the teacher, because the Gemini API terms forbid it (section 4).
3. **Cheap RL fixes are worth doing:** token-level loss (DAPO / Dr. GRPO), per-reward normalization (GDPO), graded penalties, and length reward that counts only when the content is correct. The TRL library already has most of these as settings.
4. **Eval fixes are worth doing:** a judge prompt that ignores length, length-controlled win rates, a separate dev set for experiments, and a small human calibration set.
5. **Many recommendations do not fit Second.** Several assume a model that *chooses* moves (top-1 move match, centipawn loss, puzzle accuracy, "FINAL_ANSWER: uci_move"). Second does not choose moves. Stockfish does. We reject these.

---

## 2. What our own data shows

The reports had only the short brief in [00-research-brief.md](00-research-brief.md). We checked the Phase 6 outputs and the SFT data ourselves. These numbers change the diagnosis.

| Measurement | Value | Source |
|---|---|---|
| SFT training examples | 5,522 | `data/processed/sft_train.jsonl` |
| Median words per SFT comment | 19 (only 34% have ≥ 25 words) | same |
| SFT examples with `detail: long` | **36 (0.7%)** | same |
| SFT comments in first person ("I", "my", "me") | 22.7% | same |
| Average words: SFT model → GRPO model | 23.6 → 30.7 | `eval/reports/grpo-1.0-2b.md` |
| GRPO outputs in first person | **51.3%** | `eval/results/grpo-1.0-2b_outputs.jsonl` |
| GRPO outputs that name the best move, on bad moves | **15.6%** | same |
| GRPO outputs that name a tactic (fork, pin, mate, hanging, …) | **4.3%** | same |
| Key-point score: SFT → GRPO | 0.40 → 0.39 | eval report |
| Prompts per GRPO step | **1** (8 completions per step, 8 per prompt) | `training/configs/second-1.0-2b.yaml` |
| Prompts seen in the whole GRPO run | ~450 of 20,000 | same |
| Judge swap agreement vs base Qwen | 30.7% (very noisy) | eval report |

### 2.1 What this means

1. **There was no RL length collapse.** The SFT model was already short, because the human comments are short. RL made the answers *longer* (23.6 → 30.7 words). Reports 01 and 02 explain "length collapse in GRPO" in detail. That mechanism can be real, but it is not our main cause. The short answers come from the data.
2. **The model does not use the facts.** Human commentators wrote their comments without our fact lines. So SFT taught the model to write *about* chess, not to explain *the facts in the prompt*. Example output for a blunder where the facts say "White has a check: Bxd7+": *"I was trying to get my knight to a better square, but I didn't see the point of moving it."*
3. **RL found a safe style: vague first-person text.** First-person text doubled under RL (22.7% → 51.3%). Vague text makes no checkable claims, so it rarely gets −1. This matches the "risk accumulation" idea in report 02: each concrete claim is a new chance to fail a hard check. So the policy avoids claims.
4. **The reward does not check most claims.** It checks moves, piece locations, and verdict tone. It does **not** check "protects the pawn on e4", "attacks the queen", "pins", "forks", or opening names. The judge found invented openings ("Petrov's Defence", "Sicilian Dragon", "London System") in the 20-item review.
5. **The GRPO batch was too small.** One prompt per step gives very noisy updates. The run saw only ~2% of the RL prompts. This is a simple config fix.
6. **Long mode had almost no training data.** 36 examples cannot teach a 80–200 word answer.

---

## 3. Quality of the reports

Read the reports with care. Problems found:

- **Unverified citations and numbers.** Many citations point to recent arXiv IDs and blog posts that we cannot check from here. Treat every number as "claimed", not as "known".
- **Numbers from a different task.** Report 01's LoRA vs full fine-tuning table (32.1% vs 48.1% "tactical accuracy") comes from models that *predict moves*. Second explains engine facts. The number does not transfer.
- **Goals for a move-playing model.** Report 04's release gates (top-1 move match > 85%, centipawn loss < 15, puzzle accuracy, P99 latency < 250 ms) do not apply. Second never picks a move. Our runtime (wllama on CPU) gives ~14 tokens per second on 2B, so 250 ms per turn is impossible and not needed.
- **Advice that breaks our design.** Report 01 says to remove engine facts from the student prompt. Our extension has Stockfish at inference time. The facts are the source of truth. We keep them.
- **Heavy math for small gains.** DMPO, FlowBalance, TTRL-CoCoV, Nash / SPPO preference games, and augmented Lagrangians add much code and risk. They target problems we do not have.
- **Wrong mechanism detail.** Report 01 says the 1/|o| term in GRPO makes short answers win. Per the Dr. GRPO paper, that term favors short answers when the advantage is positive and long answers when the advantage is negative. The fix (token-level loss) is still good. The explanation is too simple.

The reports are still useful. Their best ideas (teacher data, graded rewards, length-controlled eval, judge calibration) are sound and well known.

---

## 4. Teacher model and license

Reports 01 and 03 suggest "Gemini 3 Flash or GPT-4o" as the teacher. **Do not use Gemini or another closed API as the teacher.**

- The Gemini API Additional Terms say: *"You may not use the Services to develop models that compete with the Services."* Second is an open model. Training it on Gemini outputs is a license risk for the whole release.
- Most closed APIs have similar terms.
- Gemma has its own terms. A model trained on Gemma outputs can count as a "Model Derivative" under those terms. This conflicts with our Apache 2.0 release.

**Decision:** use a large **Apache 2.0 open-weight model** as the teacher. The first choice is a large Qwen 3.5 model (same family, Apache 2.0, no output restrictions). TRAINING_PLAN.md has a spike that picks the exact model. Gemini stays the **judge only**. This also prevents self-preference bias: the judge and the teacher are from different model families (report 04).

> [!NOTE]
> This is not legal advice. Check the license and terms of the chosen teacher before you publish data or models.

---

## 5. Verdict per finding

**Adopt** = implement. **Test** = run as an experiment and keep it only if it wins on the dev set. **Defer** = maybe later. **Reject** = do not do.

### 5.1 Data (reports 01 and 03)

| Finding | Verdict | Reason | Plan section |
|---|---|---|---|
| Engine-grounded teacher data ("master distillation"): teacher gets facts, writes the explanation | **Adopt** | Fixes the main cause (section 2.1). Our facts already give the teacher the truth. | TRAINING_PLAN §6 |
| Reject teacher outputs with our rule checks (rejection sampling) | **Adopt** | We already have the checks. Cheap and strong. | §6.4 |
| Taxonomy-guided coverage (forks, pins, endgames, …) | **Adopt** | Usefulness is low because the model never names ideas. Coverage must be planned. Needs motif facts first. | §5.1, §6.3 |
| "Feigned discovery" prompting (explain why the natural move fails) | **Adopt (partly)** | Good teaching style for mistakes ("Nf6 looks natural, but…"). Keep it short. | §6.2 |
| Remove engine facts from the student prompt | **Reject** | Breaks our design. Stockfish runs at inference. | — |
| Small, high-quality set (LIMA / LIMO idea, 3k–6k gold examples) | **Test** | Plausible. We test 2k / 5k / 10k sizes. | §7.2 |
| IFD / S-IFD / DEITA data selection | **Defer** | Rule checks + judge filter + dedupe do most of this. Add only if data size ablations show noise matters. | §6.5 |
| Rejection-sampling fine-tuning (RFT) on our own model's outputs | **Test** | Cheap self-improvement round. Risk: low diversity. Compare with GRPO. | §8.6 |
| Critique fine-tuning on failed RL groups | **Defer** | Complex. Teacher data should remove most failures. | — |
| Cognitive template inside `<think>` | **Test** | Fits the reasoning-mode idea. | §9 |
| Bridge mechanism with "FINAL_ANSWER: uci_move" | **Reject** | For move-choosing models. | — |

### 5.2 Training method (reports 01 and 02)

| Finding | Verdict | Reason | Plan section |
|---|---|---|---|
| DAPO token-level loss / Dr. GRPO | **Adopt** | One config line in TRL (`loss_type`). Removes length bias in the loss. Set it explicitly; Phase 6 did not. | §8.2 |
| Per-reward normalization (MO-GRPO / GDPO) | **Test** | TRL has `multi_objective_aggregation="normalize_then_sum"`. Needs split reward functions. Cheap ablation. | §8.3 |
| Graded penalties instead of flat −1 | **Adopt** | Directly attacks the "vague is safe" behavior (section 2.1). | §8.1 |
| Length reward gated on correct content (GR3 / "correct-only" idea) | **Adopt** | Stops padding. Simple to implement in our scorer. | §8.1 |
| Adaptive penalty weight (CMDP Lagrange multiplier) | **Test** | Good idea, small code. Only if fixed graded penalties fail to balance length and correctness. | §8.4 |
| Monitor length spread (σ of length per group) and entropy | **Adopt** | Cheap early warning. | §8.5 |
| Bigger RL batch (more prompts per step) | **Adopt** | Not in the reports. Phase 6 used 1 prompt per step. | §8.2 |
| Full fine-tuning instead of LoRA | **Test (low priority)** | Plausible for SFT. Needs A100 and 8-bit optimizer. Test after data is fixed: LoRA r32 / r128 / full. | §7.3 |
| S-GRPO ground-truth injection | **Defer** | Our groups are rarely all −1 (hard-fail ~3–5% during training). Low value. | — |
| LLM judge with rubric dropout inside RL | **Defer** | Breaks our "no LLM judge in RL" decision and the budget. Revisit only with a local judge model if usefulness stays low. | §8.7 |
| DMPO, FlowBalance, SimPO, TTRL-CoCoV, entropy-regularized primal-dual | **Reject** | High complexity, no clear gain for our failure modes. | — |
| PPO with a critic | **Reject** | More memory, less stable at 2B. GRPO is fine. | — |

### 5.3 Evaluation (reports 02 and 04)

| Finding | Verdict | Reason | Plan section |
|---|---|---|---|
| Length-neutral judge prompt | **Adopt** | Base Qwen ties us because it writes 287-word essays. | §4.2 |
| Length-controlled win rate (logistic regression on length difference) | **Adopt** | Small script. Makes comparisons fair. | §4.3 |
| Swap test for position bias | **Already done** | Keep. Report swap agreement per opponent. | — |
| Randomize rubric order per call | **Adopt** | Cheap. | §4.2 |
| Cross-family judge (judge ≠ teacher family) | **Adopt** | Comes free with the open-weight teacher (section 4). | §4.2 |
| Human calibration set and agreement score (Cohen's κ) | **Adopt (scaled down)** | 60–100 items rated by a human. The report's κ ≥ 0.78 target is unrealistic for us; use κ ≥ 0.6 as the bar. | §4.4 |
| Concept tiers (ChessQA-style tags) to slice results | **Adopt (light)** | Tag dev and test items by motif and phase. Report results per tag. | §4.1 |
| Separate dev set; touch the test set rarely | **Adopt** | Not in the reports. Many experiments on one test set overfit it. | §4.1 |
| Over-refusal smoke test ("attack the king") | **Adopt (tiny)** | 20 prompts, one time. | §4.5 |
| Top-1 move match, centipawn loss, puzzle accuracy | **Reject** | Second does not choose moves. | — |
| P99 latency < 250 ms | **Reject** | Not possible or needed on CPU WASM. We measure tokens per second instead. | — |
| Nash / SPPO preference models for eval | **Reject** | Overkill. Length-controlled win rates are enough. | — |
| Cross-attention chess encoder | **Reject** | Architecture change. Breaks the GGUF / wllama export. | — |

---

## 6. Findings that no report made

1. **Claim checker.** Add rule checks for relational claims: attacks, defends, protects, threatens, pins, forks, "hanging", checkmate, and opening names. Use python-chess. This is the biggest gap in our reward. (TRAINING_PLAN §5.2)
2. **Motif facts.** The fact extractor has no fork, pin, skewer, discovered attack, mate threat, or trapped piece facts. The model cannot name ideas that the facts do not contain. Add them in Python and TypeScript, with golden tests. (§5.1)
3. **No first-person narration.** Add a check for game-diary voice ("I was trying…"). (§5.2)
4. **Fix the RL batch.** Use 8–16 prompts per step. (§8.2)
5. **Long mode needs real data.** Target ~25% `detail: long` in the new SFT set. (§6.3)
6. **Base Qwen fails our format 100% of the time** (markdown). The judge does not see the format rule. Report this next to every comparison with base Qwen.
7. **"Checked" mode for the extension.** Port the hard checks to TypeScript. If an answer fails a check, sample again. This is a cheap test-time fix and pairs well with reasoning modes. (§9.6)

---

## 7. Analysis of the Reasoning Mode Idea

We also evaluated the idea to add a reasoning mode (`<think>...</think>`) during training and in the browser extension.

### 7.1 Why reasoning tokens help a 2B chess model

In Phase 5 and Phase 6, the prompt forced an empty thinking block (`<think>\n\n</think>`). The 2B model had to read the FEN, read the facts, pick the main motif, match the player level, and write the first prose word in one forward pass.

When the model writes a short structured plan inside `<think>...</think>` first, it separates **fact selection** from **prose writing**:
1. The thinking block selects the verdict, the key threat or motif from the facts, and the best move refutation.
2. The answer block turns that selected plan into clean prose for a `beginner` or `club` player.
3. The extension hides or collapses the `<think>...</think>` tokens so the user sees only the clean explanation (or can expand the reasoning if curious).

### 7.2 Latency vs intelligence in the browser

In Phase 0.2 ([runtime.md](../docs/decisions/runtime.md)), `wllama` (WebAssembly CPU) reached **14.38 tokens/s** on 2B and **28.11 tokens/s** on 0.8B. Every thinking token delays the first visible answer word.

| Reasoning Mode | Thinking Budget | Extra Wait on 2B (14.4 tok/s) | Extra Wait on 0.8B (28.1 tok/s) | Best Use Case |
|---|---|---|---|---|
| `off` (Fast) | 0 tokens | 0.0 s | 0.0 s | Default fast explanation while browsing moves. |
| `quick` | ~32–48 tokens | ~2.2–3.3 s | ~1.1–1.7 s | 3-line structured checklist (verdict, main threat, best reply). |
| `standard` | ~64–96 tokens | ~4.4–6.7 s | ~2.3–3.4 s | Full 5-step cognitive template with counterfactual check. |
| `deep` | ~128–192 tokens (or `quick` + rule-check retry) | ~8.9–13.3 s | ~4.5–6.8 s | Complex tactical positions where the user wants maximum depth. |

### 7.3 Verdict on reasoning mode

**Adopt as a unified dual-mode recipe and test budgets systematically.**
- Do **not** train free-form 500-token rambles. They take 35 seconds in the browser and cause a 2B model to hallucinate ungrounded variations.
- Train **one unified model** on mixed prompt modes (`thinking: off | quick | standard | deep`) using a strict, short checklist inside `<think>...</think>`.
- Run a Pareto frontier sweep across token budgets (`0, 32, 64, 128, 256` tokens) and compare single-sample thinking against **Best-of-2 with TypeScript rule verification**.
- Expose a **Reasoning Mode** selector in the browser extension side panel so users can choose speed (`Fast`) or deeper analysis (`Thinking`). See [TRAINING_PLAN.md](../TRAINING_PLAN.md) Section 9 for the full test protocol.

