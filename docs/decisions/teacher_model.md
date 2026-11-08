# Decision Record: Open-Weight Teacher Model Selection (Stage T2.1)

## Status
Accepted (2026-11-06).
Evaluation selected `Qwen/Qwen3.5-27B` (4-bit quantized) as the official teacher model.

## Date
2026-11-06

## Context
Second trains chess explanation models.
We require an open-weight Apache 2.0 teacher model for Stage T2 knowledge distillation.
The teacher model synthesizes engine facts into natural chess coaching prose.
It uses feigned discovery and structured cognitive traces.
We evaluate teacher candidates on Google Colab before full synthesis.

## Evaluation Protocol
We evaluated candidates on 50 stratified positions from `eval/devset-1.0.jsonl`.
Each candidate generated explanations using the Stage T2 Teacher Synthesis Prompt.
We evaluated completions with Stage T1 hard checks and soft reward scoring.
We measured five primary metrics:
1. Hard check pass rate (legal moves, grounded moves, piece-on-square, relational claims, voice).
2. Mean composite reward.
3. Length compliance rate (target word bands for short and long modes).
4. Key point coverage.
5. Inference time per item.

Selection formula: `selection_score = pass_rate * mean_reward`.

## Evaluation Results

| Rank | Model | Selection Score | Pass Rate | Mean Reward | In-Band % | Words | Key Point | Time/Item |
|---|---|---|---|---|---|---|---|---|
| #1 | `Qwen/Qwen3.5-27B` | **0.0844** | 60.0% | 0.1406 | 44.0% | 117.1 | 0.9880 | 38.21s |
| #2 | `Qwen/Qwen3.6-27B` | **0.0238** | 56.0% | 0.0426 | 34.0% | 136.1 | 0.9880 | 43.55s |
| #3 | `Qwen/Qwen3.5-9B` | **-0.1107** | 36.0% | -0.3075 | 60.0% | 105.7 | 0.9700 | 20.57s |

## Decision
We select `Qwen/Qwen3.5-27B` (4-bit quantized via BitsAndBytes on NVIDIA L4) as the official winning teacher model.
We proceed with Stage T2.3 synthesis using this model.

## Technical Rationale
1. **Highest Selection Score**: `Qwen/Qwen3.5-27B` achieves a selection score of 0.0844. It outperforms `Qwen/Qwen3.6-27B` (0.0238) and `Qwen/Qwen3.5-9B` (-0.1107).
2. **Superior Pass Rate**: The model achieves a 60.0% hard check pass rate. It passes 24.0 percentage points higher than `Qwen/Qwen3.5-9B` (36.0%).
3. **Strongest Mean Reward**: The model achieves a positive mean reward of 0.1406. It substantially exceeds `Qwen/Qwen3.6-27B` (0.0426) and `Qwen/Qwen3.5-9B` (-0.3075).
4. **Near-Perfect Key Point Coverage**: The model achieves 0.9880 key point coverage. It includes nearly all critical engine facts in the generated prose.
5. **Practical Colab Runtime**: At 38.21 seconds per item in 4-bit mode, the model operates safely within 24 GB VRAM on NVIDIA L4 GPUs.

## Synthesis Configuration
- Model: `Qwen/Qwen3.5-27B`
- Quantization: 4-bit (`BitsAndBytesConfig` with `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`)
- Hardware: Google Colab NVIDIA L4 GPU (24 GB VRAM)
- Batch Size: 4
- Maximum New Tokens: 512

## Historical Note: Initial Evaluation
An earlier benchmark evaluated `Qwen/Qwen3.5-9B` (bfloat16) against `Qwen/Qwen3.8-27B` (4-bit).
In that initial test, `Qwen/Qwen3.5-9B` scored -0.1227 (32.0% pass rate) while `Qwen/Qwen3.8-27B` scored -0.1318 (28.0% pass rate).
The team re-opened evaluation to test `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.6-27B`.
`Qwen/Qwen3.5-27B` demonstrated decisive quality improvements and won the final evaluation.

## Next Actions & Operational Outcome (Completed 2026-11-07)
1. **Stage T2.3 Operational Update (`Qwen/Qwen2.5-7B-Instruct` in `bfloat16`)**:
   - During the full 8,000-prompt synthesis run on Colab L4, 4-bit NF4 dequantization overhead on `Qwen/Qwen3.5-27B` (~38s/item) projected a ~20+ hour runtime across 8,000 prompts.
   - Switching to `Qwen/Qwen2.5-7B-Instruct` in native `bfloat16` (`batch_size=16`) ran in ~75 minutes (~7.5 Colab units) on L4 with zero quantization overhead while maintaining high Stage T1 rule-check pass rates.
2. **Stage T2.4 & T2.5 Verification**:
   - `data/pipeline/filter_teacher.py` and `data/pipeline/dedupe_diversity.py` produced `data/processed/sft_v2_train.jsonl` (`4,457` total examples: `3,657` verified teacher traces + `800` gold human examples).
   - This dataset trained `SFT-A1` (`r=32`) and `RL-B2` (`loss_type="dapo"`), which passed all Stage T4 and Stage T6 frozen test gates (`4.67%` hard-fail rate, `99.67%` in-band length, `281W–0L` vs Iteration 1).

