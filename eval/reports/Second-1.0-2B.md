# Stage T6 Final Evaluation Report: Second-1.0-2B (GRPO v2 / RL-B2)

- **Model**: `Second-1.0-2B-GRPO-v2` (`RL-B2`, DAPO + Reward v2, `r=32`)
- **Date**: 2026-11-08 11:53:11 UTC
- **Frozen Test Set**: `eval/testset-1.0.jsonl` (300 items)
- **Overall Release Gate Status**: **FAILED**

## 1. Section 3 Exit Gate Verification (Iteration 1 vs Iteration 2)

| Category | Metric | Iteration 1 (`grpo-1.0-2b`) | Stage T6 Target Gate | Iteration 2 (`grpo-1.0-2b-v2`) | Status |
|---|---|---|---|---|---|
| **Format** | Format Pass Rate | 100.0% | &ge; 99.0% | **100.00%** | PASS |
| **Grounding** | Hard-Fail Rate (All 9 Checks) | 7.7% (v1 checks) | &le; 5.0% | **4.67%** | PASS |
| **Voice Hygiene** | First-Person Diary Rate | 51.3% | &le; 2.0% | **0.00%** | PASS |
| **Completeness** | In Target Length Band | 18.7% | &ge; 80.0% | **99.67%** | PASS |
| **Substantiation** | Key-Point Soft Score | 0.3860 | &ge; 0.7000 | **0.9790** | PASS |
| **Counterfactuals** | Better-Move Soft Score | 0.1560 | &ge; 0.7500 | **1.0000** | PASS |
| **Safety** | Benign Chess Over-Refusal (20 items) | 0.0% | &le; 0.0% | **0.00%** | PASS |
| **Composite** | Mean Reward v2 / Mean Words | 26.8 words | 40–80 words | **0.9100** (62.0w) | TRACKED |
| **Judge Rubric** | Correctness (1–5) | 2.16 | &ge; 3.70 | **3.67** | FAIL |
| **Judge Rubric** | Usefulness (1–5) | 1.87 | &ge; 3.60 | **3.77** | PASS |
| **Judge Rubric** | Clarity (1–5) | 3.54 | &ge; 4.00 | **4.30** | PASS |
| **Judge Rubric** | Level Fit (1–5) | 3.53 | &ge; 3.90 | **4.09** | PASS |
| **Judge Rubric** | Overall Mean (1–5) | 2.78 | &ge; 3.80 | **3.96** | PASS |
| **Pairwise** | Win Rate vs `grpo-1.0-2b` (Excl. Ties) | — | &ge; 70.0% | **100.0%** | PASS |
| **Pairwise** | LC Win Rate vs `base-qwen-2b` | — | &ge; 65.0% | **86.2%** | PASS |

## 2. Hard Check Breakdown (All 9 Deterministic Verifiers)

| Hard Check | Failure Rate (%) | Failed Items (out of 300) |
|---|---|---|
| `legal_moves` | 1.00% | 3 |
| `grounded_moves` | 1.33% | 4 |
| `piece_on_square` | 2.00% | 6 |
| `verdict_tone` | 0.33% | 1 |
| `format` | 0.00% | 0 |
| `beginner_jargon` | 0.00% | 0 |
| `coach_voice` | 0.00% | 0 |
| `opening_claims` | 0.00% | 0 |
| `relational_claims` | 0.00% | 0 |

## 3. Subgroup & Concept Tier Breakdowns

### 3.1 Concept Tiers

| Concept Tier | Count | Hard Fail (%) | Mean Reward (v1) | Avg Words | In Band (%) |
|---|---|---|---|---|---|
| `tier_1_mechanics` | 125 | 4.0% | 0.9059 | 63.9 | 100.0% |
| `tier_2_motifs` | 17 | 0.0% | 0.9851 | 56.1 | 100.0% |
| `tier_3_calculation` | 10 | 10.0% | 0.7920 | 56.6 | 100.0% |
| `tier_4_positional` | 63 | 6.3% | 0.8492 | 63.6 | 98.4% |
| `tier_5_endgame` | 85 | 4.7% | 0.8874 | 59.9 | 100.0% |

### 3.2 Detail Length & Player Level Slices

| Slice | Count | Hard Fail (%) | Avg Words | In Target Band (%) |
|---|---|---|---|---|
| `short` (25–80w) | 255 | 3.1% | 55.7 | 100.0% |
| `long` (80–200w) | 45 | 13.3% | 97.7 | 97.8% |
| `beginner` | 150 | 4.0% | 61.5 | 100.0% |
| `club` | 150 | 5.3% | 62.6 | 99.3% |

## 4. Debiased Gemini Judge Pairwise Comparisons

| Opponent Baseline | Total Pairs | Candidate Wins | Opponent Wins | Ties | Raw Win Rate (%) | Decisive Win Rate (%) | LC-WR (%) |
|---|---|---|---|---|---|---|---|
| `grpo-1.0-2b` | 300 | 281 | 0 | 19 | 93.7% | **100.0%** | **97.5%** |
| `sft-1.0-2b` | 300 | 281 | 0 | 19 | 93.7% | **100.0%** | **97.6%** |
| `base-qwen-2b` | 300 | 293 | 0 | 7 | 97.7% | **100.0%** | **86.2%** |
| `gemini-flash` | 300 | 202 | 20 | 78 | 67.3% | **91.0%** | **74.5%** |

## 5. Stratified 30-Item Audit Summary (6 Items per Concept Tier)

- **Audited Items**: 30
- **First-Person Diary Violations**: 0
- **Unverified Opening Claim Violations**: 0
- **Relational Claim Violations**: 0
