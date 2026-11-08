# Stage T4 GRPO v2 Evaluation Report: RL-B2

**Model**: `Second-1.0-2B-GRPO-v2`  
**Starting Policy**: `SFT-A1` (`r=32`)  
**Evaluation Set**: `eval/devset-1.0.jsonl` (150 items)  
**Gate Status**: **PASSED**

## 1. Stage T4 Pass Gates vs SFT-A1 Baseline

| Gate Dimension | Stage T4 Target | SFT-A1 Baseline | GRPO v2 Result | Gate Outcome |
|---|---|---|---|---|
| **Hard Fail Rate** | &le; 5.0% | 14.67% (22 items) | **4.0%** (6 items) | PASS |
| **Length In-Band (Overall)** | &ge; 80.0% | 99.33% | **99.33%** | PASS |
| **First-Person Tone Rate** | &le; 2.0% | 0.00% | **0.0%** | PASS |
| **Key Point Score** | &ge; 0.70 | 0.9867 | **0.9907** | PASS |
| **Better Move Score** | &ge; 0.75 | 0.9600 | **1.0000** | PASS |
| **Mean Word Count** | 40–80 words | 64.0 words | **60.7 words** | TRACKED |
| **Mean Reward v2** | Informational | 0.6920 | **0.9174** | TRACKED |

## 2. Hard Check Failure Breakdown

| Hard Check | Failed Items |
|---|---|
| `grounded_moves` | 2 |
| `legal_moves` | 2 |
| `piece_on_square` | 2 |

## 3. Gate Decision

The model satisfies all Stage T4 pass gates on the development set. It is approved to proceed to Stage T5 and Stage T6 final verification.
