# Phase 5 Evaluation Report: sft-1.0-2b

- **Model**: `sft-1.0-2b`
- **Date**: 2026-11-05 22:27:15 UTC
- **Evaluation Status**: **PASS**
- **Specification**: Phase 5 (Supervised Fine-Tuning on 2B)

## 1. Executive Summary and Pass Rule Verification

Phase 5 requires two conditions for release:
1. The format must be correct in more than 95.0% of test set outputs.
2. The hard-fail rate must be lower than base Qwen 2B.

| Release Criterion | SFT Model Value | Base Qwen 2B Value | Target | Status |
|---|---|---|---|---|
| Format Pass Rate | **100.0%** (fail: 0.0%) | 0.0% | > 95.0% | PASS |
| Hard-Fail Rate | **11.3%** | 100.0% | < Base Qwen 2B | PASS |
| Overall Phase 5 Result | **PASS** | - | Both criteria satisfied | **PASS** |

## 2. Training Configuration Summary

| Parameter | Specification | Recorded Setting |
|---|---|---|
| Base Model | Qwen 3.5 2B Instruct | `Qwen/Qwen3.5-2B` |
| Reasoning Mode | Non-thinking text-only | `enable_thinking=False` |
| Method | LoRA (rank 32, alpha 32) | `r=32, lora_alpha=32, dropout=0.05` |
| Target Modules | All text linear layers | `all-linear` (excludes `lm_head`, vision) |
| Training Split | Filtered human SFT data | `data/processed/sft_train.jsonl` |
| Validation Split | Filtered human SFT data | `data/processed/sft_val.jsonl` (137 items) |
| Epochs | 1 to 2 | 2 epochs |
| Learning Rate | 1e-4 with cosine schedule | `1e-4` (warmup ratio: 0.03) |
| Max Sequence Length | 1024 tokens | 1024 tokens |
| Loss Masking | Answer tokens only | Prompt masked with `-100` |
| Hardware Target | NVIDIA L4 GPU | bfloat16 mixed precision |

### Training Run Metrics

| Training Metric | Value |
|---|---|
| `epochs` | 2 |
| `final_adapter_path` | /content/drive/MyDrive/second/checkpoints/sft-1.0-2b/final_adapter |
| `global_steps` | 1382 |
| `model_id` | Qwen/Qwen3.5-2B |
| `train_loss` | 2.3660813502740927 |
| `training_time_minutes` | 25.05 |

## 3. Validation Split Automatic Metrics

Evaluated on `data/processed/sft_val.jsonl` (137 items).

| Metric | Validation Split Result |
|---|---|
| Total Items | 137 |
| Hard-Fail Rate | **5.8%** |
| Mean Reward | **0.6101** |
| Mean Soft Score (All) | 0.7113 |
| Mean Soft Score (Passing) | 0.7099 |
| Average Word Count | 16.4 words |
| Length in Target Band | 16.8% |
| Verdict Coverage Rate | 12.4% |
| Verdict Accuracy (Covered) | 94.1% |

## 4. Frozen Test Set Comparison

Evaluated on `eval/testset-1.0.jsonl` (300 items) against `base-qwen-2b`.

### 4.1 Overall Metrics Comparison

| Metric | Second-1.0-2B-SFT | Base Qwen 2B | Difference |
|---|---|---|---|
| Hard-Fail Rate | **11.3%** | 100.0% | -88.7% |
| Mean Reward | **0.4772** | -1.0000 | +1.4772 |
| Mean Soft Score (All) | 0.6660 | 0.7427 | -0.0767 |
| Mean Soft Score (Passing) | 0.6660 | 0.0000 | +0.6660 |
| In Target Length Band | 13.0% | 0.0% | +13.0% |
| Average Word Count | 23.6 words | 286.9 words | -263.3 words |

### 4.2 Hard Check Breakdown

If any hard check fails, reward is -1.0.

| Check Name | SFT Failure Rate | Base Qwen Failure Rate | Delta |
|---|---|---|---|
| `beginner_jargon` | 0.0% | 10.7% | -10.7% |
| `format` | 0.0% | 100.0% | -100.0% |
| `grounded_moves` | 2.7% | 23.3% | -20.7% |
| `legal_moves` | 3.7% | 44.3% | -40.7% |
| `piece_on_square` | 2.7% | 59.3% | -56.7% |
| `verdict_tone` | 4.7% | 30.3% | -25.7% |

### 4.3 Soft Score Breakdown

| Soft Metric | SFT Mean Score | Base Qwen Mean Score | Delta |
|---|---|---|---|
| `better_move` | 0.6317 | 0.8700 | -0.2383 |
| `key_point` | 0.4017 | 0.9013 | -0.4996 |
| `length` | 0.6030 | 0.0073 | +0.5957 |
| `level_fit` | 0.9862 | 0.9367 | +0.0495 |
| `no_repeats` | 0.9417 | 0.5650 | +0.3767 |
| `not_copied` | 0.9816 | 0.9866 | -0.0050 |

### 4.4 Subgroup Breakdowns

#### Subgroup: Level

| Level Slice | Items | SFT Hard-Fail | SFT Mean Reward | SFT Avg Words |
|---|---|---|---|---|
| `beginner` | 150 | 12.0% | 0.4453 | 18.8 |
| `club` | 150 | 10.7% | 0.5091 | 28.5 |

#### Subgroup: Task

| Task Slice | Items | SFT Hard-Fail | SFT Mean Reward | SFT Avg Words |
|---|---|---|---|---|
| `explain_move` | 247 | 13.0% | 0.4284 | 21.3 |
| `explain_position` | 53 | 3.8% | 0.7047 | 34.6 |

#### Subgroup: Detail

| Detail Slice | Items | SFT Hard-Fail | SFT Mean Reward | SFT Avg Words |
|---|---|---|---|---|
| `long` | 45 | 17.8% | 0.3428 | 37.0 |
| `short` | 255 | 10.2% | 0.5009 | 21.3 |

## 5. Manual Review of 30 Outputs

This section lists 30 deterministically selected test outputs.
It covers both tasks, player levels, length bands, and outcome types.

### Sample 01: `test_127` (explain_move, beginner, short)

- **Verdict**: Mistake
- **Move**: `f4`
- **Word Count**: 12 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `verdict_tone`

**Completion Text**:
> I thought this was a good move, but it was a mistake.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (12) is outside the short target band (25-80 words). Grounding: FAIL. Found factual defects: verdict tone mismatch.

---

### Sample 02: `test_011` (explain_move, beginner, long)

- **Verdict**: Best
- **Move**: `Qe7`
- **Word Count**: 21 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `legal_moves`

**Completion Text**:
> This is the main line of the Petrov's Defence. The idea is to play e5 and attack the knight on e5.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (21) is outside the long target band (80-200 words). Grounding: FAIL. Found factual defects: illegal move.

---

### Sample 03: `test_013` (explain_move, club, short)

- **Verdict**: Blunder
- **Move**: `Rb2`
- **Word Count**: 21 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `legal_moves, grounded_moves`

**Completion Text**:
> Move the piece to a pivotal position, providing it with greater board command. Tactical line: Rb2 Qc4+ Kf1 Qc2+ Kf2 Qc4+

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (21) is outside the short target band (25-80 words). Grounding: FAIL. Found factual defects: illegal move, unsupported move.

---

### Sample 04: `test_029` (explain_move, club, long)

- **Verdict**: Best
- **Move**: `Kg2`
- **Word Count**: 47 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `legal_moves, grounded_moves`

**Completion Text**:
> I think this is a good move, but I am not sure if it is the best. I think that 29. Kf1 is better, because it allows the rook to get to the b file, and it also allows the king to get to the h file.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (47) is outside the long target band (80-200 words). Grounding: FAIL. Found factual defects: illegal move, unsupported move.

---

### Sample 05: `test_168` (explain_position, club, short)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 41 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `piece_on_square`

**Completion Text**:
> I think this is a good move, but I am not sure if it is the best. I think that the bishop on g7 is a bit weak, and I think that it is better to move the knight to b6.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: PASS. Word count (41) is inside the short target band (25-80 words). Grounding: FAIL. Found factual defects: wrong piece location.

---

### Sample 06: `test_006` (explain_position, club, long)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 30 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `piece_on_square`

**Completion Text**:
> This is the main line of the Nimzo-Indian Defense. The idea of the Nimzo-Indian is to play c5, to attack the e4 pawn and to gain space on the queenside.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (30) is outside the long target band (80-200 words). Grounding: FAIL. Found factual defects: wrong piece location.

---

### Sample 07: `test_007` (explain_move, beginner, short)

- **Verdict**: Inaccuracy
- **Move**: `b4`
- **Word Count**: 14 words
- **Reward**: `0.434`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I think this was a mistake. I was hoping to get a passed pawn.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (14) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 08: `test_004` (explain_move, beginner, long)

- **Verdict**: Blunder
- **Move**: `e5`
- **Word Count**: 22 words
- **Reward**: `0.3562`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> This is a very common move in the Sicilian Najdorf, and it is the most common way to play against the Najdorf.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (22) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 09: `test_001` (explain_move, club, short)

- **Verdict**: Blunder
- **Move**: `Nf6`
- **Word Count**: 23 words
- **Reward**: `0.488`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I was thinking about how to get rid of the queen, and I thought that I could just take it with my knight.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (23) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 10: `test_005` (explain_move, club, long)

- **Verdict**: Good
- **Move**: `Nf6`
- **Word Count**: 27 words
- **Reward**: `0.6906`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> This is the main line of the Petrov's Defence. The Petrov's Defence is a very solid and safe defence, but it is not the most aggressive one.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (27) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 11: `test_009` (explain_position, beginner, short)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 10 words
- **Reward**: `0.805`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> White is ready to castle queenside and attack black's king.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (10) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 12: `test_012` (explain_position, beginner, long)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 31 words
- **Reward**: `0.6956`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> This is the main line of the Petrov's Defence. The Petrov's Defence is a very solid and reliable defence, and it is one of the most played defence in the world.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (31) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 13: `test_003` (explain_position, club, short)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 14 words
- **Reward**: `0.724`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> The queen is not defended, so it is a good move to take it.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (14) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 14: `test_018` (explain_position, club, long)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 37 words
- **Reward**: `0.7644`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I was thinking about how to get my rook to the h file, and I thought that I could get my knight to a good square, and I thought that I could get my king to safety.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (37) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 15: `test_038` (explain_move, club, short)

- **Verdict**: Good
- **Move**: `Rc1`
- **Word Count**: 15 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `verdict_tone`

**Completion Text**:
> I think this is a mistake, because I think that black has a better position.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (15) is outside the short target band (25-80 words). Grounding: FAIL. Found factual defects: verdict tone mismatch.

---

### Sample 16: `test_041` (explain_move, club, long)

- **Verdict**: Best
- **Move**: `Qg5`
- **Word Count**: 25 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `piece_on_square`

**Completion Text**:
> I think this is a very good move, because it threatens to win the queen, and it also threatens to win the knight on f6.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (25) is outside the long target band (80-200 words). Grounding: FAIL. Found factual defects: wrong piece location.

---

### Sample 17: `test_047` (explain_move, beginner, long)

- **Verdict**: Good
- **Move**: `Rxd7`
- **Word Count**: 36 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `piece_on_square`

**Completion Text**:
> I think this is a good move, because it allows me to take the pawn on e5, and I think that it is better to take the pawn on d7, because it is a better pawn.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (36) is outside the long target band (80-200 words). Grounding: FAIL. Found factual defects: wrong piece location.

---

### Sample 18: `test_088` (explain_move, beginner, long)

- **Verdict**: Good
- **Move**: `Bd3`
- **Word Count**: 22 words
- **Reward**: `-1.0`
- **Hard Checks Passed**: No
- **Failed Checks**: `legal_moves`

**Completion Text**:
> This is the main line of the Petrov's Defence. The idea is to play c5 and then take the pawn on d5.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (22) is outside the long target band (80-200 words). Grounding: FAIL. Found factual defects: illegal move.

---

### Sample 19: `test_008` (explain_move, beginner, short)

- **Verdict**: Good
- **Move**: `Nc6`
- **Word Count**: 15 words
- **Reward**: `0.73`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I think this is a good move, but I would have preferred to play 15...Ra7.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (15) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 20: `test_016` (explain_move, beginner, long)

- **Verdict**: Mistake
- **Move**: `Na5`
- **Word Count**: 39 words
- **Reward**: `0.4006`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I think this is a mistake, but I don't know why. I think it is because I wanted to keep my knight on the board, and I thought that this move would be a good way to do it.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (39) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 21: `test_002` (explain_move, club, short)

- **Verdict**: Good
- **Move**: `Be6`
- **Word Count**: 19 words
- **Reward**: `0.8472`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> This is the main line of the Nimzo-Indian. Black is trying to get rid of the bishop on c4.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (19) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 22: `test_010` (explain_move, club, long)

- **Verdict**: Blunder
- **Move**: `Nd7`
- **Word Count**: 25 words
- **Reward**: `0.5719`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> Move the piece to a stronger position, enhancing its board control. Tactical line: Nd7 Rxe8 Trade the lower value piece for a higher value piece.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (25) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 23: `test_021` (explain_position, beginner, short)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 10 words
- **Reward**: `0.805`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> The rook is now free to move to the h-file.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (10) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 24: `test_024` (explain_position, beginner, long)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 20 words
- **Reward**: `0.7575`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I think this is a good position for black, because he has a good bishop pair and a good center.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (20) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 25: `test_015` (explain_position, club, short)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 22 words
- **Reward**: `0.722`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I think this is a draw, but I'm not sure. I think that black has a slight advantage, but I'm not sure.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (22) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 26: `test_030` (explain_position, club, long)

- **Verdict**: None
- **Move**: `None`
- **Word Count**: 231 words
- **Reward**: `0.6435`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> This is the position of the London System, and it is a very common position in the game of chess. The London System is a very popular opening system, and it is one of the most played openings in the world. The London System is a solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and it is a good choice for beginners. The London System is a very solid and reliable opening, and

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (231) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 27: `test_019` (explain_move, beginner, short)

- **Verdict**: Mistake
- **Move**: `Ra1`
- **Word Count**: 27 words
- **Reward**: `0.675`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> Move the piece to a pivotal position, providing it with greater board command. Tactical line: Ra1 Rxa1 Trade the lower value piece for a higher value piece.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: PASS. Word count (27) is inside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 28: `test_023` (explain_move, beginner, long)

- **Verdict**: Only move
- **Move**: `Kh1`
- **Word Count**: 21 words
- **Reward**: `0.7733`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> White is up a pawn, but black has a passed pawn on the 7th rank. White is in a difficult position.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output uses simple vocabulary and avoids engine jargon. Length: FAIL. Word count (21) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 29: `test_014` (explain_move, club, short)

- **Verdict**: Only move
- **Move**: `Kf8`
- **Word Count**: 12 words
- **Reward**: `0.712`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> The only move, but it's not enough. White has a winning advantage.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (12) is outside the short target band (25-80 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

### Sample 30: `test_017` (explain_move, club, long)

- **Verdict**: Best
- **Move**: `Nf6`
- **Word Count**: 19 words
- **Reward**: `0.6756`
- **Hard Checks Passed**: Yes
- **Failed Checks**: `None`

**Completion Text**:
> I was thinking about the possibility of a kingside attack, but I decided to play for a queenside attack.

**Review Note**: Format: PASS. Output uses clean plain text without markdown syntax or headings. Voice: PASS. Output matches standard club player tone. Length: FAIL. Word count (19) is outside the long target band (80-200 words). Grounding: PASS. All mentioned moves and piece locations match the facts.

---

## 6. Qualitative Synthesis from Manual Review

Manual inspection of the 30 test outputs confirms the expected Phase 5 behavior:

1. **Plain-Text Format Mastery**: All 30 outputs use clean prose without Markdown headings, bullet lists, or `<think>` blocks.
2. **Voice and Source Imprints**: The model reproduces two training source styles: conversational GameKnot commentary and structured MATE tactical descriptions.
3. **Factual Grounding Gaps**: When an output fails a hard check (11.3% of test items), it usually names an ungrounded move or mismatches the engine verdict tone (`I think this is a mistake` on a `Good` move).
4. **Length and Repetition on Long Prompts**: Because only 1.4% of human SFT examples use `detail: long`, long-mode outputs average 37.0 words, and one reviewed long item (`test_030`) repeats a sentence.
5. **Priorities for Phase 6 (GRPO RL)**: Phase 6 reinforcement learning will directly optimize the `key_point`, `better_move`, `length`, and `no_repeats` soft scores while penalizing hard-check failures.