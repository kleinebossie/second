# Second v1.1 Recipe Changes (Phase 11)

This document lists all recipe updates from `v1.0` to `v1.1`.
All prose follows ASD-STE100 Simplified Technical English.

## 1. Decision Scope

- Keep `Second-1.0-2B` frozen on the `v1.0` recipe.
- Keep `Second-1.0-0.8B`, `Second-1.0-4B`, and `Second-1.0-9B` skipped.
- Apply all fact, reward, evaluation, and data updates to `Second-1.1-*`.

## 2. Fact Extractor Upgrades (`core/python` and `core/ts`)

Both [facts.py](../core/python/facts.py) and [facts.ts](../core/ts/src/facts.ts) pass 260 golden cases with exact byte parity.
Those 260 cases span `golden_200.json`, `golden_tactical_30.json`, and `golden_positional_30.json`.

### 2.1 Filter Noisy Checks

Updated `compute_checks_and_captures_facts` and `computeChecksAndCapturesFacts` to use optional `engine_eval` data:

- When a Stockfish top-three line has a check within 100 centipawns of the top line, report that check.
- When no PV check qualifies, report a check only if it wins material, forks safely, strips castling, or mates.
- Avoid duplicate capture facts for the same promotion square as the selected check.
- For `explain_move`, pass `eval_after`; for `explain_position`, pass `eval_before`.

### 2.2 Add Recapture Centralization and Activity

Added `compute_recapture_activity_fact` and `computeRecaptureActivityFact`:

- Detect when an opponent recapture lands on `d4`, `e4`, `d5`, or `e5`.
- Detect when a non-pawn, non-king recapture leaves the back rank into the center or gains two attacked squares.

### 2.3 Add Positional Refutation Facts

Extended `compute_refutation_reason_fact` and `computeRefutationReasonFact` beyond captures, defenses, and opening castling:

- Report central pawn pushes that claim space, hold central tension instead of trading pawns, or attack pieces.
- Report castling refutations, safe piece retreats, centralization, open or half-open rook files, and activity gains.

### 2.4 Add Line Comparison and Quiet Position Facts

- Added `compute_line_comparison_fact` and `computeLineComparisonFact` to compare up to four-ply lines.
- Equalize compared lines to an even ply count and settle one immediate recapture ply on capture horizons.
- Added `compute_move_positional_fact` (`computeMovePositionalFact`) for castling, space, tension, development, centralization, and rook files.
- Added `compute_position_structure_fact` (`computePositionStructureFact`) for board tension, space advantage, open files, and development leads.

## 3. Expanded `v1.1` Dev and Test Sets (`eval/`)

- Created [build_eval_v1_1.py](../eval/build_eval_v1_1.py) while keeping frozen `v1.0` files unchanged.
- Built `eval/devset-1.1.jsonl` (160 items: 80 beginner, 80 club) and `eval/testset-1.1.jsonl` (320 items: 160 beginner, 160 club).
- Added 10 curated dev cases and 20 curated test cases, including the `6... exd4` case (`test_301`).
- Added `--ratings-files` support to ingest exported `second_ratings_*.json` thumbs-down entries with zero contamination across splits.

## 4. Reward `v3` Rule Checks (`rewards/`)

Added two hard checks in [claims.py](../rewards/claims.py).
Wired `reward_version="v3"` (11 hard checks total) into [scorer.py](../rewards/scorer.py), [api.py](../rewards/api.py), [metrics.py](../eval/metrics.py), and [run_eval.py](../eval/run_eval.py).

- **`vague_boilerplate` (`check_vague_boilerplate`):** Rejects unsupported phrases such as coordination cliches, harmony claims, overall-position claims, ungrounded weaknesses, and filler.
- **`positional_grounding` (`check_positional_fact_grounding`):** When no overriding tactic exists, requires explanations to reference the active positional concepts.
- Preserved `reward_version="v1"` and `reward_version="v2"` for backward compatibility.

## 5. Open-Weight Self-Training (`SFT`, `RFT`, and `GRPO`)

- Created [rft_v1_1.py](../data/pipeline/rft_v1_1.py) and [second-1.1-recipe.yaml](../training/configs/second-1.1-recipe.yaml).
- Configured `validate_open_weight_model` to accept open-weight prefixes (`Qwen/*`, `unsloth/*`), approved repos, and local checkpoints.
- Rejected proprietary APIs such as Gemini, GPT, and Claude in data synthesis.
- Validated both SFT LoRA (`sft:`) and DAPO GRPO (`grpo:`) settings in [second-1.1-recipe.yaml](../training/configs/second-1.1-recipe.yaml).
- Kept Fast mode (`enable_thinking: false`) and three-stream GDPO with `reward_version: "v3"`.
