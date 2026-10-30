---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - chess
  - stockfish
  - reinforcement-learning
  - grpo
  - sft
  - qwen3.5
pretty_name: Second Chess Explanation Dataset (second-data-1.0)
size_categories:
  - 10K<n<100K
---

# Dataset Card for `second-chess/second-data-1.0`

## 1. Dataset Summary

`second-data-1.0` is the official training, validation, test-pool, and reinforcement learning (RL) prompt dataset for **Second**. Second is a family of small language models (based on Qwen 3.5) that turn Stockfish chess evaluations and deterministic board facts into clear human explanations.

Every record uses the deterministic Phase 1 prompt contract (`core/python` and `core/ts`). Stockfish supplies all tactical and positional facts at fixed depth with `MultiPV=3`. The language model does not calculate variations on its own; it explains the provided facts for `beginner` (under ~1200 Elo) and `club` (~1200–2000 Elo) players in `short` (2–4 sentences, <= 80 words) or `long` (80–200 words) detail.

## 2. Data Sources and Licenses

We follow the licensing rules documented in [`docs/data_sources.md`](../../docs/data_sources.md):

| Data Source | License | Redistribution Policy | Included in Public HF Export (`data/processed/public/`) | Included in Local Rebuild (`data/processed/`) |
|---|---|---|---|---|
| **Lichess Open Game Database** (`data/sources/lichess_games.py`) | CC0 1.0 Universal | Full public domain rights. | Yes (`rl_prompts.*`) | Yes (`rl_prompts.*`) |
| **MATE Dataset** (`data/sources/mate.py`, Wang et al., NAACL 2025) | MIT License | Permissive redistribution with MIT notice. | Yes (`sft_*.parquet`, `sft_*.jsonl`) | Yes (`sft_*.parquet`, `sft_*.jsonl`) |
| **Lichess Studies (Open License)** (`data/sources/lichess_studies.py`) | CC0 / CC-BY / Public Domain | Redistributable when explicit open license is present. | Yes (open-license subset) | Yes |
| **Lichess Studies (Standard ToS)** (`data/sources/lichess_studies.py`) | Lichess Terms of Service | Author copyright; no third-party text redistribution. | No (rebuild script only) | Yes |
| **GameKnot Annotated Games** (`data/sources/gameknot.py`) | Proprietary (GameKnot EULA) | No redistribution rights for commentary text. | No (rebuild script only) | Yes |

### MIT License Notice for MATE Dataset Portion

Portions of the supervised fine-tuning (SFT) splits in this dataset are derived from `OutFlankShu/MATE_DATASET` (Wang et al., "Explore the Reasoning Capability of LLMs in the Chess Testbed", NAACL 2025), which is released under the MIT License:

> Copyright (c) 2024 OutFlankShu (MATE Dataset Authors)
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## 3. Pipeline Filter Counts (Phase 2.3 & 2.4)

We collected **16,000 raw human commentary records** across GameKnot (`6,000`), Lichess Studies (`6,000`), and the MATE dataset (`4,000`). We analyzed unique positions with Stockfish (`depth=12`, `MultiPV=3`) and cached **47,711 unique FEN evaluations** in `data/cache/stockfish_eval_cache.parquet`.

Each raw comment passed through seven deterministic filter stages in `data/pipeline/filter_comments.py`:

| Filter Stage | Description | Items Removed | Share of Raw Input |
|---|---|---:|---:|
| **Total Raw Input** | Raw extracted `(position, move, comment)` records | **16,000** | **100.0%** |
| 1. `cleaned_empty_or_short` | Fewer than 8 words after stripping PGN tags (`[%eval]`, `[%cal]`, `[%clk]`), move numbers, and glyphs | 4,172 | 26.1% |
| 2. `not_english` | Non-English text detected via deterministic `langdetect` (`seed=0`) | 145 | 0.9% |
| 3. `no_chess_content` | Contains no chess piece name, square coordinate, SAN move, or chess term | 931 | 5.8% |
| 4. `illegal_move_or_wrong_piece` | Names an illegal SAN move or claims a piece on a square where it does not exist (`rewards/checks.py`) | 1,387 | 8.7% |
| 5. `tone_disagrees_with_verdict` | Positive tone on `Inaccuracy`/`Mistake`/`Blunder`, or unnegated negative tone on `Best`/`Good`/`Only move` | 212 | 1.3% |
| 6. `links_usernames_offensive` | Contains URLs, `@username` handles, or profanity | 68 | 0.4% |
| 7. `duplicate` | Exact or near-duplicate normalized alphanumeric fingerprint | 3,620 | 22.6% |
| **Total Kept Comments** | Clean, verified human commentary | **5,465** | **34.2%** |
| **Expanded SFT Examples** | Short cuts (`detail: short`, `5,465`) + full long examples (`detail: long`, `85`) | **5,550** | — |

## 4. Dataset Splits and Distributions (Phase 2.5 & 2.6)

### 4.1 SFT Splits (Split Strictly by Game)

Records are partitioned by deterministic SHA-256 hash of `game_id` (`data/pipeline/splits.py`). Any common opening FEN that collides with `train` is removed from `val` and `test_pool` (`66` colliding opening records removed) so that **zero games and zero FEN positions overlap** between `train`, `val`, and `test_pool`.

| Partition | Total Records (Local Rebuild) | Public Redistributable Records (`public/`) |
|---|---:|---:|
| `sft_train` | 4,920 | 399 |
| `sft_val` (~2.8%) | 153 | 10 |
| `sft_test_pool` (~7.5%) | 411 | 33 |
| **Total SFT** | **5,484** | **442** |

### 4.2 RL Prompt Set (`rl_prompts`)

We sampled **25,000 positions** from rated games in the CC0 Lichess standard database (`data/sources/lichess_games.py`, >80% with average player Elo in `800–2000`), excluded all games and FENs in `sft_val` and `sft_test_pool`, evaluated with Stockfish (`depth=12`, `MultiPV=3`), and built **20,000 RL prompts** (`data/pipeline/rl_prompts.py`):

| Category | Count | Share | Target in `PLAN.md` |
|---|---:|---:|---:|
| **Total RL Prompts** | **20,000** | **100.0%** | 20k–50k |
| Task: `explain_move` (`Inaccuracy` / `Mistake` / `Blunder`) | 9,000 | 45.0% | ~45% |
| Task: `explain_move` (`Good` / `Best` / `Only move`) | 7,000 | 35.0% | ~35% |
| Task: `explain_position` | 4,000 | 20.0% | ~20% |
| Level: `beginner` | 10,000 | 50.0% | 50% |
| Level: `club` | 10,000 | 50.0% | 50% |
| Detail: `short` | 16,000 | 80.0% | 80% |
| Detail: `long` | 4,000 | 20.0% | 20% |

## 5. Rebuilding the Full Dataset Locally

To rebuild the full dataset including non-redistributable human commentary on your own machine or Colab runtime:

```bash
# 1. Collect raw sources
python -m data.sources.gameknot --output data/raw/gameknot.jsonl --max-items 6000
python -m data.sources.lichess_studies --output data/raw/lichess_studies.jsonl --max-items 6000 --include-all-studies
python -m data.sources.mate --output data/raw/mate.jsonl --max-items 4000
cat data/raw/gameknot.jsonl data/raw/lichess_studies.jsonl data/raw/mate.jsonl > data/raw/sft_combined.jsonl

# 2. Sample CC0 Lichess games for RL
python -m data.sources.lichess_games --output data/raw/lichess_games.jsonl --max-positions 25000

# 3. Run Stockfish analysis, filtering, splitting, and export
python -m data.pipeline.run_pipeline \
  --sft-input data/raw/sft_combined.jsonl \
  --rl-input data/raw/lichess_games.jsonl \
  --depth 12 \
  --workers 12 \
  --rl-target-count 20000
```

## 6. Known Limitations

1. **Human Commentary Grounding:** Human annotators wrote their comments without seeing our deterministic Stockfish `Facts:` lines. While Phase 2.3 filters out comments with illegal moves, wrong piece locations, or mismatched verdict tone, SFT serves primarily to teach explanation style and format. Phase 6 GRPO reinforcement learning enforces strict factual grounding.
2. **Long Commentary Scarcity:** Most move-by-move human annotations are short (15–50 words). Only ~1.6% of filtered human comments exceed 80 words (`detail: long`). The RL prompt set includes 20% `detail: long` prompts (4,000 prompts) so GRPO RL trains the `long` mode directly via the length reward check.
3. **Engine Depth:** Positions in the bulk pipeline are analyzed with Stockfish 18 at `depth=12, MultiPV=3` and cached by FEN in Parquet format. Deeper re-evaluation updates cache entries automatically when run with `--depth 18`.
