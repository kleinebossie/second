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

`second-data-1.0` provides the training, validation, test-pool, and reinforcement learning (RL) prompt splits for **Second**. Second is a family of small language models based on Qwen 3.5. These models turn Stockfish evaluations and board facts into clear human explanations.

Every record uses the Phase 1 prompt format from `core/python` and `core/ts`. Stockfish calculates all tactical and positional facts at fixed depth with `MultiPV=3`. The language model does not calculate variations on its own. Instead, the model explains the Stockfish facts for `beginner` (under 1200 Elo) and `club` (1200–2000 Elo) players. Explanations use `short` (2–4 sentences, up to 80 words) or `long` (80–200 words) detail.

## 2. Data Sources and Licenses

The pipeline follows the license rules in [`docs/data_sources.md`](../../docs/data_sources.md):

| Data Source | License | Redistribution Policy | Public HF Export (`data/processed/public/`) | Local Rebuild (`data/processed/`) |
|---|---|---|---|---|
| **Lichess Open Game Database** (`data/sources/lichess_games.py`) | CC0 1.0 Universal | Full public domain rights. | Yes (`rl_prompts.*`) | Yes (`rl_prompts.*`) |
| **MATE Dataset** (`data/sources/mate.py`, Wang et al., NAACL 2025) | MIT License | Permissive redistribution with MIT notice. | Yes (`sft_*.parquet`, `sft_*.jsonl`) | Yes (`sft_*.parquet`, `sft_*.jsonl`) |
| **Lichess Studies (Open License)** (`data/sources/lichess_studies.py`) | CC0 / CC-BY / Public Domain | Redistribute when an open license exists. | Yes (open-license subset) | Yes |
| **Lichess Studies (Standard ToS)** (`data/sources/lichess_studies.py`) | Lichess Terms of Service | Author holds copyright. Do not redistribute text. | No (rebuild script only) | Yes |
| **GameKnot Annotated Games** (`data/sources/gameknot.py`) | Proprietary (GameKnot EULA) | Do not redistribute commentary text. | No (rebuild script only) | Yes |

### MIT License Notice for MATE Dataset Portion

The supervised fine-tuning (SFT) splits include records from `OutFlankShu/MATE_DATASET` (Wang et al., NAACL 2025). The authors release the MATE dataset under the MIT License:

> Copyright (c) 2024 OutFlankShu (MATE Dataset Authors)
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

## 3. Pipeline Filter Counts (Phase 2.3 & 2.4)

The collectors extracted **16,000 raw human commentary records** from GameKnot (`6,000`), Lichess Studies (`6,000`), and the MATE dataset (`4,000`). Stockfish analyzed unique positions at `depth=12` and `MultiPV=3`. The pipeline stored **47,876 unique FEN evaluations** in `data/cache/stockfish_eval_cache.parquet`.

The filter module (`data/pipeline/filter_comments.py`) runs seven checks in order on each raw comment:

| Filter Stage | Description | Items Removed | Share of Raw Input |
|---|---|---:|---:|
| **Total Raw Input** | Raw extracted `(position, move, comment)` records | **16,000** | **100.0%** |
| 1. `cleaned_empty_or_short` | Fewer than 8 words remain after stripping PGN tags, move numbers, and glyphs | 4,165 | 26.0% |
| 2. `not_english` | `langdetect` (`seed=0`) identifies non-English text | 134 | 0.8% |
| 3. `no_chess_content` | Text lacks piece names, square coordinates, SAN moves, or chess terms | 931 | 5.8% |
| 4. `illegal_move_or_wrong_piece` | Text names an illegal SAN move or puts a piece on the wrong square (`rewards/checks.py`) | 4,164 | 26.0% |
| 5. `tone_disagrees_with_verdict` | Praise appears on an error move, or criticism appears on a good move | 191 | 1.2% |
| 6. `links_usernames_offensive` | Text contains URLs, `@username` handles, or profanity | 69 | 0.4% |
| 7. `duplicate` | Text matches an earlier fingerprint or 4-gram shingle set (Jaccard >= 0.8) | 152 | 1.0% |
| **Total Kept Examples** | Clean human commentary (`6,107` `short` + `87` `long`) | **6,194** | **38.7%** |

## 4. Dataset Splits and Distributions (Phase 2.5 & 2.6)

### 4.1 SFT Splits (Split Strictly by Game)

The split module (`data/pipeline/splits.py`) groups records by the SHA-256 hash of `game_id`. If a 4-field normalized FEN in `val` or `test_pool` also appears in `train`, the split module removes that record (`66` colliding opening records removed). Therefore, `train`, `val`, and `test_pool` share **zero games and zero board positions**.

| Partition | Total Records (Local Rebuild) | Public Redistributable Records (`public/`) |
|---|---:|---:|
| `sft_train` | 5,659 | 973 |
| `sft_val` (~2.3%) | 138 | 16 |
| `sft_test_pool` (~5.4%) | 331 | 86 |
| **Total SFT** | **6,128** | **1,075** |

### 4.2 RL Prompt Set (`rl_prompts`)

The Lichess collector (`data/sources/lichess_games.py`) sampled **25,000 positions** from rated CC0 Lichess games. More than 80% of these games have an average player Elo between `800` and `2000`. The RL prompt generator (`data/pipeline/rl_prompts.py`) removed all games and 4-field FENs that appear in `sft_val` or `sft_test_pool`. Stockfish analyzed the remaining positions at `depth=12` and `MultiPV=3` to build **20,000 RL prompts**:

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

To rebuild the full dataset on your local machine or Google Colab, run these commands:

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

1. **Human Commentary Grounding:** Human authors wrote their comments without seeing our Stockfish `Facts:` lines. Phase 2.3 removes comments with illegal moves, wrong piece squares, or mismatched verdict tone. However, SFT mainly teaches explanation style and format. Phase 6 GRPO reinforcement learning enforces strict factual grounding.
2. **Long Commentary Scarcity:** Most human move annotations are short (15–50 words). Only 1.4% of filtered human comments exceed 80 words (`detail: long`). To balance this, the RL prompt set includes 20% `detail: long` prompts (`4,000` prompts). GRPO RL trains the `long` mode directly with the length reward check.
3. **Engine Depth:** The bulk pipeline runs Stockfish 18 at `depth=12` and `MultiPV=3`. It caches evaluations by FEN in Parquet format. If you run the pipeline with `--depth 18`, the cache updates the shallow entries automatically.
