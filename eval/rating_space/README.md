---
title: Second Human Rating Arena
emoji: ♟️
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Blind human evaluation arena for chess explanations
---

# Second Human Rating Arena

This directory contains the Hugging Face Space for blind human evaluation.
Human raters compare explanation pairs blindly.
The application records which explanation teaches the move better.

## Comparison Sets

The rating arena evaluates three comparison sets:
1. `second_vs_base_qwen`: Second against the base Qwen model.
2. `second_vs_gemini_flash`: Second against Gemini Flash.
3. `second_vs_chesscom`: Second against Chess.com Coach explanations.

## Deployment to Hugging Face Spaces

Follow these steps to deploy this Space:
1. Create a new Space on Hugging Face.
2. Select the `Gradio` SDK.
3. Select the free CPU hardware tier.
4. Clone your new Space repository to your local computer.
5. Copy all files from `eval/rating_space/` into the Space repository.
6. Push the files to the `main` branch on Hugging Face.

When you push the files, Hugging Face starts the Space automatically.

## Secret Configuration

When you want to record votes to a private Hugging Face dataset, configure two Space secrets:
1. Open Space Settings on Hugging Face.
2. Navigate to the **Variables and secrets** section.
3. Add a new secret named `HF_TOKEN`. Enter your personal Hugging Face access token with write permissions.
4. Add a new secret named `RATING_DATASET_REPO`. Enter your target private dataset repository (for example, `username/second-human-ratings`).

When these secrets are present, the application uploads votes directly to your private dataset.
When these secrets are absent, the application saves votes to `votes.jsonl` on local disk only.

## Private Chess.com Coach Dataset

Follow these steps to evaluate against the Chess.com Coach:
1. Analyse 30 of your personal games with Chess.com Game Review.
2. Open `chesscom_template.jsonl` in this directory.
3. Copy `chesscom_template.jsonl` to `chesscom_pairs.jsonl`.
4. Enter the position FEN and the played move for each game moment.
5. Paste the Coach explanation text into the `model_2_text` field.
6. Paste the Second model explanation into the `model_1_text` field.
7. Upload `chesscom_pairs.jsonl` to your private dataset repository.

> [!CAUTION]
> Never commit private Chess.com coach texts to a public Git repository.
> Chess.com coach explanations are proprietary.
> Keep private rating files only in private datasets or local drives.

## Data Schema

### Pair Schema (`sample_pairs.jsonl`)

Each comparison row contains:
- `id`: Unique identifier for the position pair.
- `comparison_set`: Name of the comparison set.
- `fen`: Chess position in Forsyth-Edwards Notation.
- `move_san`: Played move in Standard Algebraic Notation.
- `move_uci`: Played move in Universal Chess Interface notation.
- `level`: Target player rating group (`beginner` or `club`).
- `detail`: Explanation length target (`short` or `long`).
- `model_1_name`: First model name.
- `model_1_text`: Explanation from the first model.
- `model_2_name`: Second model name.
- `model_2_text`: Explanation from the second model.

### Vote Record Schema (`votes.jsonl`)

Each vote contains:
- `vote_id`: Unique identifier for the vote.
- `timestamp`: UTC timestamp in ISO-8601 format.
- `session_id`: Unique identifier for the rater session.
- `rater_id`: Rater nickname or ID.
- `item_id`: Unique identifier of the evaluated pair.
- `comparison_set`: Name of the comparison set.
- `fen`: Chess position in FEN notation.
- `move_san`: Played move.
- `level`: Player rating target.
- `detail`: Detail mode.
- `model_a`: True model name shown as Explanation A.
- `model_b`: True model name shown as Explanation B.
- `choice`: Evaluator choice (`A`, `B`, or `same`).
- `winner_model`: Resolved winning model name or `tie`.
- `comment`: Optional evaluator feedback.
