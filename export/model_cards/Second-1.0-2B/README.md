---
license: apache-2.0
language:
- en
tags:
- chess
- qwen
- grpo
- dapo
- reinforcement-learning
- reasoning
- lichess
base_model: Qwen/Qwen3.5-2B
model_name: Second-1.0-2B
pipeline_tag: text-generation
---

# Second-1.0-2B: Grounded Chess Explanation Model

**Second** is an open-source line of small language models that explain chess clearly. It is based on `Qwen/Qwen3.5-2B`. It turns engine facts into clear human coaching explanations for chess players.

Stockfish supplies the ground truth. The model does not calculate; it explains.

- **Organization**: [Second Chess](https://github.com/sponsors/kleinebossie)
- **Repository**: [https://github.com/kleinebossie/second](https://github.com/kleinebossie/second)
- **Model Family**: Second 1.0 (2B, 0.8B, 4B, 9B)
- **Browser Quantization**: [joebos/Second-1.0-2B-GGUF](https://huggingface.co/joebos/Second-1.0-2B-GGUF)

---

## Model Description

Chess engines show evaluation bars and lines of engine symbols (`+2.4`, `14. Nxd4 exd4 15. Qh5+`), but amateur players cannot understand *why* a move works or fails. Second bridges this gap by converting deterministic chess engine facts (tactical motifs, hanging pieces, refutations, material balance, and king safety) into clear, instructive prose.

Second-1.0-2B supports two target audiences:
1. **`beginner`** (ratings under ~1200): Plain English, shorter sentences, no complex jargon (no "prophylaxis", "zwischenzug", or "outpost").
2. **`club`** (ratings ~1200–2000): Concrete tactical themes, refutation reasons, and strategic ideas.

---

## Input Prompt Contract

Second uses the standard Qwen chat template. The user prompt follows a strict deterministic schema:

```text
task: explain_move
level: club
detail: short

FEN: r1bqkbnr/1p1n1ppp/p2p4/2pPp1BQ/B3P3/8/PPP2PPP/RN2K1NR b KQkq - 1 7
Black to move, move 7.
Move played: Nf6.
Verdict: Blunder.
Black's winning chances: 89% → 57%.
Best move: Qa5+.
Best line: Qa5+ Nc3 b5 Bb3 Ngf6 Qe2.
Facts:
- Nf6 attacks the queen on h5.
- After Nf6, White's best reply is Bxd7+.
- The queen on h5 is not defended.
- White has a check: Bxd7+.
- White can capture the knight on f6 with Bxf6.
- Black threatens Nxh5.
- Material: equal.
- Phase: opening.
```

### Example Model Output

> *The move Nf6 is a blunder because it ignores the immediate tactical refutation. While attacking the queen on h5 looks tempting, White responds with Bxd7+, forcing Black to lose material or suffer a devastating counterattack. The best move is Qa5+, which maintains pressure and avoids the loss of the queen. This check keeps the position balanced and prevents White from gaining a decisive advantage.*

---

## Stage T6 Frozen Evaluation Benchmarks

Second-1.0-2B was evaluated on the frozen 300-item test set (`eval/testset-1.0.jsonl`) with 9 deterministic verifiers and the debiased Gemini Flash judge:

| Metric | Base Qwen 2B | Second v1 (`grpo-1.0-2b`) | Second-1.0-2B (Release) |
|---|---|---|---|
| **Deterministic Hard-Fail Rate** | 100.0% | 7.7% | **4.67%** (14 / 300) |
| **Format Pass Rate** | 82.3% | 100.0% | **100.00%** |
| **Target Length Compliance** | 21.0% | 18.7% | **99.67%** (62.0 avg words) |
| **First-Person Diary Voice** | 12.0% | 51.3% | **0.00%** |
| **Key-Point Soft Score** | 0.2100 | 0.3860 | **0.9790** |
| **Better-Move Soft Score** | 0.0800 | 0.1560 | **1.0000** |
| **Aggressive Terminology Over-Refusal** | 15.0% | 0.0% | **0.00%** (0 / 20) |
| **Pairwise Win Rate vs Base Qwen 2B** | — | — | **100.0%** (86.2% LC-WR) |
| **Pairwise Win Rate vs Second v1** | — | — | **100.0%** (97.5% LC-WR) |
| **Pairwise Win Rate vs Gemini Flash** | — | — | **91.0%** (74.5% LC-WR) |

---

## Training Recipe

Second-1.0-2B was trained using a two-stage post-training pipeline on an NVIDIA L4 GPU:

1. **Stage T3 (Engine-Grounded Distillation Warm-up `SFT-A1`)**:
   - Trained with LoRA (`r=32`, `alpha=64`, all linear layers) on 4,457 verified items (`data/processed/sft_v2_train.jsonl`).
   - Combined Apache 2.0 teacher synthetic explanations (`Qwen2.5-7B-Instruct`) filtered by 9 rule checks with curated gold human commentary.
2. **Stage T4 (GRPO RL Warm-up `RL-B2`)**:
   - Trained for 150 steps using DAPO (`Direct Advantage Policy Optimization`, `beta=0.02`) across 5,000 difficulty-filtered prompts (`data/processed/rl_prompts_v2.jsonl`).
   - Reward v2: Graded rule check penalties, multiplicative length rescaling, and anti-diary voice shaping. No LLM judge inside the RL loop.

---

## Limitations

- **Engine Dependency**: Stockfish supplies the ground truth. If the engine analysis depth is insufficient, the facts may miss deep multi-ply variations.
- **Language**: English language only.
- **Calculation**: The model does not compute chess variations beyond the engine-provided facts and best lines.
- **Ratings**: Targeted at beginner (under ~1200) and club (~1200–2000) players. Not designed for master-level opening novelty analysis.

---

## Fair Play and Integrity Notice

Second is designed exclusively for post-game review and tactical training. The official browser extension fails closed:
- Second turns completely **OFF** during active live games, blitz games, puzzles, puzzle storms, and live tournaments.
- Second only activates on finished games (`/analysis`, study boards, or ended game URLs).
- Do not use this model or its outputs for live game assistance.

---

## License

Apache License 2.0. See [LICENSE](https://github.com/kleinebossie/second/blob/main/LICENSE) for details.
