# Teacher Selection Spike (`spikes/teacher_selection`)

This directory contains code and notebook files to evaluate teacher model candidates.
All instructions follow ASD-STE100 Simplified Technical English.

## Purpose

We must select an Apache 2.0 open-weight teacher model before full data synthesis.
The teacher model generates synthetic coaching explanations from Stockfish facts.
The selected model must maximize factual compliance and coaching usefulness.
The target compute budget is 15 Google Colab compute units or less for 8,000 examples.

## Candidate Models

1. `Qwen/Qwen3.5-9B` (bf16 baseline on NVIDIA L4 GPU)
2. `Qwen/Qwen3.5-27B` (4-bit via bitsandbytes on NVIDIA L4 GPU)
3. `Qwen/Qwen3.5-35B-A3B` (MoE: 35B total parameters, 3B active, 4-bit via bitsandbytes on NVIDIA L4 GPU)
4. `Qwen/Qwen3.6-27B` (4-bit via bitsandbytes on NVIDIA L4 GPU)

We omit `Qwen/Qwen3.8-27B`.

## Hardware Memory Verification (NVIDIA L4 GPU)

An NVIDIA L4 GPU provides 24 GB of VRAM.

Candidate 3 (`Qwen/Qwen3.5-35B-A3B`) uses a Mixture of Experts (MoE) architecture:
- Total parameters: 35 billion (35B).
- Active parameters per token: 3 billion (3B).

Memory calculation in 4-bit precision (NF4 via bitsandbytes):
- Model weights: 35B parameters at 0.5 bytes per parameter require ~17.5 GB VRAM.
- Activations and KV cache: Only 3B parameters activate per token. At batch size 1, activations and KV cache require ~2.0 GB VRAM.
- Total runtime memory: ~17.5 GB + ~2.0 GB = ~19.5 GB VRAM.

Because 19.5 GB is less than 24 GB, `Qwen/Qwen3.5-35B-A3B` in 4-bit fits on an L4 GPU. It leaves ~4.5 GB of safety margin.

## Evaluation Protocol

We evaluate each candidate model on 50 items from the stratified development set (`eval/devset-1.0.jsonl`).
Each candidate uses the Teacher Synthesis System Prompt defined in `data/pipeline/synthesize_teacher.py`.
The evaluation script measures:
- Hard rule check pass rate (`legal_moves`, `grounded_moves`, `piece_on_square`, `claims`, `coach_voice`).
- Target length compliance (`short`: 35–80 words, `long`: 80–180 words).
- Key-point coverage and counterfactual best-move explanation.
- Generation latency (seconds per sample).

## How to Run on Google Colab

Follow these steps:

1. Build the Phase 2 bundle locally:
   ```bash
   python -m training.build_colab_bundle --phase 2
   ```
   This command creates `data/processed/phase2_teacher_bundle.zip`.
2. Open Google Colab: `https://colab.research.google.com`.
3. Click **File** > **Upload notebook**.
4. Select `spikes/teacher_selection/teacher_spike.ipynb`.
5. Click **Runtime** > **Change runtime type**.
6. Select **L4 GPU** (or **A100 GPU** if available) under Hardware accelerator.
7. Click **Save**.
8. Run the notebook cells sequentially from top to bottom.
9. When Step 2 prompts for `phase2_teacher_bundle.zip`, upload `data/processed/phase2_teacher_bundle.zip` from your computer.
10. The notebook runs 50 devset prompts for each candidate model and scores them.
11. Copy the final ranked comparison Markdown table into `docs/decisions/teacher_model.md`.
12. Share the winning teacher model name with the agent to start Stage T2.3 synthesis.

## How to Run Locally (Test / Dry Run)

To test the evaluation pipeline without GPU compute:

```bash
python spikes/teacher_selection/run_teacher_spike.py --mock --count 10
```

To run Candidate 1 (bf16) on a local CUDA GPU:

```bash
python spikes/teacher_selection/run_teacher_spike.py --model Qwen/Qwen3.5-9B --count 50
```

To run Candidate 2 (4-bit) on a local CUDA GPU:

```bash
python spikes/teacher_selection/run_teacher_spike.py --model Qwen/Qwen3.5-27B --load-in-4bit --count 50
```

To run Candidate 3 (MoE 4-bit) on a local CUDA GPU:

```bash
python spikes/teacher_selection/run_teacher_spike.py --model Qwen/Qwen3.5-35B-A3B --load-in-4bit --count 50
```

To run Candidate 4 (4-bit) on a local CUDA GPU:

```bash
python spikes/teacher_selection/run_teacher_spike.py --model Qwen/Qwen3.6-27B --load-in-4bit --count 50
```
