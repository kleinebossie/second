# Phase 0.3 — Training Stack Test

This directory contains the test notebook for Phase 0.3.
Before you start full training, verify the training stack on Google Colab with this notebook.

## Purpose

The notebook tests:
- GPU hardware and memory (NVIDIA L4, 24 GB VRAM).
- Google Drive mount for saving checkpoints.
- Installation of TRL, Transformers, PEFT, and optional Unsloth kernels.
- Loading `Qwen/Qwen3.5-2B` in non-thinking mode with LoRA.
- 10 steps of Supervised Fine-Tuning (SFT).
- 10 steps of Group Relative Policy Optimization (GRPO).
- vLLM fast generation and fallback to standard generation.
- Checkpoint saving to Google Drive.
- Training resumption from the saved checkpoint.

## Prerequisites

1. A Google Colab account with Colab Pro or compute units.
2. An **NVIDIA L4 GPU** runtime.
3. Access to Google Drive with at least 5 GB of free storage.

## How to Run in Google Colab

1. Open Google Colab in your browser: `https://colab.research.google.com`.
2. Click **File** > **Upload notebook**.
3. Select `phase0_3_training_test.ipynb` from this directory.
4. Click **Runtime** > **Change runtime type**.
5. Select **L4 GPU** under Hardware accelerator.
6. Click **Save**.
7. Click **Runtime** > **Run all**.
8. If Google Colab prompts you for Google Drive permission, click **Connect to Google Drive**.

## Expected Output

- SFT executes 10 steps and logs step time and peak VRAM.
- GRPO executes 10 steps with dummy word-count rewards.
- If vLLM fast generation fails, the notebook switches to standard TRL generation automatically.
- The notebook writes a checkpoint to `/content/drive/MyDrive/second/checkpoints/phase0_test`.
- The reload section loads the checkpoint and completes 2 steps.
- The final cell prints a Markdown table.
- Copy this Markdown table into `docs/decisions/compute.md`.
