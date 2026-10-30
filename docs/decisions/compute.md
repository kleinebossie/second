# Decision Record: Training Stack and Compute (Phase 0.3)

## Status
Decided.

## Context
Second fine-tunes Qwen 3.5 models with Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
We test the training stack on Google Colab before full training.
We must verify kernel compatibility, checkpoint storage, and compute efficiency on an NVIDIA L4 GPU.

## Pass Criteria
- Complete 10 steps of SFT on dummy data without memory errors.
- Complete 10 steps of GRPO with a dummy reward function.
- Save a checkpoint to Google Drive.
- Resume training successfully from the saved checkpoint in a new session.
- Measure compute unit consumption per hour.

## Test Environment
- **Platform:** Google Colab
- **Accelerator:** NVIDIA L4 GPU (24 GB VRAM)
- **Base Model:** Qwen 3.5 2B Instruct (`Qwen/Qwen3.5-2B`)
- **Adaptation Method:** LoRA (Rank 16, Alpha 16, targeting all linear projections)
- **Framework Candidate:** TRL (Transformer Reinforcement Learning) + PEFT

## Kernel Compatibility Matrix

| Dependency | Version Tested | Installation Status | Notes |
| :--- | :--- | :--- | :--- |
| PyTorch | 2.5+ | Installed | Native execution |
| CUDA | 12.4 | Active | NVIDIA L4 GPU |
| flash-linear-attention | Reference | PyTorch fallback | Native PyTorch implementation works cleanly |
| causal-conv1d | Reference | PyTorch fallback | Native PyTorch implementation works cleanly |
| vLLM | 0.7+ | Unsupported for Qwen 3.5 | Fallback to standard TRL generation works reliably |

## Training Benchmark Results

### 1. SFT Smoke Test (10 Steps)
- **Framework Used:** TRL (`SFTTrainer`) + PEFT LoRA
- **Batch Size:** 2 (sequence length: 512)
- **Sequence Length:** 512
- **Average Step Time:** 1.12 s
- **Peak VRAM Used:** 5.85 GB (within 24 GB L4 limit)
- **Errors Encountered:** TRL 0.15 missing Triton dependency. We resolved this by routing loss through standard `Trainer.compute_loss`.

### 2. GRPO Smoke Test (10 Steps)
- **Framework Used:** TRL (`GRPOTrainer`)
- **Group Size (G):** 4 generations per prompt
- **vLLM Integration:** Unsupported (Fallback to standard TRL generation active)
- **Average Step Time:** 11.85 s
- **Peak VRAM Used:** 4.38 GB (within 24 GB L4 limit)
- **Notes:** Standard generation achieves 11.85 s per step without vLLM

### 3. Checkpoint Persistence Test
- **Storage Target:** Google Drive (`/content/drive/MyDrive/second/checkpoints/phase0_test`)
- **Save Operation:** Pass
- **Save Time:** ~15 seconds
- **Resume Operation:** Pass
- **Resume Verification:** We resumed training from the saved checkpoint and completed 2 steps.

## Compute Cost Measurements

| Activity | Colab Units / Hour | Step Time | Estimated Total Steps | Estimated Total Units | Estimated Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SFT (2B) | 2.5 | 1.12 s | 1,000 | 0.8 units | 0.31 hours (19 min) |
| GRPO (2B) | 2.5 | 11.85 s | 500 | 4.1 units | 1.65 hours (99 min) |
| **Total 2B Training** | **2.5** | Combined | 1,500 | **4.9 units** | **1.96 hours** |

## Stockfish Engine Benchmark (Phase 0.4)
- **Hardware:** Intel Core Ultra 7 255H CPU (16 cores, 22 threads), Ubuntu 24.04 LTS
- **Engine:** Stockfish 19 (Linux x86-64 universal)

| Evaluation Mode | Depth | Threads | Positions Tested | Total Time | Avg Time / Position | Projected 10k Positions | Projected 50k Positions |
|---|---|---|---|---|---|---|---|
| Single-PV | 18 | 4 | 100 | 21.48 s | 0.215 s | 0.60 hours | 2.98 hours |
| MultiPV=3 | 18 | 4 | 100 | 63.46 s | 0.635 s | 1.76 hours | 8.81 hours |
| MultiPV=3 | 18 | 2 | 10 | 10.53 s | 1.052 s | 2.92 hours | 14.62 hours |

### Stockfish Compute Decision
We provide the verified Colab test notebook (`spikes/colab-stockfish/phase0_4_stockfish_test.ipynb`).
However, we select the local CPU machine for dataset generation.
Generating 10,000 positions with MultiPV=3 takes approximately 1.76 hours on 4 threads.
Generating 50,000 positions takes approximately 8.81 hours.
Local execution costs $0.00 and eliminates cloud session timeouts.

## Final Decision
- **Primary Training Framework:** TRL + PEFT LoRA (bfloat16)
- **vLLM Status for GRPO:** Disabled (uses standard TRL generation fallback)
- **Required Fast Kernels:** None mandatory (native PyTorch fallbacks run efficiently on L4)
- **Date of Decision:** 2026-10-30
- **Rationale:** The entire training stack executes reliably on an NVIDIA L4 GPU. SFT uses 5.85 GB VRAM and GRPO uses 4.38 GB VRAM, well within 24 GB. The full 2B training plan consumes 4.9 Colab units (~$1.00), far below the $50 budget limit. We saved checkpoints to Google Drive and resumed training successfully.
