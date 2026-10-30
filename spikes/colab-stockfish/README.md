# Phase 0.4 — Stockfish in Colab Test

This directory contains the Stockfish benchmark tools for Phase 0.4.
Before you build the data pipeline, use these tools to measure Stockfish evaluation speed.

## Contents

- `phase0_4_stockfish_test.ipynb`: Jupyter notebook for Google Colab CPU or GPU runtimes.
- `benchmark_stockfish.py`: Standalone Python script to run the benchmark on any machine.

## Purpose

The benchmark measures:
- Single-PV evaluation time at depth 18.
- MultiPV=3 evaluation time at depth 18 (required for Phase 2.2).
- Thread scaling between 1 thread and 2 threads.
- 100 diverse chess positions (30 openings, 40 middlegames, 30 endgames).
- Projected run times for 10,000 and 50,000 positions.

## How to Run in Google Colab

1. Open Google Colab in your browser: `https://colab.research.google.com`.
2. Click **File** > **Upload notebook**.
3. Select `phase0_4_stockfish_test.ipynb` from this folder.
4. Click **Runtime** > **Change runtime type**.
5. Select **CPU** or **T4 GPU**. A CPU runtime uses fewer compute units.
6. Click **Runtime** > **Run all**.

## How to Run Locally

If `python-chess` is not installed, install it:
```bash
pip install python-chess
```

Run the benchmark script:
```bash
python3 benchmark_stockfish.py --depth 18 --positions 100 --multipv both --threads 2
```

If you already have Stockfish installed, pass its path:
```bash
python3 benchmark_stockfish.py --stockfish-path /usr/games/stockfish
```

## Expected Output

- If Stockfish is not present, the script downloads it automatically.
- The script benchmarks all 100 positions in both Single-PV and MultiPV=3 modes.
- The script prints average time per position and total elapsed time.
- The script prints projected times for 10,000 and 50,000 positions.
- Copy the final Markdown table into `docs/decisions/compute.md`.
