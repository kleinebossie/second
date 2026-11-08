# Second

Second is an open-source chess explanation system.
Standard chess engines show numbers and move sequences.
Second explains the ideas behind chess moves in clear language.
The system helps beginners and intermediate players understand their games.

## Model Family

Second trains four model sizes based on the Qwen 3.5 architecture:

- **Second-1.0-2B:** The primary model. It balances speed and quality on consumer devices.
- **Second-1.0-0.8B:** A lightweight model. It runs fast on low-power devices and older laptops.
- **Second-1.0-4B:** An intermediate model. It provides deeper explanations on powerful graphics cards.
- **Second-1.0-9B:** The largest model. It runs locally on dedicated desktop workstations or servers.

Every model size uses the same prompt format and training recipe.

## Browser Extension

The project includes a Manifest V3 browser extension for Lichess.
The extension runs inside the Chrome side panel.
It executes the model directly on your local GPU with WebGPU.
Stockfish runs locally in WebAssembly.
Your game data stays private.
The extension sends no positions to external model servers.

### Fair-Play Rule

Second protects fair play in online chess.
The extension functions only after a game ends.
If a game is in progress, the extension disables all analysis.
The extension permits analysis only on these pages:
- Completed games.
- Analysis boards (`/analysis`).
- Study pages (`/study`).

When an active game is open, the panel displays:
> "Second is off during live games. Open the game after it ends."

## Current Status (Certified `Second-1.0-2B` Release Candidate)

Phases 0 through 7 (including the Phase 6.5 capability iteration in [`TRAINING_PLAN.md`](TRAINING_PLAN.md)) are **complete**. The winning 2B model (`RL-B2`, trained with SFT v2 `r=32` + 150 steps of token-level DAPO GRPO v2 with Reward v2) passed Stage T6 verification on the frozen 300-item test set ([`eval/reports/grpo-1.0-2b-v2.md`](eval/reports/grpo-1.0-2b-v2.md)):
- **Hard-Fail Rate (All 9 Deterministic Checks)**: `4.67%` (`14 / 300` items, gate $\le 5.0\%$)
- **In-Band Length Rate**: `99.67%` (`62.0` words mean, gate $\ge 80.0\%$)
- **First-Person Diary Voice Rate**: `0.00%` (down from `51.3%` in Iteration 1)
- **Better-Move Counterfactual Score**: `1.0000` (`100.0%` on bad moves)
- **Debiased Gemini Judge Overall**: `3.96 / 5.0` (up from `2.78` in Iteration 1)
- **Pairwise Win Rate**: `281W–0L` (`100.0%` decisive win rate, `97.5%` LC-WR) vs Iteration 1 `grpo-1.0-2b`, `293W–0L` (`86.2%` LC-WR) vs `base-qwen-2b`, and `202W–20L` (`91.0%` decisive win rate, `74.5%` LC-WR) vs `gemini-flash`.
- **Next Active Phase**: **Phase 8 (Browser GGUF `Q4_K_M` Export for `wllama`)** $\to$ **Phase 9 (Lichess Browser Extension)**.

## Repository Layout

```text
second/
├── LICENSE                  # Apache 2.0 license
├── README.md                # Project documentation
├── PLAN.md                  # Master project plan (Phases 0–12)
├── TRAINING_PLAN.md         # Phase 6.5 capability iteration plan & Stage T0–T6 results
├── docs/                    # Architecture records and specifications
│   ├── decisions/           # Runtime (wllama GGUF), compute, and teacher model records
│   ├── data_sources.md      # Data licensing and pipeline plans
│   └── prompt_format.md     # Input prompt contract (Phase 1)
├── core/                    # Shared fact extractor and prompt builder
│   ├── python/              # Python implementation + tactical motif detectors (Phase 1 / T1)
│   ├── ts/                  # TypeScript implementation with 100% parity (Phase 1 / T1)
│   └── golden/              # 230 parity test positions and generator (Phase 1 / T1)
├── data/                    # Data pipeline and teacher distillation (Phase 2 / T2)
│   ├── sources/             # Data download and extraction scripts
│   ├── pipeline/            # Stockfish analysis, teacher synthesis, filtering, and RL v2 prompts
│   └── cards/               # Dataset documentation cards
├── rewards/                 # Rule checkers, relational/opening claim verifiers, and Reward v2
├── training/                # SFT and GRPO training pipelines (Phases 5–6.5)
│   ├── configs/             # Frozen recipe configs (second-1.0-2b.yaml, sft_v2_*, grpo_v2_*)
│   ├── sft_v2.ipynb         # Stage T3 SFT v2 training notebook
│   ├── grpo_v2.ipynb        # Stage T4 GRPO v2 (DAPO + Reward v2) notebook
│   └── stage_t6_eval.ipynb  # Stage T6 self-contained frozen test-set evaluation notebook
├── eval/                    # Evaluation harness, dev set (150 items), and frozen test set (300 items)
│   ├── devset-1.0.jsonl     # Stratified 150-item development set (Stage T0)
│   ├── testset-1.0.jsonl    # Frozen 300-item evaluation set (Phase 4.1)
│   ├── metrics.py           # Automatic rule, motif, concept-tier, and length metrics
│   ├── judge.py             # Debiased Gemini Flash rubric and swap-consistent pairwise judge
│   ├── lc_winrate.py        # Length-Controlled Win Rate (LC-WR) logistic regression
│   ├── run_eval.py          # Single-command evaluation harness
│   ├── rating_space/        # Human evaluation Gradio Space
│   ├── results/             # Benchmark test results and experiment_ledger.jsonl
│   └── reports/             # Model evaluation reports (grpo-1.0-2b-v2.md, Second-1.0-2B.md)
├── export/                  # LoRA merge and GGUF Q4_K_M browser conversion (Phase 8 — Next)
├── extension/               # Lichess Chrome extension (GPLv3) (Phase 9)
│   ├── LICENSE              # GNU General Public License v3.0
│   ├── src/                 # Extension source code
│   ├── public/              # Static assets and manifest
│   └── TESTING.md           # Extension test checklist
├── spikes/                  # Research prototypes and benchmarks (Phase 0 & Stage T2.1)
└── .github/workflows/       # CI test pipelines (Phase 1)
```

## Development

### Virtual Environment Setup

To create a virtual environment and install the development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Running Tests

To run the Python tests from the repository root:

```bash
pytest core/python data/sources data/pipeline rewards eval
```

### Evaluation Harness

To run the full Phase 4 evaluation harness on a model checkpoint or pre-generated outputs:

```bash
python -m eval.run_eval --model-path Qwen/Qwen3.5-2B-Instruct --model-name base-qwen-2b
```

If you evaluate offline without a Gemini API key, add `--skip-judge-if-no-key`:

```bash
python -m eval.run_eval --model-path rule-baseline --backend rule-baseline --model-name rule-baseline --skip-judge-if-no-key
```

To run the TypeScript core tests:

```bash
cd core/ts
npm ci
npm test
```

### Data Pipeline

The data pipeline extracts, filters, and formats chess data for training.
To view all available options, run the pipeline CLI entry point from the repository root:

```bash
python -m data.pipeline.run_pipeline --help
```

The pipeline CLI supports these options:
- `--sft-input`: Input JSONL file of human SFT commentary.
- `--rl-input`: Input JSONL file of CC0 Lichess game positions.
- `--output-dir`: Output directory for full splits.
- `--public-dir`: Output directory for public splits.
- `--export-dir`: Output directory for Hugging Face export package.
- `--cache-path`: Path to Stockfish Parquet cache file.
- `--drive-dir`: Google Drive sync directory.
- `--stockfish-path`: Path to Stockfish executable.
- `--depth`: Stockfish search depth.
- `--workers`: Stockfish worker count.
- `--eval-pool-size`: Target number of evaluation pool items.
- `--rl-target-count`: Target number of RL prompts.
- `--seed`: Random seed for deterministic selection.
- `--upload-hf`: Upload public files to Hugging Face Hub.
- `--sync-drive`: Synchronize files to Google Drive.

### Local Data and Credentials

The directories `data/raw/`, `data/processed/`, and `data/cache/` store local data only.
Git ignores these directories.
The project publishes the dataset on Hugging Face.
If you use Hugging Face authentication, store `HF_ACCESS_TOKEN` in a local `.env` file.
Never commit the `.env` file to version control.

## Licenses

This project uses dual licensing:

1. **Root License (Apache License 2.0):**
   The Apache License 2.0 governs all models, training code, datasets, and shared core libraries.
   See [LICENSE](LICENSE) for the full text.

2. **Extension License (GNU General Public License v3.0):**
   The GNU General Public License v3.0 governs all files in the `extension/` directory.
   The extension uses GPLv3 because it bundles the Stockfish chess engine.
   See [extension/LICENSE](extension/LICENSE) for the full text.
