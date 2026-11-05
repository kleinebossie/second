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

## Repository Layout

```text
second/
├── LICENSE                  # Apache 2.0 license
├── README.md                # Project documentation
├── docs/                    # Architecture records and specifications
│   ├── decisions/           # Runtime and compute benchmark records
│   ├── data_sources.md      # Data licensing and pipeline plans
│   └── prompt_format.md     # Input prompt contract (Phase 1)
├── core/                    # Shared fact extractor and prompt builder
│   ├── python/              # Python implementation (Phase 1)
│   ├── ts/                  # TypeScript implementation (Phase 1)
│   └── golden/              # 200 parity test positions and generator (Phase 1)
├── data/                    # Data pipeline (Phase 2)
│   ├── sources/             # Data download and extraction scripts (Phase 2.1)
│   ├── pipeline/            # Stockfish analysis, filtering, labeling, and splitting (Phase 2.2-2.7)
│   └── cards/               # Dataset documentation cards (Phase 2.7)
├── rewards/                 # Rule checkers for Reinforcement Learning (RL) and evaluation (Phase 3)
├── training/                # Training pipelines
│   ├── configs/             # Training configurations per model size (planned for Phase 5)
│   ├── sft.ipynb            # Supervised Fine-Tuning notebook (planned for Phase 5)
│   └── grpo.ipynb           # Group Relative Policy Optimization notebook (planned for Phase 6)
├── eval/                    # Evaluation harness and frozen test set (Phase 4)
│   ├── testset-1.0.jsonl    # Frozen 300-item evaluation set (Phase 4.1)
│   ├── build_testset.py     # Deterministic test set builder (Phase 4.1)
│   ├── metrics.py           # Automatic rule and length metrics (Phase 4.2)
│   ├── judge.py             # Gemini Flash rubric and pairwise judge (Phase 4.3)
│   ├── baselines.py         # Baseline explanation generator (Phase 4.4)
│   ├── run_eval.py          # Single-command evaluation harness (Phase 4)
│   ├── rating_space/        # Human evaluation Gradio Space (Phase 4.5)
│   ├── results/             # Benchmark test results (Phase 4)
│   └── reports/             # Model evaluation reports (Phase 4 / Phase 7)
├── export/                  # Quantization and browser conversion (planned for Phase 8)
├── extension/               # Lichess Chrome extension (GPLv3) (planned for Phase 9)
│   ├── LICENSE              # GNU General Public License v3.0
│   ├── src/                 # Extension source code (planned for Phase 9)
│   ├── public/              # Static assets and manifest (planned for Phase 9)
│   └── TESTING.md           # Extension test checklist (planned for Phase 9)
├── spikes/                  # Research prototypes and benchmarks (Phase 0)
│   ├── browser-runtime/     # Browser runtime test harness (Phase 0.2)
│   ├── colab-training/      # Colab training stack test (Phase 0.3)
│   └── colab-stockfish/     # Stockfish benchmark (Phase 0.4)
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
