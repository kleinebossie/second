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
It uses WebGPU to run the model directly on your local device.
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
│   └── prompt_format.md     # Input prompt contract (planned for Phase 1)
├── core/                    # Shared fact extractor and prompt builder
│   ├── python/              # Python implementation (planned for Phase 1)
│   ├── ts/                  # TypeScript implementation (planned for Phase 1)
│   └── golden/              # Parity test suites (planned for Phase 1)
├── data/                    # Data pipeline
│   ├── sources/             # Data download scripts (planned for Phase 2)
│   ├── pipeline/            # Filtering and labeling scripts (planned for Phase 2)
│   └── cards/               # Dataset documentation cards (planned for Phase 2)
├── rewards/                 # Rule checkers for Reinforcement Learning (RL) and evaluation (planned for Phase 3)
├── training/                # Training pipelines
│   ├── configs/             # Training configurations per model size (planned for Phase 5)
│   ├── sft.ipynb            # Supervised Fine-Tuning notebook (planned for Phase 5)
│   └── grpo.ipynb           # Group Relative Policy Optimization notebook (planned for Phase 6)
├── eval/                    # Evaluation tools
│   ├── rating_space/        # Human evaluation interface (planned for Phase 4)
│   ├── results/             # Benchmark test results (planned for Phase 4)
│   └── reports/             # Model release reports (planned for Phase 7)
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
└── .github/workflows/       # CI test pipelines (planned for Phase 1)
```

## Licenses

This project uses dual licensing:

1. **Root License (Apache License 2.0):**
   The Apache License 2.0 governs all models, training code, datasets, and shared core libraries.
   See [LICENSE](LICENSE) for the full text.

2. **Extension License (GNU General Public License v3.0):**
   The GNU General Public License v3.0 governs all files in the `extension/` directory.
   The extension uses GPLv3 because it bundles the Stockfish chess engine.
   See [extension/LICENSE](extension/LICENSE) for the full text.
