---
title: Second — Chess Move Explainer
emoji: ♟️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Explain any chess move or position with Second-1.0-2B
---

# Second Demo Space

This Hugging Face Space runs **Second-1.0-2B** (`Q4_K_M` GGUF) on CPU with Stockfish.
Paste a PGN or FEN, pick a move, and read a coaching explanation.

## Deploy to Hugging Face Spaces

1. Create a new Gradio Space on Hugging Face (free CPU tier).
2. Run `python -m demo_space.prepare_hf_space` from the repository root.
3. Upload the staged folder `demo_space/dist/` to your Space repository.
