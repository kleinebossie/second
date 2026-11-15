# Privacy Policy for Second — Chess Assistant

**Last updated:** November 15, 2026

Second is a client-side chess assistant built to help players understand their games through local AI commentary without sacrificing fair play or privacy.

## 1. No Personal Data Collection

Second does not collect, sell, or transmit any personal identifying information, browsing history, or user credentials.

## 2. On-Device Local Inference

All chess position analysis, fact extraction, and language model inference run **100% locally on your computer** using WebAssembly inside your browser. No move text, board states, or explanations are sent to external AI servers.

## 3. External Network Requests

Second makes only three types of external network requests:
1. **Model Download from Hugging Face (`https://huggingface.co` and CDN hosts)**: When you explicitly consent to download model weights (~1.2 GB for Second-1.0-2B), the extension downloads the quantized weights directly from the verified Hugging Face repository and caches them in your browser.
2. **Lichess Game Verification (`https://lichess.org/game/export/...`)**: When you open a Lichess game URL, Second checks the game status to confirm the game has ended, and fetches the move list and opening metadata for finished games.
3. **Lichess Cloud Evaluation (`https://lichess.org/api/cloud-eval`)**: If enabled in settings (default on), when you request an explanation for a position, Second may query Lichess's public cloud evaluation database for the position's FEN. If Lichess has no cached eval or returns HTTP 429, or if you disable this toggle in settings, Second uses the built-in local Stockfish WASM engine instead.

## 4. Local Storage and Ratings

- Your preferences (coach style, model size, cloud eval toggle) and feedback ratings (thumbs up/down) are stored locally on your device via `chrome.storage.local`.
- You can export your ratings as a JSON file at any time using the "Export Ratings" button in Settings.
- You can delete all downloaded model weights and cached files at any time via the "Delete Downloaded Model Weights" button in Settings.

## 5. Fair Play

Second enforces strict fail-closed fair-play rules. It works only on analysis boards, studies, and games that have already ended. It will never provide analysis during active live games or puzzles.
