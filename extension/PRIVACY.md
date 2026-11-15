# Privacy Policy for Second — Chess Assistant

**Last updated:** November 15, 2026

Second is a client-side chess assistant. It helps players understand chess moves with local AI commentary.

## 1. No Personal Data Collection

Second does not collect, sell, or send personal data, browsing history, or user credentials.

## 2. On-Device Local Inference

Chess fact extraction and language model inference run locally in your browser with WebAssembly.
Move explanations and prompts stay on your computer.

## 3. External Network Requests

Second makes only three types of external network requests:

1. **Model Download (`https://huggingface.co` and Hugging Face CDN hosts):**
   If you click **Download Model**, Second downloads the quantized weights (~1.2 GB for Second 2B).
   Second caches the weights locally in your browser.
2. **Lichess Game Verification (`https://lichess.org/game/export/...`):**
   When you open a Lichess game URL, Second checks the game status to confirm the game has ended.
   For finished games, Second also fetches the official move list and opening metadata.
3. **Lichess Cloud Evaluation (`https://lichess.org/api/cloud-eval`):**
   **Use Lichess Cloud Eval** is on by default in Settings.
   When you request an explanation, Second queries the Lichess cloud evaluation database for the current position FEN.
   If Lichess has no cached evaluation or returns HTTP 429, Second uses local Stockfish WebAssembly.
   If you turn off this setting, Second uses local Stockfish WebAssembly for every position.

## 4. Local Storage and Ratings

- Second stores your settings (coach style, model size, cloud eval toggle) and feedback ratings locally via `chrome.storage.local`.
- You can export your ratings as a JSON file from Settings.
- You can delete downloaded model weights at any time from Settings.

## 5. Fair Play

Second enforces a fail-closed fair-play lock.
It works only on analysis boards, studies, and finished games.
It stays off during live games, puzzles, and tactical modes.
