# Second Extension Manual Testing Guide

Follow these steps to test the Second Chrome Extension across all target scenarios.

---

## 1. Installation and Setup

1. Run `npm run build` inside `/home/joebos/programming/Second/extension`.
2. Open Google Chrome and go to `chrome://extensions`.
3. Turn on **Developer mode**.
4. Click **Load unpacked** and select `/home/joebos/programming/Second/extension/dist`.
5. Confirm that **Second - Local AI Chess Coach** appears with version `1.0.0`.

---

## 2. Test Case Matrix

### Test 2.1: Page Type Detection and Navigation
- **Analysis Board**: Go to `https://lichess.org/analysis` and open the side panel.
  - *Expected*: Lock banner stays hidden. Before a move is selected, status shows `Select a move` and **Explain Move** stays disabled. After you play or select a move, status shows `Ready` and **Explain Move** enables.
- **Start Position (Ply 0)**: Go to ply 0 (`#0` or click `|<<`).
  - *Expected*: Move pill shows `--` and does not jump to the final move.
- **Study Page**: Go to any public study `https://lichess.org/study/...`.
  - *Expected*: Panel unlocks and tracks the selected move or variation.
- **Finished Game**: Go to a finished game (for example `https://lichess.org/<gameId>`).
  - *Expected*: Status verifies via the Lichess export API, and the panel unlocks.
- **Puzzles / Storm / Racer**: Go to `https://lichess.org/training`, `https://lichess.org/storm`, or `https://lichess.org/racer`.
  - *Expected*: Fail-closed lock turns on. Banner shows *"Second is off during puzzles and tactical challenges."* **Explain Move** stays disabled.
- **Other Pages**: Go to `https://lichess.org/tv`, `https://lichess.org/messages`, or `https://lichess.org/`.
  - *Expected*: Banner shows *"Navigate to a finished Lichess game, analysis board, or study to use Second."*

### Test 2.2: Live Game Fair-Play Lock (Must Be Off)
1. Open an active live game on Lichess (`https://lichess.org/<gameId>` with status `started`).
2. Open the Second side panel.
3. *Expected*:
   - Fail-closed lock turns on at once.
   - Red banner shows *"Second is off during live games. Open the game after it ends."*
   - **Explain Move** and **Longer** stay disabled.
   - No move suggestions or evaluations appear.
4. Wait for the game to end, then refresh or select a move.
   - *Expected*: Lock banner clears and analysis unlocks.

### Test 2.3: First-Time Model Download and Recovery
1. Open the side panel before downloading weights (or click Settings -> **Delete Downloaded Model Weights**).
2. Confirm the **Local AI Model Required** banner appears.
3. Click **Download Model**.
4. *Expected*: Progress bar updates with percentage and MB count.
5. Close and reopen the side panel during or after download.
6. *Expected*: Extension recovers cleanly without corrupted state.

### Test 2.4: Local Stockfish WASM and Cloud Eval Cooldown (HTTP 429)
1. In Settings, confirm **Use Lichess Cloud Eval** is checked.
2. Select a move and click **Explain Move**.
3. If Lichess returns HTTP 429:
   - *Expected*: Cloud Eval enters a 60-second cooldown and falls back to local Stockfish WebAssembly without error.

### Test 2.5: Streaming Generation
1. Load a model.
2. Select a move and click **Explain Move**.
   - *Expected*: Commentary streams word-by-word cleanly.
3. Click **Longer**.
   - *Expected*: A longer explanation streams and replaces the short commentary.

### Test 2.6: Stop Button and Panel Close During Active Streaming
1. Click **Explain Move** or **Longer**.
2. Click **Stop** while streaming (or close the side panel and reopen it after 5 seconds).
3. *Expected*: Side panel inference aborts cleanly and controls return to `Ready`.

### Test 2.7: Feedback Ratings and JSON Export
1. Click thumbs up or thumbs down under an explanation.
2. Confirm `aria-pressed="true"` updates and status shows `Rating saved`.
3. Open Settings and click **Export Ratings (JSON)**.
4. Open the downloaded `second_ratings_*.json` file.
5. *Expected*: JSON contains valid records with `fen`, `move`, `rating`, `prompt`, `output`, and `timestamp`.
