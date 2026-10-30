# Browser Runtime Benchmark Harness

This directory contains the browser runtime benchmark test for Second.
It tests Qwen 3.5 models on the user GPU before training starts.

## Supported Runtimes

The test harness evaluates three client-side browser runtimes:

1. **Transformers.js (WebGPU)**: ONNX Runtime Web running through WebGPU.
2. **WebLLM (MLC)**: MLC-LLM WebGPU engine.
3. **wllama (GGUF)**: llama.cpp engine compiled to WebAssembly with WebGPU support.

## Supported Models

The harness supports two model sizes:
- **0.8B**: Low memory model for weak GPUs.
- **2B**: Primary target model for Second.

You can also enter custom Hugging Face model IDs or direct GGUF URLs.

---

## Prerequisites

1. Node.js (version 20 or higher) and npm.
2. A browser with WebGPU support (Chrome 113 or higher).
3. If WebGPU is not enabled, enable `chrome://flags/#enable-unsafe-webgpu`.
4. Python 3.10 or higher with `uv` (optional, for reference benchmark).

---

## Test 1: Run Local Web Page

Follow these steps to run the benchmark on a local web page:

1. Open a terminal in this directory:
   ```bash
   cd spikes/browser-runtime
   ```

2. Install the package dependencies:
   ```bash
   npm install
   ```

3. Start the local Vite development server:
   ```bash
   npm run dev
   ```

4. Open the displayed URL (usually `http://localhost:5173`) in Google Chrome.

5. Select your target runtime engine (WebLLM, Transformers.js, or wllama).

6. Select your target model size (`2B` or `0.8B`).

7. Click **Run Benchmark**.

8. Observe the real-time download progress, token generation, and performance metrics.

9. When generation completes, click **Copy Markdown Report Row**.

10. Paste the report row into `docs/decisions/runtime.md`.

---

## Test 2: Build and Test Chrome Manifest V3 Extension

Requirement 0.2.3 requires testing inside a Chrome Manifest V3 extension.
This verifies that WebGPU and memory limits function inside extension pages.

Follow these steps to build and load the extension:

1. Build the extension package:
   ```bash
   npm run build:ext
   ```
   This command creates the complete extension in `dist-ext/`.

2. Open Google Chrome.

3. Navigate to `chrome://extensions/`.

4. Enable the **Developer mode** toggle in the top-right corner.

5. Click the **Load unpacked** button.

6. Select the folder:
   `spikes/browser-runtime/dist-ext` (or the full path to `dist-ext` in your clone).

7. Click the extension toolbar icon or open the Side Panel.

8. Run the benchmark inside the Side Panel.

9. Verify that model loading and generation operate correctly.

---

## Test 3: Run Python Reference Benchmark

Run the reference script in Python to verify text accuracy and measure native GPU speed:

1. Execute the Python reference benchmark:
   ```bash
   npm run ref
   ```
   Or run directly with `uv`:
   ```bash
   uv run --with "scipy>=1.14.0" --with torch --with "transformers>=4.48.0" --with accelerate python reference_python.py --model Qwen/Qwen3.5-0.8B
   ```

2. The script runs greedy decoding (`do_sample=False`) on the fixed ~400-token prompt.

3. The script prints:
   - Input token count
   - Model load time in seconds
   - Time to first token (TTFT in milliseconds)
   - Generation speed (tokens per second)
   - Peak VRAM memory in megabytes
   - Exact generated text

4. The script saves results to `reference_output.json`.

5. When you refresh the web harness, it loads `reference_output.json` automatically for side-by-side comparison.

---

## Performance Thresholds (Pass Rule)

- Target model: 2B instruct.
- Usability threshold: at least ~15 tokens per second for interactive side panel commentary.
- Output text: must agree with the Python reference output for the same prompt facts.
- Fallback: if 2B cannot reach ~15 tokens per second on lower-tier hardware, select the 0.8B model (28.11 tokens per second) as the primary browser target. Keep the 2B model as an optional setting.
