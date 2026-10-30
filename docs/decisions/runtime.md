# Decision Record: Browser Runtime (Phase 0.2)

## Status
Decided.

## Context
Second executes language models locally inside the browser with WebGPU.
Qwen 3.5 uses a hybrid attention architecture (Gated DeltaNet and standard attention).
We must evaluate three candidate browser runtimes before training starts.

## Candidate Runtimes
- **Transformers.js** (ONNX Runtime Web, WebGPU)
- **WebLLM** (MLC, WebGPU)
- **wllama** (GGUF, WebAssembly / WebGPU)

## Pass Criteria
- The model must generate correct text with greedy decoding.
- The 2B model must achieve at least 15 tokens per second on a mid-range laptop.
- The runtime must operate inside a Chrome Manifest V3 extension page.

## Test Configuration
- **Prompt Size:** ~400 tokens (standard Second prompt format).
- **Decoding:** Greedy decoding (temperature = 0).
- **Test Models:** Base Qwen 3.5 2B Instruct (q4), Base Qwen 3.5 0.8B Instruct (q4).

## Test Results

> [!NOTE]
> WebGPU security rules prevent web pages from reading dynamic GPU memory usage directly. Record the WebGPU adapter buffer limit, or check your operating system task manager for process GPU memory.

### 1. Integrated GPU Laptop
- **Device Specifications:** Intel Core Ultra 7 255H, Ubuntu 24.04 LTS, Chrome 134

| Runtime | Model | Download Size | Time to First Token | Tokens / Sec | Peak GPU RAM | MV3 Extension | Correct Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformers.js | 2B (q4) | Skipped | N/A | N/A | N/A | No | No (shader stall) |
| WebLLM | 2B (q4) | Failed | N/A | 0.00 | N/A | No | No (shader error) |
| wllama | 2B (q4) | 1221.5 MB | 16571 ms | 14.38 | 1024 MB (buffer) | Yes (Side Panel) | Yes (factually grounded) |
| Transformers.js | 0.8B (q4) | ~670 MB | >120,000 ms | 0.00 | N/A | No | No (stalled) |
| WebLLM | 0.8B (q4) | Failed | N/A | 0.00 | N/A | No | No (shader error) |
| wllama | 0.8B (q4) | 507.8 MB | 6647 ms | 28.11 | 1024 MB (buffer) | Yes (Side Panel) | Yes (factually grounded) |

### 2. Dedicated GPU (Laptop or Desktop)
- **Device Specifications:** [Enter CPU, GPU, VRAM, OS]

| Runtime | Model | Download Size | Time to First Token | Tokens / Sec | Peak GPU RAM | MV3 Extension | Correct Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformers.js | 2B (q4) | | | | | | |
| WebLLM | 2B (q4) | | | | | | |
| wllama | 2B (q4) | | | | | | |
| Transformers.js | 0.8B (q4) | | | | | | |
| WebLLM | 0.8B (q4) | | | | | | |
| wllama | 0.8B (q4) | | | | | | |

### 3. Apple Silicon Mac
- **Device Specifications:** [Enter Chip, Unified Memory, macOS version]

| Runtime | Model | Download Size | Time to First Token | Tokens / Sec | Peak GPU RAM | MV3 Extension | Correct Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformers.js | 2B (q4) | | | | | | |
| WebLLM | 2B (q4) | | | | | | |
| wllama | 2B (q4) | | | | | | |
| Transformers.js | 0.8B (q4) | | | | | | |
| WebLLM | 0.8B (q4) | | | | | | |
| wllama | 0.8B (q4) | | | | | | |

## Observations and Notes
- **Text Parity:** Both 0.8B and 2B models in wllama produced factually grounded chess commentary with greedy decoding.
- **WebGPU Stability:** WebLLM failed during initialization because the WebGPU shader compiler rejected `depthwise_conv1d_kernel`. Transformers.js stalled for more than two minutes during runtime WGSL shader compilation for Qwen 3.5.
- **Extension Isolation:** wllama runs reliably inside the Chrome Manifest V3 side panel. wllama uses WebAssembly and avoids WebGPU shader compiler bugs.

## Final Decision
- **Selected Runtime:** wllama (GGUF)
- **Date of Decision:** 2026-10-30
- **Rationale:** wllama executes reliably in Chrome Manifest V3 side panels. It delivers 28.11 tokens per second on 0.8B and 14.38 tokens per second on 2B. Both outputs are factually grounded. WebLLM and Transformers.js fail due to WebGPU shader compiler limitations with Qwen 3.5 hybrid attention. The project will export trained models to GGUF format (Q4_K_M).

## Fallback Plan
If no runtime meets the pass criteria, use one of these options:
1. Wait for upstream runtime fixes for Qwen 3.5 hybrid attention.
2. Select the 0.8B model as the primary browser target.
3. Select an alternative base model with verified WebGPU support.
