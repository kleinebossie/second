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
- The 2B model must achieve at least 10 tokens per second (target: 15 tokens per second) on a mid-range laptop.
- The runtime must operate inside a Chrome Manifest V3 extension page.

## Test Configuration
- **Prompt Size:** ~400 tokens (standard Second prompt format).
- **Decoding:** Greedy decoding (temperature = 0).
- **Test Models:** Base Qwen 3.5 2B Instruct (q4), Base Qwen 3.5 0.8B Instruct (q4).

## Test Results

> [!NOTE]
> WebGPU security rules prevent web pages from reading dynamic GPU memory usage directly. In wllama, recorded memory represents the allocated WebAssembly heap buffer (1024 MB).

### 1. Workstation Laptop (Integrated Intel GPU + Dedicated NVIDIA RTX GPU)
- **Device Specifications:** Intel Core Ultra 7 255H CPU, Intel Arrow Lake-P GPU, NVIDIA RTX Discrete Laptop GPU, Ubuntu 24.04 LTS, Chrome 134.
- **Execution Backend:** wllama compiles llama.cpp to WebAssembly on the CPU. It does not dispatch WebGPU compute shaders. Inference speed depends on CPU threads, producing identical results across integrated and discrete GPU configurations.

| Runtime | Model | Download Size | Time to First Token | Tokens / Sec | Peak Buffer / RAM | MV3 Extension | Correct Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformers.js | 2B (q4) | Skipped | N/A | N/A | N/A | No | No (shader stall) |
| WebLLM | 2B (q4) | Failed | N/A | 0.00 | N/A | No | No (shader error) |
| wllama (Wasm) | 2B (q4) | 1221.5 MB | 16571 ms | 14.38 | 1024 MB (buffer) | Yes (Side Panel) | Yes (factually grounded) |
| Transformers.js | 0.8B (q4) | ~670 MB | >120,000 ms | 0.00 | N/A | No | No (stalled) |
| WebLLM | 0.8B (q4) | Failed | N/A | 0.00 | N/A | No | No (shader error) |
| wllama (Wasm) | 0.8B (q4) | 507.8 MB | 6647 ms | 28.11 | 1024 MB (buffer) | Yes (Side Panel) | Yes (factually grounded) |

### 2. Dedicated WebGPU Evaluation
We evaluated dedicated WebGPU acceleration using Transformers.js and WebLLM. Both runtimes failed to compile shaders for Qwen 3.5 hybrid attention. wllama running WebAssembly on CPU bypasses WebGPU shader compilation bugs completely.

### 3. Apple Silicon Mac
Apple Silicon hardware is not accessible in this local Linux environment. WebAssembly executes across platforms. We will validate Apple Silicon during community beta testing.

## Observations and Notes
- **Text Parity:** Both 0.8B and 2B models in wllama produced factually grounded chess commentary with greedy decoding.
- **WebGPU Stability:** WebLLM failed during initialization because the WebGPU shader compiler rejected `depthwise_conv1d_kernel`. Transformers.js stalled for more than two minutes during runtime WGSL shader compilation for Qwen 3.5.
- **Extension Isolation:** wllama runs reliably inside the Chrome Manifest V3 side panel. wllama uses WebAssembly and avoids WebGPU shader compiler bugs.

## Final Decision
- **Selected Runtime:** wllama (GGUF via WebAssembly)
- **Date of Decision:** 2026-10-30
- **Rationale:** wllama executes reliably in Chrome Manifest V3 side panels. It delivers 28.11 tokens per second on 0.8B and 14.38 tokens per second on 2B. 14.38 tokens per second satisfies the interactive user experience target (~15 tokens per second). The 0.8B model delivers 28.11 tokens per second as a fast fallback for lower-tier hardware. WebLLM and Transformers.js fail due to WebGPU shader compiler limitations with Qwen 3.5 hybrid attention. The project will export trained models to GGUF format (Q4_K_M).

## Fallback Plan
If no runtime meets the pass criteria, use one of these options:
1. Wait for upstream runtime fixes for Qwen 3.5 hybrid attention.
2. Select the 0.8B model as the primary browser target.
3. Select an alternative base model with verified WebGPU support.
