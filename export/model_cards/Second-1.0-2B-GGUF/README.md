---
license: apache-2.0
language:
- en
tags:
- chess
- gguf
- wllama
- llama-cpp
- webassembly
- webgpu
- quantized
- q4_k_m
base_model: joebos/Second-1.0-2B
model_name: Second-1.0-2B-GGUF
pipeline_tag: text-generation
---

# Second-1.0-2B-GGUF: In-Browser Quantized Chess Model

This repository provides **`Q4_K_M` GGUF** weights for **Second-1.0-2B**, optimized for client-side in-browser execution with **[wllama](https://github.com/ngxson/wllama)** (WebAssembly).

- **Base Model**: [joebos/Second-1.0-2B](https://huggingface.co/joebos/Second-1.0-2B)
- **Repository**: [https://github.com/kleinebossie/second](https://github.com/kleinebossie/second)
- **Runtime Decision**: [docs/decisions/runtime.md](https://github.com/kleinebossie/second/blob/main/docs/decisions/runtime.md)

---

## Model Files and Browser Shards

Browsers and Chrome extensions impose memory limits when allocating single large contiguous buffers. To support browser caching and streaming downloads, this repository provides both a single monolithic GGUF file and `< 500 MB` shards:

| File Name | Size | Purpose |
|---|---|---|
| `Second-1.0-2B-Q4_K_M.gguf` | 1,204.9 MB | Standard monolithic file for `llama.cpp` CLI, Ollama, and desktop runtimes. |
| `Second-1.0-2B-Q4_K_M-00001-of-00003.gguf` | 469 MB | Browser shard 1 of 3 (for extension side panels and IndexedDB caching). |
| `Second-1.0-2B-Q4_K_M-00002-of-00003.gguf` | 453 MB | Browser shard 2 of 3. |
| `Second-1.0-2B-Q4_K_M-00003-of-00003.gguf` | 295 MB | Browser shard 3 of 3. |

---

## Browser Benchmark Results (`wllama`)

Tested in Chrome Manifest V3 Side Panel on Intel Core Ultra 7 CPU:

| Runtime | Quantization | Download Size | Time to First Token | Speed | Memory / Buffer | Status |
|---|---|---|---|---|---|---|
| **wllama (Wasm)** | `Q4_K_M` | 1,221.5 MB | 16,571 ms | **14.38 tok/s** | 1024 MB Wasm Heap | **Verified** |
| *wllama 0.8B fallback* | `Q4_K_M` | 507.8 MB | 6,647 ms | **28.11 tok/s** | 1024 MB Wasm Heap | Available |

*Note: WebLLM and Transformers.js failed shader compilation on Qwen 3.5's hybrid Gated DeltaNet attention kernels. wllama using WebAssembly operates reliably across all operating systems without GPU shader failures.*

---

## Post-Quantization Quality Check

Evaluated across 100 stratified test prompts from `eval/testset-1.0.jsonl` with greedy decoding (`temperature = 0`):

- **Baseline Hard-Fail Rate (`bfloat16`)**: 2.00% on the 100-item stratified sample (4.67% on full 300-item frozen test set)
- **Quantized Hard-Fail Rate (`Q4_K_M`)**: **4.00%** (+2.00% delta, &le; 6.70% release gate — **PASSED**)
- **Target Length Compliance**: 99.00% in-band (61.7 avg words)
- **Format Pass Rate**: 100.00%
- **First-Person Diary Voice**: 0.00%

---

## Usage in JavaScript / TypeScript with `wllama`

```typescript
import { Wllama } from '@wllama/wllama';

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': '/wllama/wllama-single.wasm',
  'multi-thread/wllama.wasm': '/wllama/wllama-multi.wasm',
};

const wllama = new Wllama(CONFIG_PATHS);

// Load sharded model files directly into browser cache
const shardUrls = [
  'https://huggingface.co/joebos/Second-1.0-2B-GGUF/resolve/main/Second-1.0-2B-Q4_K_M-00001-of-00003.gguf',
  'https://huggingface.co/joebos/Second-1.0-2B-GGUF/resolve/main/Second-1.0-2B-Q4_K_M-00002-of-00003.gguf',
  'https://huggingface.co/joebos/Second-1.0-2B-GGUF/resolve/main/Second-1.0-2B-Q4_K_M-00003-of-00003.gguf',
];

await wllama.loadModelFromUrl(shardUrls, {
  n_ctx: 1024,
  n_threads: navigator.hardwareConcurrency || 4,
});

// Run greedy inference with Second chat template
const output = await wllama.createCompletion({
  prompt: promptText,
  temperature: 0.0,
  max_tokens: 200,
});
console.log(output);
```

---

## Usage with `llama.cpp` (`llama-completion`)

```bash
llama-completion \
  -m Second-1.0-2B-Q4_K_M.gguf \
  -p "<|im_start|>system\nYou are Second, a chess assistant. You explain chess positions and moves clearly using the provided facts.<|im_end|>\n<|im_start|>user\ntask: explain_move\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" \
  -c 1024 \
  --temp 0.0 \
  -no-cnv \
  --no-display-prompt
```

---

## Fair Play Policy

Second is strictly for post-game chess review and study. It fails closed and refuses to operate during active live games.

---

## License

Apache License 2.0.
