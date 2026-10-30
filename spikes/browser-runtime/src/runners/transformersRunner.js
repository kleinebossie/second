// Runner for Transformers.js with WebGPU.
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

import { pipeline, TextStreamer, env } from "@huggingface/transformers";
import { createStreamTracker, createInitialResult } from "./metrics.js";
import { isExtensionEnv } from "./hardware.js";

// Configure cache and wasm settings
env.allowLocalModels = false;
env.useWasmCache = false;

// Configure local ONNX wasm paths to prevent Manifest V3 CSP violations
const ortPrefix = isExtensionEnv() ? chrome.runtime.getURL("ort/") : "/ort/";

if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.wasmPaths = ortPrefix;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.proxy = false;
}

export async function runTransformersBenchmark({
  modelId,
  promptMessages,
  rawPrompt,
  onProgress,
  onToken,
  abortSignal
}) {
  const result = createInitialResult("Transformers.js (WebGPU)", modelId);

  const fileProgressMap = new Map();

  const progressCallback = (p) => {
    if (p.status === "progress" && p.file) {
      fileProgressMap.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
      let totalLoaded = 0;
      let totalExpected = 0;
      for (const f of fileProgressMap.values()) {
        totalLoaded += f.loaded;
        totalExpected += f.total;
      }
      result.downloadBytes = totalLoaded;
      result.downloadSizeMB = +(totalLoaded / (1024 * 1024)).toFixed(1);
      const percent = totalExpected > 0 ? Math.round((totalLoaded / totalExpected) * 100) : 0;
      onProgress({
        phase: "download",
        text: `Downloading ${p.file}: ${percent}% (${result.downloadSizeMB} MB)`,
        loadedBytes: totalLoaded,
        totalBytes: totalExpected
      });
    } else if (p.status === "initiate") {
      onProgress({ phase: "initiate", text: `Fetching metadata for ${p.file || "model"}...` });
    } else if (p.status === "ready") {
      onProgress({ phase: "compile", text: "Compiling WebGPU shaders..." });
    }
  };

  try {
    if (abortSignal?.aborted) {
      throw new Error("Benchmark aborted by user");
    }

    onProgress({ phase: "start", text: `Loading pipeline for ${modelId} on WebGPU...` });
    const loadStartTime = performance.now();

    const generator = await pipeline("text-generation", modelId, {
      device: "webgpu",
      dtype: "q4",
      progress_callback: progressCallback
    });

    result.loadTimeMs = performance.now() - loadStartTime;
    onProgress({
      phase: "loaded",
      text: `Model loaded in ${(result.loadTimeMs / 1000).toFixed(2)}s. Compiling WebGPU shaders...`
    });

    if (abortSignal?.aborted) {
      throw new Error("Benchmark aborted by user");
    }

    const genStartTime = performance.now();
    const tracker = createStreamTracker({ genStartTime, result, onToken, abortSignal });

    // Show heartbeat elapsed time while WebGPU compiles shaders
    const heartbeat = setInterval(() => {
      if (result.ttftMs === 0 && !abortSignal?.aborted) {
        const elapsed = ((performance.now() - genStartTime) / 1000).toFixed(1);
        onProgress({
          phase: "generating",
          text: `Compiling WebGPU shaders and generating... (${elapsed}s elapsed)`
        });
      }
    }, 1500);

    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      callback_function: (piece) => {
        clearInterval(heartbeat);
        tracker.push(piece);
      }
    });

    let output = null;
    try {
      // Execute generation with greedy decoding (do_sample: false)
      output = await generator(rawPrompt || promptMessages, {
        max_new_tokens: 128,
        do_sample: false,
        return_full_text: false,
        streamer
      });
    } finally {
      clearInterval(heartbeat);
    }

    tracker.finish(output && output[0]?.generated_text ? output[0].generated_text : "");

    onProgress({ phase: "complete", text: "Generation completed successfully." });
    return result;
  } catch (err) {
    console.error("Transformers.js benchmark failed:", err);
    result.error = err.message || String(err);
    throw err;
  }
}
