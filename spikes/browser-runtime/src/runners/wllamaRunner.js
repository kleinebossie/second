// Runner for wllama (GGUF via WASM / WebGPU).
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

import { Wllama } from "@wllama/wllama/esm/index.js";
import { createStreamTracker, createInitialResult } from "./metrics.js";
import { isExtensionEnv } from "./hardware.js";

// In Chrome extensions, load local WASM to follow Manifest V3 rules.
// In standard web pages, use the CDN binary.
function getWasmConfig() {
  if (isExtensionEnv() && chrome?.runtime?.getURL) {
    return {
      default: chrome.runtime.getURL("wasm/wllama.wasm")
    };
  }
  return {
    default: "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.8.1/esm/wasm/wllama.wasm"
  };
}

export async function runWllamaBenchmark({
  modelUrlOrHf,
  promptMessages,
  rawPrompt,
  onProgress,
  onToken,
  abortSignal
}) {
  const modelId = typeof modelUrlOrHf === "string" ? modelUrlOrHf : `${modelUrlOrHf.repo}/${modelUrlOrHf.file}`;
  const result = createInitialResult("wllama (GGUF)", modelId);

  let wllama = null;

  try {
    if (abortSignal?.aborted) {
      throw new Error("Benchmark aborted by user");
    }

    onProgress({ phase: "start", text: "Initializing wllama WASM runtime..." });
    const loadStartTime = performance.now();

    wllama = new Wllama(getWasmConfig(), {
      suppressNativeLog: false
    });

    const progressCallback = ({ loaded, total }) => {
      result.downloadBytes = loaded;
      result.downloadSizeMB = +(loaded / (1024 * 1024)).toFixed(1);
      const percent = total > 0 ? Math.round((loaded / total) * 100) : 0;
      onProgress({
        phase: "download",
        text: `Downloading GGUF: ${percent}% (${result.downloadSizeMB} MB)`,
        loadedBytes: loaded,
        totalBytes: total
      });
    };

    if (typeof modelUrlOrHf === "string") {
      // Direct URL or HF shorthand
      if (modelUrlOrHf.startsWith("http://") || modelUrlOrHf.startsWith("https://")) {
        await wllama.loadModelFromUrl(modelUrlOrHf, {
          progressCallback
        });
      } else {
        // e.g. "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
        const parts = modelUrlOrHf.split("/");
        if (parts.length >= 3) {
          const repo = `${parts[0]}/${parts[1]}`;
          const file = parts.slice(2).join("/");
          await wllama.loadModelFromHF({ repo, file }, { progressCallback });
        } else {
          throw new Error(`Invalid GGUF model path or URL: ${modelUrlOrHf}`);
        }
      }
    } else {
      await wllama.loadModelFromHF(modelUrlOrHf, { progressCallback });
    }

    result.loadTimeMs = performance.now() - loadStartTime;
    onProgress({
      phase: "loaded",
      text: `Model loaded in ${(result.loadTimeMs / 1000).toFixed(2)}s. Starting generation...`
    });

    if (abortSignal?.aborted) {
      throw new Error("Benchmark aborted by user");
    }

    const genStartTime = performance.now();
    const tracker = createStreamTracker({ genStartTime, result, onToken, abortSignal });

    // Use raw formatted prompt with closed thinking tags to bypass GGUF Jinja macro bugs
    const stream = rawPrompt
      ? await wllama.createCompletion({
          prompt: rawPrompt,
          temperature: 0.0,
          max_tokens: 128,
          stream: true,
          abortSignal
        })
      : await wllama.createChatCompletion({
          messages: promptMessages,
          temperature: 0.0,
          max_tokens: 128,
          stream: true,
          abortSignal
        });

    for await (const chunk of stream) {
      if (abortSignal?.aborted) {
        break;
      }

      const delta = typeof chunk === "string"
        ? chunk
        : (chunk?.choices?.[0]?.text ?? chunk?.choices?.[0]?.delta?.content ?? chunk?.token ?? "");

      if (delta) {
        tracker.push(delta);
      }
    }

    tracker.finish();

    onProgress({ phase: "complete", text: "Generation completed successfully." });
    return result;
  } catch (err) {
    console.error("wllama benchmark failed:", err);
    result.error = err.message || String(err);
    throw err;
  } finally {
    if (wllama) {
      try {
        await wllama.exit();
      } catch (e) {
        console.warn("Error exiting wllama:", e);
      }
    }
  }
}
