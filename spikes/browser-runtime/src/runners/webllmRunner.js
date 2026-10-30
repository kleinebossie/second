// Runner for WebLLM (MLC) with WebGPU.
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

import { CreateMLCEngine, prebuiltAppConfig } from "@mlc-ai/web-llm";
import { createStreamTracker, createInitialResult } from "./metrics.js";

export async function runWebLLMBenchmark({
  modelId,
  promptMessages,
  onProgress,
  onToken,
  abortSignal
}) {
  const result = createInitialResult("WebLLM (MLC)", modelId);

  try {
    if (abortSignal?.aborted) {
      throw new Error("Benchmark aborted by user");
    }

    onProgress({ phase: "start", text: `Initializing WebLLM engine with ${modelId}...` });
    const loadStartTime = performance.now();

    // Check if modelId is in prebuiltAppConfig or custom
    const isPrebuilt = prebuiltAppConfig.model_list.some((m) => m.model_id === modelId);
    let appConfig = prebuiltAppConfig;

    if (!isPrebuilt && (modelId.startsWith("http://") || modelId.startsWith("https://"))) {
      // Custom model URL
      const customModelId = "custom-model-" + Date.now();
      appConfig = {
        ...prebuiltAppConfig,
        model_list: [
          ...prebuiltAppConfig.model_list,
          {
            model: modelId,
            model_id: customModelId,
            model_lib: prebuiltAppConfig.model_list.find((m) => m.model_id.includes("Qwen"))?.model_lib
          }
        ]
      };
      modelId = customModelId;
    }

    const initProgressCallback = (report) => {
      // report has { progress: 0-1, text: string, timeElapsed: number }
      const percent = Math.round((report.progress || 0) * 100);
      onProgress({
        phase: "download",
        text: `[${percent}%] ${report.text || "Loading model..."}`,
        progress: report.progress
      });
    };

    const engine = await CreateMLCEngine(modelId, {
      appConfig,
      initProgressCallback
    });

    result.loadTimeMs = performance.now() - loadStartTime;
    onProgress({
      phase: "loaded",
      text: `Engine initialized in ${(result.loadTimeMs / 1000).toFixed(2)}s. Starting generation...`
    });

    if (abortSignal?.aborted) {
      throw new Error("Benchmark aborted by user");
    }

    const genStartTime = performance.now();
    const tracker = createStreamTracker({ genStartTime, result, onToken, abortSignal });

    // Streaming completion with greedy decoding (temperature: 0)
    const stream = await engine.chat.completions.create({
      messages: promptMessages,
      temperature: 0.0,
      max_tokens: 128,
      stream: true,
      stream_options: { include_usage: true }
    });

    for await (const chunk of stream) {
      if (abortSignal?.aborted) {
        await engine.interruptGenerate();
        break;
      }

      const delta = chunk.choices?.[0]?.delta?.content || "";
      if (delta) {
        tracker.push(delta);
      }
    }

    tracker.finish();

    onProgress({ phase: "complete", text: "Generation completed successfully." });
    return result;
  } catch (err) {
    console.error("WebLLM benchmark failed:", err);
    result.error = err.message || String(err);
    throw err;
  }
}
