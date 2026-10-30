// Main application coordinator for browser runtime test harness.
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

import { PROMPT_DATA } from "./prompt.js";
import { getHardwareInfo } from "./runners/hardware.js";
import { runTransformersBenchmark } from "./runners/transformersRunner.js";
import { runWebLLMBenchmark } from "./runners/webllmRunner.js";
import { runWllamaBenchmark } from "./runners/wllamaRunner.js";

// Model presets for each runtime and size
export const PRESETS = {
  transformers: {
    "0.8B": "onnx-community/Qwen3.5-0.8B-ONNX",
    "2B": "onnx-community/Qwen3.5-2B-ONNX"
  },
  webllm: {
    "0.8B": "Qwen3.5-0.8B-q4f16_1-MLC",
    "2B": "Qwen3.5-2B-q4f16_1-MLC"
  },
  wllama: {
    "0.8B": "unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
    "2B": "unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf"
  }
};

let currentAbortController = null;
let currentBenchmarkResult = null;
let hardwareData = null;
let pythonReferenceData = null;

// Initialize DOM and hardware listeners
export async function initApp() {
  const elements = getDOMElements();

  // Populate prompt preview
  if (elements.promptPreview) {
    elements.promptPreview.textContent = PROMPT_DATA.userMessage;
  }
  if (elements.promptTokenBadge) {
    elements.promptTokenBadge.textContent = `~${PROMPT_DATA.approxTokenCount} tokens`;
  }

  // Detect hardware capabilities
  hardwareData = await getHardwareInfo();
  renderHardwareInfo(hardwareData, elements);

  // Load Python reference JSON if present
  await loadPythonReference(elements);

  // Update model ID input when user changes runtime or size
  const syncModelId = () => {
    const runtime = elements.runtimeSelect.value;
    const size = elements.sizeSelect.value;
    const defaultModel = PRESETS[runtime]?.[size] || "";
    elements.modelInput.value = defaultModel;
    elements.modelInput.placeholder = defaultModel;
  };

  elements.runtimeSelect.addEventListener("change", syncModelId);
  elements.sizeSelect.addEventListener("change", syncModelId);
  syncModelId();

  // Setup Run button
  elements.runBtn.addEventListener("click", () => startBenchmark(elements));

  // Setup Stop button
  elements.stopBtn.addEventListener("click", () => stopBenchmark(elements));

  // Setup Copy Markdown Report Row button
  elements.copyReportBtn.addEventListener("click", () => copyMarkdownRow(elements));

  // Setup Load Python Reference button
  if (elements.loadPythonRefBtn) {
    elements.loadPythonRefBtn.addEventListener("click", () => loadPythonReference(elements, true));
  }
}

function getDOMElements() {
  return {
    runtimeSelect: document.getElementById("runtime-select"),
    sizeSelect: document.getElementById("size-select"),
    modelInput: document.getElementById("model-input"),
    runBtn: document.getElementById("run-btn"),
    stopBtn: document.getElementById("stop-btn"),
    copyReportBtn: document.getElementById("copy-report-btn"),
    loadPythonRefBtn: document.getElementById("load-python-ref-btn"),
    progressBar: document.getElementById("progress-bar"),
    progressText: document.getElementById("progress-text"),
    logContainer: document.getElementById("log-container"),
    outputText: document.getElementById("output-text"),
    pythonRefText: document.getElementById("python-ref-text"),
    promptPreview: document.getElementById("prompt-preview"),
    promptTokenBadge: document.getElementById("prompt-token-badge"),
    hwBadge: document.getElementById("hw-badge"),
    hwDetails: document.getElementById("hw-details"),
    // Metric cards
    metricLoadTime: document.getElementById("metric-load-time"),
    metricDownloadSize: document.getElementById("metric-download-size"),
    metricTtft: document.getElementById("metric-ttft"),
    metricSpeed: document.getElementById("metric-speed"),
    metricTokens: document.getElementById("metric-tokens"),
    metricVram: document.getElementById("metric-vram"),
    reportRowPreview: document.getElementById("report-row-preview")
  };
}

function renderHardwareInfo(info, elements) {
  if (info.webgpuSupported) {
    elements.hwBadge.className = "badge badge-success";
    elements.hwBadge.textContent = "WebGPU Available";
    elements.hwDetails.textContent = `${info.adapterName} | Max Buffer: ${info.maxBufferSizeMB} MB | Max Workgroup Storage: ${info.maxComputeWorkgroupStorageKB} KB`;
    elements.metricVram.textContent = `${info.maxBufferSizeMB} MB (buffer limit)`;
  } else {
    elements.hwBadge.className = "badge badge-danger";
    elements.hwBadge.textContent = "WebGPU Unavailable";
    elements.hwDetails.textContent = "Browser does not expose WebGPU adapter. Ensure flag chrome://flags/#enable-unsafe-webgpu is enabled.";
    elements.metricVram.textContent = "Unavailable";
  }
}

async function loadPythonReference(elements, alertOnMissing = false) {
  try {
    const res = await fetch("./reference_output.json");
    if (res.ok) {
      pythonReferenceData = await res.json();
      if (elements.pythonRefText) {
        elements.pythonRefText.value = pythonReferenceData.generated_text || "";
      }
      addLog(elements, `Loaded Python reference (${pythonReferenceData.model_id}, ${pythonReferenceData.tokens_per_second} tok/s)`);
    } else if (alertOnMissing) {
      alert("reference_output.json not found. Run 'python reference_python.py' first.");
    }
  } catch (err) {
    if (alertOnMissing) {
      alert("Could not load reference_output.json: " + err.message);
    }
  }
}

async function startBenchmark(elements) {
  const runtime = elements.runtimeSelect.value;
  const size = elements.sizeSelect.value;
  const modelId = elements.modelInput.value.trim() || PRESETS[runtime]?.[size];

  // Reset UI metrics
  resetMetricsUI(elements);
  elements.runBtn.disabled = true;
  elements.stopBtn.disabled = false;
  elements.copyReportBtn.disabled = true;

  currentAbortController = new AbortController();
  const abortSignal = currentAbortController.signal;

  addLog(elements, `Starting benchmark: Runtime=${runtime}, Size=${size}, Model=${modelId}`);
  updateProgress(elements, 5, "Initializing runtime environment...");

  try {
    let result;
    const onProgress = (p) => {
      let percent = p.progress ? Math.round(p.progress * 100) : 50;
      updateProgress(elements, percent, p.text);
      addLog(elements, p.text);
    };

    const onToken = ({ token, allText, tokenCount, ttftMs }) => {
      elements.outputText.value = allText;
      elements.metricTokens.textContent = `${tokenCount}`;
      if (ttftMs) {
        elements.metricTtft.textContent = `${ttftMs} ms`;
      }
    };

    const RUNNER_DISPATCH = {
      transformers: (opts) => runTransformersBenchmark({
        modelId: opts.modelId,
        promptMessages: PROMPT_DATA.messages,
        rawPrompt: PROMPT_DATA.rawPromptWithThinkingClosed,
        onProgress: opts.onProgress,
        onToken: opts.onToken,
        abortSignal: opts.abortSignal
      }),
      webllm: (opts) => runWebLLMBenchmark({
        modelId: opts.modelId,
        promptMessages: PROMPT_DATA.messages,
        onProgress: opts.onProgress,
        onToken: opts.onToken,
        abortSignal: opts.abortSignal
      }),
      wllama: (opts) => runWllamaBenchmark({
        modelUrlOrHf: opts.modelId,
        promptMessages: PROMPT_DATA.messages,
        rawPrompt: PROMPT_DATA.rawPromptWithThinkingClosed,
        onProgress: opts.onProgress,
        onToken: opts.onToken,
        abortSignal: opts.abortSignal
      })
    };

    const runner = RUNNER_DISPATCH[runtime];
    if (!runner) {
      throw new Error(`Unsupported runtime: ${runtime}`);
    }

    result = await runner({ modelId, onProgress, onToken, abortSignal });

    currentBenchmarkResult = result;
    renderFinalMetrics(result, elements);
    updateProgress(elements, 100, "Benchmark completed successfully.");
    addLog(elements, `Finished. Speed: ${result.tokensPerSecond} tok/s, TTFT: ${result.ttftMs} ms.`);
  } catch (err) {
    if (abortSignal.aborted) {
      updateProgress(elements, 0, "Benchmark stopped by user.");
      addLog(elements, "Benchmark stopped by user.");
    } else {
      updateProgress(elements, 0, `Error: ${err.message}`);
      addLog(elements, `Execution failed: ${err.message}`);
      alert(`Benchmark execution failed:\n${err.message}`);
    }
  } finally {
    elements.runBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.copyReportBtn.disabled = !currentBenchmarkResult;
    currentAbortController = null;
  }
}

function stopBenchmark(elements) {
  if (currentAbortController) {
    addLog(elements, "Stop request sent...");
    currentAbortController.abort();
  }
}

function resetMetricsUI(elements) {
  elements.metricLoadTime.textContent = "--";
  elements.metricDownloadSize.textContent = "--";
  elements.metricTtft.textContent = "--";
  elements.metricSpeed.textContent = "--";
  elements.metricTokens.textContent = "--";
  elements.outputText.value = "";
  elements.reportRowPreview.textContent = "";
}

function renderFinalMetrics(result, elements) {
  elements.metricLoadTime.textContent = `${(result.loadTimeMs / 1000).toFixed(2)} s`;
  elements.metricDownloadSize.textContent = result.downloadSizeMB ? `${result.downloadSizeMB} MB` : "Cached/Unknown";
  elements.metricTtft.textContent = `${result.ttftMs} ms`;
  elements.metricSpeed.textContent = `${result.tokensPerSecond} tok/s`;
  elements.metricTokens.textContent = `${result.tokenCount}`;
  elements.outputText.value = result.outputText;

  // Build markdown report row
  const row = generateMarkdownRow(result);
  elements.reportRowPreview.textContent = row;
}

function evaluateCorrectness(outputText, referenceText) {
  if (!outputText || outputText.length < 30) {
    return "No (empty or incomplete)";
  }

  // Compare with Python greedy reference output if loaded
  if (referenceText && typeof referenceText === "string") {
    const cleanOutput = outputText.trim();
    const cleanRef = referenceText.trim();
    if (cleanOutput === cleanRef) {
      return "Yes (exact match with Python)";
    }
    let matchCount = 0;
    const minLength = Math.min(cleanOutput.length, cleanRef.length);
    while (matchCount < minLength && cleanOutput[matchCount] === cleanRef[matchCount]) {
      matchCount++;
    }
    if (matchCount >= 40) {
      return `Yes (${matchCount}-char prefix match with Python)`;
    }
  }

  const hasMove = outputText.includes("Ng5");
  const hasSquare = outputText.includes("f7");
  const hasReply = outputText.includes("d5") || outputText.includes("Bc5");
  if (hasMove && hasSquare) {
    return hasReply ? "Yes (factually grounded)" : "Partial (mentions Ng5 and f7)";
  }
  return "Review required (deviates from facts)";
}

function generateMarkdownRow(result) {
  const isMV3 = (typeof chrome !== "undefined" && Boolean(chrome.runtime && chrome.runtime.id)) || window.location.protocol === "chrome-extension:";
  const mv3Text = isMV3 ? "Yes (Side Panel)" : "No (Web Tab)";
  const sizeMb = result.downloadSizeMB > 0 ? `${result.downloadSizeMB} MB` : "Cached/N/A";
  const ttft = result.ttftMs > 0 ? `${result.ttftMs} ms` : "N/A";
  const speed = result.tokensPerSecond ? `${result.tokensPerSecond}` : "0.00";
  const peakRam = hardwareData?.maxBufferSizeMB ? `${hardwareData.maxBufferSizeMB} MB (buffer)` : "N/A";
  const correct = evaluateCorrectness(result.outputText, pythonReferenceData?.generated_text);

  return `| ${result.runtime} | ${result.modelId} | ${sizeMb} | ${ttft} | ${speed} | ${peakRam} | ${mv3Text} | ${correct} |`;
}

function copyMarkdownRow(elements) {
  if (!currentBenchmarkResult) return;
  const row = generateMarkdownRow(currentBenchmarkResult);
  navigator.clipboard.writeText(row).then(() => {
    addLog(elements, "Copied Markdown report row to clipboard.");
    const originalText = elements.copyReportBtn.textContent;
    elements.copyReportBtn.textContent = "Copied!";
    setTimeout(() => {
      elements.copyReportBtn.textContent = originalText;
    }, 2000);
  }).catch((err) => {
    console.error("Clipboard copy failed:", err);
    alert("Clipboard copy failed. Please select text manually:\n\n" + row);
  });
}

function updateProgress(elements, percent, text) {
  if (elements.progressBar) {
    elements.progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  }
  if (elements.progressText) {
    elements.progressText.textContent = text;
  }
}

function addLog(elements, message) {
  if (!elements.logContainer) return;
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.textContent = `[${time}] ${message}`;
  elements.logContainer.appendChild(entry);
  elements.logContainer.scrollTop = elements.logContainer.scrollHeight;
}
