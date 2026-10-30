// Shared metrics calculation and result finalization for benchmark runners.
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

export function finalizeResult({
  result,
  tokens,
  generatedChunks,
  genStartTime,
  firstTokenTime,
  fallbackOutputText = ""
}) {
  const genEndTime = performance.now();
  result.generationTimeMs = genEndTime - (firstTokenTime || genStartTime);
  result.totalTimeMs = genEndTime - genStartTime;
  result.tokenCount = tokens;

  if (result.generationTimeMs > 0 && tokens > 1) {
    result.tokensPerSecond = +((tokens - 1) / (result.generationTimeMs / 1000)).toFixed(2);
  } else if (result.totalTimeMs > 0 && tokens > 0) {
    result.tokensPerSecond = +(tokens / (result.totalTimeMs / 1000)).toFixed(2);
  } else {
    result.tokensPerSecond = 0.0;
  }

  if (generatedChunks && generatedChunks.length > 0) {
    result.outputText = generatedChunks.join("").trim();
  } else if (fallbackOutputText) {
    result.outputText = fallbackOutputText.trim();
  } else {
    result.outputText = "";
  }

  return result;
}

export function createStreamTracker({ genStartTime, result, onToken, abortSignal = null }) {
  let tokens = 0;
  let firstTokenTime = null;
  const generatedChunks = [];

  function push(chunk) {
    if (abortSignal?.aborted) return;
    const now = performance.now();
    if (firstTokenTime === null) {
      firstTokenTime = now;
      result.ttftMs = Math.round(now - genStartTime);
    }
    tokens++;
    generatedChunks.push(chunk);
    if (onToken) {
      onToken({
        token: chunk,
        allText: generatedChunks.join(""),
        tokenCount: tokens,
        ttftMs: result.ttftMs
      });
    }
  }

  function finish(fallbackOutputText = "") {
    return finalizeResult({
      result,
      tokens,
      generatedChunks,
      genStartTime,
      firstTokenTime,
      fallbackOutputText
    });
  }

  return { push, finish };
}
