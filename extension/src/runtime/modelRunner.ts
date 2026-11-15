/**
 * Model Runtime Controller using wllama for Second Chrome Extension.
 *
 * Implements in-browser local inference per PLAN.md Sections 9.5 and 9.6:
 * - Checks WebGPU / WebAssembly support.
 * - Detects weak hardware / suggests 0.8B fallback.
 * - Handles first-time user consent to download weights.
 * - Streams tokens word-by-word for move explanations.
 * - Caches weights and allows deleting model files.
 */

import { Wllama } from '@wllama/wllama';

export interface HardwareSupport {
  webGpuSupported: boolean;
  webAssemblySupported: boolean;
  adapterInfo?: string;
  isWeakDevice: boolean;
  reason?: string;
}

export interface ModelDownloadProgress {
  phase: 'idle' | 'downloading' | 'loading' | 'ready' | 'error';
  percent: number;
  loadedBytes: number;
  totalBytes: number;
  message: string;
}

export type ProgressCallback = (progress: ModelDownloadProgress) => void;
export type TokenCallback = (token: string) => void;

export const MODEL_CONFIGS = {
  '2B': {
    name: 'Second-1.0-2B (Recommended)',
    approxSizeMB: 1205,
    url: 'https://huggingface.co/joebos/Second-1.0-2B-GGUF/resolve/main/Second-1.0-2B-Q4_K_M-00001-of-00003.gguf',
    shards: [
      'https://huggingface.co/joebos/Second-1.0-2B-GGUF/resolve/main/Second-1.0-2B-Q4_K_M-00001-of-00003.gguf',
      'https://huggingface.co/joebos/Second-1.0-2B-GGUF/resolve/main/Second-1.0-2B-Q4_K_M-00002-of-00003.gguf',
      'https://huggingface.co/joebos/Second-1.0-2B-GGUF/resolve/main/Second-1.0-2B-Q4_K_M-00003-of-00003.gguf',
    ],
  },
  '0.8B': {
    name: 'Second-1.0-0.8B (Fast / Low Memory)',
    approxSizeMB: 508,
    url: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf',
    shards: [
      'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf',
    ],
  },
};

/**
 * Finds an ASCII byte sequence inside a Uint8Array buffer.
 */
function findAsciiBytes(buf: Uint8Array, str: string): number {
  const target = new TextEncoder().encode(str);
  outer: for (let i = 0; i <= buf.length - target.length; i++) {
    for (let j = 0; j < target.length; j++) {
      if (buf[i + j] !== target[j]) continue outer;
    }
    return i;
  }
  return -1;
}

/**
 * Patches the GGUF metadata header of shard 1 in-memory (zero-copy for the bulk of the blob)
 * when a Qwen3.5 model was exported with `qwen35.nextn_predict_layers = 1` (`block_count = 25`)
 * in config.json but without the `blk.24.*` MTP tensors in the safetensors/GGUF weights.
 *
 * Specifically:
 * 1. Decrements `qwen35.block_count` by `nextn_predict_layers` (e.g., 25 -> 24).
 * 2. Sets `qwen35.nextn_predict_layers` to 0.
 * 3. Renames `qwen35.attention.recurrent_layers` (33 bytes) in-place to
 *    `qwen35.attention.recurrent_unused` (33 bytes) so llama.cpp uses its built-in
 *    `qwen35.full_attention_interval` formula for the 24 layers instead of failing array length validation.
 *
 * Because all replacements are strictly in-place with identical byte lengths,
 * all GGUF tensor offsets and 32-byte alignments remain 100% unchanged.
 */
export async function patchGgufHeaderBlob(blob: Blob): Promise<Blob> {
  if (!blob || blob.size < 64) return blob;
  const headerSize = Math.min(blob.size, 65536);
  const headerSlice = blob.slice(0, headerSize);
  const ab =
    typeof headerSlice.arrayBuffer === 'function'
      ? await headerSlice.arrayBuffer()
      : await new Response(headerSlice).arrayBuffer();
  const headerBytes = new Uint8Array(ab);

  // Verify GGUF magic ('GGUF' = 0x47, 0x47, 0x55, 0x46)
  if (
    headerBytes[0] !== 0x47 ||
    headerBytes[1] !== 0x47 ||
    headerBytes[2] !== 0x55 ||
    headerBytes[3] !== 0x46
  ) {
    return blob;
  }

  const dv = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);
  const nextnKey = 'qwen35.nextn_predict_layers';
  const nextnIdx = findAsciiBytes(headerBytes, nextnKey);
  if (nextnIdx < 0) return blob;

  // In GGUF v3, key string is followed by uint32 value_type (4 = UINT32) and uint32 value
  const nextnTypeOffset = nextnIdx + nextnKey.length;
  if (nextnTypeOffset + 8 > headerBytes.length) return blob;
  const nextnType = dv.getUint32(nextnTypeOffset, true);
  const nextnVal = dv.getUint32(nextnTypeOffset + 4, true);
  if (nextnType !== 4 || nextnVal === 0) return blob;

  const blockKey = 'qwen35.block_count';
  const blockIdx = findAsciiBytes(headerBytes, blockKey);
  if (blockIdx < 0) return blob;
  const blockTypeOffset = blockIdx + blockKey.length;
  if (blockTypeOffset + 8 > headerBytes.length) return blob;
  const blockType = dv.getUint32(blockTypeOffset, true);
  const blockVal = dv.getUint32(blockTypeOffset + 4, true);
  if (blockType !== 4 || blockVal <= nextnVal) return blob;

  // 1. Patch qwen35.block_count (e.g., 25 -> 24)
  dv.setUint32(blockTypeOffset + 4, blockVal - nextnVal, true);

  // 2. Patch qwen35.nextn_predict_layers (1 -> 0)
  dv.setUint32(nextnTypeOffset + 4, 0, true);

  // 3. Rename qwen35.attention.recurrent_layers -> qwen35.attention.recurrent_unused (same 33-byte length)
  const recKey = 'qwen35.attention.recurrent_layers';
  const recReplacement = 'qwen35.attention.recurrent_unused';
  const recIdx = findAsciiBytes(headerBytes, recKey);
  if (recIdx >= 0) {
    headerBytes.set(new TextEncoder().encode(recReplacement), recIdx);
  }

  const patchedBlob = new Blob([headerBytes, blob.slice(headerSize)], { type: blob.type });
  if ('name' in blob && (blob as any).name) {
    Object.defineProperty(patchedBlob, 'name', {
      value: (blob as any).name,
      writable: false,
      configurable: true,
    });
  }
  return patchedBlob;
}

const STOP_TOKENS = ['<|im_end|>', '<|endoftext|>'];

/**
 * Returns the length of the longest suffix of `text` that is a non-empty prefix
 * of any stop token, so partial stop tokens split across stream chunks are held back.
 */
function trailingStopPrefixLength(text: string): number {
  let maxMatch = 0;
  for (const token of STOP_TOKENS) {
    const limit = Math.min(text.length, token.length - 1);
    for (let len = limit; len >= 1; len--) {
      if (text.endsWith(token.slice(0, len))) {
        if (len > maxMatch) maxMatch = len;
        break;
      }
    }
  }
  return maxMatch;
}

export class ModelRunner {
  private wllama: any = null;
  private currentModelKey: '2B' | '0.8B' | null = null;
  private isLoaded: boolean = false;
  private isGenerating: boolean = false;
  private abortController: AbortController | null = null;

  private customWllamaInstance: any = null;
  private loadPromise: Promise<void> | null = null;
  private loadingModelKey: '2B' | '0.8B' | null = null;
  private loadAbortController: AbortController | null = null;
  private activeLoadProgressCallback?: ProgressCallback;
  private forceCpuFallback: boolean = false;
  private usingWebGpu: boolean = false;
  private cachedModelStatus: Partial<Record<'2B' | '0.8B', boolean>> = {};

  constructor(wllamaInstance?: any) {
    if (wllamaInstance) {
      this.customWllamaInstance = wllamaInstance;
      this.wllama = wllamaInstance;
      this.isLoaded = true;
      this.currentModelKey = null;
    }
  }

  private createWllamaInstance(suppressNativeLog: boolean): Wllama {
    const wasmUrl =
      typeof chrome !== 'undefined' && chrome.runtime?.getURL
        ? chrome.runtime.getURL('wasm/wllama.wasm')
        : '/wasm/wllama.wasm';
    const wllamaConfig = {
      'single-thread/wllama.wasm': wasmUrl,
      'multi-thread/wllama.wasm': wasmUrl,
      default: wasmUrl,
    };
    return new Wllama(
      wllamaConfig,
      suppressNativeLog
        ? { suppressNativeLog: true }
        : {
            suppressNativeLog: false,
            logger: {
              debug: (...args: any[]) => console.debug('[wllama]', ...args),
              log: (...args: any[]) => console.debug('[wllama]', ...args),
              warn: (...args: any[]) => console.debug('[wllama:warn]', ...args),
              error: (...args: any[]) => console.warn('[wllama:error]', ...args),
            },
          }
    );
  }

  private async teardownWllama(): Promise<void> {
    const instance = this.wllama;
    this.isLoaded = false;
    this.usingWebGpu = false;
    this.currentModelKey = null;
    if (!instance) return;

    try {
      if (typeof instance.unloadModel === 'function') {
        await instance.unloadModel();
      }
      if (!this.customWllamaInstance && typeof instance.exit === 'function') {
        await instance.exit();
      } else if (this.customWllamaInstance && typeof instance.unloadModel !== 'function' && typeof instance.exit === 'function') {
        await instance.exit();
      }
    } catch {
      // ignore teardown errors
    }
    if (!this.customWllamaInstance && this.wllama === instance) {
      this.wllama = null;
    }
  }

  /**
   * Check browser hardware capabilities for WebGPU and WebAssembly.
   */
  public static async checkHardware(): Promise<HardwareSupport> {
    const wasmSupported = typeof WebAssembly === 'object' && typeof WebAssembly.validate === 'function';

    let webGpuSupported = false;
    let adapterInfo: string | undefined;
    let isWeakDevice = false;

    if (typeof navigator !== 'undefined' && 'gpu' in navigator && (navigator as any).gpu) {
      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter) {
          webGpuSupported = true;
          if ('info' in adapter && (adapter as any).info) {
            const info = (adapter as any).info;
            adapterInfo = `${info.vendor || ''} ${info.architecture || ''} ${info.description || ''}`.trim();
          }
          // Detect integrated or low-memory adapter
          const maxBufferSize = adapter.limits?.maxBufferSize ?? 0;
          if (maxBufferSize < 1024 * 1024 * 1024) {
            isWeakDevice = true;
          }
        }
      } catch {
        webGpuSupported = false;
      }
    }

    const threads = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
    if (threads < 4) {
      isWeakDevice = true;
    }

    return {
      webGpuSupported,
      webAssemblySupported: wasmSupported,
      adapterInfo,
      isWeakDevice,
      reason: !wasmSupported
        ? 'WebAssembly is disabled or not supported in this browser.'
        : !webGpuSupported
          ? 'WebGPU is not enabled or supported on this device. Inference will run via WebAssembly CPU.'
          : undefined,
    };
  }

  public isModelLoaded(): boolean {
    return this.isLoaded && this.wllama !== null;
  }

  public getCurrentModel(): '2B' | '0.8B' | null {
    return this.currentModelKey;
  }

  /**
   * Checks whether the requested model's shards are already downloaded in OPFS cache.
   */
  public async isModelCached(modelKey: '2B' | '0.8B'): Promise<boolean> {
    if (this.isLoaded && this.currentModelKey === modelKey) return true;
    if (this.cachedModelStatus[modelKey] !== undefined) {
      return this.cachedModelStatus[modelKey]!;
    }
    const config = MODEL_CONFIGS[modelKey];
    if (!config) return false;
    try {
      const tempWllama: any = this.wllama || this.createWllamaInstance(true);
      // Prefer cacheManager.list() when available because wllama's modelManager.getModels()
      // throws if a different multi-shard model has an incomplete download in OPFS.
      if (typeof tempWllama?.cacheManager?.list === 'function') {
        const entries = await tempWllama.cacheManager.list();
        if (Array.isArray(entries)) {
          const expectedShards = config.shards?.length ? config.shards : [config.url];
          const isCached = expectedShards.every((shardUrl) =>
            entries.some(
              (entry: any) =>
                entry?.metadata?.originalURL === shardUrl &&
                typeof entry?.metadata?.originalSize === 'number' &&
                entry.metadata.originalSize > 0 &&
                entry.size === entry.metadata.originalSize
            )
          );
          this.cachedModelStatus[modelKey] = isCached;
          return isCached;
        }
      }
      if (tempWllama?.modelManager?.getModels) {
        const models = await tempWllama.modelManager.getModels();
        const isCached = models.some((m: any) => m.url === config.url);
        this.cachedModelStatus[modelKey] = isCached;
        return isCached;
      }
    } catch {
      // ignore OPFS access errors in non-browser test environments
    }
    return false;
  }

  /**
   * Aborts an in-flight model download or initialization if one is running.
   */
  public abortLoad(): void {
    if (this.loadingModelKey) {
      delete this.cachedModelStatus[this.loadingModelKey];
    }
    if (this.loadAbortController) {
      this.loadAbortController.abort();
    }
  }

  /**
   * Loads the model weights into wllama.
   *
   * Concurrent calls for the same model share a single load. This lets the
   * side panel warm the model in the background while still making
   * "Explain Move" wait for (rather than duplicate) that work.
   */
  public async loadModel(
    modelKey: '2B' | '0.8B',
    onProgress?: ProgressCallback,
    forceReload: boolean | { forceReload?: boolean; signal?: AbortSignal } = false
  ): Promise<void> {
    const isForce = typeof forceReload === 'boolean' ? forceReload : Boolean(forceReload?.forceReload);
    const callerSignal = typeof forceReload === 'object' ? forceReload?.signal : undefined;

    if (callerSignal?.aborted) {
      throw new Error('Model loading aborted');
    }

    if (isForce) {
      delete this.cachedModelStatus[modelKey];
    }

    if (!isForce && this.isLoaded && this.currentModelKey === modelKey && this.wllama) {
      onProgress?.({
        phase: 'ready',
        percent: 100,
        loadedBytes: 0,
        totalBytes: 0,
        message: 'Model is already loaded.',
      });
      return;
    }

    if (this.loadPromise) {
      if (!isForce && this.loadingModelKey === modelKey) {
        return this.loadPromise;
      }
      // A different model (or a forced reload) is being loaded. Abort & wait for it
      // to settle before starting the new load so only one runtime exists.
      this.abortLoad();
      try {
        await this.loadPromise;
      } catch {
        // Ignore the previous load's error; the caller asked for a new load.
      }
    }

    const promise = this.loadModelInternal(modelKey, onProgress, callerSignal);
    this.loadPromise = promise;
    this.loadingModelKey = modelKey;
    try {
      await promise;
    } finally {
      if (this.loadPromise === promise) {
        this.loadPromise = null;
        this.loadingModelKey = null;
      }
    }
  }

  private async loadModelInternal(
    modelKey: '2B' | '0.8B',
    onProgress?: ProgressCallback,
    callerSignal?: AbortSignal
  ): Promise<void> {
    const config = MODEL_CONFIGS[modelKey];
    if (!config) throw new Error(`Unknown model key: ${modelKey}`);

    const localLoadController = new AbortController();
    this.loadAbortController = localLoadController;
    this.activeLoadProgressCallback = onProgress;
    const onCallerAbort = () => localLoadController.abort();
    callerSignal?.addEventListener('abort', onCallerAbort, { once: true });

    try {
      if (localLoadController.signal.aborted) {
        throw new Error('Model loading aborted');
      }

      onProgress?.({
        phase: 'downloading',
        percent: 0,
        loadedBytes: 0,
        totalBytes: config.approxSizeMB * 1024 * 1024,
        message: `Connecting to Hugging Face for ${config.name}...`,
      });

      // Unload existing model if different (keep single-microtask hop for concurrent callers)
      if (this.wllama) {
        const prevWllama = this.wllama;
        this.isLoaded = false;
        this.usingWebGpu = false;
        this.currentModelKey = null;
        try {
          if (typeof prevWllama.unloadModel === 'function') {
            await prevWllama.unloadModel();
          } else if (typeof prevWllama.exit === 'function') {
            await prevWllama.exit();
          }
        } catch {
          // ignore
        }
        if (!this.customWllamaInstance && this.wllama === prevWllama) {
          this.wllama = null;
        }
      }

      const progressCallback = ({ loaded, total }: { loaded: number; total: number }) => {
        const rawPct = total > 0 ? Math.round((loaded / total) * 95) : 0;
        const pct = Math.min(95, Math.max(0, rawPct));
        const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
        const totalMB = total > 0 ? (total / (1024 * 1024)).toFixed(1) : `${config.approxSizeMB}`;
        onProgress?.({
          phase: 'downloading',
          percent: pct,
          loadedBytes: loaded,
          totalBytes: total || config.approxSizeMB * 1024 * 1024,
          message: `Downloading model weights: ${pct}% (${loadedMB} / ${totalMB} MB)`,
        });
      };

      const hasGpuApi =
        !this.forceCpuFallback &&
        typeof navigator !== 'undefined' &&
        'gpu' in navigator &&
        Boolean((navigator as any).gpu);
      const hw = hasGpuApi ? await ModelRunner.checkHardware() : null;
      if (localLoadController.signal.aborted) {
        throw new Error('Model loading aborted');
      }

      const canUseWebGpu = Boolean(hw && hw.webGpuSupported && !hw.isWeakDevice && !this.forceCpuFallback);
      const rawThreads = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
      // Leave one core free on 4+ core CPUs to keep browser and UI responsive.
      const threads = Math.min(4, Math.max(1, rawThreads >= 4 ? rawThreads - 1 : rawThreads));
      const modelUrl = config.url || (Array.isArray(config.shards) ? config.shards[0] : config.shards);

      const runLoadAttempt = async (useWebGpu: boolean): Promise<void> => {
        if (!this.customWllamaInstance) {
          this.wllama = this.createWllamaInstance(true);
        }

        // Intercept loadModel to patch GGUF header metadata before WASM parses it.
        if (this.wllama && typeof this.wllama.loadModel === 'function' && !this.wllama.__ggufPatchHooked) {
          const origLoadModel = this.wllama.loadModel.bind(this.wllama);
          this.wllama.loadModel = async (blobs: Blob[], loadParams?: any) => {
            this.activeLoadProgressCallback?.({
              phase: 'loading',
              percent: 95,
              loadedBytes: 0,
              totalBytes: 0,
              message: 'Initializing WebAssembly runtime buffers...',
            });
            const patchedBlobs = Array.isArray(blobs) ? [...blobs] : blobs;
            if (Array.isArray(patchedBlobs) && patchedBlobs.length > 0) {
              patchedBlobs[0] = await patchGgufHeaderBlob(patchedBlobs[0]);
            }
            return origLoadModel(patchedBlobs, loadParams);
          };
          this.wllama.__ggufPatchHooked = true;
        }

        await this.wllama.loadModelFromUrl(modelUrl, {
          progressCallback,
          signal: localLoadController.signal,
          n_ctx: 2048,
          n_parallel: 1,
          n_threads: threads,
          n_batch: 512,
          n_ubatch: useWebGpu ? 512 : 256,
          n_gpu_layers: useWebGpu ? 99999 : 0,
          flash_attn: true,
          cache_type_k: 'q8_0',
          cache_type_v: 'q8_0',
          cache_idle_slots: true,
        });

        if (localLoadController.signal.aborted) {
          throw new Error('Model loading aborted');
        }

        // Verify that the native llama.cpp server loaded model tensors.
        // wllama 3.8.1 ignores loadResult.success, so failed loads leave nLayer or nVocab at 0.
        if (typeof this.wllama.getModelMetadata === 'function') {
          const meta = this.wllama.getModelMetadata();
          const nLayer = meta?.hparams?.nLayer ?? 0;
          const nVocab = meta?.hparams?.nVocab ?? 0;
          if (nLayer <= 0 || nVocab <= 0) {
            throw new Error(
              `Failed to initialize ${config.name} in runtime (nLayer=${nLayer}, nVocab=${nVocab}).`
            );
          }
        }
      };

      if (canUseWebGpu) {
        try {
          await runLoadAttempt(true);
          this.usingWebGpu = true;
        } catch (webGpuErr: any) {
          const isAbort =
            localLoadController.signal.aborted ||
            webGpuErr?.name === 'AbortError' ||
            webGpuErr?.message === 'Model loading aborted';
          const isNetworkErr = /Failed to fetch|HTTP \d+|NetworkError/i.test(
            String(webGpuErr?.message || '')
          );
          if (isAbort || isNetworkErr) {
            throw webGpuErr;
          }
          console.warn('[ModelRunner] WebGPU load failed. Falling back to CPU runtime:', webGpuErr);
          this.forceCpuFallback = true;
          await this.teardownWllama();
          onProgress?.({
            phase: 'loading',
            percent: 95,
            loadedBytes: 0,
            totalBytes: 0,
            message: 'WebGPU failed. Falling back to CPU runtime...',
          });
          await runLoadAttempt(false);
          this.usingWebGpu = false;
        }
      } else {
        await runLoadAttempt(false);
        this.usingWebGpu = false;
      }

      this.currentModelKey = modelKey;
      this.isLoaded = true;
      this.cachedModelStatus[modelKey] = true;

      onProgress?.({
        phase: 'ready',
        percent: 100,
        loadedBytes: 0,
        totalBytes: 0,
        message: `${config.name} loaded and ready.`,
      });
    } catch (err) {
      delete this.cachedModelStatus[modelKey];
      await this.teardownWllama();
      throw err;
    } finally {
      callerSignal?.removeEventListener('abort', onCallerAbort);
      if (this.loadAbortController === localLoadController) {
        this.loadAbortController = null;
      }
      this.activeLoadProgressCallback = undefined;
    }
  }

  /**
   * Run streaming generation.
   */
  public async generate(
    prompt: string,
    onToken: TokenCallback,
    options: {
      maxTokens?: number;
      temperature?: number;
      userMessage?: string;
      systemMessage?: string;
    } = {}
  ): Promise<{ text: string }> {
    if (!this.isLoaded || !this.wllama) {
      throw new Error('Model is not loaded. Please download or load the model first.');
    }

    if (this.isGenerating) {
      this.stop();
    }

    const localController = new AbortController();
    this.isGenerating = true;
    this.abortController = localController;

    const maxTokens = options.maxTokens ?? 256;
    const temperature = options.temperature ?? 0.0;

    let fullText = '';
    let answerBuffer = '';
    let receivedAnyChunk = false;
    let emittedAnyToken = false;

    const emitToken = (token: string) => {
      emittedAnyToken = true;
      onToken(token);
    };

    const flushAnswerChunk = (raw: string, flushAll: boolean) => {
      if (!raw && !flushAll) return;
      answerBuffer += raw;
      if (answerBuffer.includes('<')) {
        answerBuffer = answerBuffer.replace(/<\|im_end\|>|<\|endoftext\|>/g, '');
      }
      const holdBack =
        flushAll || !answerBuffer.includes('<') ? 0 : trailingStopPrefixLength(answerBuffer);
      const safeLen = answerBuffer.length - holdBack;
      if (safeLen > 0) {
        const emit = answerBuffer.slice(0, safeLen);
        answerBuffer = answerBuffer.slice(safeLen);
        fullText += emit;
        emitToken(emit);
      }
    };

    const processChunk = (chunk: any) => {
      if (localController.signal.aborted) {
        throw new Error('Generation aborted');
      }
      const piece: string =
        chunk?.choices?.[0]?.text ??
        chunk?.choices?.[0]?.delta?.content ??
        chunk?.token ??
        chunk?.piece ??
        '';
      if (!piece) return;
      receivedAnyChunk = true;
      flushAnswerChunk(piece, false);
    };

    const startStream = async () => {
      const hasFormattedPrefill = prompt.includes('<|im_start|>assistant');
      const canChat = typeof this.wllama.createChatCompletion === 'function';
      const canComplete = typeof this.wllama.createCompletion === 'function';
      const onData = (chunk: any) => processChunk(chunk);

      if (hasFormattedPrefill && canComplete) {
        // wllama's createCompletion calls llama-server directly and reads n_predict and temp.
        return await this.wllama.createCompletion({
          prompt,
          temperature,
          temp: temperature,
          max_tokens: maxTokens,
          n_predict: maxTokens,
          stop: STOP_TOKENS,
          cache_prompt: true,
          stream: true,
          onData,
          abortSignal: localController.signal,
        } as any);
      } else if (canChat) {
        let userMsg = options.userMessage;
        let systemMsg = options.systemMessage;

        if (!userMsg) {
          const userMatch = prompt.match(/<\|im_start\|>user\n([\s\S]*?)<\|im_end\|>/);
          userMsg = userMatch ? userMatch[1].trim() : prompt;
        }

        if (!systemMsg && prompt.includes('<|im_start|>system')) {
          const sysMatch = prompt.match(/<\|im_start\|>system\n([\s\S]*?)<\|im_end\|>/);
          if (sysMatch) systemMsg = sysMatch[1].trim();
        }

        const messages: any[] = [];
        if (systemMsg) {
          messages.push({ role: 'system', content: systemMsg });
        }
        messages.push({ role: 'user', content: userMsg });

        return await this.wllama.createChatCompletion({
          messages,
          temperature,
          max_tokens: maxTokens,
          stop: STOP_TOKENS,
          cache_prompt: true,
          stream: true,
          onData,
          abortSignal: localController.signal,
        } as any);
      } else {
        return await this.wllama.createCompletion({
          prompt,
          temperature,
          temp: temperature,
          max_tokens: maxTokens,
          n_predict: maxTokens,
          stop: STOP_TOKENS,
          cache_prompt: true,
          stream: true,
          onData,
          abortSignal: localController.signal,
        } as any);
      }
    };

    const runInferenceAttempt = async () => {
      let stream: any;
      try {
        stream = await startStream();
      } catch (startErr: any) {
        // With n_parallel: 1, restarting after stop() can briefly hit slot contention.
        if (
          !localController.signal.aborted &&
          startErr?.message?.includes('Model failed to start inference')
        ) {
          await new Promise((r) => setTimeout(r, 40));
          if (localController.signal.aborted) {
            throw new Error('Generation aborted');
          }
          stream = await startStream();
        } else {
          throw startErr;
        }
      }

      // Support AsyncIterable fallback for test mocks that return a stream instead of calling onData.
      if (stream && typeof (stream as any)[Symbol.asyncIterator] === 'function') {
        for await (const chunk of stream) {
          processChunk(chunk);
        }
      }

      if (localController.signal.aborted) {
        throw new Error('Generation aborted');
      }

      // Flush any trailing buffers at end of stream.
      if (answerBuffer.length > 0) {
        flushAnswerChunk('', true);
      }
    };

    try {
      await runInferenceAttempt();
    } catch (err: any) {
      let activeErr = err;
      const isUserAbort =
        localController.signal.aborted ||
        err?.name === 'AbortError' ||
        err?.message === 'Generation aborted' ||
        err?.message === 'Model loading aborted';

      if (
        this.usingWebGpu &&
        this.currentModelKey &&
        !this.forceCpuFallback &&
        !receivedAnyChunk &&
        !emittedAnyToken &&
        !isUserAbort
      ) {
        const modelToReload = this.currentModelKey;
        console.warn('[ModelRunner] WebGPU inference failed. Falling back to CPU runtime:', err);
        this.forceCpuFallback = true;
        try {
          await this.teardownWllama();
          await this.loadModelInternal(modelToReload, undefined, localController.signal);
          fullText = '';
          answerBuffer = '';
          receivedAnyChunk = false;
          emittedAnyToken = false;
          await runInferenceAttempt();
          activeErr = null;
        } catch (retryErr: any) {
          activeErr = retryErr;
        }
      }

      if (activeErr) {
        console.warn('[ModelRunner] Inference error:', activeErr);
        const isWasmCrash =
          activeErr instanceof RangeError ||
          activeErr?.name === 'RangeError' ||
          activeErr?.name === 'RuntimeError' ||
          activeErr?.name === 'WllamaRuntimeError' ||
          activeErr?.message?.includes('1163217991') ||
          activeErr?.message?.includes('(ABORT)') ||
          activeErr?.message?.includes('RuntimeError');

        if (isWasmCrash) {
          // Teardown poisoned runtime so subsequent requests can reinitialize cleanly.
          await this.teardownWllama();
          throw new Error(
            `WebAssembly runtime encountered an abort or memory limit (${activeErr?.message || activeErr}). The runtime has been reset. Please click "Explain Move" again to reload.`
          );
        }
        throw activeErr;
      }
    } finally {
      if (this.abortController === localController) {
        this.isGenerating = false;
        this.abortController = null;
      }
    }

    return {
      text: fullText.trim(),
    };
  }

  /**
   * Stop any in-flight inference.
   */
  public stop(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.isGenerating = false;
  }

  /**
   * Unload model and clear browser cache (OPFS + CacheStorage).
   */
  public async deleteModelCache(): Promise<void> {
    this.stop();
    this.abortLoad();
    this.cachedModelStatus = {};
    if (this.loadPromise) {
      try {
        await this.loadPromise;
      } catch {
        // ignore aborted load errors
      }
    }

    const activeWllama = this.wllama;
    await this.teardownWllama();

    if (activeWllama) {
      try {
        if (activeWllama.modelManager?.clear) {
          await activeWllama.modelManager.clear();
        } else if (activeWllama.cacheManager?.clear) {
          await activeWllama.cacheManager.clear();
        }
      } catch {
        // ignore
      }
    } else {
      try {
        const tempWllama: any = this.createWllamaInstance(true);
        if (tempWllama?.modelManager?.clear) {
          await tempWllama.modelManager.clear();
        } else if (tempWllama?.cacheManager?.clear) {
          await tempWllama.cacheManager.clear();
        }
      } catch {
        // ignore
      }
    }

    // Clear CacheStorage if available
    if (typeof caches !== 'undefined') {
      try {
        const keys = await caches.keys();
        for (const k of keys) {
          if (k.includes('wllama') || k.includes('second')) {
            await caches.delete(k);
          }
        }
      } catch (err) {
        console.warn('Failed to delete cache storage:', err);
      }
    }
  }
}
