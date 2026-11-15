import { describe, it, expect, vi } from 'vitest';
import { ModelRunner, MODEL_CONFIGS, patchGgufHeaderBlob } from '../src/runtime/modelRunner.js';

describe('ModelRunner Inference Controller', () => {
  it('detects hardware support structure', async () => {
    const hw = await ModelRunner.checkHardware();
    expect(hw).toHaveProperty('webAssemblySupported');
    expect(hw).toHaveProperty('webGpuSupported');
    expect(hw).toHaveProperty('isWeakDevice');
  });

  it('correctly extracts tokens from OpenAI-compatible choices[0].text chunks', async () => {
    const mockChunks = [
      { id: '1', choices: [{ text: 'The ' }] },
      { id: '2', choices: [{ text: 'move ' }] },
      { id: '3', choices: [{ text: 'Nf3 ' }] },
      { id: '4', choices: [{ text: 'controls ' }] },
      { id: '5', choices: [{ text: 'the center.' }] },
    ];

    async function* generateStream() {
      for (const c of mockChunks) {
        yield c;
      }
    }

    const mockWllama = {
      createCompletion: vi.fn().mockResolvedValue(generateStream()),
    };

    const runner = new ModelRunner(mockWllama);
    const streamedTokens: string[] = [];

    const result = await runner.generate('prompt test', (token) => {
      streamedTokens.push(token);
    });

    expect(streamedTokens).toEqual(['The ', 'move ', 'Nf3 ', 'controls ', 'the center.']);
    expect(result.text).toBe('The move Nf3 controls the center.');
  });

  it('correctly handles delta content format chunks', async () => {
    const mockChunks = [
      { choices: [{ delta: { content: 'White ' } }] },
      { choices: [{ delta: { content: 'attacks.' } }] },
    ];

    async function* generateStream() {
      for (const c of mockChunks) {
        yield c;
      }
    }

    const mockWllama = {
      createCompletion: vi.fn().mockResolvedValue(generateStream()),
    };

    const runner = new ModelRunner(mockWllama);
    const streamedTokens: string[] = [];

    const result = await runner.generate('prompt test', (token) => {
      streamedTokens.push(token);
    });

    expect(streamedTokens).toEqual(['White ', 'attacks.']);
    expect(result.text).toBe('White attacks.');
  });

  it('uses createChatCompletion when available and passes formatted messages', async () => {
    const mockChunks = [
      { choices: [{ delta: { content: 'Excell' } }] },
      { choices: [{ delta: { content: 'ent move.' } }] },
    ];

    async function* generateStream() {
      for (const c of mockChunks) {
        yield c;
      }
    }

    const mockWllama = {
      createChatCompletion: vi.fn().mockResolvedValue(generateStream()),
    };

    const runner = new ModelRunner(mockWllama);
    const streamedTokens: string[] = [];

    const prompt = '<|im_start|>system\nYou are Second.<|im_end|>\n<|im_start|>user\nExplain e4.<|im_end|>\n<|im_start|>assistant\n';
    const result = await runner.generate(prompt, (token) => {
      streamedTokens.push(token);
    });

    expect(mockWllama.createChatCompletion).toHaveBeenCalledTimes(1);
    const callArgs = mockWllama.createChatCompletion.mock.calls[0][0];
    expect(callArgs.messages).toEqual([
      { role: 'system', content: 'You are Second.' },
      { role: 'user', content: 'Explain e4.' },
    ]);
    expect(result.text).toBe('Excellent move.');
  });

  it('strips stop tokens <|im_end|> and <|endoftext|> from final output', async () => {
    const mockChunks = [
      { choices: [{ text: 'Great knight move.' }] },
      { choices: [{ text: '<|im_end|>' }] },
    ];

    async function* generateStream() {
      for (const c of mockChunks) {
        yield c;
      }
    }

    const mockWllama = {
      createCompletion: vi.fn().mockResolvedValue(generateStream()),
    };

    const runner = new ModelRunner(mockWllama);
    const tokens: string[] = [];

    const result = await runner.generate('prompt', (token) => {
      tokens.push(token);
    });

    expect(result.text).toBe('Great knight move.');
    expect(tokens.join('')).toBe('Great knight move.');
    expect(tokens).not.toContain('<|im_end|>');
  });

  it('resets loaded state and instance on WASM crash or abort', async () => {
    const mockWllama = {
      createCompletion: vi.fn().mockRejectedValue(new Error('(ABORT)')),
    };

    const runner = new ModelRunner(mockWllama);
    expect(runner.isModelLoaded()).toBe(true);

    await expect(runner.generate('prompt', () => {})).rejects.toThrow(
      'WebAssembly runtime encountered an abort or memory limit'
    );

    expect(runner.isModelLoaded()).toBe(false);
  });

  it('resets loaded state on RangeError (Invalid typed array length)', async () => {
    const mockWllama = {
      createCompletion: vi.fn().mockRejectedValue(new RangeError('Invalid typed array length: 1163217991')),
    };

    const runner = new ModelRunner(mockWllama);
    expect(runner.isModelLoaded()).toBe(true);

    await expect(runner.generate('prompt', () => {})).rejects.toThrow(
      'WebAssembly runtime encountered an abort or memory limit'
    );

    expect(runner.isModelLoaded()).toBe(false);
  });

  it('aborts active generation when stop() is called', async () => {
    let aborted = false;

    async function* generateStream() {
      yield { choices: [{ text: 'Start' }] };
      // simulate long running stream
      await new Promise((_, reject) => {
        const check = setInterval(() => {
          if (aborted) {
            clearInterval(check);
            reject(new Error('Generation aborted'));
          }
        }, 10);
      });
    }

    const mockWllama = {
      createCompletion: vi.fn().mockImplementation(({ abortSignal }) => {
        abortSignal.addEventListener('abort', () => {
          aborted = true;
        });
        return generateStream();
      }),
    };

    const runner = new ModelRunner(mockWllama);
    const genPromise = runner.generate('prompt', () => {});

    setTimeout(() => {
      runner.stop();
    }, 20);

    await expect(genPromise).rejects.toThrow('Generation aborted');
  });

  describe('Model Loading and Configuration', () => {
    it('MODEL_CONFIGS contains valid .gguf URLs', () => {
      expect(MODEL_CONFIGS['2B'].url).toMatch(/\.gguf$/);
      expect(MODEL_CONFIGS['2B'].url).toMatch(/-00001-of-00003\.gguf$/);
      expect(MODEL_CONFIGS['0.8B'].url).toMatch(/\.gguf$/);
    });

    it('loadModel passes a single .gguf URL string to wllama.loadModelFromUrl with optimized CPU runtime params', async () => {
      const mockWllama = {
        unloadModel: vi.fn().mockResolvedValue(undefined),
        loadModelFromUrl: vi.fn().mockImplementation(async (url, opts) => {
          expect(typeof url).toBe('string');
          expect(url.endsWith('.gguf')).toBe(true);
          expect(opts.n_ctx).toBe(2048);
          expect(opts.n_batch).toBe(512);
          expect(opts.n_ubatch).toBe(256);
          expect(opts.n_parallel).toBe(1);
          expect(opts.n_gpu_layers).toBe(0);
          expect(opts.flash_attn).toBe(true);
          expect(opts.cache_type_k).toBe('q8_0');
          expect(opts.cache_type_v).toBe('q8_0');
          expect(opts.cache_idle_slots).toBe(true);
          opts.progressCallback({ loaded: 500, total: 1000 });
        }),
      };

      const runner = new ModelRunner(mockWllama);
      const progressMessages: string[] = [];

      await runner.loadModel('2B', (p) => {
        progressMessages.push(p.message);
      });

      expect(mockWllama.loadModelFromUrl).toHaveBeenCalledTimes(1);
      const calledUrl = mockWllama.loadModelFromUrl.mock.calls[0][0];
      expect(calledUrl).toBe(MODEL_CONFIGS['2B'].url);
      expect(calledUrl.endsWith('.gguf')).toBe(true);
      expect(progressMessages.some((msg) => msg.includes('Downloading model weights'))).toBe(true);
      expect(progressMessages.some((msg) => msg.includes('loaded and ready'))).toBe(true);
    });

    it('loadModel works for 0.8B model with .gguf URL string', async () => {
      const mockWllama = {
        unloadModel: vi.fn().mockResolvedValue(undefined),
        loadModelFromUrl: vi.fn().mockResolvedValue(undefined),
      };

      const runner = new ModelRunner(mockWllama);
      await runner.loadModel('0.8B');

      expect(mockWllama.loadModelFromUrl).toHaveBeenCalledTimes(1);
      const calledUrl = mockWllama.loadModelFromUrl.mock.calls[0][0];
      expect(calledUrl).toBe(MODEL_CONFIGS['0.8B'].url);
      expect(calledUrl.endsWith('.gguf')).toBe(true);
    });

    it('shares a single in-flight load between concurrent loadModel calls', async () => {
      let releaseLoad!: () => void;
      const gate = new Promise<void>((resolve) => {
        releaseLoad = resolve;
      });
      const mockWllama = {
        unloadModel: vi.fn().mockResolvedValue(undefined),
        loadModelFromUrl: vi.fn().mockImplementation(async () => {
          await gate;
        }),
      };

      const runner = new ModelRunner(mockWllama);
      const warmup = runner.loadModel('2B');
      const click = runner.loadModel('2B');

      await Promise.resolve();
      expect(mockWllama.loadModelFromUrl).toHaveBeenCalledTimes(1);

      releaseLoad();
      await Promise.all([warmup, click]);
      expect(mockWllama.loadModelFromUrl).toHaveBeenCalledTimes(1);
      expect(runner.isModelLoaded()).toBe(true);
    });

    it('throws when native wllama model load silently fails (nLayer = 0)', async () => {
      const mockWllama = {
        unloadModel: vi.fn().mockResolvedValue(undefined),
        loadModelFromUrl: vi.fn().mockResolvedValue(undefined),
        getModelMetadata: vi.fn().mockReturnValue({ hparams: { nLayer: 0, nVocab: 0 } }),
      };

      const runner = new ModelRunner(mockWllama);
      await expect(runner.loadModel('2B')).rejects.toThrow('Failed to initialize');
      expect(runner.isModelLoaded()).toBe(false);
    });

    it('patchGgufHeaderBlob patches qwen35.block_count (25->24), nextn_predict_layers (1->0), and recurrent_layers key in-place', async () => {
      // Construct a synthetic GGUF v3 header buffer
      const buf = new Uint8Array(512);
      const dv = new DataView(buf.buffer);
      const enc = new TextEncoder();

      // Magic 'GGUF'
      buf.set(enc.encode('GGUF'), 0);

      // Place qwen35.block_count = 25 at offset 64
      const blockKey = enc.encode('qwen35.block_count');
      buf.set(blockKey, 64);
      dv.setUint32(64 + blockKey.length, 4, true); // type = UINT32
      dv.setUint32(64 + blockKey.length + 4, 25, true); // value = 25

      // Place qwen35.nextn_predict_layers = 1 at offset 128
      const nextnKey = enc.encode('qwen35.nextn_predict_layers');
      buf.set(nextnKey, 128);
      dv.setUint32(128 + nextnKey.length, 4, true); // type = UINT32
      dv.setUint32(128 + nextnKey.length + 4, 1, true); // value = 1

      // Place qwen35.attention.recurrent_layers at offset 200
      const recKey = enc.encode('qwen35.attention.recurrent_layers');
      buf.set(recKey, 200);

      const originalBlob = new Blob([buf]);
      Object.defineProperty(originalBlob, 'name', { value: 'shard-00001-of-00003.gguf' });

      const patched = await patchGgufHeaderBlob(originalBlob);
      expect(patched.size).toBe(originalBlob.size);
      expect((patched as any).name).toBe('shard-00001-of-00003.gguf');

      const patchedBuf = new Uint8Array(await patched.arrayBuffer());
      const patchedDv = new DataView(patchedBuf.buffer);

      expect(patchedDv.getUint32(64 + blockKey.length + 4, true)).toBe(24);
      expect(patchedDv.getUint32(128 + nextnKey.length + 4, true)).toBe(0);

      const patchedRecKey = new TextDecoder().decode(patchedBuf.slice(200, 200 + recKey.length));
      expect(patchedRecKey).toBe('qwen35.attention.recurrent_unused');
    });

    it('never emits a premature 95% loading spike before shard downloading progresses', async () => {
      const mockWllama: any = {
        unloadModel: vi.fn().mockResolvedValue(undefined),
        loadModel: vi.fn().mockResolvedValue(undefined),
        loadModelFromUrl: vi.fn().mockImplementation(async (_url: string, opts: any) => {
          opts.progressCallback({ loaded: 500, total: 1000 });
          await mockWllama.loadModel([new Blob(['GGUF'])]);
        }),
      };

      const runner = new ModelRunner(mockWllama);
      const percents: number[] = [];
      const phases: string[] = [];

      await runner.loadModel('2B', (p) => {
        percents.push(p.percent);
        phases.push(p.phase);
      });

      // Expect strictly non-decreasing progress: 0% -> ~48% downloading -> 95% loading -> 100% ready
      for (let i = 1; i < percents.length; i++) {
        expect(percents[i]).toBeGreaterThanOrEqual(percents[i - 1]);
      }
      expect(phases).toEqual(['downloading', 'downloading', 'loading', 'ready']);
    });

    it('does not let a late-rejecting aborted generate() clobber a newer active generate()', async () => {
      let rejectRun1!: (err: Error) => void;
      let resolveRun2Chunk!: () => void;

      async function* stream2() {
        yield { choices: [{ text: 'Fresh ' }] };
        await new Promise<void>((r) => {
          resolveRun2Chunk = r;
        });
        yield { choices: [{ text: 'answer.' }] };
      }

      const mockWllama = {
        createCompletion: vi
          .fn()
          .mockImplementationOnce(
            () =>
              (async function* () {
                yield { choices: [{ text: 'Old' }] };
                await new Promise((_res, rej) => {
                  rejectRun1 = rej;
                });
              })()
          )
          .mockImplementationOnce(() => stream2()),
      };

      const runner = new ModelRunner(mockWllama);
      const p1 = runner.generate('p1', () => {});
      await new Promise((r) => setTimeout(r, 5));

      runner.stop();
      const p2 = runner.generate('p2', () => {});
      await new Promise((r) => setTimeout(r, 5));

      // Late rejection of run 1 must not clear run 2's active controller/generation flag
      rejectRun1(new Error('Aborted'));
      await expect(p1).rejects.toThrow('Aborted');

      // Stopping now must still abort run 2
      runner.stop();
      resolveRun2Chunk();
      await expect(p2).rejects.toThrow('Generation aborted');
    });

    it('forwards native llama-server n_predict and temp alongside max_tokens and temperature', async () => {
      async function* answerStream() {
        yield { choices: [{ text: 'Solid move.' }] };
      }

      const mockWllama = {
        createCompletion: vi.fn().mockImplementation(() => answerStream()),
      };

      const runner = new ModelRunner(mockWllama);
      const prompt = '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';
      await runner.generate(prompt, () => {});

      expect(mockWllama.createCompletion).toHaveBeenCalledWith(
        expect.objectContaining({
          prompt,
          n_predict: 256,
          max_tokens: 256,
          temp: 0.0,
          temperature: 0.0,
          stop: ['<|im_end|>', '<|endoftext|>'],
        })
      );
    });

    it('holds back split stop tokens across chunks without leaking partial stop markers', async () => {
      async function* splitAnswerStream() {
        yield { choices: [{ text: 'Solid ' }] };
        yield { choices: [{ text: 'move.<|im_' }] };
        yield { choices: [{ text: 'end|>' }] };
      }

      const mockWllama = {
        createCompletion: vi.fn().mockImplementationOnce(() => splitAnswerStream()),
      };

      const runner = new ModelRunner(mockWllama);
      const tokens: string[] = [];
      const res = await runner.generate('<|im_start|>assistant\n<think>\n\n</think>\n\n', (t) => {
        tokens.push(t);
      });
      expect(res.text).toBe('Solid move.');
      expect(tokens.join('')).toBe('Solid move.');
    });

    it('resets runtime on WllamaRuntimeError (name === "RuntimeError")', async () => {
      const err = new Error('WASM OOM');
      err.name = 'RuntimeError';
      const mockWllama = {
        createCompletion: vi.fn().mockRejectedValue(err),
        exit: vi.fn().mockResolvedValue(undefined),
      };

      const runner = new ModelRunner(mockWllama);
      await expect(runner.generate('prompt', () => {})).rejects.toThrow(
        'WebAssembly runtime encountered an abort or memory limit'
      );
      expect(runner.isModelLoaded()).toBe(false);
    });

    it('isModelCached checks cacheManager.list() safely even if another model has missing shards', async () => {
      const mockWllama = {
        cacheManager: {
          list: vi.fn().mockResolvedValue([
            {
              name: '0.8b.gguf',
              size: 500,
              metadata: {
                originalURL: MODEL_CONFIGS['0.8B'].url,
                originalSize: 500,
              },
            },
            {
              name: '2b-shard1.gguf',
              size: 400,
              metadata: {
                originalURL: MODEL_CONFIGS['2B'].shards[0],
                originalSize: 400,
              },
            },
          ]),
        },
        modelManager: {
          getModels: vi.fn().mockRejectedValue(new Error('Model file not found: shard 2')),
        },
      };

      const runner = new ModelRunner(mockWllama);
      await expect(runner.isModelCached('0.8B')).resolves.toBe(true);
      await expect(runner.isModelCached('2B')).resolves.toBe(false);
    });

    it('updates loading progress callback across subsequent loads and supports abortLoad()', async () => {
      const mockWllama = {
        unloadModel: vi.fn().mockResolvedValue(undefined),
        loadModel: vi.fn().mockResolvedValue(undefined),
        loadModelFromUrl: vi.fn().mockImplementation(async (_url: string, opts: any) => {
          opts.progressCallback({ loaded: 500, total: 1000 });
          await mockWllama.loadModel([new Blob(['GGUF'])]);
          if (opts.signal?.aborted) {
            throw new Error('Aborted');
          }
        }),
      };

      const runner = new ModelRunner(mockWllama);
      const firstPhases: string[] = [];
      const secondPhases: string[] = [];

      await runner.loadModel('0.8B', (p) => firstPhases.push(p.phase));
      await runner.loadModel('2B', (p) => secondPhases.push(p.phase), { forceReload: true });

      expect(firstPhases).toContain('loading');
      expect(secondPhases).toContain('loading');

      mockWllama.loadModelFromUrl.mockImplementationOnce(async (_url: string, opts: any) => {
        await new Promise<void>((resolve) => {
          opts.signal?.addEventListener('abort', () => resolve(), { once: true });
        });
      });

      const abortedLoad = runner.loadModel('0.8B', undefined, { forceReload: true });
      await Promise.resolve();
      runner.abortLoad();
      await expect(abortedLoad).rejects.toThrow('Model loading aborted');
      expect(runner.isModelLoaded()).toBe(false);
    });

    it('uses WebGPU offloading when supported and falls back to CPU on load failure', async () => {
      vi.stubGlobal('navigator', {
        ...globalThis.navigator,
        hardwareConcurrency: 8,
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue({
            limits: { maxBufferSize: 2 * 1024 * 1024 * 1024 },
          }),
        },
      });

      try {
        const calls: any[] = [];
        const mockWllama = {
          unloadModel: vi.fn().mockResolvedValue(undefined),
          loadModelFromUrl: vi.fn().mockImplementation(async (_url: string, opts: any) => {
            calls.push({
              n_gpu_layers: opts.n_gpu_layers,
              n_ubatch: opts.n_ubatch,
              n_threads: opts.n_threads,
            });
            if (opts.n_gpu_layers > 0) {
              throw new Error('WebGPU shader compilation failed');
            }
          }),
          getModelMetadata: vi.fn().mockReturnValue({ hparams: { nLayer: 24, nVocab: 151936 } }),
        };

        const runner = new ModelRunner(mockWllama);
        await runner.loadModel('0.8B');

        expect(calls).toEqual([
          { n_gpu_layers: 99999, n_ubatch: 512, n_threads: 4 },
          { n_gpu_layers: 0, n_ubatch: 256, n_threads: 4 },
        ]);
        expect(runner.isModelLoaded()).toBe(true);

        // Subsequent loads remember the CPU fallback for the rest of the session
        await runner.loadModel('2B', undefined, { forceReload: true });
        expect(calls[2]).toEqual({ n_gpu_layers: 0, n_ubatch: 256, n_threads: 4 });
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it('falls back to CPU and retries transparently when first WebGPU completion fails', async () => {
      vi.stubGlobal('navigator', {
        ...globalThis.navigator,
        hardwareConcurrency: 4,
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue({
            limits: { maxBufferSize: 2 * 1024 * 1024 * 1024 },
          }),
        },
      });

      try {
        const loadCalls: any[] = [];
        const mockWllama = {
          unloadModel: vi.fn().mockResolvedValue(undefined),
          loadModelFromUrl: vi.fn().mockImplementation(async (_url: string, opts: any) => {
            loadCalls.push({
              n_gpu_layers: opts.n_gpu_layers,
              n_ubatch: opts.n_ubatch,
              n_threads: opts.n_threads,
            });
          }),
          getModelMetadata: vi.fn().mockReturnValue({ hparams: { nLayer: 24, nVocab: 151936 } }),
          createCompletion: vi
            .fn()
            .mockRejectedValueOnce(new Error('WebGPU device lost during inference'))
            .mockImplementationOnce(async (opts: any) => {
              opts.onData?.({ choices: [{ text: 'Fallback CPU answer.' }] });
            }),
        };

        const runner = new ModelRunner(mockWllama);
        await runner.loadModel('0.8B');
        expect(loadCalls[0]).toEqual({ n_gpu_layers: 99999, n_ubatch: 512, n_threads: 3 });

        const streamed: string[] = [];
        const res = await runner.generate(
          '<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n',
          (tok) => streamed.push(tok)
        );

        expect(loadCalls).toHaveLength(2);
        expect(loadCalls[1]).toEqual({ n_gpu_layers: 0, n_ubatch: 256, n_threads: 3 });
        expect(res.text).toBe('Fallback CPU answer.');
        expect(streamed.join('')).toBe('Fallback CPU answer.');
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it('streams via onData callback and passes cache_prompt: true without AsyncIterable', async () => {
      const mockWllama = {
        createCompletion: vi.fn().mockImplementation(async (opts: any) => {
          expect(opts.cache_prompt).toBe(true);
          expect(typeof opts.onData).toBe('function');
          opts.onData({ choices: [{ text: 'Play ' }] });
          opts.onData({ choices: [{ text: 'Nf3.' }] });
        }),
      };

      const runner = new ModelRunner(mockWllama);
      const streamed: string[] = [];
      const res = await runner.generate(
        '<|im_start|>user\nBest move?<|im_end|>\n<|im_start|>assistant\n',
        (tok) => streamed.push(tok)
      );

      expect(res.text).toBe('Play Nf3.');
      expect(streamed).toEqual(['Play ', 'Nf3.']);
    });

    it('caches isModelCached() result in memory and invalidates it on deleteModelCache()', async () => {
      const listMock = vi.fn().mockResolvedValue([
        {
          name: '0.8b.gguf',
          size: 500,
          metadata: {
            originalURL: MODEL_CONFIGS['0.8B'].url,
            originalSize: 500,
          },
        },
      ]);
      const mockWllama = {
        cacheManager: {
          list: listMock,
          clear: vi.fn().mockResolvedValue(undefined),
        },
        exit: vi.fn().mockResolvedValue(undefined),
      };

      const runner = new ModelRunner(mockWllama);
      await expect(runner.isModelCached('0.8B')).resolves.toBe(true);
      await expect(runner.isModelCached('0.8B')).resolves.toBe(true);
      expect(listMock).toHaveBeenCalledTimes(1);

      await runner.deleteModelCache();
      await expect(runner.isModelCached('0.8B')).resolves.toBe(true);
      expect(listMock).toHaveBeenCalledTimes(2);
    });
  });
});


