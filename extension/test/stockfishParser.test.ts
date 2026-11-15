import { describe, it, expect } from 'vitest';
import { parseUciInfoLine } from '../src/engine/stockfish.js';

describe('Stockfish UCI Info Line Parser', () => {
  it('parses multiPV line for White to move with cp score', () => {
    const line = 'info depth 18 seldepth 24 multipv 1 score cp 45 nodes 12345 nps 500000 time 25 pv e2e4 e7e5 g1f3';
    const parsed = parseUciInfoLine(line, false); // isBlackToMove = false

    expect(parsed).not.toBeNull();
    expect(parsed?.multiPvIndex).toBe(1);
    expect(parsed?.score).toEqual({ type: 'cp', value: 45 });
    expect(parsed?.moves).toEqual(['e2e4', 'e7e5', 'g1f3']);
  });

  it('inverts cp score to White perspective when Black is to move', () => {
    // If Black is winning by 120 cp, UCI outputs score cp 120
    // From White perspective, this is -120 cp
    const line = 'info depth 18 seldepth 22 multipv 1 score cp 120 nodes 45000 pv c7c5 g1f3';
    const parsed = parseUciInfoLine(line, true); // isBlackToMove = true

    expect(parsed).not.toBeNull();
    expect(parsed?.score).toEqual({ type: 'cp', value: -120 });
    expect(parsed?.moves).toEqual(['c7c5', 'g1f3']);
  });

  it('inverts mate score to White perspective when Black is to move', () => {
    const line = 'info depth 10 seldepth 12 multipv 2 score mate 4 nodes 5000 pv d8h4 g2g3';
    const parsed = parseUciInfoLine(line, true); // isBlackToMove = true

    expect(parsed).not.toBeNull();
    expect(parsed?.multiPvIndex).toBe(2);
    expect(parsed?.score).toEqual({ type: 'mate', value: -4 });
  });

  it('parses promotion moves in PV line', () => {
    const line = 'info depth 15 multipv 1 score cp 900 pv e7e8q d8e8';
    const parsed = parseUciInfoLine(line, false);

    expect(parsed?.moves).toEqual(['e7e8q', 'd8e8']);
  });

  it('returns null for non-multipv info lines', () => {
    expect(parseUciInfoLine('info string starting search', false)).toBeNull();
    expect(parseUciInfoLine('info currmove e2e4 currmovenumber 1', false)).toBeNull();
    expect(parseUciInfoLine('info depth 5 nodes 200', false)).toBeNull();
  });

  it('parses single-PV or terminal mate-0 info lines without multipv', () => {
    const parsed = parseUciInfoLine('info depth 0 score mate 0', false);
    expect(parsed).toEqual({
      multiPvIndex: 1,
      score: { type: 'mate', value: 0 },
      moves: [],
    });
  });
});

import { vi, afterEach } from 'vitest';
import { StockfishEngine } from '../src/engine/stockfish.js';

describe('StockfishEngine Abort Serialization & Lifecycle', () => {
  const originalWorker = (globalThis as any).Worker;

  afterEach(() => {
    (globalThis as any).Worker = originalWorker;
  });

  function createMockWorkerClass(options: {
    ignoreStop?: boolean;
    failInit?: boolean;
    failSecondWorker?: boolean;
    onCreate?: (instance: any) => void;
  } = {}) {
    const sentCommands: string[] = [];
    let workerCount = 0;

    class MockWorker {
      private listeners = new Map<string, Set<(ev: any) => void>>();
      public terminated = false;
      private index: number;

      constructor() {
        this.index = ++workerCount;
        options.onCreate?.(this);
      }

      addEventListener(type: string, fn: (ev: any) => void) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type)!.add(fn);
      }

      removeEventListener(type: string, fn: (ev: any) => void) {
        this.listeners.get(type)?.delete(fn);
      }

      postMessage(msg: string) {
        if (this.terminated) return;
        sentCommands.push(msg);
        if (msg === 'uci') {
          if (options.failInit && this.index === 1) {
            queueMicrotask(() => this.emitType('error', new Error('wasm load failed')));
          } else if (options.failSecondWorker && this.index === 2) {
            queueMicrotask(() => this.emitType('error', new Error('js fallback failed')));
          } else {
            this.emit('uciok');
          }
        } else if (msg === 'isready') {
          this.emit('readyok');
        }
      }

      terminate() {
        this.terminated = true;
        this.listeners.clear();
      }

      emit(line: string) {
        this.emitType('message', { data: line });
      }

      emitType(type: string, payload: any) {
        for (const fn of [...(this.listeners.get(type) ?? [])]) {
          fn(payload);
        }
      }
    }

    return { MockWorker, sentCommands };
  }

  it('keeps evalQueue locked after abort until Worker emits bestmove so stale UCI output never bleeds into the next search', async () => {
    let activeWorker: any;
    const { MockWorker, sentCommands } = createMockWorkerClass({
      onCreate: (w) => {
        activeWorker = w;
      },
    });

    (globalThis as any).Worker = MockWorker;
    const engine = new StockfishEngine({ workerScriptUrl: '/mock.js', defaultTimeMs: 500 });

    const ac1 = new AbortController();
    const fen1 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const fen2 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';

    const p1 = engine.evaluate(fen1, { signal: ac1.signal });
    const p2 = engine.evaluate(fen2, {});

    await new Promise((r) => setTimeout(r, 15));
    expect(sentCommands).toContain(`position fen ${fen1}`);
    expect(sentCommands.filter((c) => c.startsWith('go ')).length).toBe(1);

    // Abort first evaluation; p1 resolves cleanly, Worker receives 'stop',
    // and fen2 MUST NOT start until Stockfish emits bestmove for fen1.
    ac1.abort();
    await expect(p1).resolves.toMatchObject({ fen: fen1 });
    expect(sentCommands).toContain('stop');
    expect(sentCommands).not.toContain(`position fen ${fen2}`);

    // Delayed stale info + bestmove from aborted fen1 search:
    activeWorker.emit('info depth 10 multipv 1 score cp 999 pv a2a3');
    activeWorker.emit('bestmove a2a3');

    await new Promise((r) => setTimeout(r, 10));
    // Now fen2 starts cleanly
    expect(sentCommands).toContain(`position fen ${fen2}`);
    activeWorker.emit('info depth 12 multipv 1 score cp -35 pv e7e5 g1f3');
    activeWorker.emit('bestmove e7e5');

    const res2 = await p2;
    expect(res2.fen).toBe(fen2);
    expect(res2.pvs).toEqual([{ moves: ['e7e5', 'g1f3'], score: { type: 'cp', value: 35 } }]);
    engine.terminate();
  });

  it('ignores bestmove (none) and keeps exact PVs from being overwritten by bound lines or empty PVs', async () => {
    let activeWorker: any;
    const { MockWorker } = createMockWorkerClass({
      onCreate: (w) => {
        activeWorker = w;
      },
    });
    (globalThis as any).Worker = MockWorker;

    const engine = new StockfishEngine({ workerScriptUrl: '/mock.js', defaultTimeMs: 500 });
    const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

    const p1 = engine.evaluate(fen);
    await new Promise((r) => setTimeout(r, 15));
    activeWorker.emit('info depth 14 multipv 1 score cp 30 pv e2e4 e7e5');
    activeWorker.emit('info depth 15 multipv 1 score cp 90 lowerbound pv d2d4');
    activeWorker.emit('info depth 15 multipv 1 score cp 32'); // no pv
    activeWorker.emit('bestmove e2e4');
    const res1 = await p1;
    expect(res1.pvs).toEqual([{ moves: ['e2e4', 'e7e5'], score: { type: 'cp', value: 32 } }]);

    const fen2 = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1';
    const p2 = engine.evaluate(fen2);
    await new Promise((r) => setTimeout(r, 15));
    activeWorker.emit('bestmove (none)');
    const res2 = await p2;
    expect(res2.pvs).toEqual([{ moves: [], score: { type: 'cp', value: 0 } }]);
    engine.terminate();
  });

  it('recovers and releases evalQueue when safety timer fires and worker never emits bestmove', async () => {
    const workers: any[] = [];
    const { MockWorker } = createMockWorkerClass({
      onCreate: (w) => workers.push(w),
    });
    (globalThis as any).Worker = MockWorker;

    const engine = new StockfishEngine({ workerScriptUrl: '/mock.js', defaultTimeMs: 30 });
    const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

    // First search hangs past timeMs + 100 + 500ms stop timeout
    const p1 = engine.evaluate(fen, { timeMs: 20 });
    const res1 = await p1;
    expect(res1.fen).toBe(fen);
    expect(workers[0].terminated).toBe(true);

    // Second search re-initializes a fresh worker cleanly
    const p2 = engine.evaluate(fen, { timeMs: 200 });
    await new Promise((r) => setTimeout(r, 20));
    workers[1].emit('info depth 10 multipv 1 score cp 15 pv d2d4 d7d5');
    workers[1].emit('bestmove d2d4');
    const res2 = await p2;
    expect(res2.pvs[0].moves).toEqual(['d2d4', 'd7d5']);
    engine.terminate();
  });

  it('settles in-flight evaluate() immediately when terminate() is called', async () => {
    const { MockWorker } = createMockWorkerClass();
    (globalThis as any).Worker = MockWorker;

    const engine = new StockfishEngine({ workerScriptUrl: '/mock.js', defaultTimeMs: 5000 });
    const p1 = engine.evaluate('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    await new Promise((r) => setTimeout(r, 15));
    engine.terminate();

    await expect(p1).resolves.toMatchObject({ pvs: [{ moves: [], score: { type: 'cp', value: 0 } }] });
  });

  it('clears readyPromise when both primary and fallback workers fail to load so retries work', async () => {
    let fail = true;
    const { MockWorker } = createMockWorkerClass({
      failInit: true,
      failSecondWorker: true,
      onCreate: () => {},
    });

    class RetryableMockWorker extends MockWorker {
      override postMessage(msg: string) {
        if (!fail) {
          if (msg === 'uci') this.emit('uciok');
          else if (msg === 'isready') this.emit('readyok');
          return;
        }
        super.postMessage(msg);
      }
    }

    (globalThis as any).Worker = RetryableMockWorker;
    const engine = new StockfishEngine({ workerScriptUrl: '/engine/stockfish.wasm.js' });

    await expect(engine.init()).rejects.toThrow();
    fail = false;
    await expect(engine.init()).resolves.toBeUndefined();
    engine.terminate();
  });

  it('caches evaluations by normalized FEN, allows multiPv=1 to reuse multiPv=3, and sends MultiPV only when changed', async () => {
    let activeWorker: any;
    const { MockWorker, sentCommands } = createMockWorkerClass({
      onCreate: (w) => {
        activeWorker = w;
      },
    });
    (globalThis as any).Worker = MockWorker;

    const engine = new StockfishEngine({ workerScriptUrl: '/mock.js', defaultTimeMs: 500 });
    const fen1 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
    const fen1DiffClock = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 4 12';

    const p1 = engine.evaluate(fen1, { multiPv: 3 });
    await new Promise((r) => setTimeout(r, 15));
    activeWorker.emit('info depth 18 multipv 1 score cp -20 pv c7c5 g1f3');
    activeWorker.emit('info depth 18 multipv 2 score cp -25 pv e7e5 g1f3');
    activeWorker.emit('bestmove c7c5');
    const r1 = await p1;
    expect(r1.pvs).toHaveLength(2);

    // Same normalized FEN with multiPv=1 hits cache immediately without Worker command
    const commandsBeforeCacheHit = sentCommands.length;
    const r2 = await engine.evaluate(fen1DiffClock, { multiPv: 1 });
    expect(r2.fen).toBe(fen1DiffClock);
    expect(r2.pvs).toEqual([{ moves: ['c7c5', 'g1f3'], score: { type: 'cp', value: 20 } }]);
    expect(sentCommands.length).toBe(commandsBeforeCacheHit);

    // Different FEN with same multiPv=3 skips redundant setoption MultiPV
    const fen2 = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
    const p3 = engine.evaluate(fen2, { multiPv: 3 });
    await new Promise((r) => setTimeout(r, 15));
    activeWorker.emit('info depth 18 multipv 1 score cp 25 pv g1f3');
    activeWorker.emit('bestmove g1f3');
    await p3;

    const multiPvCmds = sentCommands.filter((c) => c.startsWith('setoption name MultiPV'));
    expect(multiPvCmds).toEqual(['setoption name MultiPV value 3']);

    // Requesting multiPv=1 for a new FEN DOES send MultiPV=1, and subsequent multiPv=3 misses 1-PV cache
    const fen3 = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2';
    const p4 = engine.evaluate(fen3, { multiPv: 1 });
    await new Promise((r) => setTimeout(r, 15));
    activeWorker.emit('info depth 18 multipv 1 score cp -15 pv d7d6');
    activeWorker.emit('bestmove d7d6');
    await p4;

    const p5 = engine.evaluate(fen3, { multiPv: 3 });
    await new Promise((r) => setTimeout(r, 15));
    activeWorker.emit('info depth 18 multipv 1 score cp -15 pv d7d6');
    activeWorker.emit('info depth 18 multipv 2 score cp -22 pv b8c6');
    activeWorker.emit('bestmove d7d6');
    const r5 = await p5;
    expect(r5.pvs).toHaveLength(2);

    const multiPvCmdsAfter = sentCommands.filter((c) => c.startsWith('setoption name MultiPV'));
    expect(multiPvCmdsAfter).toEqual([
      'setoption name MultiPV value 3',
      'setoption name MultiPV value 1',
      'setoption name MultiPV value 3',
    ]);

    engine.terminate();
  });
});
