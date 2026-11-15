import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CloudEvalClient } from '../src/engine/cloudEval.js';

describe('Polite Lichess Cloud Eval Client', () => {
  let mockFetch: any;
  let client: CloudEvalClient;

  beforeEach(() => {
    mockFetch = vi.fn();
    client = new CloudEvalClient({
      fetchFn: mockFetch,
      cooldownMs: 60_000,
    });
  });

  it('respects the settings switch when disabled', async () => {
    client.setEnabled(false);
    const res = await client.getEvaluation('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    expect(res).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('parses valid cloud eval response with cp score', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        depth: 35,
        pvs: [
          { moves: 'e2e4 e7e5 g1f3', cp: 25 },
          { moves: 'd2d4 d7d5', cp: 18 },
        ],
      }),
    });

    const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const res = await client.getEvaluation(fen);

    expect(res).not.toBeNull();
    expect(res?.pvs).toHaveLength(2);
    expect(res?.pvs[0].score).toEqual({ type: 'cp', value: 25 });
    expect(res?.pvs[0].moves).toEqual(['e2e4', 'e7e5', 'g1f3']);
    expect(res?.pvs[1].score).toEqual({ type: 'cp', value: 18 });
  });

  it('parses mate score properly', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        depth: 40,
        pvs: [{ moves: 'd1h5 e8f8 h5f7', mate: 2 }],
      }),
    });

    const res = await client.getEvaluation('rnbqkbnr/ppppp2p/5p2/6p1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1');
    expect(res?.pvs[0].score).toEqual({ type: 'mate', value: 2 });
  });

  it('caches evaluations by normalized FEN and does not re-fetch', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        depth: 20,
        pvs: [{ moves: 'e2e4', cp: 15 }],
      }),
    });

    const fen1 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const fen2 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 10'; // different ply clock, same position

    const res1 = await client.getEvaluation(fen1);
    const res2 = await client.getEvaluation(fen2);

    expect(res1).not.toBeNull();
    expect(res2).not.toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('triggers 60-second cooldown on HTTP 429 and suppresses further requests', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 429,
    });

    const res1 = await client.getEvaluation('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    expect(res1).toBeNull();
    expect(client.isInCooldown()).toBe(true);
    expect(client.getCooldownRemainingMs()).toBeGreaterThan(50_000);

    // Second request while in cooldown must NOT call fetch
    mockFetch.mockClear();
    const res2 = await client.getEvaluation('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3');
    expect(res2).toBeNull();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('returns null gracefully on 404 (not in cloud db) without triggering cooldown', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
    });

    const res = await client.getEvaluation('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3');
    expect(res).toBeNull();
    expect(client.isInCooldown()).toBe(false);
  });

  it('enforces single concurrency (serializes concurrent requests)', async () => {
    let activeCalls = 0;
    let maxConcurrent = 0;

    mockFetch.mockImplementation(async () => {
      activeCalls++;
      maxConcurrent = Math.max(maxConcurrent, activeCalls);
      await new Promise((r) => setTimeout(r, 20));
      activeCalls--;
      return {
        ok: true,
        status: 200,
        json: async () => ({ pvs: [{ moves: 'e2e4', cp: 10 }] }),
      };
    });

    await Promise.all([
      client.getEvaluation('fen_pos_1 w - - 0 1'),
      client.getEvaluation('fen_pos_2 w - - 0 1'),
      client.getEvaluation('fen_pos_3 w - - 0 1'),
    ]);

    expect(maxConcurrent).toBe(1);
    expect(mockFetch).toHaveBeenCalledTimes(3);
  });

  it('aborts in-flight fetch and skips queued callers when AbortSignal fires', async () => {
    mockFetch.mockImplementation((_url: string, init?: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          const err = new Error('Aborted');
          err.name = 'AbortError';
          reject(err);
        });
      });
    });

    const ac1 = new AbortController();
    const ac2 = new AbortController();

    const p1 = client.getEvaluation('fen_1 w - - 0 1', { signal: ac1.signal });
    const p2 = client.getEvaluation('fen_2 w - - 0 1', { signal: ac2.signal });

    await Promise.resolve();
    ac2.abort();
    expect(await p2).toBeNull();

    ac1.abort();
    expect(await p1).toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('respects Retry-After header on HTTP 429 and skips malformed PV entries', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 429,
      headers: new Headers({ 'Retry-After': '90' }),
    });

    await client.getEvaluation('fen_429 w - - 0 1');
    expect(client.getCooldownRemainingMs()).toBeGreaterThan(80_000);

    const client2 = new CloudEvalClient({ fetchFn: mockFetch });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        pvs: [
          { moves: 'e2e4' }, // missing cp/mate
          { moves: '', cp: 20 }, // missing moves
          { moves: 'd2d4', cp: 15 },
        ],
      }),
    });

    const res = await client2.getEvaluation('fen_partial w - - 0 1');
    expect(res?.pvs).toEqual([{ moves: ['d2d4'], score: { type: 'cp', value: 15 } }]);
  });

  it('uses 4-part normalized FEN in query URL and caches 404 misses without caching 500 errors', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
    });

    const fenA = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3';
    const fenB = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 8 15';

    expect(await client.getEvaluation(fenA)).toBeNull();
    expect(await client.getEvaluation(fenB)).toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const calledUrl = mockFetch.mock.calls[0][0] as string;
    expect(calledUrl).toContain(
      encodeURIComponent('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -')
    );

    // 500 error must NOT negative-cache
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ pvs: [{ moves: 'd2d4', cp: 20 }] }),
    });
    const fenErr = 'rnbqkbnr/pppppppp/8/8/3P4/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
    expect(await client.getEvaluation(fenErr)).toBeNull();
    const recovered = await client.getEvaluation(fenErr);
    expect(recovered?.pvs[0].moves).toEqual(['d2d4']);
  });

  it('deduplicates concurrent in-flight fetches for the same normalized FEN and isolates caller aborts', async () => {
    let resolveFetch!: (val: any) => void;
    mockFetch.mockImplementation((_url: string, init?: RequestInit) => {
      return new Promise((resolve, reject) => {
        resolveFetch = resolve;
        init?.signal?.addEventListener('abort', () => {
          const err = new Error('Aborted');
          err.name = 'AbortError';
          reject(err);
        });
      });
    });

    const ac1 = new AbortController();
    const ac2 = new AbortController();
    const fen1 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
    const fen2 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 5 9';

    const p1 = client.getEvaluation(fen1, { signal: ac1.signal });
    const p2 = client.getEvaluation(fen2, { signal: ac2.signal });

    await Promise.resolve();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    ac1.abort();
    expect(await p1).toBeNull();

    // Shared fetch was NOT aborted because caller 2 is still waiting
    resolveFetch({
      ok: true,
      status: 200,
      json: async () => ({ pvs: [{ moves: 'c7c5', cp: 12 }] }),
    });
    const r2 = await p2;
    expect(r2?.fen).toBe(fen2);
    expect(r2?.pvs).toEqual([{ moves: ['c7c5'], score: { type: 'cp', value: 12 } }]);
  });

  it('refreshes LRU order on cache hit so hot entries are not evicted', async () => {
    const lruClient = new CloudEvalClient({ fetchFn: mockFetch, maxCacheSize: 2 });
    mockFetch.mockImplementation(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => ({ pvs: [{ moves: url.includes('pos1') ? 'e2e4' : 'd2d4', cp: 10 }] }),
    }));

    await lruClient.getEvaluation('pos1 w - - 0 1');
    await lruClient.getEvaluation('pos2 w - - 0 1');
    // Touch pos1 so pos2 becomes the oldest
    await lruClient.getEvaluation('pos1 w - - 0 1');
    // Insert pos3 -> evicts pos2, keeps pos1
    await lruClient.getEvaluation('pos3 w - - 0 1');

    expect(lruClient.getCached('pos1 w - - 0 1')).toBeDefined();
    expect(lruClient.getCached('pos2 w - - 0 1')).toBeUndefined();
    expect(lruClient.getCached('pos3 w - - 0 1')).toBeDefined();
  });
});
