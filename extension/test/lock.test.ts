import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  detectPageType,
  checkGameStatus,
  evaluatePageAccess,
  clearGameStatusCache,
  getCachedFinishedGameData,
  LOCK_OFF_MESSAGE,
  PUZZLE_OFF_MESSAGE,
  OTHER_PAGE_MESSAGE,
  FINISHED_GAME_STATUSES,
} from '../src/lock.js';

beforeEach(() => {
  clearGameStatusCache();
});

describe('Live-Game Lock: URL & Page Type Detection', () => {
  it('identifies analysis board URLs as allowed', () => {
    expect(detectPageType('https://lichess.org/analysis')).toEqual({
      pageType: 'analysis',
      isAllowedType: true,
    });
    expect(detectPageType('https://lichess.org/analysis/standard')).toEqual({
      pageType: 'analysis',
      isAllowedType: true,
    });
    expect(detectPageType('https://lichess.org/analysis/r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')).toEqual({
      pageType: 'analysis',
      isAllowedType: true,
    });
  });

  it('identifies studies as allowed', () => {
    expect(detectPageType('https://lichess.org/study/abcd1234')).toEqual({
      pageType: 'study',
      isAllowedType: true,
    });
    expect(detectPageType('https://lichess.org/study/abcd1234/efgh5678')).toEqual({
      pageType: 'study',
      isAllowedType: true,
    });
  });

  it('identifies game URLs and extracts 8-character game ID', () => {
    expect(detectPageType('https://lichess.org/aBcD1234')).toEqual({
      pageType: 'game',
      gameId: 'aBcD1234',
      isAllowedType: true,
    });
    expect(detectPageType('https://lichess.org/aBcD1234/white')).toEqual({
      pageType: 'game',
      gameId: 'aBcD1234',
      isAllowedType: true,
    });
    expect(detectPageType('https://lichess.org/aBcD1234/black')).toEqual({
      pageType: 'game',
      gameId: 'aBcD1234',
      isAllowedType: true,
    });
    // /game/:id format
    expect(detectPageType('https://lichess.org/game/GGRmK9Q0')).toEqual({
      pageType: 'game',
      gameId: 'GGRmK9Q0',
      isAllowedType: true,
    });
    // /game/:id with query and hash
    expect(detectPageType('https://lichess.org/game/GGRmK9Q0?pov=joebossie#12')).toEqual({
      pageType: 'game',
      gameId: 'GGRmK9Q0',
      isAllowedType: true,
    });
    // /game/:id/white and /round/:id
    expect(detectPageType('https://lichess.org/game/GGRmK9Q0/white')).toEqual({
      pageType: 'game',
      gameId: 'GGRmK9Q0',
      isAllowedType: true,
    });
    expect(detectPageType('https://lichess.org/round/GGRmK9Q0')).toEqual({
      pageType: 'game',
      gameId: 'GGRmK9Q0',
      isAllowedType: true,
    });
    // 12-char player URL (8-char game id + 4 secret chars)
    expect(detectPageType('https://lichess.org/aBcD1234wxyz')).toEqual({
      pageType: 'game',
      gameId: 'aBcD1234',
      isAllowedType: true,
    });
  });

  it('identifies puzzle and training pages as disallowed', () => {
    const puzzleUrls = [
      'https://lichess.org/training',
      'https://lichess.org/training/12345',
      'https://lichess.org/training/daily',
      'https://lichess.org/puzzle',
      'https://lichess.org/puzzle/12345',
      'https://lichess.org/storm',
      'https://lichess.org/storm/dashboard',
      'https://lichess.org/streak',
      'https://lichess.org/streak/123',
      'https://lichess.org/racer',
      'https://lichess.org/racer/abcde',
    ];
    for (const url of puzzleUrls) {
      const res = detectPageType(url);
      expect(res.pageType).toBe('puzzle');
      expect(res.isAllowedType).toBe(false);
    }
  });

  it('classifies other pages as disallowed including 8-char Lichess routes', () => {
    const otherUrls = [
      'https://lichess.org/',
      'https://lichess.org/tv',
      'https://lichess.org/tournament',
      'https://lichess.org/messages',
      'https://lichess.org/timeline',
      'https://lichess.org/calendar',
      'https://lichess.org/marathon',
      'https://lichess.org/feedback',
      'https://lichess.org/@/magnuscarlsen',
      'https://chess.com/game/live/123',
      'invalid-url',
    ];
    for (const url of otherUrls) {
      const res = detectPageType(url);
      expect(res.pageType).toBe('other');
      expect(res.isAllowedType).toBe(false);
    }
  });
});

describe('Live-Game Lock: Fail-Closed Game Status Verification', () => {
  it('approves all recognized finished game statuses', async () => {
    for (const status of Array.from(FINISHED_GAME_STATUSES)) {
      clearGameStatusCache();
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status }),
      });

      const res = await checkGameStatus('aBcD1234', mockFetch as any);
      expect(res.isFinished).toBe(true);
      expect(res.status).toBe(status);
    }
  });

  it('strictly locks live games (status: started, created)', async () => {
    for (const liveStatus of ['started', 'created', 'playing']) {
      clearGameStatusCache();
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: liveStatus }),
      });

      const res = await checkGameStatus('aBcD1234', mockFetch as any);
      expect(res.isFinished).toBe(false);
      expect(res.status).toBe(liveStatus);
    }
  });

  it('fails closed when API returns HTTP 404 or 500 and caches error briefly', async () => {
    for (const [idx, statusCode] of [404, 500, 503].entries()) {
      clearGameStatusCache();
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: statusCode,
      });

      const gameId = `aBcD123${idx}`;
      const res1 = await checkGameStatus(gameId, mockFetch as any);
      const res2 = await checkGameStatus(gameId, mockFetch as any);
      expect(res1.isFinished).toBe(false);
      expect(res1.error).toContain(`${statusCode}`);
      expect(res2.isFinished).toBe(false);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    }
  });

  it('enters cooldown and fails closed on HTTP 429', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 429,
      headers: { get: () => '30' },
    });

    const res1 = await checkGameStatus('RateLim1', mockFetch as any);
    const res2 = await checkGameStatus('RateLim2', mockFetch as any);
    expect(res1.isFinished).toBe(false);
    expect(res1.error).toContain('429');
    expect(res2.isFinished).toBe(false);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('fails closed on network error or timeout', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('Network failure'));
    const res = await checkGameStatus('aBcD1234', mockFetch as any);
    expect(res.isFinished).toBe(false);
    expect(res.error).toBe('Network failure');
  });

  it('fails closed when API returns malformed or empty json', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    });
    const res = await checkGameStatus('aBcD1234', mockFetch as any);
    expect(res.isFinished).toBe(false);
  });

  it('rejects invalid game ID without network calls', async () => {
    const mockFetch = vi.fn();
    const res = await checkGameStatus('bad_id!!', mockFetch as any);
    expect(res.isFinished).toBe(false);
    expect(mockFetch).not.toHaveBeenCalled();
  });
});

describe('Live-Game Lock: evaluatePageAccess End-to-End', () => {
  it('allows analysis boards immediately without network call', async () => {
    const mockFetch = vi.fn();
    const evalRes = await evaluatePageAccess('https://lichess.org/analysis', mockFetch as any);
    expect(evalRes.allowed).toBe(true);
    expect(evalRes.pageType).toBe('analysis');
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('allows studies immediately without network call', async () => {
    const mockFetch = vi.fn();
    const evalRes = await evaluatePageAccess('https://lichess.org/study/12345678', mockFetch as any);
    expect(evalRes.allowed).toBe(true);
    expect(evalRes.pageType).toBe('study');
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('blocks puzzle pages with exact puzzle lock message', async () => {
    const mockFetch = vi.fn();
    const evalRes = await evaluatePageAccess('https://lichess.org/storm', mockFetch as any);
    expect(evalRes.allowed).toBe(false);
    expect(evalRes.reason).toBe(PUZZLE_OFF_MESSAGE);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('blocks other non-chess pages with guidance message', async () => {
    const mockFetch = vi.fn();
    const evalRes = await evaluatePageAccess('https://lichess.org/tv', mockFetch as any);
    expect(evalRes.allowed).toBe(false);
    expect(evalRes.reason).toBe(OTHER_PAGE_MESSAGE);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('allows finished game with mate', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: 'mate' }),
    });

    const evalRes = await evaluatePageAccess('https://lichess.org/AbCd1234', mockFetch as any);
    expect(evalRes.allowed).toBe(true);
    expect(evalRes.status).toBe('mate');
  });

  it('locks live game with exact fair-play message', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: 'started' }),
    });

    const evalRes = await evaluatePageAccess('https://lichess.org/AbCd1234', mockFetch as any);
    expect(evalRes.allowed).toBe(false);
    expect(evalRes.reason).toBe(LOCK_OFF_MESSAGE);
  });

  it('caches finished game status and avoids redundant network calls', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: 'resign' }),
    });

    const res1 = await checkGameStatus('AbCd1234', mockFetch as any);
    const res2 = await checkGameStatus('AbCd1234', mockFetch as any);

    expect(res1.isFinished).toBe(true);
    expect(res2.isFinished).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('requests slim game fields and caches finished game data for reuse', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        id: 'ZyXw9876',
        status: 'mate',
        moves: 'e4 e5 Nf3 Nc6',
        opening: { eco: 'C44', name: "King's Knight Opening" },
      }),
    });

    await checkGameStatus('ZyXw9876', mockFetch as any);
    expect(mockFetch).toHaveBeenCalledWith(
      'https://lichess.org/game/export/ZyXw9876?moves=true&pgnInJson=false&clocks=false&evals=false&opening=true',
      expect.objectContaining({ headers: { Accept: 'application/json' } })
    );

    const cached = getCachedFinishedGameData('ZyXw9876');
    expect(cached).toMatchObject({
      moves: ['e4', 'e5', 'Nf3', 'Nc6'],
      opening: "King's Knight Opening",
    });
  });
});
