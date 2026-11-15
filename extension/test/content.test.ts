// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  fetchLichessGameApi,
  clearApiGameCache,
  extractMovesFromDom,
  extractFenFromDom,
  extractInitialFenFromUrlAndDom,
  isLikelyFen,
  getLichessGameState,
  broadcastState,
} from '../src/content.js';
import { clearGameStatusCache } from '../src/lock.js';

describe('Content Script: Lichess API & Move Extraction', () => {
  beforeEach(() => {
    clearApiGameCache();
    clearGameStatusCache();
  });

  it('fetches and parses official game moves and metadata from Lichess API', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        id: 'AbCd1234',
        moves: 'e4 e5 Nf3 Nc6 Bb5 a6',
        players: {
          white: { user: { name: 'PlayerOne' } },
          black: { user: { name: 'PlayerTwo' } },
        },
        opening: { name: 'Ruy Lopez, Morphy Defense' },
      }),
    });

    const data = await fetchLichessGameApi('AbCd1234', mockFetch as any);

    expect(data).not.toBeNull();
    expect(data?.moves).toEqual(['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6']);
    expect(data?.whitePlayer).toBe('PlayerOne');
    expect(data?.blackPlayer).toBe('PlayerTwo');
    expect(data?.opening).toBe('Ruy Lopez, Morphy Defense');
  });

  it('caches API game data and avoids redundant fetches', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        moves: 'd4 d5 c4',
      }),
    });

    const d1 = await fetchLichessGameApi('Game1234', mockFetch as any);
    const d2 = await fetchLichessGameApi('Game1234', mockFetch as any);

    expect(d1?.moves).toEqual(['d4', 'd5', 'c4']);
    expect(d2?.moves).toEqual(['d4', 'd5', 'c4']);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('handles API errors gracefully by returning null', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
    });

    const data = await fetchLichessGameApi('NotFound1', mockFetch as any);
    expect(data).toBeNull();
  });

  it('extracts initialFen and variant from Chess960 / From-Position API responses', async () => {
    const customFen = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        moves: 'Nf3 Nc6',
        initialFen: customFen,
        variant: { key: 'fromPosition' },
      }),
    });

    const data = await fetchLichessGameApi('PosGame1', mockFetch as any);
    expect(data?.initialFen).toBe(customFen);
    expect(data?.variant).toBe('fromPosition');
  });

  it('extracts initialFen from PGN FEN header when initialFen field is absent', async () => {
    const customFen = '4k3/8/8/8/8/8/4P3/4K3 w - - 0 1';
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        moves: 'e4',
        pgn: `[Event "Rated"]\n[FEN "${customFen}"]\n\n1. e4`,
      }),
    });

    const data = await fetchLichessGameApi('PgnGame1', mockFetch as any);
    expect(data?.initialFen).toBe(customFen);
  });

  it('enforces HTTP 429 cooldown in fetchLichessGameApi', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 429,
      headers: { get: () => '30' },
    });

    const d1 = await fetchLichessGameApi('RateApi1', mockFetch as any);
    const d2 = await fetchLichessGameApi('RateApi2', mockFetch as any);
    expect(d1).toBeNull();
    expect(d2).toBeNull();
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

describe('Content Script: Lichess move-list extraction', () => {
  const mainline: Array<{ path: string; san: string; ply: number }> = (() => {
    const sans = [
      'e4', 'c6', 'd3', 'd5', 'exd5', 'cxd5', 'Nc3', 'e5', 'Nf3', 'Nc6',
      'd4', 'exd4', 'Nxd4', 'Nf6', 'Nxc6', 'bxc6', 'Bg5', 'Bb4', 'Bxf6',
      'Qxf6', 'Qd2', 'd4', 'Qe2+', 'Qe6', 'Qxe6+', 'Bxe6', 'Kd2', 'dxc3+',
      'bxc3', 'Bc5', 'f3',
    ];
    const ids = [
      '00', '01', '02', '03', '04', '05', '06', '07', '08', '0a', '0b',
      '0c', '0d', '0e', '0f', '10', '11', '12', '13', '14', '15', '16',
      '17', '18', '19', '1a', '1b', '1c', '1d', '1e', '1f',
    ];
    let path = '';
    return sans.map((san, i) => {
      path += ids[i];
      return { path, san, ply: i + 1 };
    });
  })();

  function moveMarkup(
    path: string,
    san: string,
    opts: { mainline?: boolean; active?: boolean; current?: boolean; evalText?: string } = {}
  ): string {
    const classes = [
      opts.mainline ? 'mainline' : '',
      opts.active ? 'active' : '',
      opts.current ? 'current' : '',
      san.endsWith('?!') ? 'inaccuracy' : '',
    ]
      .filter(Boolean)
      .join(' ');
    return `<move p="${path}" class="${classes}"><index>${Math.ceil((path.length / 2 + 1) / 2)}.</index><san>${san.replace(/[?!]+$/, '')}</san><glyph>${san.match(/[?!]+$/)?.[0] || ''}</glyph><eval>${opts.evalText || '+0.1'}</eval></move>`;
  }

  function buildMoveList(activePath?: string, currentPath?: string): void {
    const prefix = mainline.slice(0, 8); // e4 ... e5
    const variationPrefix = prefix[prefix.length - 1].path + '09';
    const variationMoves = [
      { path: variationPrefix, san: 'd4' },
      { path: variationPrefix + '0a', san: 'e4' },
      { path: variationPrefix + '0a0b', san: 'f3' },
      { path: variationPrefix + '0a0b0c', san: 'Nf6' },
      { path: variationPrefix + '0a0b0c0d', san: 'fxe4' },
      { path: variationPrefix + '0a0b0c0d0e', san: 'Nxe4' },
    ];

    const parts: string[] = ['<div class="analyse__moves"><div class="tview2 tview2-inline">'];
    for (const move of mainline.slice(0, 9)) {
      parts.push(
        moveMarkup(move.path, move.san + (move.san === 'Nf3' ? '?!' : ''), {
          mainline: true,
          active: move.path === activePath,
          current: move.path === currentPath,
        })
      );
    }
    parts.push('<comment>Inaccuracy. d4 was best.</comment>');
    parts.push('<interrupt><lines><line>');
    for (const move of variationMoves) {
      parts.push(
        moveMarkup(move.path, move.san, {
          active: move.path === activePath,
          current: move.path === currentPath,
        })
      );
    }
    parts.push('</line></lines></interrupt>');
    for (const move of mainline.slice(9)) {
      parts.push(
        moveMarkup(move.path, move.san, {
          mainline: true,
          active: move.path === activePath,
          current: move.path === currentPath,
        })
      );
    }
    parts.push('</div></div>');
    document.body.innerHTML = parts.join('');
  }

  it('extracts only mainline moves and ignores variation lines', () => {
    buildMoveList();

    const result = extractMovesFromDom();
    expect(result.moves[0]).toBe('e4');
    expect(result.moves[9]).toBe('Nc6');
    expect(result.moves[23]).toBe('Qe6');
    expect(result.moves).toHaveLength(mainline.length);
    // Variation moves must not leak into the game move list.
    expect(result.moves).not.toContain('Nxe4');
  });

  it('resolves an active mainline move by its tree path, not by counting DOM nodes', () => {
    const active = mainline[9]; // 5...Nc6
    buildMoveList(active.path);

    const result = extractMovesFromDom();
    expect(result.selectedMoveSan).toBe('Nc6');
    expect(result.activePly).toBe(10);
    expect(result.activeIsVariation).toBe(false);
    expect(result.activeLine).toHaveLength(10);
    expect(result.activeLine?.[9]).toBe('Nc6');
  });

  it('rebuilds the exact variation line when a variation move is active', () => {
    const variationPrefix = mainline[7].path + '09';
    const activePath = variationPrefix + '0a0b'; // 5.d4 e4 6.f3
    buildMoveList(activePath);

    const result = extractMovesFromDom();
    expect(result.selectedMoveSan).toBe('f3');
    expect(result.activeIsVariation).toBe(true);
    expect(result.activeLine).toEqual([
      'e4', 'c6', 'd3', 'd5', 'exd5', 'cxd5', 'Nc3', 'e5', 'd4', 'e4', 'f3',
    ]);
    expect(result.activePly).toBe(11);
  });

  it('keeps the API mainline authoritative and uses the selected SAN to fix a stale ply', async () => {
    const active = mainline[9]; // 5...Nc6
    buildMoveList(active.path);

    const mockFetch = vi.fn((url: string) => {
      if (url.includes('/game/export/')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: 'resign',
            moves: mainline.map((m) => m.san).join(' '),
            players: { white: { user: { name: 'LaKdarma' } }, black: { user: { name: 'joebossie' } } },
            opening: { name: 'Caro-Kann Defense: Breyer Variation' },
          }),
        });
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
    });

    // Point the jsdom URL at the real game URL from the report.
    window.history.replaceState(null, '', '/GGRmK9Q0?pov=joebossie#10');
    const state = await getLichessGameState(mockFetch as any);

    expect(state.moves?.[9]).toBe('Nc6');
    expect(state.selectedMoveSan).toBe('Nc6');
    expect(state.activePly).toBe(10);
    expect(state.activeLine).toEqual(mainline.slice(0, 10).map((m) => m.san));
  });

  it('rejects Lichess share URLs in input.copyable and extracts valid FENs from DOM', () => {
    expect(isLikelyFen('https://lichess.org/AbCd1234')).toBe(false);
    expect(isLikelyFen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1')).toBe(true);

    document.body.innerHTML = `
      <input class="copyable" value="https://lichess.org/AbCd1234" />
      <div class="analyse__fen">
        <input class="copyable" value="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" />
      </div>
    `;
    expect(extractFenFromDom()).toBe(
      'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'
    );

    // Falls back cleanly to starting FEN when only share URL exists
    document.body.innerHTML = `<input class="copyable" value="https://lichess.org/AbCd1234" />`;
    expect(extractFenFromDom()).toBe(
      'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    );
  });

  it('extracts initialFen from /analysis/fromPosition/ and direct /analysis/<FEN> URLs', () => {
    document.body.innerHTML = '';
    const fen = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3';
    const encoded = fen.replace(/ /g, '_');

    expect(extractInitialFenFromUrlAndDom(`https://lichess.org/analysis/fromPosition/${encoded}`)).toBe(fen);
    expect(extractInitialFenFromUrlAndDom(`https://lichess.org/analysis/${encoded}`)).toBe(fen);
    expect(extractInitialFenFromUrlAndDom('https://lichess.org/analysis/standard')).toBeUndefined();
  });

  it('extracts selected move from standard round l4x/rm6 move list without tview2', () => {
    document.body.innerHTML = `
      <div class="rm6">
        <l4x>
          <i5z>1</i5z><kwdb>e4</kwdb><kwdb>e5</kwdb>
          <i5z>2</i5z><kwdb class="a1t">Nf3</kwdb><kwdb>Nc6</kwdb>
        </l4x>
      </div>
    `;
    const result = extractMovesFromDom();
    expect(result.moves).toEqual(['e4', 'e5', 'Nf3', 'Nc6']);
    expect(result.selectedMoveSan).toBe('Nf3');
    expect(result.activePly).toBe(3);
    expect(result.activeLine).toEqual(['e4', 'e5', 'Nf3']);
  });

  it('emits activePly 0 and undefined selectedMoveSan at start position instead of jumping to last move', async () => {
    buildMoveList(); // No active or current move
    window.history.replaceState(null, '', '/analysis#0');

    const domRes = extractMovesFromDom();
    expect(domRes.activePly).toBe(0);
    expect(domRes.selectedMoveSan).toBeUndefined();
    expect(domRes.activeLine).toEqual([]);

    const state = await getLichessGameState();
    expect(state.activePly).toBe(0);
    expect(state.selectedMoveSan).toBeUndefined();
    expect(state.activeLine).toEqual([]);
  });

  it('maps ruser-top and ruser-bottom according to board orientation', async () => {
    document.body.innerHTML = `
      <div class="cg-wrap orientation-white"></div>
      <div class="ruser-top"><a class="user-link">BlackPlayer</a></div>
      <div class="ruser-bottom"><a class="user-link">WhitePlayer</a></div>
    `;
    window.history.replaceState(null, '', '/analysis');
    const whiteBottom = await getLichessGameState();
    expect(whiteBottom.whitePlayer).toBe('WhitePlayer');
    expect(whiteBottom.blackPlayer).toBe('BlackPlayer');

    document.body.innerHTML = `
      <div class="cg-wrap orientation-black"></div>
      <div class="ruser-top"><a class="user-link">WhiteFlipped</a></div>
      <div class="ruser-bottom"><a class="user-link">BlackFlipped</a></div>
    `;
    const blackBottom = await getLichessGameState();
    expect(blackBottom.whitePlayer).toBe('WhiteFlipped');
    expect(blackBottom.blackPlayer).toBe('BlackFlipped');
  });

  it('drops stale out-of-order broadcastState responses', async () => {
    buildMoveList(mainline[0].path);
    window.history.replaceState(null, '', '/SlowGam1');

    let resolveSlow!: (val: any) => void;
    const slowPromise = new Promise((r) => {
      resolveSlow = r;
    });

    const sendMessage = vi.fn().mockResolvedValue(undefined);
    (globalThis as any).chrome = { runtime: { sendMessage } };

    // Mock fetch: first game hangs on status check, second game resolves immediately via /analysis
    const origFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(() => slowPromise as any);

    try {
      const p1 = broadcastState();
      window.history.replaceState(null, '', '/analysis');
      const p2 = broadcastState();
      await p2;

      resolveSlow({ ok: true, status: 200, json: async () => ({ status: 'started' }) });
      await p1;

      expect(sendMessage).toHaveBeenCalledTimes(1);
      expect(sendMessage.mock.calls[0][0].payload.pageType).toBe('analysis');
    } finally {
      globalThis.fetch = origFetch;
      delete (globalThis as any).chrome;
    }
  });
});
