import { describe, it, expect } from 'vitest';
import { resolveMoveContext } from '../src/core/moveResolution.js';

describe('Move and Board Timing Resolution (resolveMoveContext)', () => {
  it('correctly resolves position BEFORE move 3 (Nf3) from game move list', () => {
    const moves = ['e4', 'e5', 'Nf3', 'Nc6'];
    const activePly = 3; // White's move: Nf3

    const ctx = resolveMoveContext(moves, activePly);

    // fenBefore must have White to move, with e4 and e5 on board
    expect(ctx.fenBefore).toContain('w KQkq');
    expect(ctx.moveSan).toBe('Nf3');
    expect(ctx.moveUci).toBe('g1f3');
    // fenAfter must have Black to move, with Nf3 on board
    expect(ctx.fenAfter).toContain('b KQkq');
  });

  it('correctly resolves first move 1. e4', () => {
    const moves = ['e4', 'c5'];
    const activePly = 1;

    const ctx = resolveMoveContext(moves, activePly);

    expect(ctx.fenBefore).toBe('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    expect(ctx.moveSan).toBe('e4');
    expect(ctx.moveUci).toBe('e2e4');
    expect(ctx.fenAfter).toBe('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
  });

  it('correctly defaults to the last move when activePly is omitted', () => {
    const moves = ['d4', 'd5', 'c4'];
    const ctx = resolveMoveContext(moves);

    expect(ctx.moveSan).toBe('c4');
    expect(ctx.moveUci).toBe('c2c4');
    expect(ctx.fenBefore).toContain('w KQkq');
    expect(ctx.fenAfter).toContain('b KQkq');
  });

  it('handles custom FEN positions with legal move matching', () => {
    const startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const ctx = resolveMoveContext([], undefined, 'e4', startFen);

    expect(ctx.fenBefore).toBe(startFen);
    expect(ctx.moveSan).toBe('e4');
    expect(ctx.moveUci).toBe('e2e4');
    expect(ctx.fenAfter).toBe('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1');
  });
});

describe('Move/ply consistency', () => {
  const gameMoves = [
    'e4', 'c6', 'd3', 'd5', 'exd5', 'cxd5', 'Nc3', 'e5', 'Nf3', 'Nc6',
    'd4', 'exd4', 'Nxd4', 'Nf6', 'Nxc6', 'bxc6', 'Bg5', 'Bb4', 'Bxf6',
    'Qxf6', 'Qd2', 'd4', 'Qe2+', 'Qe6', 'Qxe6+',
  ];

  it('uses the selected SAN when a stale ply points at a different move', () => {
    // Regression: a polluted DOM move index produced ply 24 (Qe6) while the
    // user had actually selected 5...Nc6 (ply 10).
    const ctx = resolveMoveContext(gameMoves, 24, 'Nc6');

    expect(ctx.moveSan).toBe('Nc6');
    expect(ctx.moveUci).toBe('b8c6');
    expect(ctx.fenBefore).toBe(
      'rnbqkbnr/pp3ppp/8/3pp3/8/2NP1N2/PPP2PPP/R1BQKB1R b KQkq - 1 5'
    );
    expect(ctx.fenAfter).toContain(' w KQkq ');
  });

  it('treats an explicit activePly of 0 as the initial position', () => {
    const ctx = resolveMoveContext(gameMoves, 0);
    expect(ctx.moveSan).toBeUndefined();
    expect(ctx.fenBefore).toBe('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  });

  it('resolves a reconstructed variation line correctly', () => {
    // 1.e4 c6 2.d3 d5 3.exd5 cxd5 4.Nc3 e5 5.d4 e4 6.f3
    const variation = ['e4', 'c6', 'd3', 'd5', 'exd5', 'cxd5', 'Nc3', 'e5', 'd4', 'e4', 'f3'];
    const ctx = resolveMoveContext(variation, variation.length, 'f3');

    expect(ctx.moveSan).toBe('f3');
    expect(ctx.moveUci).toBe('f2f3');
    expect(ctx.fenBefore).toContain(' w KQkq ');
    expect(ctx.fenAfter).toContain(' b KQkq ');
  });

  it('replays moves accurately from a custom initialFen (Study / fromPosition)', () => {
    // 1... e5 from after 1. e4, or custom tactical setup:
    const customStartFen = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
    const ctx = resolveMoveContext(['Nf3', 'Nc6'], 2, 'Nc6', undefined, customStartFen);

    expect(ctx.moveSan).toBe('Nc6');
    expect(ctx.moveUci).toBe('b8c6');
    expect(ctx.fenBefore).toBe('rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2');
    expect(ctx.fenAfter).toContain('2n5');
  });

  it('falls back cleanly when standard replay breaks due to an unknown custom starting position', () => {
    const customCurrentFen = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKBNR b KQkq - 1 2';
    // Suppose moves list contains moves from an uncaptured custom start where ply 1 is illegal from Chess.default()
    const ctx = resolveMoveContext(['Qh4+', 'g3'], 1, 'Qh4+', customCurrentFen);
    // Does not falsely return Chess.default() as fenBefore with mismatched moves
    expect(ctx.fenBefore).toBe(customCurrentFen);
  });

  it('normalizes zero-castling and NAG annotations in SAN strings', () => {
    const moves = ['e4!', 'e5', 'Nf3?!', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6', '0-0'];
    const ctx = resolveMoveContext(moves, 9, '0-0!');

    expect(ctx.moveSan).toBe('O-O');
    expect(ctx.moveUci).toBe('e1g1');
    expect(ctx.fenAfter).toContain('RNBQ1RK1');
  });

  it('prefers valid initialFen on activePly 0 even if currentFen is stale', () => {
    const customStartFen = 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2';
    const staleFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    const ctx = resolveMoveContext(['Nf3'], 0, undefined, staleFen, customStartFen);

    expect(ctx.fenBefore).toBe(customStartFen);
    expect(ctx.moveSan).toBeUndefined();
    expect(ctx.fenAfter).toBeUndefined();
  });

  it('resolves queenside castling and returns independent memoized copies', () => {
    const moves = ['e4', 'e5', 'd4', 'exd4', 'Nc3', 'Nc6', 'Bf4', 'Nf6', 'Qd2', 'Be7', '0-0-0'];
    const ctx1 = resolveMoveContext(moves, 11, 'O-O-O');
    expect(ctx1.moveSan).toBe('O-O-O');
    expect(ctx1.moveUci).toBe('e1c1');
    expect(ctx1.fenAfter).toContain('2KR1BNR');

    ctx1.moveSan = 'MUTATED';
    const ctx2 = resolveMoveContext(moves, 11, 'O-O-O');
    expect(ctx2.moveSan).toBe('O-O-O');
    expect(ctx2.moveUci).toBe('e1c1');
  });

  it('formats knight underpromotion UCI as n rather than k', () => {
    const startFen = '8/4P1k1/8/8/8/8/8/6K1 w - - 0 1';
    const ctx = resolveMoveContext(['e8=N+'], 1, 'e8=N+', undefined, startFen);
    expect(ctx.moveSan).toBe('e8=N+');
    expect(ctx.moveUci).toBe('e7e8n');
    expect(ctx.fenAfter).toContain('4N3');
  });
});
