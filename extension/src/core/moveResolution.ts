/**
 * Board and Move Timing Resolution.
 *
 * Solves Lichess FEN timing desynchronization by replaying moves from start position:
 * - Computes accurate fenBefore, moveUci, and fenAfter for any active ply.
 * - Enables precise win-rate drop calculation and move verdicts (Blunder, Mistake, Inaccuracy, Good, Best).
 */

import { Chess } from 'chessops';
import { parseFen, makeFen } from 'chessops/fen';
import { makeSan, makeSanAndPlay, parseSan } from 'chessops/san';
import type { NormalMove } from 'chessops/types';
import { isNormal } from 'chessops/types';
import { kingCastlesTo, roleToChar } from 'chessops/util';
import { getLegalMoves } from '../../../core/ts/src/facts.js';

export interface MoveContext {
  fenBefore: string;
  fenAfter?: string;
  moveUci?: string;
  moveSan?: string;
}

const DEFAULT_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const MAX_MOVE_CONTEXT_CACHE = 256;
const moveContextCache = new Map<string, MoveContext>();

export function clearMoveContextCache(): void {
  moveContextCache.clear();
}

export function squareToString(sq: number): string {
  const file = String.fromCharCode('a'.charCodeAt(0) + (sq % 8));
  const rank = String.fromCharCode('1'.charCodeAt(0) + Math.floor(sq / 8));
  return file + rank;
}

function normalizeSan(san: string): string {
  let s = san.trim().replace(/[!?]+$/g, '');
  if (s.startsWith('0-0-0')) {
    s = 'O-O-O' + s.slice(5);
  } else if (s.startsWith('0-0')) {
    s = 'O-O' + s.slice(3);
  }
  return s;
}

function sanMatches(candidateSan: string, targetSan: string): boolean {
  const a = normalizeSan(candidateSan);
  const b = normalizeSan(targetSan);
  if (a === b) return true;
  return a.replace(/[+#]+$/, '') === b.replace(/[+#]+$/, '');
}

function resolveSanMove(chess: Chess, san: string): NormalMove | undefined {
  const normalized = normalizeSan(san);
  if (normalized) {
    const parsed = parseSan(chess, normalized);
    if (parsed && isNormal(parsed)) {
      return parsed;
    }
  }
  // Fallback for any non-standard SAN notation
  for (const m of getLegalMoves(chess)) {
    if (sanMatches(makeSan(chess, m.move), san)) {
      return m.move;
    }
  }
  return undefined;
}

function formatMoveUci(chess: Chess, move: NormalMove): string {
  const piece = chess.board.get(move.from);
  const toSq =
    piece?.role === 'king' && chess.board[chess.turn].has(move.to)
      ? kingCastlesTo(chess.turn, move.to > move.from ? 'h' : 'a')
      : move.to;
  const promo = move.promotion ? roleToChar(move.promotion) : '';
  return `${squareToString(move.from)}${squareToString(toSq)}${promo}`;
}

/**
 * Reconstructs accurate fenBefore, moveUci, and fenAfter by replaying moves.
 */
export function resolveMoveContext(
  moves: string[] = [],
  activePly?: number,
  selectedMoveSan?: string,
  currentFen?: string,
  initialFen?: string
): MoveContext {
  const cacheKey = `${initialFen ?? ''}\0${currentFen ?? ''}\0${activePly ?? ''}\0${selectedMoveSan ?? ''}\0${moves.join('\x01')}`;
  const cached = moveContextCache.get(cacheKey);
  if (cached) {
    moveContextCache.delete(cacheKey);
    moveContextCache.set(cacheKey, cached);
    return { ...cached };
  }

  const result = computeMoveContext(moves, activePly, selectedMoveSan, currentFen, initialFen);
  if (moveContextCache.size >= MAX_MOVE_CONTEXT_CACHE) {
    const oldestKey = moveContextCache.keys().next().value;
    if (oldestKey !== undefined) {
      moveContextCache.delete(oldestKey);
    }
  }
  moveContextCache.set(cacheKey, { ...result });
  return result;
}

function computeMoveContext(
  moves: string[] = [],
  activePly?: number,
  selectedMoveSan?: string,
  currentFen?: string,
  initialFen?: string
): MoveContext {
  let startChess = Chess.default();
  let hasValidInitialFen = false;
  if (initialFen) {
    const parseRes = parseFen(initialFen);
    if (parseRes.isOk) {
      const fromSetup = Chess.fromSetup(parseRes.value);
      if (fromSetup.isOk) {
        startChess = fromSetup.value;
        hasValidInitialFen = true;
      }
    }
  }
  const startingFenStr = hasValidInitialFen ? makeFen(startChess.toSetup()) : DEFAULT_FEN;

  if (moves.length > 0) {
    // `activePly === 0` explicitly means the initial position (no move yet).
    if (activePly === 0 && !selectedMoveSan?.trim()) {
      return {
        fenBefore: hasValidInitialFen ? startingFenStr : (currentFen || startingFenStr),
      };
    }

    let targetPly =
      activePly !== undefined && activePly > 0 && activePly <= moves.length
        ? activePly
        : activePly === 0
          ? 1
          : moves.length;

    // A stale/lossy ply must never silently analyze a different move. If the
    // requested ply does not match the selected SAN, locate the SAN in the
    // move list instead.
    if (selectedMoveSan && !sanMatches(moves[targetPly - 1], selectedMoveSan)) {
      const matches: number[] = [];
      for (let i = 0; i < moves.length; i++) {
        if (sanMatches(moves[i], selectedMoveSan)) matches.push(i + 1);
      }

      if (matches.length === 0) {
        // The selected move is not part of this line (e.g. a variation the
        // caller could not reconstruct). Fall back to a custom FEN position.
        return resolveCustomPosition(currentFen, selectedMoveSan);
      }

      // Prefer the occurrence closest to the requested ply.
      targetPly = matches.reduce((best, ply) =>
        Math.abs(ply - targetPly) < Math.abs(best - targetPly) ? ply : best
      );
    }

    const chess = startChess;
    let replayBroken = false;
    for (let i = 0; i < targetPly - 1; i++) {
      const matchedMove = resolveSanMove(chess, moves[i]);
      if (!matchedMove) {
        replayBroken = true;
        break;
      }
      chess.play(matchedMove);
    }

    if (replayBroken) {
      // Moves likely start from a custom FEN not provided as initialFen;
      // fall back to currentFen resolution rather than returning a bogus default FEN.
      return resolveCustomPosition(currentFen, selectedMoveSan || moves[targetPly - 1]);
    }

    const fenBefore = makeFen(chess.toSetup());
    const targetSan = selectedMoveSan || moves[targetPly - 1];
    let moveUci: string | undefined;
    let fenAfter: string | undefined;
    let canonicalSan: string = targetSan;

    const targetMove = resolveSanMove(chess, targetSan);
    if (targetMove) {
      moveUci = formatMoveUci(chess, targetMove);
      canonicalSan = makeSanAndPlay(chess, targetMove);
      fenAfter = makeFen(chess.toSetup());
    }

    if (!moveUci && !hasValidInitialFen && currentFen) {
      return resolveCustomPosition(currentFen, targetSan);
    }

    return {
      fenBefore,
      fenAfter,
      moveUci,
      moveSan: canonicalSan,
    };
  }

  // Fallback for custom positions without move history
  return resolveCustomPosition(
    currentFen || (hasValidInitialFen ? startingFenStr : DEFAULT_FEN),
    selectedMoveSan
  );
}

/**
 * Resolves a single move from a custom FEN position.
 */
function resolveCustomPosition(currentFen?: string, selectedMoveSan?: string): MoveContext {
  const fen = currentFen || DEFAULT_FEN;
  let moveUci: string | undefined;
  let fenAfter: string | undefined;
  let canonicalSan = selectedMoveSan;

  if (selectedMoveSan) {
    try {
      const parseRes = parseFen(fen);
      if (parseRes.isOk) {
        const chessRes = Chess.fromSetup(parseRes.value);
        if (chessRes.isOk) {
          const chess = chessRes.value;
          const targetMove = resolveSanMove(chess, selectedMoveSan);
          if (targetMove) {
            moveUci = formatMoveUci(chess, targetMove);
            canonicalSan = makeSanAndPlay(chess, targetMove);
            fenAfter = makeFen(chess.toSetup());
          }
        }
      }
    } catch {
      // ignore
    }
  }

  return {
    fenBefore: fen,
    fenAfter,
    moveUci,
    moveSan: canonicalSan,
  };
}
