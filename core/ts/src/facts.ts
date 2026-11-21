/**
 * Fact extractor for chess positions.
 * This module computes deterministic facts from positions and engine evaluations.
 * It exports functions to evaluate winning chances, classify moves, and extract facts.
 */

import { Chess } from "chessops";
import { attacks, ray } from "chessops/attacks";
import { parseFen } from "chessops/fen";
import { makeSan, makeSanAndPlay } from "chessops/san";
import { Color, isNormal, NormalMove, Role } from "chessops/types";
import {
  kingCastlesTo,
  makeSquare,
  opposite,
  parseUci,
  roleToChar,
  squareFile,
  squareRank,
} from "chessops/util";
import { lookupOpening } from "./openings.js";

export interface EngineScore {
  type: "cp" | "mate";
  value: number;
}

export interface EnginePv {
  moves: string[];
  score: EngineScore;
}

export interface EngineEval {
  fen: string;
  pvs: EnginePv[];
}

export type Verdict =
  | "Best"
  | "Only move"
  | "Good"
  | "Inaccuracy"
  | "Mistake"
  | "Blunder";

export const PIECE_VALUES: Record<Role, number> = {
  pawn: 1,
  knight: 3,
  bishop: 3,
  rook: 5,
  queen: 9,
  king: 0,
};

const CENTRAL_SQUARES = new Set<number>([27, 28, 35, 36]); // d4, e4, d5, e5
const WING_CENTER_45_SQUARES = new Set<number>([26, 29, 34, 37]); // c4, f4, c5, f5
const WHITE_START_MINORS = new Set<number>([1, 2, 5, 6]); // b1, c1, f1, g1
const BLACK_START_MINORS = new Set<number>([57, 58, 61, 62]); // b8, c8, f8, g8
const FILE_LETTERS = "abcdefgh";

/** Check if sq lies inside the central c3-f6 box. */
function isExtendedCenter(sq: number): boolean {
  const f = squareFile(sq);
  const r = squareRank(sq);
  return f >= 2 && f <= 5 && r >= 2 && r <= 5;
}

/** Check if sq lies on central fourth or fifth rank files c through f. */
function isCenter45(sq: number): boolean {
  const f = squareFile(sq);
  const r = squareRank(sq);
  return f >= 2 && f <= 5 && r >= 3 && r <= 4;
}

/** Check whether color still retains any castling rights. */
function hasCastlingRights(pos: Chess, color: Color): boolean {
  return pos.castles.rook[color].a !== undefined || pos.castles.rook[color].h !== undefined;
}

/** Check whether move represents castling in standard or Chess960 UCI. */
function isCastlingMove(pos: Chess, move: NormalMove): boolean {
  const p = pos.board.get(move.from);
  if (!p || p.role !== "king") return false;
  return (
    pos.board[pos.turn].has(move.to) ||
    Math.abs(squareFile(move.to) - squareFile(move.from)) === 2
  );
}

/** Check whether move captures an enemy piece or en passant pawn. */
function isCaptureMove(pos: Chess, move: NormalMove): boolean {
  const p = pos.board.get(move.from);
  if (!p) return false;
  if (p.role === "king" && pos.board[pos.turn].has(move.to)) return false;
  const target = pos.board.get(move.to);
  if (target && target.color !== pos.turn) return true;
  return p.role === "pawn" && pos.epSquare !== undefined && move.to === pos.epSquare;
}

/** Return true if move encodes castling in Chess960 king-takes-friendly-rook notation. */
export function isKingTakesFriendlyRook(pos: Chess, move: NormalMove): boolean {
  const p = pos.board.get(move.from);
  return Boolean(p && p.role === "king" && pos.board[pos.turn].has(move.to));
}

/** Normalize king-takes-friendly-rook UCI to standard castling UCI when applicable. */
function canonicalMoveUci(pos: Chess | undefined, uci: string): string {
  if (!pos || !uci) return uci;
  const move = parseUci(uci);
  if (!move || !isNormal(move) || !isKingTakesFriendlyRook(pos, move)) return uci;
  const kingDest = kingCastlesTo(pos.turn, move.to > move.from ? "h" : "a");
  return `${makeSquare(move.from)}${makeSquare(kingDest)}`;
}

/** Normalize castling target squares to the king destination square. */
function normalizeMoveTo(pos: Chess, move: NormalMove, color: Color = pos.turn): number {
  const pieceBefore = pos.board.get(move.from);
  if (pieceBefore && pieceBefore.role === "king" && pos.board[color].has(move.to)) {
    return kingCastlesTo(color, move.to > move.from ? "h" : "a");
  }
  return move.to;
}

/** Count squares that the piece on sq attacks, excluding friendly pieces. */
function pieceMobility(pos: Chess, sq: number): number {
  const piece = pos.board.get(sq);
  if (!piece) return 0;
  return attacks(piece, sq, pos.board.occupied).diff(pos.board[piece.color]).size();
}

/** Count pawns on fileIdx for one color or both colors. */
function countPawnsOnFile(pos: Chess, fileIdx: number, color?: Color): number {
  let count = 0;
  const targetColors: Color[] = color ? [color] : ["white", "black"];
  for (const c of targetColors) {
    for (const sq of pos.board.pawn.intersect(pos.board[c])) {
      if (squareFile(sq) === fileIdx) count++;
    }
  }
  return count;
}

/** Classify whether a rook move claims an open or half-open file. */
function classifyRookFileMove(
  posBefore: Chess,
  posAfter: Chess,
  fromSq: number,
  toSq: number,
  moverColor: Color
): "open" | "half-open" | null {
  const fromFile = squareFile(fromSq);
  const toFile = squareFile(toSq);
  if (fromFile === toFile) return null;
  if (countPawnsOnFile(posBefore, toFile, moverColor) > 0) return null;

  const oppColor = opposite(moverColor);
  const fromFriendly = countPawnsOnFile(posBefore, fromFile, moverColor);
  const fromEnemy = countPawnsOnFile(posBefore, fromFile, oppColor);
  const toEnemy = countPawnsOnFile(posAfter, toFile, oppColor);

  if (toEnemy === 0 && (fromFriendly >= 1 || fromEnemy >= 1)) {
    return "open";
  }
  if (toEnemy >= 1 && fromFriendly >= 1) {
    return "half-open";
  }
  return null;
}

/** Return true when move is a pawn capturing an enemy pawn or en passant. */
function isPawnTakesPawn(posBefore: Chess, move: NormalMove): boolean {
  const mover = posBefore.board.get(move.from);
  if (!mover || mover.role !== "pawn") return false;
  if (!isCaptureMove(posBefore, move)) return false;
  if (
    posBefore.epSquare !== undefined &&
    move.to === posBefore.epSquare &&
    !posBefore.board.get(move.to)
  ) {
    return true;
  }
  const target = posBefore.board.get(move.to);
  return target !== undefined && target.role === "pawn";
}

export interface LegalMoveInfo {
  move: NormalMove;
  from: number;
  to: number;
  promotionVal: number;
}

/**
 * Enumerate all legal moves for the current position.
 * Castling destinations use king squares to match standard order.
 */
export function getLegalMoves(pos: Chess): LegalMoveInfo[] {
  const moves: LegalMoveInfo[] = [];
  const turn = pos.turn;
  const promoRank = turn === "white" ? 7 : 0;

  for (const [from, dests] of pos.allDests()) {
    const piece = pos.board.get(from);
    if (!piece) continue;

    for (const dest of dests) {
      if (piece.role === "king" && pos.board[turn].has(dest)) {
        const kingDest = kingCastlesTo(turn, dest > from ? "h" : "a");
        moves.push({
          move: { from, to: kingDest },
          from,
          to: kingDest,
          promotionVal: 0,
        });
      } else if (piece.role === "pawn" && squareRank(dest) === promoRank) {
        const promos: Array<{ role: Role; val: number }> = [
          { role: "queen", val: 5 },
          { role: "rook", val: 4 },
          { role: "bishop", val: 3 },
          { role: "knight", val: 2 },
        ];
        for (const p of promos) {
          moves.push({
            move: { from, to: dest, promotion: p.role },
            from,
            to: dest,
            promotionVal: p.val,
          });
        }
      } else {
        moves.push({
          move: { from, to: dest },
          from,
          to: dest,
          promotionVal: 0,
        });
      }
    }
  }
  return moves;
}

/**
 * Convert an engine score to White centipawns.
 * If score type is cp, return the centipawn value.
 * If score type is mate, compute a bounded centipawn value.
 * When score type is unknown, return zero.
 */
export function scoreToWhiteCp(score: EngineScore): number {
  const scoreType = score.type ?? "cp";
  if (scoreType === "cp") {
    return Number(score.value ?? 0);
  }
  if (scoreType === "mate") {
    const m = Math.trunc(Number(score.value ?? 0));
    if (m > 0) return 10000.0 - m * 10.0;
    if (m < 0) return -10000.0 - m * 10.0;
    return 0.0;
  }
  return 0.0;
}

/**
 * Calculate the White winning chance percentage.
 * This function uses the standard Lichess winning chances formula.
 */
export function winChanceWhite(score: EngineScore): number {
  const cp = Math.max(-10000.0, Math.min(10000.0, scoreToWhiteCp(score)));
  return 50.0 + 50.0 * (2.0 / (1.0 + Math.exp(-0.00368208 * cp)) - 1.0);
}

/**
 * Calculate the winning chance percentage for one side.
 * If isWhite is true, return the White winning chance.
 * Otherwise, return one hundred minus the White winning chance.
 */
export function winChance(score: EngineScore, isWhite: boolean = true): number {
  const w = winChanceWhite(score);
  return isWhite ? w : 100.0 - w;
}

export const winChanceSide = winChance;

/**
 * Round a percentage value to the nearest integer.
 * This function rounds half values upward.
 */
export function roundPct(val: number): number {
  return Math.floor(val + 0.5);
}

function hasSingleLegalMove(pos: Chess): boolean {
  let count = 0;
  const turn = pos.turn;
  const promoRank = turn === "white" ? 7 : 0;

  for (const [from, dests] of pos.allDests()) {
    const piece = pos.board.get(from);
    if (!piece) continue;
    for (const dest of dests) {
      if (piece.role === "pawn" && squareRank(dest) === promoRank && !pos.board[turn].has(dest)) {
        count += 4;
      } else {
        count += 1;
      }
      if (count > 1) return false;
    }
  }
  return count === 1;
}

/**
 * Classify the quality of a played move.
 * If the move is best and other moves lose win chance, return Only move.
 * If the win chance drop is at least twenty percent, return Blunder.
 * If the drop is at least ten percent, return Mistake.
 * If the drop is at least five percent, return Inaccuracy.
 * When the drop is small, return Best or Good.
 */
export function classifyVerdict(
  evalBefore: EngineEval,
  evalAfter: EngineEval | null | undefined,
  moveUci: string,
  parsedPos?: Chess
): Verdict {
  const parseRes = parsedPos ? null : parseFen(evalBefore.fen);
  let verdictPos = parsedPos;
  if (!verdictPos && parseRes && parseRes.isOk) {
    const chessRes = Chess.fromSetup(parseRes.value);
    if (chessRes.isOk) {
      verdictPos = chessRes.value;
    }
  }
  const isWhite = verdictPos
    ? verdictPos.turn === "white"
    : parseRes && parseRes.isOk
      ? parseRes.value.turn === "white"
      : evalBefore.fen.split(" ")[1] === "w";
  const pvs = evalBefore.pvs ?? [];
  const bestMoveUci =
    pvs.length > 0 && pvs[0].moves && pvs[0].moves.length > 0
      ? pvs[0].moves[0]
      : "";
  const normMoveUci = canonicalMoveUci(verdictPos, moveUci);
  const normBestUci = canonicalMoveUci(verdictPos, bestMoveUci);
  const wcBefore =
    pvs.length > 0 && pvs[0].score ? winChanceSide(pvs[0].score, isWhite) : 50.0;

  let wcAfter = wcBefore;
  if (
    evalAfter &&
    evalAfter.pvs &&
    evalAfter.pvs.length > 0 &&
    evalAfter.pvs[0].score
  ) {
    wcAfter = winChanceSide(evalAfter.pvs[0].score, isWhite);
  }

  if (normMoveUci === normBestUci) {
    if (verdictPos && hasSingleLegalMove(verdictPos)) {
      return "Only move";
    }
    if (pvs.length >= 2 && pvs[1].score) {
      const wcSecond = winChanceSide(pvs[1].score, isWhite);
      if (wcBefore - wcSecond >= 15.0) {
        return "Only move";
      }
    }
  }

  const drop = wcBefore - wcAfter;
  if (normMoveUci === normBestUci && drop <= 2.0) {
    return "Best";
  }
  if (drop >= 20.0) {
    return "Blunder";
  }
  if (drop >= 10.0) {
    return "Mistake";
  }
  if (drop >= 5.0) {
    return "Inaccuracy";
  }
  return "Good";
}

/**
 * Convert a sequence of UCI moves to SAN strings.
 * If a move is illegal on the board, stop processing.
 * Return at most maxPlies SAN strings.
 */
export function uciLineToSan(
  fen: string,
  uciMoves: string[],
  maxPlies = 6,
  parsedPos?: Chess
): string[] {
  let pos: Chess;
  if (parsedPos) {
    pos = parsedPos.clone();
  } else {
    const parseRes = parseFen(fen);
    if (parseRes.isErr) return [];
    const chessRes = Chess.fromSetup(parseRes.value);
    if (chessRes.isErr) return [];
    pos = chessRes.value;
  }

  const sanList: string[] = [];
  const limit = Math.min(uciMoves.length, maxPlies);
  for (let i = 0; i < limit; i++) {
    const move = parseUci(uciMoves[i]);
    if (!move || !pos.isLegal(move)) break;
    sanList.push(makeSanAndPlay(pos, move));
  }
  return sanList;
}

/**
 * Compute the material balance fact.
 * When material is equal, return material equal.
 * When one side has more material, name the advantage.
 */
export function computeMaterialFact(pos: Chess): string {
  const b = pos.board;
  const wMask = b.white;
  const bMask = b.black;

  const wP = b.pawn.intersect(wMask).size();
  const wN = b.knight.intersect(wMask).size();
  const wB = b.bishop.intersect(wMask).size();
  const wR = b.rook.intersect(wMask).size();
  const wQ = b.queen.intersect(wMask).size();

  const bP = b.pawn.intersect(bMask).size();
  const bN = b.knight.intersect(bMask).size();
  const bB = b.bishop.intersect(bMask).size();
  const bR = b.rook.intersect(bMask).size();
  const bQ = b.queen.intersect(bMask).size();

  const whitePts = wP + 3 * (wN + wB) + 5 * wR + 9 * wQ;
  const blackPts = bP + 3 * (bN + bB) + 5 * bR + 9 * bQ;
  const diff = whitePts - blackPts;

  if (diff === 0) {
    return "Material: equal.";
  }

  const winnerName = diff > 0 ? "White" : "Black";
  const adv = Math.abs(diff);
  const wWinner = diff > 0;

  const netP = Math.max(0, wWinner ? wP - bP : bP - wP);
  const netN = Math.max(0, wWinner ? wN - bN : bN - wN);
  const netB = Math.max(0, wWinner ? wB - bB : bB - wB);
  const netR = Math.max(0, wWinner ? wR - bR : bR - wR);
  const netQ = Math.max(0, wWinner ? wQ - bQ : bQ - wQ);

  const loserNetP = Math.max(0, wWinner ? bP - wP : wP - bP);
  const loserNetN = Math.max(0, wWinner ? bN - wN : wN - bN);
  const loserNetB = Math.max(0, wWinner ? bB - wB : wB - bB);
  const loserNetR = Math.max(0, wWinner ? bR - wR : wR - bR);
  const loserNetQ = Math.max(0, wWinner ? bQ - wQ : wQ - bQ);

  // Single-type clean advantages
  if (
    loserNetP === 0 &&
    loserNetN === 0 &&
    loserNetB === 0 &&
    loserNetR === 0 &&
    loserNetQ === 0
  ) {
    if (adv === 1 && netP === 1) return `${winnerName} is up a pawn.`;
    if (adv === 2 && netP === 2) return `${winnerName} is up two pawns.`;
    if (adv === 3 && netN === 1) return `${winnerName} is up a knight.`;
    if (adv === 3 && netB === 1) return `${winnerName} is up a bishop.`;
    if (adv === 5 && netR === 1) return `${winnerName} is up a rook.`;
    if (adv === 9 && netQ === 1) return `${winnerName} is up a queen.`;
  }

  // Exchange advantage
  if (
    adv === 2 &&
    netR === 1 &&
    loserNetN + loserNetB === 1 &&
    netP === 0 &&
    loserNetP === 0 &&
    netQ === 0 &&
    loserNetQ === 0
  ) {
    return `${winnerName} is up the exchange.`;
  }

  // Fallback descriptions
  if (adv === 1) return `${winnerName} is up a pawn in material.`;
  if (adv === 2) return `${winnerName} is up two pawns in material.`;
  if (adv === 3) return `${winnerName} is up a minor piece in material.`;
  if (adv === 4) return `${winnerName} is up a minor piece and a pawn in material.`;
  if (adv === 5) return `${winnerName} is up a rook in material.`;
  if (adv === 6) return `${winnerName} is up two minor pieces in material.`;
  if (adv === 7) return `${winnerName} is up a rook and two pawns in material.`;
  if (adv === 8) return `${winnerName} is up a rook and a minor piece in material.`;
  return `${winnerName} is up a queen in material.`;
}

/**
 * Compute the current game phase fact.
 * When piece count is low, return endgame.
 * When early moves have undeveloped back ranks, return opening.
 * Otherwise, return middlegame.
 */
export function computePhaseFact(pos: Chess): string {
  let majorMinor = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = pos.board.get(sq);
    if (p && p.role !== "pawn" && p.role !== "king") {
      majorMinor++;
    }
  }

  const initialMinors: Array<{ sq: number; color: Color; role: Role }> = [
    { sq: 1, color: "white", role: "knight" }, // b1
    { sq: 2, color: "white", role: "bishop" }, // c1
    { sq: 5, color: "white", role: "bishop" }, // f1
    { sq: 6, color: "white", role: "knight" }, // g1
    { sq: 57, color: "black", role: "knight" }, // b8
    { sq: 58, color: "black", role: "bishop" }, // c8
    { sq: 61, color: "black", role: "bishop" }, // f8
    { sq: 62, color: "black", role: "knight" }, // g8
  ];

  let backrankDeveloped = 0;
  for (const m of initialMinors) {
    const p = pos.board.get(m.sq);
    if (!p || p.color !== m.color || p.role !== m.role) {
      backrankDeveloped++;
    }
  }

  if (majorMinor <= 6) {
    return "Phase: endgame.";
  }
  if (pos.fullmoves <= 10 && majorMinor >= 12 && backrankDeveloped < 6) {
    return "Phase: opening.";
  }
  return "Phase: middlegame.";
}

/**
 * Compute king safety facts for both kings.
 * If a king is in check, return a check fact.
 * If a king has not castled after move nine, return a castling fact.
 */
export function computeKingSafetyFacts(pos: Chess): string[] {
  const facts: string[] = [];
  const entries: Array<{ color: Color; sideName: string; startSq: number }> = [
    { color: "white", sideName: "White", startSq: 4 }, // e1
    { color: "black", sideName: "Black", startSq: 60 }, // e8
  ];

  for (const entry of entries) {
    const ksq = pos.board.kingOf(entry.color);
    if (pos.turn === entry.color && pos.isCheck()) {
      if (ksq !== undefined) {
        facts.push(`${entry.sideName} king is in check on ${makeSquare(ksq)}.`);
      }
    } else if (pos.fullmoves >= 10) {
      const hasCastlingRights =
        pos.castles.rook[entry.color].a !== undefined ||
        pos.castles.rook[entry.color].h !== undefined;
      if (ksq === entry.startSq && hasCastlingRights) {
        facts.push(`${entry.sideName} has not castled.`);
      }
    }
  }
  return facts;
}

/**
 * Find undefended pieces that enemy pieces attack.
 * Return at most two facts for hanging pieces.
 */
export function computeHangingFacts(pos: Chess): string[] {
  const facts: string[] = [];
  for (let sq = 0; sq < 64; sq++) {
    const p = pos.board.get(sq);
    if (!p || p.role === "pawn" || p.role === "king") continue;
    const enemy = opposite(p.color);
    const friendly = p.color;
    if (
      pos.kingAttackers(sq, enemy, pos.board.occupied).nonEmpty() &&
      pos.kingAttackers(sq, friendly, pos.board.occupied).isEmpty()
    ) {
      facts.push(`The ${p.role} on ${makeSquare(sq)} is not defended.`);
    }
  }
  return facts.slice(0, 2);
}

/** Order legal moves deterministically with higher promotions first. */
function compareLegalMoves(a: LegalMoveInfo, b: LegalMoveInfo): number {
  return a.from !== b.from
    ? a.from - b.from
    : a.to !== b.to
    ? a.to - b.to
    : b.promotionVal - a.promotionVal;
}

/** Quickly filter moves that cannot deliver direct or discovered check. */
function mightGiveCheck(pos: Chess, move: NormalMove, enemyKingSq: number | undefined): boolean {
  if (enemyKingSq === undefined) return false;
  const moverRole = pos.board.getRole(move.from);
  if (!moverRole) return true;
  const effectiveRole = move.promotion ?? moverRole;
  const occAfter = pos.board.occupied.without(move.from).with(move.to);
  if (attacks({ color: pos.turn, role: effectiveRole }, move.to, occAfter).has(enemyKingSq)) {
    return true;
  }
  if (ray(move.from, enemyKingSq).nonEmpty()) {
    return true;
  }
  if (moverRole === "pawn" && pos.epSquare !== undefined && move.to === pos.epSquare) {
    return true;
  }
  if (moverRole === "king" && isCastlingMove(pos, move)) {
    return true;
  }
  return false;
}

/** Return the top Stockfish PV check within 100 centipawns of PV zero. */
function getQualifyingPvCheck(pos: Chess, engineEval?: EngineEval | null): NormalMove | null {
  if (!engineEval || !engineEval.pvs || engineEval.pvs.length === 0) {
    return null;
  }
  const pvs = engineEval.pvs;
  const sideSign = pos.turn === "white" ? 1.0 : -1.0;
  const pv0Score = pvs[0].score;
  const pv0Cp = pv0Score ? scoreToWhiteCp(pv0Score) * sideSign : null;

  const limit = Math.min(3, pvs.length);
  for (let idx = 0; idx < limit; idx++) {
    const pv = pvs[idx];
    const moves = pv.moves ?? [];
    if (moves.length === 0) continue;
    if (idx > 0) {
      if (pv0Cp === null || !pv.score) continue;
      const pvCp = scoreToWhiteCp(pv.score) * sideSign;
      if (pv0Cp - pvCp > 100.0) continue;
    }
    const parsed = parseUci(moves[0]);
    if (!parsed || !isNormal(parsed) || !pos.isLegal(parsed)) continue;
    const after = pos.clone();
    after.play(parsed);
    if (after.isCheck()) {
      return parsed;
    }
  }
  return null;
}

/** Check if a legal check forces mate, wins material, or wins a clear advantage. */
function isForcingCheck(pos: Chess, move: NormalMove): { isMate: boolean; qualifies: boolean } {
  const moverPiece = pos.board.get(move.from);
  if (!moverPiece) return { isMate: false, qualifies: false };

  const side = pos.turn;
  const opp = opposite(side);
  const origMoverVal = PIECE_VALUES[moverPiece.role];
  const effectiveMover = move.promotion ?? moverPiece.role;
  const moverVal = PIECE_VALUES[effectiveMover];
  const toSq = normalizeMoveTo(pos, move);

  const target = pos.board.get(move.to);
  const isEp =
    moverPiece.role === "pawn" &&
    pos.epSquare !== undefined &&
    move.to === pos.epSquare &&
    !target;
  const isCap = (target !== undefined && target.color !== side) || isEp;
  const targetVal = isEp ? PIECE_VALUES.pawn : target ? PIECE_VALUES[target.role] : 0;

  const after = pos.clone();
  after.play(move);
  if (after.isCheckmate()) {
    return { isMate: true, qualifies: true };
  }

  const oppLegal = getLegalMoves(after);
  const recaptures = oppLegal.filter((r) => normalizeMoveTo(after, r) === toSq);

  // Priority 2: captures material safely or wins an exchange or piece
  if (isCap && (recaptures.length === 0 || targetVal > origMoverVal)) {
    return { isMate: false, qualifies: true };
  }

  // Confirm the checking piece is safe before evaluating non-winning-capture forcing checks
  let isSafe = false;
  if (recaptures.length === 0) {
    isSafe = true;
  } else {
    const defended =
      after.kingAttackers(toSq, side, after.board.occupied).size() >= recaptures.length;
    let allGe = true;
    for (const r of recaptures) {
      const rp = after.board.get(r.from);
      if (!rp || PIECE_VALUES[rp.role] < moverVal) {
        allGe = false;
        break;
      }
    }
    isSafe = defended && allGe;
  }

  if (!isSafe) {
    return { isMate: false, qualifies: false };
  }

  // Priority 3a: royal fork or double attack against an undefended or higher-value piece
  const recapturerFromSqs = new Set<number>(recaptures.map((r) => r.from));
  const checkerAfter = after.board.get(toSq);
  if (checkerAfter) {
    for (const tSq of attacks(checkerAfter, toSq, after.board.occupied)) {
      const tPiece = after.board.get(tSq);
      if (!tPiece || tPiece.color !== opp) continue;
      if (tPiece.role === "pawn" || tPiece.role === "king") continue;
      if (recapturerFromSqs.has(tSq)) continue;
      const isUndefended = after.kingAttackers(tSq, opp, after.board.occupied).isEmpty();
      const isHigher = PIECE_VALUES[tPiece.role] > moverVal;
      if (isUndefended || isHigher) {
        return { isMate: false, qualifies: true };
      }
    }
  }

  // Priority 3b: safe check strips castling rights by forcing a king move
  if (hasCastlingRights(after, opp) && oppLegal.length > 0) {
    const allKingReplies = oppLegal.every((r) => after.board.get(r.from)?.role === "king");
    if (allKingReplies) {
      return { isMate: false, qualifies: true };
    }
  }

  return { isMate: false, qualifies: false };
}

/**
 * Find Stockfish-supported or forcing checks and high value captures.
 * If Stockfish top lines include a check within 100 centipawns, report that check.
 * If no PV check qualifies, report a check only when it wins an advantage.
 * If a capture of a non-pawn piece is legal, record the top capture.
 */
export function computeChecksAndCapturesFacts(
  pos: Chess,
  engineEval?: EngineEval | null
): string[] {
  const sideName = pos.turn === "white" ? "White" : "Black";
  const facts: string[] = [];
  let checkMoveSan: string | null = null;
  let checkMoveFromTo: [number, number] | null = null;

  const legalMoves = getLegalMoves(pos);
  legalMoves.sort(compareLegalMoves);

  const pvCheck = getQualifyingPvCheck(pos, engineEval);
  if (pvCheck) {
    checkMoveSan = makeSan(pos, pvCheck);
    checkMoveFromTo = [pvCheck.from, normalizeMoveTo(pos, pvCheck)];
    facts.push(`${sideName} has a check: ${checkMoveSan}.`);
  } else {
    const enemyKingSq = pos.board.kingOf(opposite(pos.turn));
    let mateMove: NormalMove | null = null;
    let firstForcing: NormalMove | null = null;

    for (const m of legalMoves) {
      if (!mightGiveCheck(pos, m.move, enemyKingSq)) continue;
      const clone = pos.clone();
      clone.play(m.move);
      if (!clone.isCheck()) continue;
      const res = isForcingCheck(pos, m.move);
      if (res.isMate) {
        mateMove = m.move;
        break;
      }
      if (res.qualifies && !firstForcing) {
        firstForcing = m.move;
      }
    }

    const chosenCheck = mateMove ?? firstForcing;
    if (chosenCheck) {
      checkMoveSan = makeSan(pos, chosenCheck);
      checkMoveFromTo = [chosenCheck.from, normalizeMoveTo(pos, chosenCheck)];
      facts.push(`${sideName} has a check: ${checkMoveSan}.`);
    }
  }

  // Captures of non-pawn pieces
  interface CaptureItem {
    targetVal: number;
    from: number;
    to: number;
    promotionVal: number;
    move: NormalMove;
    role: Role;
  }
  const captures: CaptureItem[] = [];

  for (const m of legalMoves) {
    const target = pos.board.get(m.move.to);
    if (target && target.color !== pos.turn && target.role !== "pawn") {
      const targetVal = PIECE_VALUES[target.role];
      captures.push({
        targetVal,
        from: m.from,
        to: m.to,
        promotionVal: m.promotionVal,
        move: m.move,
        role: target.role,
      });
    }
  }

  captures.sort((a, b) =>
    a.targetVal !== b.targetVal
      ? b.targetVal - a.targetVal
      : a.from !== b.from
      ? a.from - b.from
      : a.to !== b.to
      ? a.to - b.to
      : b.promotionVal - a.promotionVal
  );

  for (const item of captures) {
    if (
      checkMoveFromTo !== null &&
      item.from === checkMoveFromTo[0] &&
      item.to === checkMoveFromTo[1]
    ) {
      continue;
    }
    const san = makeSan(pos, item.move);
    if (san !== checkMoveSan) {
      facts.push(`${sideName} can capture the ${item.role} on ${makeSquare(item.to)} with ${san}.`);
      break;
    }
  }

  return facts;
}

/**
 * Find the most dangerous threat for the specified color.
 * If the position has check, return null.
 * If the setup is invalid, return null.
 * Check for checkmate threats, winning captures, and safe checks.
 */
export function computeThreatFact(pos: Chess, color: Color): string | null {
  if (pos.isCheck()) return null;

  const setup = pos.toSetup();
  setup.turn = color;
  setup.epSquare = undefined;
  const res = Chess.fromSetup(setup);
  if (res.isErr) return null;
  const temp = res.value;

  const colorName = color === "white" ? "White" : "Black";
  const enemyColor = opposite(color);
  const legalMoves = getLegalMoves(temp);
  legalMoves.sort(compareLegalMoves);

  // Category A + C pre-scan: only checking moves can deliver checkmate or a safe check.
  // Because legalMoves is sorted by (from, to, promotionVal), the first mate and first safe
  // check encountered match the exact deterministic tie-breaking order.
  const enemyKingSq = temp.board.kingOf(enemyColor);
  let firstSafeCheck: LegalMoveInfo | null = null;
  for (const m of legalMoves) {
    if (!mightGiveCheck(temp, m.move, enemyKingSq)) continue;
    const afterM = temp.clone();
    afterM.play(m.move);
    if (afterM.isCheck()) {
      if (afterM.isCheckmate()) {
        return `${colorName} threatens ${makeSan(temp, m.move)}.`;
      }
      if (firstSafeCheck === null) {
        const isSafe =
          afterM.kingAttackers(m.move.to, enemyColor, afterM.board.occupied).isEmpty() ||
          afterM.kingAttackers(m.move.to, color, afterM.board.occupied).nonEmpty();
        if (isSafe) {
          firstSafeCheck = m;
        }
      }
    }
  }

  // Category B: Winning capture
  interface WinningCaptureItem {
    targetVal: number;
    attackerVal: number;
    from: number;
    to: number;
    promotionVal: number;
    move: NormalMove;
  }
  const winningCaptures: WinningCaptureItem[] = [];

  for (const m of legalMoves) {
    const target = temp.board.get(m.move.to);
    const attacker = temp.board.get(m.from);
    if (target && target.color !== color && target.role !== "king" && attacker) {
      const targetVal = PIECE_VALUES[target.role];
      const attackerVal = PIECE_VALUES[attacker.role];
      const isSafeCapture =
        targetVal >= 3 &&
        temp.kingAttackers(m.move.to, enemyColor, temp.board.occupied).isEmpty();
      const isHigherValue = targetVal > attackerVal;
      if (isSafeCapture || isHigherValue) {
        winningCaptures.push({
          targetVal,
          attackerVal,
          from: m.from,
          to: m.to,
          promotionVal: m.promotionVal,
          move: m.move,
        });
      }
    }
  }

  if (winningCaptures.length > 0) {
    winningCaptures.sort((a, b) =>
      a.targetVal !== b.targetVal
        ? b.targetVal - a.targetVal
        : a.attackerVal !== b.attackerVal
        ? a.attackerVal - b.attackerVal
        : a.from !== b.from
        ? a.from - b.from
        : a.to !== b.to
        ? a.to - b.to
        : b.promotionVal - a.promotionVal
    );
    return `${colorName} threatens ${makeSan(temp, winningCaptures[0].move)}.`;
  }

  // Category C: Safe check
  if (firstSafeCheck !== null) {
    return `${colorName} threatens ${makeSan(temp, firstSafeCheck.move)}.`;
  }

  return null;
}

/**
 * Find enemy pieces that the moved piece attacks.
 * If another friendly piece supports the attack, mention that piece.
 * When no attacked piece exists, return null.
 */
export function computeMovedPieceAttackFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess
): string | null {
  if (boardAfter.isCheck()) {
    return null;
  }

  const pieceBefore = posBefore.board.get(move.from);
  if (pieceBefore && pieceBefore.role === "king") {
    return null;
  }

  const moverColor = posBefore.turn;
  const toSq = normalizeMoveTo(posBefore, move, moverColor);

  const moveSan = makeSan(posBefore, move);
  const movedPiece = boardAfter.board.get(toSq);
  if (!movedPiece) return null;

  const att = attacks(movedPiece, toSq, boardAfter.board.occupied);
  interface AttackCandidate {
    score: number;
    sq: number;
    text: string;
  }
  const candidates: AttackCandidate[] = [];

  for (const sq of att) {
    const target = boardAfter.board.get(sq);
    if (target && target.color !== moverColor && target.role !== "king") {
      const targetVal = PIECE_VALUES[target.role];
      const otherAttackers = [...boardAfter.kingAttackers(sq, moverColor, boardAfter.board.occupied).without(toSq)];
      const otherNonPawns = otherAttackers.filter((s) => {
        const p = boardAfter.board.get(s);
        return p && p.role !== "pawn" && p.role !== "king";
      });
      otherNonPawns.sort((a, b) => a - b);
      if (otherNonPawns.length > 0) {
        const otherSq = otherNonPawns[0];
        const otherPiece = boardAfter.board.get(otherSq)!;
        candidates.push({
          score: -targetVal,
          sq,
          text: `${moveSan} attacks the ${target.role} on ${makeSquare(sq)} together with the ${otherPiece.role} on ${makeSquare(otherSq)}.`,
        });
      } else if (target.role !== "pawn") {
        candidates.push({
          score: -targetVal,
          sq,
          text: `${moveSan} attacks the ${target.role} on ${makeSquare(sq)}.`,
        });
      }
    }
  }

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => (a.score !== b.score ? a.score - b.score : a.sq - b.sq));
  return candidates[0].text;
}

/**
 * Compute opponent reply and line material changes.
 * Identify the opponent best reply from engine variations.
 * Trace the variation to detect material gain.
 */
export function computeReplyAndMaterialChangeFacts(
  posBefore: Chess,
  move: NormalMove,
  posAfter: Chess,
  evalAfter: EngineEval | null | undefined
): string[] {
  if (!evalAfter || !evalAfter.pvs || evalAfter.pvs.length === 0) {
    return [];
  }
  const pvsMoves = evalAfter.pvs[0].moves;
  if (!pvsMoves || pvsMoves.length === 0) {
    return [];
  }

  const replyUci = pvsMoves[0];
  const replyMove = parseUci(replyUci);
  if (!replyMove || !isNormal(replyMove) || !posAfter.isLegal(replyMove)) {
    return [];
  }

  const oppColor = posAfter.turn;
  const oppName = oppColor === "white" ? "White" : "Black";
  const moveSan = makeSan(posBefore, move);
  const replySan = makeSan(posAfter, replyMove);

  const facts: string[] = [
    `After ${moveSan}, ${oppName}'s best reply is ${replySan}.`,
  ];

  function getNetPts(b: Chess): number {
    let oppPts = 0;
    let moverPts = 0;
    for (let sq = 0; sq < 64; sq++) {
      const p = b.board.get(sq);
      if (!p || p.role === "king") continue;
      if (p.color === oppColor) oppPts += PIECE_VALUES[p.role];
      else moverPts += PIECE_VALUES[p.role];
    }
    return oppPts - moverPts;
  }

  const initialNet = getNetPts(posAfter);

  const sim = posAfter.clone();
  let cap = sim.board.get(replyMove.to);
  let capSqIdx = replyMove.to;
  if (
    !cap &&
    posAfter.epSquare !== undefined &&
    replyMove.to === posAfter.epSquare &&
    posAfter.board.getRole(replyMove.from) === "pawn"
  ) {
    const epDelta = oppColor === "white" ? -8 : 8;
    capSqIdx = replyMove.to + epDelta;
    cap = sim.board.get(capSqIdx);
  }
  sim.play(replyMove);

  for (const uci of pvsMoves.slice(1, 3)) {
    const m = parseUci(uci);
    if (m && sim.isLegal(m)) {
      sim.play(m);
    } else {
      break;
    }
  }

  // Case 1: replyMove captures a piece
  if (cap && cap.color !== oppColor && cap.role !== "king") {
    const capVal = PIECE_VALUES[cap.role];
    const netGain = getNetPts(sim) - initialNet;
    if (netGain >= capVal) {
      const capSq = makeSquare(capSqIdx);
      facts.push(`After ${moveSan}, ${oppName} captures the ${cap.role} on ${capSq} with ${replySan}.`);
      return facts;
    }
  }

  // Case 2: replyMove does not capture a non-pawn piece
  if ((!cap || cap.role === "pawn") && pvsMoves.length >= 3) {
    const sim2 = posAfter.clone();
    sim2.play(replyMove);
    const m1 = parseUci(pvsMoves[1]);
    if (m1 && sim2.isLegal(m1)) {
      sim2.play(m1);
      const m2 = parseUci(pvsMoves[2]);
      if (m2 && sim2.isLegal(m2)) {
        const cap3 = sim2.board.get(m2.to);
        if (cap3 && cap3.color !== oppColor && cap3.role !== "pawn" && cap3.role !== "king") {
          sim2.play(m2);
          if (pvsMoves.length >= 4) {
            const m3 = parseUci(pvsMoves[3]);
            if (m3 && sim2.isLegal(m3)) {
              sim2.play(m3);
            }
          }
          const netGain = getNetPts(sim2) - initialNet;
          if (netGain >= 2) {
            const sqName = makeSquare(m2.to);
            const pBefore = posBefore.board.get(m2.to);
            const pAfter = posAfter.board.get(m2.to);
            const stoodOnSq =
              (pBefore !== undefined && pBefore.role === cap3.role && pBefore.color === cap3.color) ||
              (pAfter !== undefined && pAfter.role === cap3.role && pAfter.color === cap3.color);
            if (stoodOnSq) {
              facts.push(`After ${replySan}, ${oppName} wins the ${cap3.role} on ${sqName}.`);
            } else {
              facts.push(`After ${replySan}, ${oppName} wins the ${cap3.role}.`);
            }
          }
        }
      }
    }
  }

  return facts;
}

/**
 * Check if sqMid lies strictly between sq1 and sq2 along a ray.
 */
export function isBetween(sqMid: number, sq1: number, sq2: number): boolean {
  const r1 = Math.floor(sq1 / 8);
  const c1 = sq1 % 8;
  const rm = Math.floor(sqMid / 8);
  const cm = sqMid % 8;
  const r2 = Math.floor(sq2 / 8);
  const c2 = sq2 % 8;
  const dr = r2 - r1;
  const dc = c2 - c1;

  if (dr === 0 && dc !== 0) {
    return r1 === rm && Math.min(c1, c2) < cm && cm < Math.max(c1, c2);
  }
  if (dc === 0 && dr !== 0) {
    return c1 === cm && Math.min(r1, r2) < rm && rm < Math.max(r1, r2);
  }
  if (Math.abs(dr) === Math.abs(dc) && dr !== 0) {
    const drm = rm - r1;
    const dcm = cm - c1;
    if (Math.abs(drm) !== Math.abs(dcm)) {
      return false;
    }
    if ((dr > 0) !== (drm > 0) || (dc > 0) !== (dcm > 0)) {
      return false;
    }
    return Math.abs(drm) < Math.abs(dr);
  }
  return false;
}

/**
 * Return direction vectors for sliding pieces.
 */
export function getRayDirections(role: Role): Array<[number, number]> {
  const diag: Array<[number, number]> = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const orth: Array<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  if (role === "bishop") return diag;
  if (role === "rook") return orth;
  if (role === "queen") return [...diag, ...orth];
  return [];
}

/**
 * Scan along a direction vector for the first two pieces.
 */
export function scanRay(
  pos: Chess,
  startSq: number,
  dr: number,
  dc: number
): {
  firstSq: number | null;
  firstPiece: { role: Role; color: Color } | null;
  secondSq: number | null;
  secondPiece: { role: Role; color: Color } | null;
} {
  let r = Math.floor(startSq / 8) + dr;
  let c = (startSq % 8) + dc;
  let firstSq: number | null = null;
  let firstPiece: { role: Role; color: Color } | null = null;
  let secondSq: number | null = null;
  let secondPiece: { role: Role; color: Color } | null = null;

  while (r >= 0 && r < 8 && c >= 0 && c < 8) {
    const sq = r * 8 + c;
    const p = pos.board.get(sq);
    if (p) {
      if (firstSq === null) {
        firstSq = sq;
        firstPiece = p;
      } else if (secondSq === null) {
        secondSq = sq;
        secondPiece = p;
        break;
      }
    }
    r += dr;
    c += dc;
  }
  return { firstSq, firstPiece, secondSq, secondPiece };
}

/** Find the top two enemy pieces forked by the piece on attackerSq. */
function checkForkOnBoard(
  pos: Chess,
  attackerSq: number,
  attackerColor: Color
): Array<{ target: { role: Role; color: Color }; sq: number }> | null {
  const attackerPiece = pos.board.get(attackerSq);
  if (!attackerPiece || attackerPiece.role === "king") return null;

  const attackerVal = PIECE_VALUES[attackerPiece.role];
  const oppColor = opposite(attackerColor);
  const att = attacks(attackerPiece, attackerSq, pos.board.occupied);

  const candidates: Array<{ target: { role: Role; color: Color }; sq: number }> = [];
  for (const sq of att) {
    const target = pos.board.get(sq);
    if (!target || target.color !== oppColor) continue;

    if (target.role === "king") {
      candidates.push({ target, sq });
    } else {
      const targetVal = PIECE_VALUES[target.role];
      const isUndefended = pos.kingAttackers(sq, oppColor, pos.board.occupied).isEmpty();
      const isHigherVal = targetVal > attackerVal;
      if (isUndefended || isHigherVal) {
        candidates.push({ target, sq });
      }
    }
  }

  if (candidates.length === 0) return null;

  const hasKing = candidates.some((c) => c.target.role === "king");
  let qualifying = candidates;
  if (!hasKing) {
    qualifying = candidates.filter((c) => PIECE_VALUES[c.target.role] >= 3);
  }

  if (qualifying.length < 2) return null;

  qualifying.sort((a, b) => {
    const rankA = a.target.role === "king" ? 100 : PIECE_VALUES[a.target.role];
    const rankB = b.target.role === "king" ? 100 : PIECE_VALUES[b.target.role];
    return rankB !== rankA ? rankB - rankA : a.sq - b.sq;
  });

  return [qualifying[0], qualifying[1]];
}

/**
 * Detect a fork by the moved piece or the best reply piece.
 * A fork attacks two or more high value or undefended pieces.
 * If the moved piece forks, return the fork fact.
 * If the best reply forks, return the reply fork fact.
 */
export function computeForkFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess,
  evalAfter?: EngineEval | null
): string | null {
  const moverColor = posBefore.turn;
  const toSq = normalizeMoveTo(posBefore, move, moverColor);

  // Check moved piece
  const pair = checkForkOnBoard(boardAfter, toSq, moverColor);
  if (pair) {
    const moveSan = makeSan(posBefore, move);
    const p1Name = pair[0].target.role;
    const sq1Name = makeSquare(pair[0].sq);
    const p2Name = pair[1].target.role;
    const sq2Name = makeSquare(pair[1].sq);
    return `${moveSan} forks the ${p1Name} on ${sq1Name} and the ${p2Name} on ${sq2Name}.`;
  }

  // Check best reply piece
  if (
    evalAfter &&
    evalAfter.pvs &&
    evalAfter.pvs.length > 0 &&
    evalAfter.pvs[0].moves &&
    evalAfter.pvs[0].moves.length > 0
  ) {
    const replyUci = evalAfter.pvs[0].moves[0];
    const replyMove = parseUci(replyUci);
    if (replyMove && isNormal(replyMove) && boardAfter.isLegal(replyMove)) {
      const sim = boardAfter.clone();
      const replySan = makeSan(boardAfter, replyMove);
      const oppColor = opposite(moverColor);
      const replyTo = normalizeMoveTo(boardAfter, replyMove, oppColor);
      sim.play(replyMove);
      const pairReply = checkForkOnBoard(sim, replyTo, oppColor);
      if (pairReply) {
        const p1Name = pairReply[0].target.role;
        const sq1Name = makeSquare(pairReply[0].sq);
        const p2Name = pairReply[1].target.role;
        const sq2Name = makeSquare(pairReply[1].sq);
        return `${replySan} forks the ${p1Name} on ${sq1Name} and the ${p2Name} on ${sq2Name}.`;
      }
    }
  }

  return null;
}

/** Find the highest priority enemy piece pinned along a ray from sliderSq. */
function checkPinOnBoard(
  pos: Chess,
  sliderSq: number,
  sliderColor: Color
): { fSq: number; fP: { role: Role; color: Color }; sSq: number; sP: { role: Role; color: Color } } | null {
  const slider = pos.board.get(sliderSq);
  if (!slider || (slider.role !== "bishop" && slider.role !== "rook" && slider.role !== "queen")) {
    return null;
  }

  const oppColor = opposite(sliderColor);
  const directions = getRayDirections(slider.role);
  interface PinCandidate {
    rank: number;
    fSq: number;
    fP: { role: Role; color: Color };
    sSq: number;
    sP: { role: Role; color: Color };
  }
  const candidates: PinCandidate[] = [];

  for (const [dr, dc] of directions) {
    const { firstSq, firstPiece, secondSq, secondPiece } = scanRay(pos, sliderSq, dr, dc);
    if (firstSq === null || !firstPiece || firstPiece.color !== oppColor) continue;
    if (firstPiece.role === "king") continue;
    if (secondSq === null || !secondPiece || secondPiece.color !== oppColor) continue;

    const fVal = PIECE_VALUES[firstPiece.role];
    const sVal = PIECE_VALUES[secondPiece.role];

    const isAbsolute = secondPiece.role === "king";
    const isRelative = (secondPiece.role === "queen" || secondPiece.role === "rook") && sVal > fVal;

    if (isAbsolute || isRelative) {
      const rank = isAbsolute ? 100 : sVal;
      candidates.push({ rank, fSq: firstSq, fP: firstPiece, sSq: secondSq, sP: secondPiece });
    }
  }

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => (a.rank !== b.rank ? b.rank - a.rank : a.fSq - b.fSq));
  const top = candidates[0];
  return { fSq: top.fSq, fP: top.fP, sSq: top.sSq, sP: top.sP };
}

/**
 * Detect a pin by the moved piece or the best reply piece.
 * A pin holds a piece in front of a more valuable piece.
 * If the moved piece pins, return the pin fact.
 * If the best reply pins, return the reply pin fact.
 */
export function computePinFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess,
  evalAfter?: EngineEval | null
): string | null {
  const moverColor = posBefore.turn;
  const toSq = normalizeMoveTo(posBefore, move, moverColor);

  // Check moved piece
  const res = checkPinOnBoard(boardAfter, toSq, moverColor);
  if (res) {
    const moveSan = makeSan(posBefore, move);
    const pName = res.fP.role;
    const sqName = makeSquare(res.fSq);
    const tgtName = res.sP.role;
    const tgtSq = makeSquare(res.sSq);
    return `${moveSan} pins the ${pName} on ${sqName} to the ${tgtName} on ${tgtSq}.`;
  }

  // Check best reply piece
  if (
    evalAfter &&
    evalAfter.pvs &&
    evalAfter.pvs.length > 0 &&
    evalAfter.pvs[0].moves &&
    evalAfter.pvs[0].moves.length > 0
  ) {
    const replyUci = evalAfter.pvs[0].moves[0];
    const replyMove = parseUci(replyUci);
    if (replyMove && isNormal(replyMove) && boardAfter.isLegal(replyMove)) {
      const sim = boardAfter.clone();
      const replySan = makeSan(boardAfter, replyMove);
      const oppColor = opposite(moverColor);
      const replyTo = normalizeMoveTo(boardAfter, replyMove, oppColor);
      sim.play(replyMove);
      const resReply = checkPinOnBoard(sim, replyTo, oppColor);
      if (resReply) {
        const pName = resReply.fP.role;
        const sqName = makeSquare(resReply.fSq);
        const tgtName = resReply.sP.role;
        const tgtSq = makeSquare(resReply.sSq);
        return `${replySan} pins the ${pName} on ${sqName} to the ${tgtName} on ${tgtSq}.`;
      }
    }
  }

  return null;
}

/** Find the highest priority enemy piece skewered along a ray from sliderSq. */
function checkSkewerOnBoard(
  pos: Chess,
  sliderSq: number,
  sliderColor: Color
): { fSq: number; fP: { role: Role; color: Color }; sSq: number; sP: { role: Role; color: Color } } | null {
  const slider = pos.board.get(sliderSq);
  if (!slider || (slider.role !== "bishop" && slider.role !== "rook" && slider.role !== "queen")) {
    return null;
  }

  const oppColor = opposite(sliderColor);
  const directions = getRayDirections(slider.role);
  interface SkewerCandidate {
    fVal: number;
    sVal: number;
    fSq: number;
    fP: { role: Role; color: Color };
    sSq: number;
    sP: { role: Role; color: Color };
  }
  const candidates: SkewerCandidate[] = [];

  for (const [dr, dc] of directions) {
    const { firstSq, firstPiece, secondSq, secondPiece } = scanRay(pos, sliderSq, dr, dc);
    if (firstSq === null || !firstPiece || firstPiece.color !== oppColor) continue;
    if (firstPiece.role !== "king" && firstPiece.role !== "queen" && firstPiece.role !== "rook") continue;
    if (secondSq === null || !secondPiece || secondPiece.color !== oppColor) continue;
    if (secondPiece.role === "king") continue;

    const fVal = firstPiece.role === "king" ? 100 : PIECE_VALUES[firstPiece.role];
    const sVal = PIECE_VALUES[secondPiece.role];

    if (sVal <= fVal && sVal >= 1) {
      candidates.push({ fVal, sVal, fSq: firstSq, fP: firstPiece, sSq: secondSq, sP: secondPiece });
    }
  }

  if (candidates.length === 0) return null;

  candidates.sort((a, b) =>
    a.fVal !== b.fVal ? b.fVal - a.fVal : a.sVal !== b.sVal ? b.sVal - a.sVal : a.fSq - b.fSq
  );
  const top = candidates[0];
  return { fSq: top.fSq, fP: top.fP, sSq: top.sSq, sP: top.sP };
}

/**
 * Detect a skewer by the moved piece or the best reply piece.
 * A skewer attacks a high value piece with a piece behind it.
 * If the moved piece skewers, return the skewer fact.
 * If the best reply skewers, return the reply skewer fact.
 */
export function computeSkewerFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess,
  evalAfter?: EngineEval | null
): string | null {
  const moverColor = posBefore.turn;
  const toSq = normalizeMoveTo(posBefore, move, moverColor);

  // Check moved piece
  const res = checkSkewerOnBoard(boardAfter, toSq, moverColor);
  if (res) {
    const moveSan = makeSan(posBefore, move);
    const p1Name = res.fP.role;
    const sq1Name = makeSquare(res.fSq);
    const p2Name = res.sP.role;
    const sq2Name = makeSquare(res.sSq);
    return `${moveSan} skewers the ${p1Name} on ${sq1Name} and the ${p2Name} on ${sq2Name}.`;
  }

  // Check best reply piece
  if (
    evalAfter &&
    evalAfter.pvs &&
    evalAfter.pvs.length > 0 &&
    evalAfter.pvs[0].moves &&
    evalAfter.pvs[0].moves.length > 0
  ) {
    const replyUci = evalAfter.pvs[0].moves[0];
    const replyMove = parseUci(replyUci);
    if (replyMove && isNormal(replyMove) && boardAfter.isLegal(replyMove)) {
      const sim = boardAfter.clone();
      const replySan = makeSan(boardAfter, replyMove);
      const oppColor = opposite(moverColor);
      const replyTo = normalizeMoveTo(boardAfter, replyMove, oppColor);
      sim.play(replyMove);
      const resReply = checkSkewerOnBoard(sim, replyTo, oppColor);
      if (resReply) {
        const p1Name = resReply.fP.role;
        const sq1Name = makeSquare(resReply.fSq);
        const p2Name = resReply.sP.role;
        const sq2Name = makeSquare(resReply.sSq);
        return `${replySan} skewers the ${p1Name} on ${sq1Name} and the ${p2Name} on ${sq2Name}.`;
      }
    }
  }

  return null;
}

/**
 * Detect a discovered attack created by moving a piece.
 * Moving a piece unblocks a friendly sliding piece attack.
 * If an attack opens against a king, queen, rook, or undefended minor piece, return the fact.
 */
export function computeDiscoveredAttackFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess
): string | null {
  const moverColor = posBefore.turn;
  const oppColor = opposite(moverColor);
  const fromSq = move.from;
  const toSq = normalizeMoveTo(posBefore, move, moverColor);

  interface DiscCandidate {
    targetVal: number;
    sq1: number;
    slider: { role: Role; color: Color };
    sq2: number;
    target: { role: Role; color: Color };
  }
  const candidates: DiscCandidate[] = [];

  for (let sq1 = 0; sq1 < 64; sq1++) {
    const slider = boardAfter.board.get(sq1);
    if (!slider || slider.color !== moverColor) continue;
    if (slider.role !== "bishop" && slider.role !== "rook" && slider.role !== "queen") continue;
    if (sq1 === toSq) continue;

    const att = attacks(slider, sq1, boardAfter.board.occupied);
    for (const sq2 of att) {
      const target = boardAfter.board.get(sq2);
      if (!target || target.color !== oppColor) continue;

      let targetVal = 0;
      if (target.role === "king") {
        targetVal = 100;
      } else if (target.role === "queen" || target.role === "rook") {
        targetVal = PIECE_VALUES[target.role];
      } else if (target.role === "bishop" || target.role === "knight") {
        if (boardAfter.kingAttackers(sq2, oppColor, boardAfter.board.occupied).isEmpty()) {
          targetVal = PIECE_VALUES[target.role];
        } else {
          continue;
        }
      } else {
        continue;
      }

      if (isBetween(fromSq, sq1, sq2)) {
        const attBefore = attacks(slider, sq1, posBefore.board.occupied);
        if (!attBefore.has(sq2)) {
          candidates.push({ targetVal, sq1, slider, sq2, target });
        }
      }
    }
  }

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => (a.targetVal !== b.targetVal ? b.targetVal - a.targetVal : a.sq1 - b.sq1));
  const top = candidates[0];
  const moveSan = makeSan(posBefore, move);
  const sliderName = top.slider.role;
  const sq1Name = makeSquare(top.sq1);
  const targetName = top.target.role;
  const sq2Name = makeSquare(top.sq2);
  return `${moveSan} opens a discovered attack from the ${sliderName} on ${sq1Name} against the ${targetName} on ${sq2Name}.`;
}

/**
 * Detect a forced checkmate or checkmate threat from engine evaluations.
 * If the mover has forced mate, report the moves to mate.
 * If the opponent has forced mate, report the checkmate threat.
 */
export function computeMateThreatFact(
  posBefore: Chess,
  _move: NormalMove,
  boardAfter: Chess,
  evalAfter?: EngineEval | null
): string | null {
  if (!evalAfter || !evalAfter.pvs || evalAfter.pvs.length === 0) return null;
  const pv = evalAfter.pvs[0];
  if (!pv.score || pv.score.type !== "mate") return null;
  const val = Math.trunc(Number(pv.score.value ?? 0));
  if (val === 0) return null;
  if (!pv.moves || pv.moves.length === 0) return null;

  const replyMove = parseUci(pv.moves[0]);
  if (!replyMove || !isNormal(replyMove) || !boardAfter.isLegal(replyMove)) return null;

  const replySan = makeSan(boardAfter, replyMove);
  const moverColor = posBefore.turn;
  const moverName = moverColor === "white" ? "White" : "Black";
  const oppName = moverColor === "white" ? "Black" : "White";
  const moverHasMate = moverColor === "white" ? val > 0 : val < 0;
  const n = Math.abs(val);
  const movesStr = n === 1 ? "move" : "moves";

  if (moverHasMate) {
    return `${moverName} has a forced checkmate in ${n} ${movesStr} starting with ${replySan}.`;
  }
  return `${oppName} threatens checkmate with ${replySan}.`;
}

/**
 * Detect whether an opponent recapture centralizes or activates a piece.
 * Check whether the played move is a capture and the top reply recaptures there.
 * When that recapture enters the center or gains activity, return a fact.
 */
export function computeRecaptureActivityFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess,
  evalAfter?: EngineEval | null
): string | null {
  if (!isCaptureMove(posBefore, move)) return null;
  if (!evalAfter || !evalAfter.pvs || evalAfter.pvs.length === 0) return null;
  const pv0 = evalAfter.pvs[0];
  if (!pv0.moves || pv0.moves.length === 0) return null;

  const replyMove = parseUci(pv0.moves[0]);
  if (!replyMove || !isNormal(replyMove) || !boardAfter.isLegal(replyMove)) return null;
  if (replyMove.to !== move.to || !isCaptureMove(boardAfter, replyMove)) return null;

  const recapPiece = boardAfter.board.get(replyMove.from);
  if (
    !recapPiece ||
    (recapPiece.role !== "knight" &&
      recapPiece.role !== "bishop" &&
      recapPiece.role !== "rook" &&
      recapPiece.role !== "queen")
  ) {
    return null;
  }

  const moveSan = makeSan(posBefore, move);
  const replySan = makeSan(boardAfter, replyMove);
  const oppColor = boardAfter.turn;
  const oppName = oppColor === "white" ? "White" : "Black";
  const pieceName = recapPiece.role;

  if (CENTRAL_SQUARES.has(replyMove.to) && !CENTRAL_SQUARES.has(replyMove.from)) {
    return `After ${moveSan}, ${oppName}'s recapture ${replySan} centralizes the ${pieceName}.`;
  }

  const mobilityBefore = pieceMobility(boardAfter, replyMove.from);
  const simReply = boardAfter.clone();
  simReply.play(replyMove);
  const mobilityAfter = pieceMobility(simReply, replyMove.to);

  const backRank = oppColor === "white" ? 0 : 7;
  const developedFromBack =
    squareRank(replyMove.from) === backRank && isExtendedCenter(replyMove.to);
  if (developedFromBack || mobilityAfter >= mobilityBefore + 2) {
    return `After ${moveSan}, ${oppName}'s recapture ${replySan} increases ${pieceName} activity.`;
  }

  return null;
}

/**
 * Explain what the best move accomplishes when the played move is bad.
 * When the verdict is Inaccuracy, Mistake, or Blunder, check the best move.
 * Report tactical gains, central pawn space or tension, king safety, or activity.
 */
export function computeRefutationReasonFact(
  posBefore: Chess,
  move: NormalMove,
  _boardAfter: Chess,
  evalBefore: EngineEval,
  evalAfter?: EngineEval | null,
  moveUci?: string | null
): string | null {
  const actualMoveUci =
    moveUci ??
    `${makeSquare(move.from)}${makeSquare(move.to)}${move.promotion ? roleToChar(move.promotion) : ""}`;
  const verdict = classifyVerdict(evalBefore, evalAfter, actualMoveUci, posBefore);
  if (verdict !== "Inaccuracy" && verdict !== "Mistake" && verdict !== "Blunder") {
    return null;
  }

  const pvs = evalBefore.pvs ?? [];
  if (pvs.length === 0 || !pvs[0].moves || pvs[0].moves.length === 0) return null;
  const bestUci = pvs[0].moves[0];
  if (bestUci === actualMoveUci) return null;

  const bestMove = parseUci(bestUci);
  if (!bestMove || !isNormal(bestMove) || !posBefore.isLegal(bestMove)) return null;

  const bestSan = makeSan(posBefore, bestMove);
  if (bestSan === makeSan(posBefore, move)) return null;
  const moverColor = posBefore.turn;
  const oppColor = opposite(moverColor);
  const bestTo = normalizeMoveTo(posBefore, bestMove, moverColor);
  const pieceBefore = posBefore.board.get(bestMove.from);

  // 1. Best move captures a piece
  const targetBefore = posBefore.board.get(bestTo);
  const isEnPassant =
    pieceBefore?.role === "pawn" &&
    posBefore.epSquare !== undefined &&
    bestTo === posBefore.epSquare &&
    !targetBefore;
  if (targetBefore && targetBefore.color === oppColor) {
    const targetName = targetBefore.role;
    const sqName = makeSquare(bestTo);
    return `Best move ${bestSan} captures the ${targetName} on ${sqName}.`;
  }
  if (isEnPassant) {
    const capSq = bestTo + (moverColor === "white" ? -8 : 8);
    const sqName = makeSquare(capSq);
    return `Best move ${bestSan} captures the pawn on ${sqName}.`;
  }

  // 2. Best move defends a hanging piece
  const sim = posBefore.clone();
  sim.play(bestMove);
  const bestPieceAfter = sim.board.get(bestTo);
  if (bestPieceAfter) {
    const bestAttacks = attacks(bestPieceAfter, bestTo, sim.board.occupied);
    interface DefCandidate {
      val: number;
      sq: number;
      role: Role;
    }
    const defCandidates: DefCandidate[] = [];

    for (const sq of bestAttacks) {
      if (sq !== bestTo) {
        const p = sim.board.get(sq);
        if (p && p.color === moverColor && p.role !== "king") {
          const wasAttacked = posBefore.kingAttackers(sq, oppColor, posBefore.board.occupied).nonEmpty();
          const wasDefended = posBefore.kingAttackers(sq, moverColor, posBefore.board.occupied).nonEmpty();
          if (wasAttacked && !wasDefended) {
            defCandidates.push({ val: PIECE_VALUES[p.role], sq, role: p.role });
          }
        }
      }
    }

    if (defCandidates.length > 0) {
      defCandidates.sort((a, b) => (a.val !== b.val ? b.val - a.val : a.sq - b.sq));
      const top = defCandidates[0];
      const pieceName = top.role;
      const sqName = makeSquare(top.sq);
      return `Best move ${bestSan} defends the ${pieceName} on ${sqName}.`;
    }
  }

  // 3. Best move develops a piece and prepares castling
  if (hasCastlingRights(posBefore, moverColor) && posBefore.fullmoves <= 12) {
    if (pieceBefore && (pieceBefore.role === "knight" || pieceBefore.role === "bishop")) {
      const startSqs = moverColor === "white" ? WHITE_START_MINORS : BLACK_START_MINORS;
      if (startSqs.has(bestMove.from)) {
        const pieceName = pieceBefore.role;
        return `Best move ${bestSan} develops the ${pieceName} and prepares castling.`;
      }
    }
  }

  // 4. Central pawn advance: space, tension, or pawn attack on a piece
  if (pieceBefore && pieceBefore.role === "pawn") {
    const playedIsPawnCapture = isPawnTakesPawn(posBefore, move);
    const playedTo = move.to;
    const playedToSq = makeSquare(playedTo);

    // 4a. Central pawn advance attacking an enemy non-pawn, non-king piece
    if (isExtendedCenter(bestTo) && bestPieceAfter) {
      interface AttackedTarget {
        val: number;
        sq: number;
        role: Role;
      }
      const attackedTargets: AttackedTarget[] = [];
      for (const tSq of attacks(bestPieceAfter, bestTo, sim.board.occupied)) {
        const target = sim.board.get(tSq);
        if (
          target &&
          target.color === oppColor &&
          target.role !== "pawn" &&
          target.role !== "king" &&
          posBefore.board.get(tSq) !== undefined
        ) {
          attackedTargets.push({ val: PIECE_VALUES[target.role], sq: tSq, role: target.role });
        }
      }
      if (attackedTargets.length > 0) {
        attackedTargets.sort((a, b) => (a.val !== b.val ? b.val - a.val : a.sq - b.sq));
        const top = attackedTargets[0];
        const tSqName = makeSquare(top.sq);
        const targetName = top.role;
        if (isCenter45(bestTo)) {
          if (playedIsPawnCapture && CENTRAL_SQUARES.has(playedTo)) {
            return `Best move ${bestSan} gains space and attacks the ${targetName} on ${tSqName} instead of trading in the center.`;
          }
          return `Best move ${bestSan} gains central space and attacks the ${targetName} on ${tSqName}.`;
        }
        return `Best move ${bestSan} attacks the ${targetName} on ${tSqName}.`;
      }
    }

    // 4b. Space-gaining central or 4th/5th-rank pawn push
    if (isCenter45(bestTo)) {
      if (playedIsPawnCapture && isExtendedCenter(playedTo)) {
        return `Best move ${bestSan} keeps central space and tension instead of trading on ${playedToSq}.`;
      }
      if (CENTRAL_SQUARES.has(bestTo)) {
        return `Best move ${bestSan} claims central space on ${makeSquare(bestTo)}.`;
      }
      return `Best move ${bestSan} gains space and challenges the center.`;
    }

    // 4c. Solid tension-holding pawn move
    if (isExtendedCenter(bestTo) && playedIsPawnCapture && CENTRAL_SQUARES.has(playedTo)) {
      return `Best move ${bestSan} maintains central tension instead of trading on ${playedToSq}.`;
    }
  }

  // 5. Best move castles to keep the king safe
  if (isCastlingMove(posBefore, bestMove)) {
    return `Best move ${bestSan} castles to keep the king safe.`;
  }

  // 6. Best move saves an attacked piece
  if (
    pieceBefore &&
    (pieceBefore.role === "knight" ||
      pieceBefore.role === "bishop" ||
      pieceBefore.role === "rook" ||
      pieceBefore.role === "queen")
  ) {
    const pieceName = pieceBefore.role;
    const wasAttacked = posBefore.kingAttackers(bestMove.from, oppColor, posBefore.board.occupied).nonEmpty();
    const stillAttacked = sim.kingAttackers(bestTo, oppColor, sim.board.occupied).nonEmpty();
    if (wasAttacked && !stillAttacked) {
      return `Best move ${bestSan} moves the attacked ${pieceName} to safety.`;
    }

    // 7. Piece centralization, open file, or activity gain
    if (CENTRAL_SQUARES.has(bestTo) && !CENTRAL_SQUARES.has(bestMove.from)) {
      return `Best move ${bestSan} centralizes the ${pieceName}.`;
    }

    if (pieceBefore.role === "rook") {
      const rookFileKind = classifyRookFileMove(posBefore, sim, bestMove.from, bestTo, moverColor);
      if (rookFileKind === "open") {
        return `Best move ${bestSan} places the rook on an open file.`;
      }
      if (rookFileKind === "half-open") {
        return `Best move ${bestSan} places the rook on a half-open file.`;
      }
    }

    const mobilityBefore = pieceMobility(posBefore, bestMove.from);
    const mobilityAfter = pieceMobility(sim, bestTo);
    if (mobilityAfter >= mobilityBefore + 2) {
      return `Best move ${bestSan} increases ${pieceName} activity.`;
    }
  }

  return null;
}

/** Compute material point difference from moverColor perspective. */
function moverNetMaterial(pos: Chess, moverColor: Color): number {
  let net = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = pos.board.get(sq);
    if (!p || p.role === "king") continue;
    const val = PIECE_VALUES[p.role];
    net += p.color === moverColor ? val : -val;
  }
  return net;
}

/** Count opponent non-pawn, non-king pieces on the four center squares. */
function countOppCentralPieces(pos: Chess, oppColor: Color): number {
  let count = 0;
  for (const sq of CENTRAL_SQUARES) {
    const p = pos.board.get(sq);
    if (p && p.color === oppColor && p.role !== "pawn" && p.role !== "king") {
      count++;
    }
  }
  return count;
}

/** Count mover pawns on the center squares and fifth-rank extended center. */
function countMoverCentralPawns(pos: Chess, moverColor: Color): number {
  const rank5 = moverColor === "white" ? 4 : 3;
  let count = 0;
  for (const sq of pos.board.pawn.intersect(pos.board[moverColor])) {
    const f = squareFile(sq);
    const r = squareRank(sq);
    if (CENTRAL_SQUARES.has(sq) || (r === rank5 && f >= 2 && f <= 5)) {
      count++;
    }
  }
  return count;
}

/** Compute net non-pawn, non-king mobility difference for moverColor. */
function netPieceMobility(pos: Chess, moverColor: Color): number {
  const oppColor = opposite(moverColor);
  let moverMob = 0;
  let oppMob = 0;
  const pieceSet = pos.board.knight.union(pos.board.bishop).union(pos.board.rook).union(pos.board.queen);
  for (const sq of pieceSet.intersect(pos.board[moverColor])) {
    moverMob += pieceMobility(pos, sq);
  }
  for (const sq of pieceSet.intersect(pos.board[oppColor])) {
    oppMob += pieceMobility(pos, sq);
  }
  return moverMob - oppMob;
}

/** Settle one recapture ply when the horizon ends in a capture. */
function settledMaterial(
  statesMeta: Array<{ pos: Chess; wasCap: boolean; toSq: number }>,
  pvMoves: string[],
  commonPlies: number,
  moverColor: Color,
  nextPvIdx: number
): number {
  const { pos: eqBoard, wasCap, toSq: capToSq } = statesMeta[commonPlies - 1];
  if (wasCap && nextPvIdx >= 0 && nextPvIdx < pvMoves.length) {
    const nextMove = parseUci(pvMoves[nextPvIdx]);
    if (
      nextMove &&
      isNormal(nextMove) &&
      !isKingTakesFriendlyRook(eqBoard, nextMove) &&
      eqBoard.isLegal(nextMove) &&
      isCaptureMove(eqBoard, nextMove) &&
      normalizeMoveTo(eqBoard, nextMove) === capToSq
    ) {
      const settled = eqBoard.clone();
      settled.play(nextMove);
      return moverNetMaterial(settled, moverColor);
    }
  }
  return moverNetMaterial(eqBoard, moverColor);
}

/**
 * Compare key positional or material differences between played and best lines.
 * Simulate up to four plies from posBefore for both lines on bad moves.
 * Report the highest priority concrete difference.
 */
export function computeLineComparisonFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess,
  evalBefore: EngineEval,
  evalAfter?: EngineEval | null,
  moveUci?: string | null
): string | null {
  const actualMoveUci =
    moveUci ??
    `${makeSquare(move.from)}${makeSquare(move.to)}${move.promotion ? roleToChar(move.promotion) : ""}`;
  const verdict = classifyVerdict(evalBefore, evalAfter, actualMoveUci, posBefore);
  if (verdict !== "Inaccuracy" && verdict !== "Mistake" && verdict !== "Blunder") {
    return null;
  }
  if (!evalAfter) return null;

  const pvsBefore = evalBefore.pvs ?? [];
  const pvsAfter = evalAfter.pvs ?? [];
  if (pvsBefore.length === 0 || !pvsBefore[0].moves || pvsBefore[0].moves.length === 0) return null;
  if (pvsAfter.length === 0 || !pvsAfter[0].moves || pvsAfter[0].moves.length === 0) return null;

  const bestUci = pvsBefore[0].moves[0];
  if (bestUci === actualMoveUci) return null;

  const bestMove = parseUci(bestUci);
  if (!bestMove || !isNormal(bestMove) || !posBefore.isLegal(bestMove)) return null;

  const moveSan = makeSan(posBefore, move);
  const bestSan = makeSan(posBefore, bestMove);
  if (bestSan === moveSan) return null;

  const moverColor = posBefore.turn;
  const oppColor = opposite(moverColor);
  const moverName = moverColor === "white" ? "White" : "Black";
  const oppName = moverColor === "white" ? "Black" : "White";

  const playedPvMoves = pvsAfter[0].moves;
  const playedSim = boardAfter.clone();
  let moverCastledPlayed = isCastlingMove(posBefore, move);
  const playedStatesMeta: Array<{ pos: Chess; wasCap: boolean; toSq: number }> = [
    {
      pos: playedSim.clone(),
      wasCap: isCaptureMove(posBefore, move),
      toSq: normalizeMoveTo(posBefore, move, moverColor),
    },
  ];
  const playedCastledAtPly: boolean[] = [moverCastledPlayed];
  for (const uci of playedPvMoves.slice(0, 3)) {
    const m = parseUci(uci);
    if (!m || !isNormal(m) || !playedSim.isLegal(m)) break;
    if (playedSim.turn === moverColor && isCastlingMove(playedSim, m)) {
      moverCastledPlayed = true;
    }
    const wasCap = isCaptureMove(playedSim, m);
    const toSq = normalizeMoveTo(playedSim, m, playedSim.turn);
    playedSim.play(m);
    playedStatesMeta.push({ pos: playedSim.clone(), wasCap, toSq });
    playedCastledAtPly.push(moverCastledPlayed);
  }

  const bestPvMoves = pvsBefore[0].moves;
  const bestSim = posBefore.clone();
  let moverCastledBest = false;
  const bestStatesMeta: Array<{ pos: Chess; wasCap: boolean; toSq: number }> = [];
  const bestCastledAtPly: boolean[] = [];
  for (const uci of bestPvMoves.slice(0, 4)) {
    const m = parseUci(uci);
    if (!m || !isNormal(m) || !bestSim.isLegal(m)) break;
    if (bestSim.turn === moverColor && isCastlingMove(bestSim, m)) {
      moverCastledBest = true;
    }
    const wasCap = isCaptureMove(bestSim, m);
    const toSq = normalizeMoveTo(bestSim, m, bestSim.turn);
    bestSim.play(m);
    bestStatesMeta.push({ pos: bestSim.clone(), wasCap, toSq });
    bestCastledAtPly.push(moverCastledBest);
  }

  if (bestStatesMeta.length === 0) return null;

  // 0. Checkmate delivered or avoided
  const bestFinal = bestStatesMeta[bestStatesMeta.length - 1].pos;
  const playedFinal = playedStatesMeta[playedStatesMeta.length - 1].pos;
  const bestMateByMover = bestFinal.isCheckmate() && bestFinal.turn === oppColor;
  const playedMateByMover = playedFinal.isCheckmate() && playedFinal.turn === oppColor;
  const playedMateByOpp = playedFinal.isCheckmate() && playedFinal.turn === moverColor;
  const bestMateByOpp = bestFinal.isCheckmate() && bestFinal.turn === moverColor;

  if (bestMateByMover && !playedMateByMover) {
    return `Unlike ${moveSan}, the best line with ${bestSan} leads to checkmate.`;
  }
  if (playedMateByOpp && !bestMateByOpp) {
    return `Unlike ${moveSan}, ${bestSan} avoids checkmate.`;
  }

  // Equalize ply count across simulated lines before comparing state metrics
  let commonPlies = Math.min(playedStatesMeta.length, bestStatesMeta.length);
  if (commonPlies % 2 === 1) {
    commonPlies -= 1;
  }
  if (commonPlies < 2) return null;

  const playedEq = playedStatesMeta[commonPlies - 1].pos;
  const bestEq = bestStatesMeta[commonPlies - 1].pos;
  moverCastledPlayed = playedCastledAtPly[commonPlies - 1];
  moverCastledBest = bestCastledAtPly[commonPlies - 1];

  // 1. Material difference (settle 1 immediate recapture ply when horizon ends in a capture)
  const matBefore = moverNetMaterial(posBefore, moverColor);
  const matPlayed = settledMaterial(
    playedStatesMeta,
    playedPvMoves,
    commonPlies,
    moverColor,
    commonPlies - 1
  );
  const matBest = settledMaterial(
    bestStatesMeta,
    bestPvMoves,
    commonPlies,
    moverColor,
    commonPlies
  );
  const matDiff = matBest - matPlayed;
  if (matDiff >= 1) {
    if (matDiff >= 2) {
      if (matPlayed < matBefore) {
        return `Unlike ${moveSan}, the best line with ${bestSan} avoids losing material.`;
      }
      return `Unlike ${moveSan}, the best line with ${bestSan} wins material.`;
    }
    if (matPlayed < matBefore) {
      return `Unlike ${moveSan}, the best line with ${bestSan} avoids losing a pawn.`;
    }
    return `Unlike ${moveSan}, the best line with ${bestSan} wins a pawn.`;
  }

  // 2. Opponent centralization
  if (countOppCentralPieces(playedEq, oppColor) > countOppCentralPieces(bestEq, oppColor)) {
    return `Unlike ${moveSan}, ${bestSan} prevents ${oppName} from centralizing pieces.`;
  }

  // 3. Mover central pawn control
  if (countMoverCentralPawns(bestEq, moverColor) > countMoverCentralPawns(playedEq, moverColor)) {
    return `Unlike ${moveSan}, ${bestSan} keeps stronger central pawn control.`;
  }

  // 4. King safety
  const playedInCheck = playedEq.turn === moverColor && playedEq.isCheck();
  const bestInCheck = bestEq.turn === moverColor && bestEq.isCheck();
  if ((moverCastledBest && !moverCastledPlayed) || (playedInCheck && !bestInCheck)) {
    return `Unlike ${moveSan}, ${bestSan} gives ${moverName} better king safety.`;
  }

  // 5. Piece activity
  if (netPieceMobility(bestEq, moverColor) - netPieceMobility(playedEq, moverColor) >= 4) {
    return `Unlike ${moveSan}, ${bestSan} leaves ${moverName} with more active pieces.`;
  }

  return null;
}

/**
 * Compute a positional fact describing what a quiet or strategic move does.
 */
export function computeMovePositionalFact(
  posBefore: Chess,
  move: NormalMove,
  posAfter: Chess
): string | null {
  const moveSan = makeSan(posBefore, move);
  const moverColor = posBefore.turn;
  const piece = posBefore.board.get(move.from);
  if (!piece) return null;

  if (isCastlingMove(posBefore, move)) {
    return `${moveSan} castles to safeguard the king.`;
  }

  const toSq = normalizeMoveTo(posBefore, move, moverColor);
  const toSqName = makeSquare(toSq);

  if (piece.role === "pawn") {
    const isCap = isCaptureMove(posBefore, move);
    if (isPawnTakesPawn(posBefore, move) && CENTRAL_SQUARES.has(toSq)) {
      return `${moveSan} releases central tension on ${toSqName}.`;
    }
    if (!isCap && CENTRAL_SQUARES.has(toSq)) {
      return `${moveSan} claims central space on ${toSqName}.`;
    }
    if (!isCap && WING_CENTER_45_SQUARES.has(toSq)) {
      return `${moveSan} gains space and challenges the center.`;
    }
    return null;
  }

  const pieceName = piece.role;
  if (piece.role === "knight" || piece.role === "bishop") {
    const startSqs = moverColor === "white" ? WHITE_START_MINORS : BLACK_START_MINORS;
    if (startSqs.has(move.from) && isExtendedCenter(toSq)) {
      return `${moveSan} develops the ${pieceName} toward the center.`;
    }
  }

  if (
    piece.role === "knight" ||
    piece.role === "bishop" ||
    piece.role === "rook" ||
    piece.role === "queen"
  ) {
    if (CENTRAL_SQUARES.has(toSq) && !CENTRAL_SQUARES.has(move.from)) {
      return `${moveSan} centralizes the ${pieceName}.`;
    }
  }

  if (piece.role === "rook") {
    const rookFileKind = classifyRookFileMove(posBefore, posAfter, move.from, toSq, moverColor);
    if (rookFileKind !== null) {
      const fileLetter = FILE_LETTERS[squareFile(toSq)];
      if (rookFileKind === "open") {
        return `${moveSan} places the rook on the open ${fileLetter}-file.`;
      }
      return `${moveSan} places the rook on the half-open ${fileLetter}-file.`;
    }
  }

  return null;
}

/**
 * Compute one structural or positional fact about the current board.
 */
export function computePositionStructureFact(pos: Chess): string | null {
  // 1. Central pawn tension
  const whiteC4F4 = [26, 27, 28, 29]; // c4, d4, e4, f4
  const blackC5F5 = [34, 35, 36, 37]; // c5, d5, e5, f5
  for (const wSq of whiteC4F4) {
    const wp = pos.board.get(wSq);
    if (wp && wp.color === "white" && wp.role === "pawn") {
      const wFile = squareFile(wSq);
      for (const bSq of blackC5F5) {
        const bp = pos.board.get(bSq);
        if (bp && bp.color === "black" && bp.role === "pawn") {
          if (Math.abs(wFile - squareFile(bSq)) === 1) {
            return `Central pawn tension exists between ${makeSquare(wSq)} and ${makeSquare(bSq)}.`;
          }
        }
      }
    }
  }

  // 2. Central space advantage
  let wSpace = 0;
  for (const sq of pos.board.pawn.intersect(pos.board.white)) {
    const f = squareFile(sq);
    const r = squareRank(sq);
    if (f >= 2 && f <= 5 && r >= 3 && r <= 5) wSpace++;
  }
  let bSpace = 0;
  for (const sq of pos.board.pawn.intersect(pos.board.black)) {
    const f = squareFile(sq);
    const r = squareRank(sq);
    if (f >= 2 && f <= 5 && r >= 2 && r <= 4) bSpace++;
  }
  if (Math.abs(wSpace - bSpace) >= 2) {
    const leader = wSpace > bSpace ? "White" : "Black";
    return `${leader} has more central space.`;
  }

  // 3. Open central file (d-file, then e-file)
  for (const fileIdx of [3, 4]) {
    if (countPawnsOnFile(pos, fileIdx) === 0) {
      return `The ${FILE_LETTERS[fileIdx]}-file is open.`;
    }
  }

  // 4. Opening development lead
  if (pos.fullmoves <= 12) {
    const minors = pos.board.knight.union(pos.board.bishop);
    let wDev = 0;
    for (const sq of minors.intersect(pos.board.white)) {
      if (!WHITE_START_MINORS.has(sq)) wDev++;
    }
    let bDev = 0;
    for (const sq of minors.intersect(pos.board.black)) {
      if (!BLACK_START_MINORS.has(sq)) bDev++;
    }
    if (Math.abs(wDev - bDev) >= 2) {
      const leader = wDev > bDev ? "White" : "Black";
      return `${leader} leads in piece development.`;
    }
  }

  return null;
}

/**
 * Append shared position facts to rawFacts.
 * Extract hanging pieces, checks, captures, threats, and king safety.
 * Extract position structure, material balance, phase, and opening facts.
 */
export function appendPositionFacts(
  board: Chess,
  fen: string,
  moveUci: string | null | undefined,
  rawFacts: Array<string | null>,
  parsedPosBefore?: Chess,
  engineEval?: EngineEval | null
): void {
  let posBefore = parsedPosBefore;
  if (!posBefore) {
    const parseRes = parseFen(fen);
    if (parseRes.isErr) return;
    const chessRes = Chess.fromSetup(parseRes.value);
    if (chessRes.isErr) return;
    posBefore = chessRes.value;
  }

  for (const f of computeHangingFacts(board)) {
    rawFacts.push(f);
  }

  for (const f of computeChecksAndCapturesFacts(board, engineEval)) {
    rawFacts.push(f);
  }

  rawFacts.push(computeThreatFact(board, posBefore.turn));

  for (const f of computeKingSafetyFacts(board)) {
    rawFacts.push(f);
  }

  rawFacts.push(computePositionStructureFact(board));
  rawFacts.push(computeMaterialFact(board));
  rawFacts.push(computePhaseFact(posBefore));

  const opening = lookupOpening(fen, moveUci);
  if (opening) {
    rawFacts.push(`Opening: ${opening}.`);
  }
}

/**
 * Extract ordered facts for a position or move task.
 * If the task is explain_move, extract move-specific facts.
 * If the task is explain_position, extract candidate moves and position facts.
 * Remove duplicate facts while preserving order.
 */
export function extractFacts(
  task: string,
  fen: string,
  evalBefore: EngineEval,
  evalAfter?: EngineEval | null,
  moveUci?: string | null,
  parsedPos?: Chess
): string[] {
  let posBefore = parsedPos;
  if (!posBefore) {
    const parseRes = parseFen(fen);
    if (parseRes.isErr) {
      throw new Error(`Invalid FEN: ${fen}`);
    }
    const chessRes = Chess.fromSetup(parseRes.value);
    if (chessRes.isErr) {
      throw new Error(`Invalid setup for FEN: ${fen}`);
    }
    posBefore = chessRes.value;
  }
  const rawFacts: Array<string | null> = [];

  if (task === "explain_move") {
    if (!moveUci) {
      throw new Error("move_uci is required for task explain_move");
    }
    const move = parseUci(moveUci);
    if (!move || !isNormal(move) || isKingTakesFriendlyRook(posBefore, move) || !posBefore.isLegal(move)) {
      throw new Error(`Invalid or illegal move_uci: ${moveUci}`);
    }
    const posAfter = posBefore.clone();
    posAfter.play(move);

    // 1. Moved-piece attack fact
    rawFacts.push(computeMovedPieceAttackFact(posBefore, move, posAfter));

    // 2 & 3. Reply and material change in line
    for (const f of computeReplyAndMaterialChangeFacts(posBefore, move, posAfter, evalAfter)) {
      rawFacts.push(f);
    }

    // 4. Opponent recapture centralization or activity gain
    rawFacts.push(computeRecaptureActivityFact(posBefore, move, posAfter, evalAfter));

    // 5. Tactical motifs
    rawFacts.push(computeForkFact(posBefore, move, posAfter, evalAfter));
    rawFacts.push(computePinFact(posBefore, move, posAfter, evalAfter));
    rawFacts.push(computeSkewerFact(posBefore, move, posAfter, evalAfter));
    rawFacts.push(computeDiscoveredAttackFact(posBefore, move, posAfter));
    rawFacts.push(computeMateThreatFact(posBefore, move, posAfter, evalAfter));

    // 6. Refutation reason and line comparison
    rawFacts.push(computeRefutationReasonFact(posBefore, move, posAfter, evalBefore, evalAfter, moveUci));
    rawFacts.push(computeLineComparisonFact(posBefore, move, posAfter, evalBefore, evalAfter, moveUci));

    // 7. Move positional fact
    rawFacts.push(computeMovePositionalFact(posBefore, move, posAfter));

    // 8. Shared position facts
    appendPositionFacts(posAfter, fen, moveUci, rawFacts, posBefore, evalAfter);
  } else if (task === "explain_position") {
    // 1. Candidate moves fact
    const pvs = evalBefore.pvs ?? [];
    if (pvs.length >= 2) {
      const candidateSans: string[] = [];
      for (const pv of pvs.slice(0, 3)) {
        const moves = pv.moves ?? [];
        if (moves.length > 0) {
          const m = parseUci(moves[0]);
          if (m && posBefore.isLegal(m)) {
            const san = makeSan(posBefore, m);
            if (!candidateSans.includes(san)) {
              candidateSans.push(san);
            }
          }
        }
      }
      if (candidateSans.length >= 2) {
        rawFacts.push(`Top candidate moves: ${candidateSans.join(", ")}.`);
      }
    }

    // 2-9. Shared position facts
    appendPositionFacts(posBefore, fen, null, rawFacts, posBefore, evalBefore);
  } else {
    throw new Error(`Unknown task: ${task}`);
  }

  const uniqueFacts: string[] = [];
  const seen = new Set<string>();
  for (const f of rawFacts) {
    if (f && !seen.has(f)) {
      seen.add(f);
      uniqueFacts.push(f);
    }
  }

  return uniqueFacts;
}
