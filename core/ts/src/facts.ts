/**
 * Fact extractor for chess positions.
 * This module computes deterministic facts from positions and engine evaluations.
 * It exports functions to evaluate winning chances, classify moves, and extract facts.
 */

import { Chess } from "chessops";
import { attacks } from "chessops/attacks";
import { parseFen } from "chessops/fen";
import { makeSan, makeSanAndPlay } from "chessops/san";
import { Color, isNormal, NormalMove, Role } from "chessops/types";
import { kingCastlesTo, makeSquare, opposite, parseUci, squareRank } from "chessops/util";
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
  moveUci: string
): Verdict {
  const parseRes = parseFen(evalBefore.fen);
  const isWhite = parseRes.isOk
    ? parseRes.value.turn === "white"
    : evalBefore.fen.split(" ")[1] === "w";
  const pvs = evalBefore.pvs ?? [];
  const bestMoveUci =
    pvs.length > 0 && pvs[0].moves && pvs[0].moves.length > 0
      ? pvs[0].moves[0]
      : "";
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

  if (moveUci === bestMoveUci) {
    if (parseRes.isOk) {
      const chessRes = Chess.fromSetup(parseRes.value);
      if (chessRes.isOk && getLegalMoves(chessRes.value).length === 1) {
        return "Only move";
      }
    }
    if (pvs.length >= 2 && pvs[1].score) {
      const wcSecond = winChanceSide(pvs[1].score, isWhite);
      if (wcBefore - wcSecond >= 15.0) {
        return "Only move";
      }
    }
  }

  const drop = wcBefore - wcAfter;
  if (moveUci === bestMoveUci && drop <= 2.0) {
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
  maxPlies = 6
): string[] {
  const parseRes = parseFen(fen);
  if (parseRes.isErr) return [];
  const chessRes = Chess.fromSetup(parseRes.value);
  if (chessRes.isErr) return [];
  const pos = chessRes.value;

  const sanList: string[] = [];
  for (const uci of uciMoves.slice(0, maxPlies)) {
    const move = parseUci(uci);
    if (!move || !pos.isLegal(move)) break;
    sanList.push(makeSanAndPlay(pos, move));
  }
  return sanList;
}

/**
 * Compute total material points for one color.
 * Kings have zero material points.
 */
function totalMaterialPoints(pos: Chess, color: Color): number {
  let pts = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = pos.board.get(sq);
    if (p && p.color === color && p.role !== "king") {
      pts += PIECE_VALUES[p.role];
    }
  }
  return pts;
}

/**
 * Count pieces of a specific role and color.
 */
function countPieceRole(pos: Chess, color: Color, role: Role): number {
  let count = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = pos.board.get(sq);
    if (p && p.color === color && p.role === role) {
      count++;
    }
  }
  return count;
}

/**
 * Compute the material balance fact.
 * When material is equal, return material equal.
 * When one side has more material, name the advantage.
 */
export function computeMaterialFact(pos: Chess): string {
  const whitePts = totalMaterialPoints(pos, "white");
  const blackPts = totalMaterialPoints(pos, "black");
  const diff = whitePts - blackPts;

  if (diff === 0) {
    return "Material: equal.";
  }

  const winnerName = diff > 0 ? "White" : "Black";
  const winnerColor: Color = diff > 0 ? "white" : "black";
  const loserColor: Color = diff > 0 ? "black" : "white";
  const adv = Math.abs(diff);

  const netP = Math.max(0, countPieceRole(pos, winnerColor, "pawn") - countPieceRole(pos, loserColor, "pawn"));
  const netN = Math.max(0, countPieceRole(pos, winnerColor, "knight") - countPieceRole(pos, loserColor, "knight"));
  const netB = Math.max(0, countPieceRole(pos, winnerColor, "bishop") - countPieceRole(pos, loserColor, "bishop"));
  const netR = Math.max(0, countPieceRole(pos, winnerColor, "rook") - countPieceRole(pos, loserColor, "rook"));
  const netQ = Math.max(0, countPieceRole(pos, winnerColor, "queen") - countPieceRole(pos, loserColor, "queen"));

  const loserNetP = Math.max(0, countPieceRole(pos, loserColor, "pawn") - countPieceRole(pos, winnerColor, "pawn"));
  const loserNetN = Math.max(0, countPieceRole(pos, loserColor, "knight") - countPieceRole(pos, winnerColor, "knight"));
  const loserNetB = Math.max(0, countPieceRole(pos, loserColor, "bishop") - countPieceRole(pos, winnerColor, "bishop"));
  const loserNetR = Math.max(0, countPieceRole(pos, loserColor, "rook") - countPieceRole(pos, winnerColor, "rook"));
  const loserNetQ = Math.max(0, countPieceRole(pos, loserColor, "queen") - countPieceRole(pos, winnerColor, "queen"));

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

/**
 * Find available checks and high value captures.
 * If a check is legal, record the first check.
 * If a capture of a non-pawn piece is legal, record the top capture.
 */
export function computeChecksAndCapturesFacts(pos: Chess): string[] {
  const sideName = pos.turn === "white" ? "White" : "Black";
  const facts: string[] = [];
  let checkMoveSan: string | null = null;

  const legalMoves = getLegalMoves(pos);

  // Check moves
  const checks: LegalMoveInfo[] = [];
  for (const m of legalMoves) {
    const clone = pos.clone();
    clone.play(m.move);
    if (clone.isCheck()) {
      checks.push(m);
    }
  }

  if (checks.length > 0) {
    checks.sort((a, b) =>
      a.from !== b.from
        ? a.from - b.from
        : a.to !== b.to
        ? a.to - b.to
        : a.promotionVal - b.promotionVal
    );
    const checkMove = checks[0];
    checkMoveSan = makeSan(pos, checkMove.move);
    facts.push(`${sideName} has a check: ${checkMoveSan}.`);
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
      : a.promotionVal - b.promotionVal
  );

  for (const item of captures) {
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

  // Category A: Checkmate
  const mateMoves: LegalMoveInfo[] = [];
  for (const m of legalMoves) {
    const afterM = temp.clone();
    afterM.play(m.move);
    if (afterM.isCheckmate()) {
      mateMoves.push(m);
    }
  }

  if (mateMoves.length > 0) {
    mateMoves.sort((a, b) =>
      a.from !== b.from
        ? a.from - b.from
        : a.to !== b.to
        ? a.to - b.to
        : a.promotionVal - b.promotionVal
    );
    return `${colorName} threatens ${makeSan(temp, mateMoves[0].move)}.`;
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
        : a.promotionVal - b.promotionVal
    );
    return `${colorName} threatens ${makeSan(temp, winningCaptures[0].move)}.`;
  }

  // Category C: Safe check
  const safeChecks: LegalMoveInfo[] = [];
  for (const m of legalMoves) {
    const afterM = temp.clone();
    afterM.play(m.move);
    if (afterM.isCheck()) {
      const isSafe =
        afterM.kingAttackers(m.move.to, enemyColor, afterM.board.occupied).isEmpty() ||
        afterM.kingAttackers(m.move.to, color, afterM.board.occupied).nonEmpty();
      if (isSafe) {
        safeChecks.push(m);
      }
    }
  }

  if (safeChecks.length > 0) {
    safeChecks.sort((a, b) =>
      a.from !== b.from
        ? a.from - b.from
        : a.to !== b.to
        ? a.to - b.to
        : a.promotionVal - b.promotionVal
    );
    return `${colorName} threatens ${makeSan(temp, safeChecks[0].move)}.`;
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
  // Normalize castling destination to king square
  let toSq = move.to;
  if (pieceBefore && pieceBefore.role === "king" && posBefore.board[moverColor].has(move.to)) {
    toSq = kingCastlesTo(moverColor, move.to > move.from ? "h" : "a");
  }

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
  if (
    !cap &&
    posAfter.epSquare !== undefined &&
    replyMove.to === posAfter.epSquare &&
    posAfter.board.getRole(replyMove.from) === "pawn"
  ) {
    const epDelta = oppColor === "white" ? -8 : 8;
    cap = sim.board.get(replyMove.to + epDelta);
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
      const capSq = makeSquare(replyMove.to);
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
            facts.push(`After ${replySan}, ${oppName} wins the ${cap3.role} on ${sqName}.`);
          }
        }
      }
    }
  }

  return facts;
}

/**
 * Append shared position facts to rawFacts.
 * Extract hanging pieces, checks, captures, threats, and king safety.
 * Extract material balance, phase, and opening facts.
 */
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
  let toSq = move.to;
  const pieceBefore = posBefore.board.get(move.from);
  if (pieceBefore && pieceBefore.role === "king" && posBefore.board[moverColor].has(move.to)) {
    toSq = kingCastlesTo(moverColor, move.to > move.from ? "h" : "a");
  }

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
      sim.play(replyMove);
      let replyTo = replyMove.to;
      const rPieceBefore = boardAfter.board.get(replyMove.from);
      const oppColor = opposite(moverColor);
      if (rPieceBefore && rPieceBefore.role === "king" && boardAfter.board[oppColor].has(replyMove.to)) {
        replyTo = kingCastlesTo(oppColor, replyMove.to > replyMove.from ? "h" : "a");
      }
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
  let toSq = move.to;
  const pieceBefore = posBefore.board.get(move.from);
  if (pieceBefore && pieceBefore.role === "king" && posBefore.board[moverColor].has(move.to)) {
    toSq = kingCastlesTo(moverColor, move.to > move.from ? "h" : "a");
  }

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
      sim.play(replyMove);
      let replyTo = replyMove.to;
      const rPieceBefore = boardAfter.board.get(replyMove.from);
      const oppColor = opposite(moverColor);
      if (rPieceBefore && rPieceBefore.role === "king" && boardAfter.board[oppColor].has(replyMove.to)) {
        replyTo = kingCastlesTo(oppColor, replyMove.to > replyMove.from ? "h" : "a");
      }
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
  let toSq = move.to;
  const pieceBefore = posBefore.board.get(move.from);
  if (pieceBefore && pieceBefore.role === "king" && posBefore.board[moverColor].has(move.to)) {
    toSq = kingCastlesTo(moverColor, move.to > move.from ? "h" : "a");
  }

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
      sim.play(replyMove);
      let replyTo = replyMove.to;
      const rPieceBefore = boardAfter.board.get(replyMove.from);
      const oppColor = opposite(moverColor);
      if (rPieceBefore && rPieceBefore.role === "king" && boardAfter.board[oppColor].has(replyMove.to)) {
        replyTo = kingCastlesTo(oppColor, replyMove.to > replyMove.from ? "h" : "a");
      }
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
  let toSq = move.to;
  const pieceBefore = posBefore.board.get(move.from);
  if (pieceBefore && pieceBefore.role === "king" && posBefore.board[moverColor].has(move.to)) {
    toSq = kingCastlesTo(moverColor, move.to > move.from ? "h" : "a");
  }

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
  move: NormalMove,
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
 * Explain what the best move accomplishes when the played move is bad.
 * When the verdict is Inaccuracy, Mistake, or Blunder, check the best move.
 * Report if the best move captures a piece, defends a hanging piece, or prepares castling.
 */
export function computeRefutationReasonFact(
  posBefore: Chess,
  move: NormalMove,
  boardAfter: Chess,
  evalBefore: EngineEval,
  evalAfter?: EngineEval | null,
  moveUci?: string | null
): string | null {
  const actualMoveUci =
    moveUci ??
    `${makeSquare(move.from)}${makeSquare(move.to)}${move.promotion ? move.promotion[0] : ""}`;
  const verdict = classifyVerdict(evalBefore, evalAfter, actualMoveUci);
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
  const moverColor = posBefore.turn;
  const oppColor = opposite(moverColor);
  let bestTo = bestMove.to;
  const pieceBefore = posBefore.board.get(bestMove.from);
  if (pieceBefore && pieceBefore.role === "king" && posBefore.board[moverColor].has(bestMove.to)) {
    bestTo = kingCastlesTo(moverColor, bestMove.to > bestMove.from ? "h" : "a");
  }

  // 1. Best move captures a piece
  const targetBefore = posBefore.board.get(bestTo);
  const isEnPassant = pieceBefore?.role === "pawn" && posBefore.epSquare === bestTo;
  if (targetBefore && targetBefore.color === oppColor) {
    const targetName = targetBefore.role;
    const sqName = makeSquare(bestTo);
    return `Best move ${bestSan} captures the ${targetName} on ${sqName}.`;
  }
  if (isEnPassant) {
    const sqName = makeSquare(bestTo);
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
  const hasCastlingRights =
    posBefore.castles.rook[moverColor].a !== undefined ||
    posBefore.castles.rook[moverColor].h !== undefined;
  if (hasCastlingRights && posBefore.fullmoves <= 12) {
    if (pieceBefore && (pieceBefore.role === "knight" || pieceBefore.role === "bishop")) {
      const startSqs =
        moverColor === "white"
          ? [1, 2, 5, 6] // b1, c1, f1, g1
          : [57, 58, 61, 62]; // b8, c8, f8, g8
      if (startSqs.includes(bestMove.from)) {
        const pieceName = pieceBefore.role;
        return `Best move ${bestSan} develops the ${pieceName} and prepares castling.`;
      }
    }
  }

  return null;
}

/**
 * Append shared position facts to rawFacts.
 * Extract hanging pieces, checks, captures, threats, and king safety.
 * Extract material balance, phase, and opening facts.
 */
export function appendPositionFacts(
  board: Chess,
  fen: string,
  moveUci: string | null | undefined,
  rawFacts: Array<string | null>
): void {
  const parseRes = parseFen(fen);
  if (parseRes.isErr) return;
  const chessRes = Chess.fromSetup(parseRes.value);
  if (chessRes.isErr) return;
  const posBefore = chessRes.value;

  for (const f of computeHangingFacts(board)) {
    rawFacts.push(f);
  }

  for (const f of computeChecksAndCapturesFacts(board)) {
    rawFacts.push(f);
  }

  rawFacts.push(computeThreatFact(board, posBefore.turn));

  for (const f of computeKingSafetyFacts(board)) {
    rawFacts.push(f);
  }

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
  moveUci?: string | null
): string[] {
  const parseRes = parseFen(fen);
  if (parseRes.isErr) {
    throw new Error(`Invalid FEN: ${fen}`);
  }
  const chessRes = Chess.fromSetup(parseRes.value);
  if (chessRes.isErr) {
    throw new Error(`Invalid setup for FEN: ${fen}`);
  }
  const posBefore = chessRes.value;
  const rawFacts: Array<string | null> = [];

  if (task === "explain_move") {
    if (!moveUci) {
      throw new Error("move_uci is required for task explain_move");
    }
    const move = parseUci(moveUci);
    if (!move || !isNormal(move)) {
      throw new Error(`Invalid UCI move: ${moveUci}`);
    }
    const posAfter = posBefore.clone();
    posAfter.play(move);

    // 1. Moved-piece attack fact
    rawFacts.push(computeMovedPieceAttackFact(posBefore, move, posAfter));

    // 2 & 3. Reply and material change in line
    for (const f of computeReplyAndMaterialChangeFacts(posBefore, move, posAfter, evalAfter)) {
      rawFacts.push(f);
    }

    // Tactical motifs
    rawFacts.push(computeForkFact(posBefore, move, posAfter, evalAfter));
    rawFacts.push(computePinFact(posBefore, move, posAfter, evalAfter));
    rawFacts.push(computeSkewerFact(posBefore, move, posAfter, evalAfter));
    rawFacts.push(computeDiscoveredAttackFact(posBefore, move, posAfter));
    rawFacts.push(computeMateThreatFact(posBefore, move, posAfter, evalAfter));
    rawFacts.push(computeRefutationReasonFact(posBefore, move, posAfter, evalBefore, evalAfter, moveUci));

    // Shared position facts
    appendPositionFacts(posAfter, fen, moveUci, rawFacts);
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

    // 2-8. Shared position facts
    appendPositionFacts(posBefore, fen, null, rawFacts);
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
