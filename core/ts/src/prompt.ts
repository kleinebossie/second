/**
 * Prompt builder for the Second chess assistant.
 * This module formats prompt headers and facts into user messages.
 * It formats full chat prompts for the Qwen model architecture.
 */

import { Chess } from "chessops";
import { parseFen } from "chessops/fen";
import { makeSan } from "chessops/san";
import { isNormal } from "chessops/types";
import { parseUci } from "chessops/util";
import {
  classifyVerdict,
  EngineEval,
  extractFacts,
  isKingTakesFriendlyRook,
  roundPct,
  uciLineToSan,
  winChanceSide,
} from "./facts.js";

export const SYSTEM_MESSAGE =
  "You are Second, a chess assistant. " +
  "You explain chess positions and moves clearly using the provided facts.";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * Build the user message string for the model prompt.
 * If task is explain_move, format move header and verdict.
 * If task is explain_position, format position evaluation lines.
 * Append extracted facts as bullet points.
 */
export function buildUserMessage(
  task: string,
  level: string,
  detail: string,
  fen: string,
  evalBefore: EngineEval,
  evalAfter?: EngineEval | null,
  moveUci?: string | null,
  facts?: string[] | null
): string {
  const parseRes = parseFen(fen);
  if (parseRes.isErr) {
    throw new Error(`Invalid FEN: ${fen}`);
  }
  const chessRes = Chess.fromSetup(parseRes.value);
  if (chessRes.isErr) {
    throw new Error(`Invalid setup for FEN: ${fen}`);
  }
  const pos = chessRes.value;

  const isWhite = pos.turn === "white";
  const sideName = isWhite ? "White" : "Black";
  const pvs = evalBefore.pvs ?? [];

  const wcBefore =
    pvs.length > 0 && pvs[0].score ? winChanceSide(pvs[0].score, isWhite) : 50.0;
  const wcBeforePct = roundPct(wcBefore);

  let bestMoveSan = "";
  let bestLineSan = "";
  if (pvs.length > 0 && pvs[0].moves && pvs[0].moves.length > 0) {
    const bestMovesUci = pvs[0].moves;
    const bestM = parseUci(bestMovesUci[0]);
    if (bestM && pos.isLegal(bestM)) {
      bestMoveSan = makeSan(pos, bestM);
    }
    const bestLineSans = uciLineToSan(fen, bestMovesUci, 6, pos);
    bestLineSan = bestLineSans.join(" ");
  }

  const resolvedFacts =
    facts ?? extractFacts(task, fen, evalBefore, evalAfter, moveUci, pos);
  const factsLines = resolvedFacts.map((f) => `- ${f}`);

  if (task === "explain_move") {
    if (!moveUci) {
      throw new Error("move_uci is required for task explain_move");
    }
    const moveObj = parseUci(moveUci);
    if (!moveObj || !isNormal(moveObj) || isKingTakesFriendlyRook(pos, moveObj) || !pos.isLegal(moveObj)) {
      throw new Error(`Invalid or illegal move_uci: ${moveUci}`);
    }
    const moveSan = makeSan(pos, moveObj);
    const verdict = classifyVerdict(
      evalBefore,
      evalAfter,
      moveUci,
      evalBefore.fen === fen ? pos : undefined
    );

    let wcAfter = wcBefore;
    if (
      evalAfter &&
      evalAfter.pvs &&
      evalAfter.pvs.length > 0 &&
      evalAfter.pvs[0].score
    ) {
      wcAfter = winChanceSide(evalAfter.pvs[0].score, isWhite);
    }
    const wcAfterPct = roundPct(wcAfter);

    const lines = [
      `task: ${task}`,
      `level: ${level}`,
      `detail: ${detail}`,
      "",
      `FEN: ${fen}`,
      `${sideName} to move, move ${pos.fullmoves}.`,
      `Move played: ${moveSan}.`,
      `Verdict: ${verdict}.`,
      `${sideName}'s winning chances: ${wcBeforePct}% → ${wcAfterPct}%.`,
      `Best move: ${bestMoveSan}.`,
      `Best line: ${bestLineSan}.`,
      "Facts:",
    ];
    lines.push(...factsLines);
    return lines.join("\n");
  }

  if (task === "explain_position") {
    const lines = [
      `task: ${task}`,
      `level: ${level}`,
      `detail: ${detail}`,
      "",
      `FEN: ${fen}`,
      `${sideName} to move, move ${pos.fullmoves}.`,
      `${sideName}'s winning chances: ${wcBeforePct}%.`,
      `Best move: ${bestMoveSan}.`,
      `Best line: ${bestLineSan}.`,
      "Facts:",
    ];
    lines.push(...factsLines);
    return lines.join("\n");
  }

  throw new Error(`Unknown task: ${task}`);
}

/**
 * Build standard role-based chat messages.
 * Return an array containing system and user message objects.
 */
export function buildPromptMessages(
  task: string,
  level: string,
  detail: string,
  fen: string,
  evalBefore: EngineEval,
  evalAfter?: EngineEval | null,
  moveUci?: string | null,
  systemMessage: string = SYSTEM_MESSAGE
): ChatMessage[] {
  const userContent = buildUserMessage(
    task,
    level,
    detail,
    fen,
    evalBefore,
    evalAfter,
    moveUci
  );
  return [
    { role: "system", content: systemMessage },
    { role: "user", content: userContent },
  ];
}

/**
 * Format chat messages into the exact Qwen chat template.
 * Return the complete input string with reasoning markers.
 */
export function formatQwenChatPrompt(
  userMessage: string,
  systemMessage: string = SYSTEM_MESSAGE
): string {
  return (
    `<|im_start|>system\n${systemMessage}<|im_end|>\n` +
    `<|im_start|>user\n${userMessage}<|im_end|>\n` +
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  );
}
