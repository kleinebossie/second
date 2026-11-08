/**
 * Second chess shared core module.
 * This module exports fact extraction and prompt formatting functions.
 * It achieves byte-for-byte parity with the Python implementation.
 */

export { OPENING_BOOK, lookupOpening } from "./openings.js";

export {
  EngineScore,
  EnginePv,
  EngineEval,
  Verdict,
  LegalMoveInfo,
  PIECE_VALUES,
  getLegalMoves,
  scoreToWhiteCp,
  winChanceWhite,
  winChanceSide,
  winChance,
  roundPct,
  classifyVerdict,
  uciLineToSan,
  computeMaterialFact,
  computePhaseFact,
  computeKingSafetyFacts,
  computeHangingFacts,
  computeChecksAndCapturesFacts,
  computeThreatFact,
  computeMovedPieceAttackFact,
  computeReplyAndMaterialChangeFacts,
  computeForkFact,
  computePinFact,
  computeSkewerFact,
  computeDiscoveredAttackFact,
  computeMateThreatFact,
  computeRefutationReasonFact,
  appendPositionFacts,
  extractFacts,
} from "./facts.js";

export {
  SYSTEM_MESSAGE,
  ChatMessage,
  buildUserMessage,
  buildPromptMessages,
  formatQwenChatPrompt,
} from "./prompt.js";
