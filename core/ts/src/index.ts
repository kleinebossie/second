/**
 * Second chess shared core module.
 * This module exports fact extraction and prompt formatting functions.
 * It achieves byte-for-byte parity with the Python implementation.
 */

export { OPENING_BOOK, lookupOpening } from "./openings.js";

export type {
  EngineScore,
  EnginePv,
  EngineEval,
  Verdict,
  LegalMoveInfo,
} from "./facts.js";

export {
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
  computeRecaptureActivityFact,
  computeRefutationReasonFact,
  computeLineComparisonFact,
  computeMovePositionalFact,
  computePositionStructureFact,
  appendPositionFacts,
  extractFacts,
} from "./facts.js";

export type { ChatMessage } from "./prompt.js";

export {
  SYSTEM_MESSAGE,
  buildUserMessage,
  buildPromptMessages,
  formatQwenChatPrompt,
} from "./prompt.js";
