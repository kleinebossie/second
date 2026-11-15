/**
 * Prompt Builder wrapper for Second Chrome Extension.
 * Connects directly to the shared TypeScript core.
 */

import {
  buildUserMessage,
  buildPromptMessages,
  formatQwenChatPrompt,
  SYSTEM_MESSAGE,
  ChatMessage,
} from '../../../core/ts/src/prompt.js';
import {
  extractFacts,
  classifyVerdict,
  EngineEval,
  EnginePv,
  EngineScore,
  Verdict,
} from '../../../core/ts/src/facts.js';

export {
  buildUserMessage,
  buildPromptMessages,
  formatQwenChatPrompt,
  extractFacts,
  classifyVerdict,
  SYSTEM_MESSAGE,
};
export type { ChatMessage, EngineEval, EnginePv, EngineScore, Verdict };

export interface ExplanationRequest {
  task?: 'explain_move' | 'explain_position';
  fen: string;
  moveUci?: string;
  evalBefore: EngineEval;
  evalAfter?: EngineEval | null;
  level?: 'beginner' | 'club';
  detail?: 'short' | 'long';
}

/**
 * Builds the complete prompt string ready for wllama inference.
 */
export function buildPromptForInference(req: ExplanationRequest): {
  prompt: string;
  userMessage: string;
  systemMessage: string;
} {
  const task = req.task ?? (req.moveUci ? 'explain_move' : 'explain_position');
  const level = req.level ?? 'club';
  const detail = req.detail ?? 'short';

  const userMessage = buildUserMessage(
    task,
    level,
    detail,
    req.fen,
    req.evalBefore,
    req.evalAfter,
    req.moveUci
  );

  // Use pre-closed <think> tags for direct answer streaming
  const prompt = formatQwenChatPrompt(userMessage, SYSTEM_MESSAGE);

  return {
    prompt,
    userMessage,
    systemMessage: SYSTEM_MESSAGE,
  };
}
