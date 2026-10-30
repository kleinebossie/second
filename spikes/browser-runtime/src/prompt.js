// This file exports the fixed ~400-token benchmark prompt.
// Both the web harness and Python reference script load from prompt.json.
import promptJson from "../prompt.json";

const systemMessage = promptJson.system_message;
const userMessage = promptJson.user_message;

export const PROMPT_DATA = {
  name: promptJson.name,
  task: promptJson.task,
  level: promptJson.level,
  detail: promptJson.detail,
  fen: promptJson.fen,
  approxTokenCount: promptJson.approx_token_count,
  systemMessage,
  userMessage,
  messages: promptJson.messages,
  // Pre-formatted chat template string with non-thinking tags closed
  rawPromptWithThinkingClosed: `<|im_start|>system\n${systemMessage}<|im_end|>\n<|im_start|>user\n${userMessage}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
};
