// This file exports the fixed ~400-token benchmark prompt.
// Both the web harness and Python reference script use this exact prompt.

const systemMessage = "You are Second, a chess assistant. You explain chess positions and moves clearly using the provided facts.";

const userMessage = `task: explain_move
level: club
detail: short

FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4
White to move, move 4.
Move played: Ng5.
Verdict: Good.
White's winning chances: 54% → 53%.
Best move: d3.
Best line: d3 Bc5 c3 d6.
Facts:
- Ng5 attacks the pawn on f7 together with the bishop on c4.
- Pawn on f7 is defended only by the Black king on e8.
- Black's best reply is d5.
- After d5, White plays exd5, attacking the knight on c6.
- The knight on c6 must move, typically to Na5.
- If 4... Bc5, White has the Fried Liver Attack with Nxf7.
- White king is safe and ready to castle kingside.
- Black has not castled; king safety is compromised.
- Material: equal.
- Phase: opening.
- Opening: Italian Game, Two Knights Defense.
- Checks and captures available: Ng5xf7 forks queen on d8 and rook on h8.
- Strategic plan: force central concession with d5.`;

export const PROMPT_DATA = {
  name: "Second ~400-token benchmark prompt",
  task: "explain_move",
  level: "club",
  detail: "short",
  fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  approxTokenCount: 409,
  systemMessage,
  userMessage,
  messages: [
    { role: "system", content: systemMessage },
    { role: "user", content: userMessage }
  ],
  // Pre-formatted chat template string with non-thinking tags closed
  rawPromptWithThinkingClosed: `<|im_start|>system\n${systemMessage}<|im_end|>\n<|im_start|>user\n${userMessage}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
};
