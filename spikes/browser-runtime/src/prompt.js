// This file exports the fixed ~400-token benchmark prompt.
// Both the web harness and Python reference script use this exact prompt.

export const PROMPT_DATA = {
  name: "Second ~400-token benchmark prompt",
  task: "explain_move",
  level: "club",
  detail: "short",
  fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  approxTokenCount: 409,
  systemMessage: `You are Second, a chess coach that explains moves clearly.
Stockfish provides the facts. You explain why the move is good or bad in simple English.
Follow these rules:
- Rely strictly on the provided engine facts.
- Adapt your language to the requested level (beginner or club).
- Produce 2 to 4 concise sentences for short explanations.`,
  userMessage: `task: explain_move
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
- Strategic plan: force central concession with d5.`,
  messages: [
    {
      role: "system",
      content: `You are Second, a chess coach that explains moves clearly.
Stockfish provides the facts. You explain why the move is good or bad in simple English.
Follow these rules:
- Rely strictly on the provided engine facts.
- Adapt your language to the requested level (beginner or club).
- Produce 2 to 4 concise sentences for short explanations.`
    },
    {
      role: "user",
      content: `task: explain_move
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
- Strategic plan: force central concession with d5.`
    }
  ],
  // Pre-formatted chat template string with non-thinking tags closed
  rawPromptWithThinkingClosed: `<|im_start|>system
You are Second, a chess coach that explains moves clearly.
Stockfish provides the facts. You explain why the move is good or bad in simple English.
Follow these rules:
- Rely strictly on the provided engine facts.
- Adapt your language to the requested level (beginner or club).
- Produce 2 to 4 concise sentences for short explanations.<|im_end|>
<|im_start|>user
task: explain_move
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
- Strategic plan: force central concession with d5.<|im_end|>
<|im_start|>assistant
<think>

</think>

`
};
