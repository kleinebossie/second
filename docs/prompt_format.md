# Prompt Format Contract

This document defines the prompt format for the Second chess model.
Python training code and TypeScript browser extension code must build identical text.
Both implementations must match byte for byte.

## Overview

The prompt uses the Qwen chat template.
The prompt contains two messages:
1. A fixed system message.
2. A structured user message.

## System Message

The system message is fixed:

```text
You are Second, a chess assistant. You explain chess positions and moves clearly using the provided facts.
```

## User Message Schema

The user message contains these parts in this exact order:

1. **Task header:** three key-value lines.
2. **Blank line.**
3. **FEN position line.**
4. **Game state facts:** fixed sentence lines.
5. **Facts section:** bullet points with concrete positional observations.

### 1. Task Header Fields

| Field | Allowed Values | Description |
|---|---|---|
| `task` | `explain_move`, `explain_position` | The explanation mode. |
| `level` | `beginner`, `club` | The target player audience. |
| `detail` | `short`, `long` | The desired response length. |

### 2. Task Variations

#### Task: `explain_move`
Use this task to explain a specific move that a player made:
- It includes the `Move played` line.
- It includes the `Verdict` line.
- It includes the `winning chances` before and after the move.
- It includes the `Best move` line and the `Best line`.
- For this task, the material fact describes the position after the played move.

#### Task: `explain_position`
Use this task to explain the current board position:
- It does not include a `Move played` line.
- It does not include a `Verdict` line.
- It lists the top candidate moves for the side to move.
- For this task, the material fact describes the current board position.

## Canonical Example

### Header and State
```text
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
- After Ng5, Black's best reply is d5.
- White threatens Bxf7+.
- Material: equal.
- Phase: opening.
- Opening: Italian Game, Two Knights Defense.
```

## Rules for Fact Sentences

Follow these rules when you generate fact lines:

1. Use fixed sentence templates.
2. Do not use numbers in fact text, except win chances and move numbers.
3. Use standard algebraic notation (SAN) for moves (for example: `Nxe5`, `O-O`, `e8=Q`).
4. Name squares with lower case letters and numbers (for example: `e4`, `f7`).
5. Name pieces with lower case words in prose (for example: "the knight on e5", "the bishop on c4").
6. Keep sentences short and clear.
7. For `explain_move`, calculate the material fact on the board after the played move.
8. For `explain_position`, calculate the material fact on the current board.
9. Never name a piece on a square unless that piece stands on that square before or after the played move.

### v1.1 Fact Templates and Check Filtering

Version `1.1` removes noisy checks and adds positional fact templates:

- **Filtered checks:** When Stockfish top three lines include a check within 100 centipawns of the top line, emit `{Side} has a check: {move}.`. When no PV check qualifies and a legal check forces a clear gain, emit `{Side} has a check: {move}.`.
- **Recapture activity:** When the opponent's best recapture centralizes or becomes more active, emit `After {move}, {Side}'s recapture {reply} centralizes the {piece}.` or `After {move}, {Side}'s recapture {reply} increases {piece} activity.`.
- **Positional refutations (pawn):** When the best move claims space or attacks a piece in the center, emit one of these templates:
  - `Best move {best} claims central space on {sq}.`
  - `Best move {best} gains space and challenges the center.`
  - `Best move {best} gains central space and attacks the {piece} on {sq}.`
  - `Best move {best} attacks the {piece} on {sq}.`
- **Positional refutations (tension and trade avoidance):** When the best move avoids a premature trade, emit one of these templates:
  - `Best move {best} keeps central space and tension instead of trading on {sq}.`
  - `Best move {best} maintains central tension instead of trading on {sq}.`
  - `Best move {best} gains space and attacks the {piece} on {sq} instead of trading in the center.`
- **Positional refutations (pieces and king):** When the best move improves king safety, piece safety, or file control, emit one of these templates:
  - `Best move {best} castles to keep the king safe.`
  - `Best move {best} moves the attacked {piece} to safety.`
  - `Best move {best} centralizes the {piece}.`
  - `Best move {best} places the rook on an open file.`
  - `Best move {best} places the rook on a half-open file.`
  - `Best move {best} increases {piece} activity.`
- **Line comparison (checkmate and material):** After equalizing up to four-ply lines to an even ply count, emit one of these templates:
  - `Unlike {move}, the best line with {best} leads to checkmate.`
  - `Unlike {move}, {best} avoids checkmate.`
  - `Unlike {move}, the best line with {best} avoids losing material.`
  - `Unlike {move}, the best line with {best} wins material.`
  - `Unlike {move}, the best line with {best} avoids losing a pawn.`
  - `Unlike {move}, the best line with {best} wins a pawn.`
- **Line comparison (positional):** When material is equal across the settled even-ply lines, emit one of these templates:
  - `Unlike {move}, {best} prevents {Side} from centralizing pieces.`
  - `Unlike {move}, {best} keeps stronger central pawn control.`
  - `Unlike {move}, {best} gives {Side} better king safety.`
  - `Unlike {move}, {best} leaves {Side} with more active pieces.`
- **Quiet move facts:** When the played move improves safety, structure, or development, emit one of these templates:
  - `{move} castles to safeguard the king.`
  - `{move} releases central tension on {sq}.`
  - `{move} claims central space on {sq}.`
  - `{move} gains space and challenges the center.`
  - `{move} develops the {piece} toward the center.`
  - `{move} centralizes the {piece}.`
  - `{move} places the rook on the open {file}-file.`
  - `{move} places the rook on the half-open {file}-file.`
- **Board structure facts:** When no higher tactic overrides structure, emit one of these templates:
  - `Central pawn tension exists between {sq1} and {sq2}.`
  - `{Side} has more central space.`
  - `The {file}-file is open.`
  - `{Side} leads in piece development.`

## Level Guidelines

### Level: `beginner`
- The model must write short sentences (prefer <= 15 words).
- The model must not use chess jargon (for example: "prophylaxis", "zwischenzug", "outpost", "centipawn").
- The model must focus on immediate threats and piece safety.

### Level: `club`
- The model may use standard tactical and strategic chess terms.
- The model may include deeper plan concepts.
