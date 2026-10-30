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

#### Task: `explain_position`
Use this task to explain the current board position:
- It does not include a `Move played` line.
- It does not include a `Verdict` line.
- It lists the top candidate moves for the side to move.

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
- Strategic plan: force central concession with d5.
```

## Rules for Fact Sentences

Follow these rules when you generate fact lines:

1. Use fixed sentence templates.
2. Do not use numbers in fact text, except win chances and move numbers.
3. Use standard algebraic notation (SAN) for moves (for example: `Nxe5`, `O-O`, `e8=Q`).
4. Name squares with lower case letters and numbers (for example: `e4`, `f7`).
5. Name pieces with lower case words in prose (for example: "the knight on e5", "the bishop on c4").
6. Keep sentences short and clear.

## Level Guidelines

### Level: `beginner`
- The model must write short sentences (prefer <= 15 words).
- The model must not use chess jargon (for example: "prophylaxis", "zwischenzug", "outpost", "centipawn").
- The model must focus on immediate threats and piece safety.

### Level: `club`
- The model may use standard tactical and strategic chess terms.
- The model may include deeper plan concepts.
