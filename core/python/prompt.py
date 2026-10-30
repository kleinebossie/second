"""Prompt builder for the Second chess assistant.

This module formats prompt headers and facts into user messages.
It formats full chat prompts for the Qwen model architecture.
"""

from __future__ import annotations

from typing import Any

import chess

from .facts import (
    classify_verdict,
    extract_facts,
    round_pct,
    uci_line_to_san,
    win_chance_side,
)

SYSTEM_MESSAGE: str = (
    "You are Second, a chess assistant. "
    "You explain chess positions and moves clearly using the provided facts."
)


def build_user_message(
    task: str,
    level: str,
    detail: str,
    fen: str,
    eval_before: dict[str, Any],
    eval_after: dict[str, Any] | None = None,
    move_uci: str | None = None,
) -> str:
    """Build the user message string for the model prompt.

    If task is explain_move, format move header and verdict.
    If task is explain_position, format position evaluation lines.
    Append extracted facts as bullet points.
    """
    board = chess.Board(fen)
    is_white = (board.turn == chess.WHITE)
    side_name = "White" if is_white else "Black"
    pvs = eval_before.get("pvs", [])

    wc_before = win_chance_side(pvs[0]["score"], is_white) if (pvs and "score" in pvs[0]) else 50.0
    wc_before_pct = round_pct(wc_before)

    best_move_san = ""
    best_line_san = ""
    if pvs and pvs[0].get("moves"):
        best_moves_uci = pvs[0]["moves"]
        try:
            best_m = chess.Move.from_uci(best_moves_uci[0])
            if best_m in board.legal_moves:
                best_move_san = board.san(best_m)
        except ValueError:
            pass
        best_line_sans = uci_line_to_san(fen, best_moves_uci, max_plies=6)
        best_line_san = " ".join(best_line_sans)

    facts = extract_facts(
        task=task,
        fen=fen,
        eval_before=eval_before,
        eval_after=eval_after,
        move_uci=move_uci,
    )
    facts_lines = [f"- {f}" for f in facts]

    if task == "explain_move":
        if not move_uci:
            raise ValueError("move_uci is required for task explain_move")
        move_obj = chess.Move.from_uci(move_uci)
        move_san = board.san(move_obj)
        verdict = classify_verdict(eval_before, eval_after, move_uci)

        if eval_after and eval_after.get("pvs") and "score" in eval_after["pvs"][0]:
            wc_after = win_chance_side(eval_after["pvs"][0]["score"], is_white)
        else:
            wc_after = wc_before
        wc_after_pct = round_pct(wc_after)

        lines = [
            f"task: {task}",
            f"level: {level}",
            f"detail: {detail}",
            "",
            f"FEN: {fen}",
            f"{side_name} to move, move {board.fullmove_number}.",
            f"Move played: {move_san}.",
            f"Verdict: {verdict}.",
            f"{side_name}'s winning chances: {wc_before_pct}% \u2192 {wc_after_pct}%.",
            f"Best move: {best_move_san}.",
            f"Best line: {best_line_san}.",
            "Facts:",
        ]
        lines.extend(facts_lines)
        return "\n".join(lines)

    if task == "explain_position":
        lines = [
            f"task: {task}",
            f"level: {level}",
            f"detail: {detail}",
            "",
            f"FEN: {fen}",
            f"{side_name} to move, move {board.fullmove_number}.",
            f"{side_name}'s winning chances: {wc_before_pct}%.",
            f"Best move: {best_move_san}.",
            f"Best line: {best_line_san}.",
            "Facts:",
        ]
        lines.extend(facts_lines)
        return "\n".join(lines)

    raise ValueError(f"Unknown task: {task}")


def build_prompt_messages(
    task: str,
    level: str,
    detail: str,
    fen: str,
    eval_before: dict[str, Any],
    eval_after: dict[str, Any] | None = None,
    move_uci: str | None = None,
    system_message: str = SYSTEM_MESSAGE,
) -> list[dict[str, str]]:
    """Build standard role-based chat messages.

    Return a list containing system and user message dictionaries.
    """
    user_content = build_user_message(
        task=task,
        level=level,
        detail=detail,
        fen=fen,
        eval_before=eval_before,
        eval_after=eval_after,
        move_uci=move_uci,
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def format_qwen_chat_prompt(
    user_message: str,
    system_message: str = SYSTEM_MESSAGE,
) -> str:
    """Format chat messages into the exact Qwen chat template.

    Return the complete input string with reasoning markers.
    """
    return (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
