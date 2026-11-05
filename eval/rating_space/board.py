"""Chess board rendering module for the rating arena.

This module converts FEN positions into SVG chess boards.
The renderer marks played moves with arrows and square highlights.
"""

from __future__ import annotations

from typing import Any

import chess
import chess.svg


def parse_move(board: chess.Board, move_text: str | None) -> chess.Move | None:
    """Parse a move string into a chess Move object.

    When move_text is empty, return None.
    When move_text matches SAN notation, return the legal move.
    When move_text matches UCI notation, return the legal move.
    When the move is illegal or invalid, return None.
    """
    if not move_text:
        return None

    clean_text = move_text.strip()
    if not clean_text:
        return None

    # First attempt: parse move as Standard Algebraic Notation (SAN).
    try:
        return board.parse_san(clean_text)
    except ValueError:
        pass

    # Second attempt: parse move as Universal Chess Interface (UCI) notation.
    try:
        uci_move = chess.Move.from_uci(clean_text)
        if uci_move in board.legal_moves:
            return uci_move
    except ValueError:
        pass

    return None


def render_board_svg(
    fen: str,
    move_san: str | None = None,
    move_uci: str | None = None,
    orientation: chess.Color | None = None,
    size: int = 400,
    show_post_move: bool = False,
) -> str:
    """Render a chess board to SVG format.

    When the FEN string is invalid, return an error SVG banner.
    When orientation is None, orient the board toward the side to move.
    When a played move is provided, highlight the move on the board.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {size} {size}">'
            f'<rect width="100%" height="100%" fill="#fee2e2" rx="8"/>'
            f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
            f'fill="#991b1b" font-family="sans-serif" font-size="16">Invalid chess position</text>'
            f'</svg>'
        )

    board_orientation = board.turn if orientation is None else orientation

    move: chess.Move | None = None
    if move_uci:
        move = parse_move(board, move_uci)
    if move is None and move_san:
        move = parse_move(board, move_san)

    if move is not None:
        arrow = chess.svg.Arrow(move.from_square, move.to_square, color="#2563ebcc")
        if show_post_move:
            board.push(move)
            return chess.svg.board(
                board=board,
                orientation=board_orientation,
                lastmove=move,
                size=size,
            )
        return chess.svg.board(
            board=board,
            orientation=board_orientation,
            arrows=[arrow],
            lastmove=move,
            size=size,
        )

    return chess.svg.board(
        board=board,
        orientation=board_orientation,
        size=size,
    )


def extract_position_metadata(fen: str, move_san: str | None = None) -> dict[str, Any]:
    """Extract chess position metadata from a FEN string.

    Return the side to move, move number, check state, and move label.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return {
            "valid": False,
            "side_to_move": "Unknown",
            "move_number": 0,
            "is_check": False,
            "is_game_over": False,
            "move_san": move_san or "None",
        }

    side = "White" if board.turn == chess.WHITE else "Black"
    return {
        "valid": True,
        "side_to_move": side,
        "move_number": board.fullmove_number,
        "is_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "move_san": move_san or "None",
    }
