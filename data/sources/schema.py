"""Shared record schema and utilities for data sources.

This module defines the standard record schema for all collected chess data.
It provides functions to create, validate, and format records.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

import chess


class ChessRecord(TypedDict):
    """Schema for a single chess commentary record.

    Attributes:
        source: Name of the data source.
        license: License name of the record.
        redistributable: Flag indicating if the record can be published.
        game_id: Unique identifier for the game or study.
        pgn: PGN move history up to the current move.
        fen: Board FEN representation before the move is played.
        move_uci: Move in UCI format.
        move_san: Move in SAN format.
        comment: Extracted commentary or explanation text.
        white_elo: White player rating if available.
        black_elo: Black player rating if available.
    """

    source: str
    license: str
    redistributable: bool
    game_id: str
    pgn: str
    fen: str
    move_uci: str
    move_san: str
    comment: str
    white_elo: int | None
    black_elo: int | None


RECORD_KEYS: set[str] = {
    "source",
    "license",
    "redistributable",
    "game_id",
    "pgn",
    "fen",
    "move_uci",
    "move_san",
    "comment",
    "white_elo",
    "black_elo",
}


def make_record(
    source: str,
    license_name: str,
    redistributable: bool,
    game_id: str,
    pgn: str,
    fen: str,
    move_uci: str,
    move_san: str,
    comment: str,
    white_elo: int | None = None,
    black_elo: int | None = None,
) -> ChessRecord:
    """Create a chess record dictionary.

    If optional Elo values are not provided, set them to None.
    Return the typed dictionary.
    """
    return {
        "source": source,
        "license": license_name,
        "redistributable": redistributable,
        "game_id": game_id,
        "pgn": pgn,
        "fen": fen,
        "move_uci": move_uci,
        "move_san": move_san,
        "comment": comment,
        "white_elo": white_elo,
        "black_elo": black_elo,
    }


def validate_record(record: dict[str, Any]) -> bool:
    """Validate a record against the schema.

    If any required key is missing, return False.
    If any field type is invalid, return False.
    When all fields match the schema, return True.
    """
    if set(record.keys()) != RECORD_KEYS:
        return False
    if not isinstance(record["source"], str):
        return False
    if not isinstance(record["license"], str):
        return False
    if not isinstance(record["redistributable"], bool):
        return False
    if not isinstance(record["game_id"], str):
        return False
    if not isinstance(record["pgn"], str):
        return False
    if not isinstance(record["fen"], str):
        return False
    if not isinstance(record["move_uci"], str):
        return False
    if not isinstance(record["move_san"], str):
        return False
    if not isinstance(record["comment"], str):
        return False
    if record["white_elo"] is not None and not isinstance(record["white_elo"], int):
        return False
    if record["black_elo"] is not None and not isinstance(record["black_elo"], int):
        return False
    return True


def parse_elo(raw_elo: Any) -> int | None:
    """Parse an Elo rating from raw input.

    If the value contains digits, convert it to an integer.
    When the value is empty or invalid, return None.
    """
    if raw_elo is None:
        return None
    text = str(raw_elo).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"\b\d{3,4}\b", text)
    if match:
        return int(match.group(0))
    return None


def format_moves_up_to(initial_board: chess.Board, moves: list[chess.Move]) -> str:
    """Format moves up to the current position as standard PGN notation.

    Replay the moves from the initial board state.
    Return the formatted PGN move string.
    """
    board = initial_board.copy()
    tokens: list[str] = []
    for move in moves:
        num = board.fullmove_number
        san = board.san(move)
        if board.turn == chess.WHITE:
            tokens.append(f"{num}. {san}")
        else:
            if not tokens or tokens[-1].endswith("."):
                tokens.append(f"{num}... {san}")
            else:
                tokens.append(san)
        board.push(move)
    return " ".join(tokens)
