"""Hard rule checks for chess explanations.

This module evaluates hard factual and format constraints on explanations.
Each check returns a pass or fail result with an explanation reason.
A failure on any hard check assigns a negative reward to the explanation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

import chess
from langdetect import DetectorFactory, detect

from .parser import (
    find_tone_words,
    parse_moves,
    parse_piece_square_pairs,
)

# Set deterministic seed for language detection
DetectorFactory.seed = 0

# Piece name mapping to python-chess PieceType
CHESS_PIECE_TYPES: Final[dict[str, chess.PieceType]] = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}

# Engine jargon words prohibited for beginner level
ENGINE_JARGON_WORDS: Final[list[str]] = [
    "centipawn",
    "centipawns",
    "eval",
    "evaluation",
    "depth",
    "engine",
    "stockfish",
    "multipv",
    "nodes",
]

# Decimal and signed evaluation numbers regex
EVAL_NUMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:[+-]\d+(?:\.\d+)?|\b\d+\.\d+)\b"
)

# Negation and opponent context regex for negative tone words
NEGATED_OR_OPPONENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b(?:avoid(?:s|ed|ing)?|prevent(?:s|ed|ing)?|stop(?:s|ped|ping)?|"
    r"punish(?:es|ed|ing)?|exploit(?:s|ed|ing)?|take(?:s|ing)?\s+advantage\s+of|"
    r"capitaliz(?:e|es|ed|ing)\s+on|without|not|instead\s+of)\s+(?:a|an|the|any)?\s*|"
    r"\b(?:opponent(?:'s)?|black(?:'s)?|white(?:'s)?)\s+)$",
    re.IGNORECASE,
)

# Alternative move context regex for positive tone words
ALTERNATIVE_MOVE_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b(?:instead\s+of|better\s+is|better\s+was|best\s+was|best\s+is|best\s+move\s+was)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckResult:
    """Result of a rule check evaluation.

    The passed attribute indicates check success or failure.
    The reason attribute explains the result in plain language.
    """

    passed: bool
    reason: str


def strip_san_suffix(san: str) -> str:
    """Remove check, mate, and annotation suffixes from a SAN move."""
    return san.rstrip("+#?!")


def check_legal_moves(
    text: str,
    fen: str,
    move_uci: str | None = None,
    facts: list[str] | None = None,
) -> CheckResult:
    """Verify that all moves named in text are legal or fact-grounded.

    If a move is legal before the move, accept the move.
    If a move is legal after the played move, accept the move.
    If a move appears in the provided facts, accept the move.
    If any named move is illegal, fail the check.
    """
    named_moves = parse_moves(text)
    if not named_moves:
        return CheckResult(passed=True, reason="No moves named in text.")

    board = chess.Board(fen)
    allowed_sans: set[str] = set()
    allowed_bases: set[str] = set()

    # Add all legal moves in the position before the move
    for m in board.legal_moves:
        san = board.san(m)
        allowed_sans.add(san)
        allowed_bases.add(strip_san_suffix(san))

    # Add the played move and all legal moves after the played move
    if move_uci:
        try:
            played_move = chess.Move.from_uci(move_uci)
            if played_move in board.legal_moves:
                played_san = board.san(played_move)
                allowed_sans.add(played_san)
                allowed_bases.add(strip_san_suffix(played_san))

                board_after = board.copy()
                board_after.push(played_move)
                for m in board_after.legal_moves:
                    san = board_after.san(m)
                    allowed_sans.add(san)
                    allowed_bases.add(strip_san_suffix(san))
        except ValueError:
            pass

    # Add moves parsed from provided facts
    if facts:
        for f in facts:
            for f_move in parse_moves(f):
                allowed_sans.add(f_move)
                allowed_bases.add(strip_san_suffix(f_move))

    for move in named_moves:
        base = strip_san_suffix(move)
        if move not in allowed_sans and base not in allowed_bases:
            return CheckResult(
                passed=False,
                reason=f"Named move is not legal in position: {move}.",
            )

    return CheckResult(passed=True, reason="All named moves are legal.")


def check_grounded_moves(
    text: str,
    prompt_or_facts: str | list[str],
) -> CheckResult:
    """Verify that all moves in text appear in prompt or facts.

    Extract all moves from the prompt or fact lines.
    Fail when the text mentions a move that is not present in facts.
    """
    named_moves = parse_moves(text)
    if not named_moves:
        return CheckResult(passed=True, reason="No moves named in text.")

    grounded_moves: set[str] = set()
    grounded_bases: set[str] = set()

    if isinstance(prompt_or_facts, list):
        fact_text = " ".join(prompt_or_facts)
    else:
        fact_text = prompt_or_facts

    for m in parse_moves(fact_text):
        grounded_moves.add(m)
        grounded_bases.add(strip_san_suffix(m))

    for move in named_moves:
        base = strip_san_suffix(move)
        if move not in grounded_moves and base not in grounded_bases:
            return CheckResult(
                passed=False,
                reason=f"Move is not grounded in facts: {move}.",
            )

    return CheckResult(passed=True, reason="All named moves are grounded.")


def check_piece_on_square(
    text: str,
    fen: str,
    move_uci: str | None = None,
) -> CheckResult:
    """Verify that claimed pieces exist on claimed squares.

    Check position before the move, position after the move, and captured piece.
    Fail when the text claims a wrong piece on a square.
    """
    pairs = parse_piece_square_pairs(text)
    if not pairs:
        return CheckResult(passed=True, reason="No piece on square claims made.")

    board_before = chess.Board(fen)
    board_after: chess.Board | None = None
    captured_piece_type: chess.PieceType | None = None
    captured_square: int | None = None

    if move_uci:
        try:
            m = chess.Move.from_uci(move_uci)
            if m in board_before.legal_moves:
                if board_before.is_en_passant(m):
                    captured_piece_type = chess.PAWN
                    captured_square = m.to_square + (
                        -8 if board_before.turn == chess.WHITE else 8
                    )
                elif board_before.is_capture(m):
                    target = board_before.piece_at(m.to_square)
                    if target:
                        captured_piece_type = target.piece_type
                        captured_square = m.to_square

                board_after = board_before.copy()
                board_after.push(m)
        except ValueError:
            pass

    for piece_name, sq_name in pairs:
        pt = CHESS_PIECE_TYPES.get(piece_name)
        if pt is None:
            continue
        try:
            sq = chess.parse_square(sq_name)
        except ValueError:
            return CheckResult(
                passed=False,
                reason=f"Invalid square name: {sq_name}.",
            )

        # Check before move
        p_before = board_before.piece_at(sq)
        if p_before and p_before.piece_type == pt:
            continue

        # Check after move
        if board_after:
            p_after = board_after.piece_at(sq)
            if p_after and p_after.piece_type == pt:
                continue

        # Check captured piece
        if captured_piece_type == pt and captured_square == sq:
            continue

        return CheckResult(
            passed=False,
            reason=f"No {piece_name} on square {sq_name}.",
        )

    return CheckResult(passed=True, reason="All piece locations are valid.")


def check_verdict_tone(text: str, verdict: str) -> CheckResult:
    """Verify that explanation tone matches the move verdict.

    If verdict is bad, fail on positive tone directed at the played move.
    If verdict is good, fail on unnegated negative move verdict terms.
    """
    verdict_clean = verdict.strip().capitalize()
    is_bad_verdict = verdict_clean in ("Inaccuracy", "Mistake", "Blunder")
    is_good_verdict = verdict_clean in ("Best", "Good", "Only move")

    tone_words = find_tone_words(text)
    positive_words = tone_words["positive"]
    negative_words = tone_words["negative"]

    if is_bad_verdict and positive_words:
        # Check if positive words describe an alternative instead of played move
        for p in positive_words:
            # If word is best move, check if it refers to alternative move
            if p == "best move":
                match = re.search(r"\bbest\s+move\b", text, re.IGNORECASE)
                if match:
                    preceding = text[: match.start()]
                    if ALTERNATIVE_MOVE_CONTEXT_RE.search(preceding):
                        continue
                    following = text[match.end() :]
                    if re.match(r"^\s+(?:was|is)\s+[A-Za-z0-9]", following):
                        continue
            return CheckResult(
                passed=False,
                reason=f"Positive tone used for {verdict}: '{p}'.",
            )

    if is_good_verdict and negative_words:
        for n in negative_words:
            # Check if negative word is negated or refers to opponent
            pattern = rf"\b{re.escape(n)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                preceding = text[: match.start()]
                if not NEGATED_OR_OPPONENT_RE.search(preceding):
                    return CheckResult(
                        passed=False,
                        reason=f"Negative tone used for {verdict}: '{n}'.",
                    )

    return CheckResult(passed=True, reason="Verdict tone is consistent.")


def check_format(text: str, prompt_text: str = "") -> CheckResult:
    """Verify formatting constraints for the explanation.

    Fail when text is empty.
    Fail when text contains Markdown headings, lists, bold markers, or code blocks.
    Fail when text is not English.
    Fail when text repeats prompt headers or facts verbatim.
    """
    clean_text = text.strip()
    if not clean_text:
        return CheckResult(passed=False, reason="Explanation is empty.")

    # Check Markdown headings
    if re.search(r"^\s*#{1,6}\s+", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown heading detected.")

    # Check Markdown bullet lists
    if re.search(r"^\s*[\*\-]\s+", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown bullet list detected.")

    # Check Markdown numbered lists
    if re.search(r"^\s*\d+\.\s+", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown numbered list detected.")

    # Check bold formatting
    if "**" in text:
        return CheckResult(passed=False, reason="Markdown bold formatting detected.")

    # Check code blocks
    if "```" in text:
        return CheckResult(passed=False, reason="Markdown code block detected.")

    # Check English language for sentences with sufficient length
    words = clean_text.split()
    if len(words) >= 6:
        try:
            detected_lang = detect(clean_text)
            if detected_lang != "en":
                return CheckResult(
                    passed=False,
                    reason=f"Text detected as non-English: {detected_lang}.",
                )
        except Exception:
            pass

    # Check repetition of prompt headers
    for header in ("task: explain", "FEN:", "Verdict:", "Facts:"):
        if header in text:
            return CheckResult(
                passed=False,
                reason="Explanation repeats prompt headers.",
            )

    # Check verbatim prompt repetition
    if prompt_text and len(clean_text) >= 30 and clean_text in prompt_text:
        return CheckResult(
            passed=False,
            reason="Explanation repeats prompt text verbatim.",
        )

    return CheckResult(passed=True, reason="Format is valid.")


def check_beginner_jargon(text: str, level: str) -> CheckResult:
    """Verify that beginner explanations contain no engine jargon or decimal evals.

    When level is beginner, search for forbidden engine terms and numbers.
    When level is club, pass the check.
    """
    if level.lower() != "beginner":
        return CheckResult(passed=True, reason="Club level allows technical terms.")

    text_lower = text.lower()
    for word in ENGINE_JARGON_WORDS:
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, text_lower):
            return CheckResult(
                passed=False,
                reason=f"Engine jargon not allowed for beginners: '{word}'.",
            )

    eval_match = EVAL_NUMBER_RE.search(text)
    if eval_match:
        val = eval_match.group(0)
        return CheckResult(
            passed=False,
            reason=f"Evaluation number not allowed for beginners: '{val}'.",
        )

    return CheckResult(passed=True, reason="No beginner jargon found.")
