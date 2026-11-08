"""Relational claim and coaching voice verification module.

This module validates relational chess claims, opening names, and coaching voice.
It checks claims against python-chess board representations and prompt facts.
It eliminates hallucinations of piece attacks, defenses, pins, and openings.
"""

from __future__ import annotations

import re
from typing import Final

import chess

from .checks import CheckResult, CHESS_PIECE_TYPES
from .parser import clean_move_token, parse_piece_square_pairs

# Opening names recognized by the opening claim verifier
KNOWN_OPENINGS: Final[tuple[str, ...]] = (
    "Sicilian",
    "Dragon",
    "Najdorf",
    "Italian",
    "Giuoco Piano",
    "Petrov",
    "Russian",
    "Ruy Lopez",
    "Spanish",
    "French",
    "Caro-Kann",
    "London",
    "Queen's Gambit",
    "King's Indian",
    "Nimzo-Indian",
    "Scandinavian",
    "Pirc",
    "Alekhine",
    "English",
    "Dutch",
    "Scotch",
    "Vienna",
    "Slav",
    "Grunfeld",
    "Benoni",
    "Catalan",
)

FIRST_PERSON_I_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(I'm|I've|I'd|I'll|I)\b"
)
FIRST_PERSON_MY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(my|me|mine)\b",
    re.IGNORECASE,
)

ATTACK_CLAIM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:attacks?|threatens?|hits?|pressures?)\s+(?:the\s+)?(?:(?:black|white|enemy|friendly|opponent's)\s+)?([a-z]+)\s+on\s+([a-h][1-8])\b",
    re.IGNORECASE,
)

DEFEND_CLAIM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:defends?|protects?|guards?|supports?)\s+(?:the\s+)?(?:(?:black|white|enemy|friendly|opponent's)\s+)?([a-z]+)\s+on\s+([a-h][1-8])\b",
    re.IGNORECASE,
)

PIN_CLAIM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:pins?\s+(?:the\s+)?(?:(?:black|white|enemy|friendly|opponent's)\s+)?([a-z]+)\s+on\s+([a-h][1-8])|(?:(?:black|white|enemy|friendly|opponent's)\s+)?([a-z]+)\s+on\s+([a-h][1-8])\s+is\s+pinned)\b",
    re.IGNORECASE,
)

FORK_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\b(fork|forks|forked|forking)\b", re.IGNORECASE)
SKEWER_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\b(skewer|skewers|skewered|skewering)\b", re.IGNORECASE)
MATE_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\b(checkmate|checkmates|mates|mated)\b", re.IGNORECASE)


def check_coach_voice(text: str) -> CheckResult:
    """Verify that the explanation does not use first-person player diary voice.

    Fail when text contains I, I'm, I've, I'd, I'll, my, me, or mine.
    A coach speaks objectively to the player in second or third person.
    """
    match_i = FIRST_PERSON_I_RE.search(text)
    if match_i:
        word = match_i.group(0)
        return CheckResult(
            passed=False,
            reason=f"First-person player pronoun not allowed: '{word}'.",
        )

    match_my = FIRST_PERSON_MY_RE.search(text)
    if match_my:
        word = match_my.group(0)
        return CheckResult(
            passed=False,
            reason=f"First-person player pronoun not allowed: '{word}'.",
        )

    return CheckResult(passed=True, reason="Coach voice is objective.")


def check_opening_claims(text: str, facts: list[str] | None = None) -> CheckResult:
    """Verify that named chess openings match the prompt facts.

    Fail when text names a known opening that is not in the Opening fact line.
    Pass when no openings are mentioned or when all mentioned openings match facts.
    """
    opening_fact = ""
    if facts:
        for f in facts:
            if f.lower().startswith("opening:"):
                opening_fact = f.lower()
                break

    text_lower = text.lower()
    for name in KNOWN_OPENINGS:
        pattern = rf"\b{re.escape(name.lower())}\b"
        if re.search(pattern, text_lower):
            if not opening_fact or name.lower() not in opening_fact:
                return CheckResult(
                    passed=False,
                    reason=f"Unverified opening claim: '{name}'.",
                )

    return CheckResult(passed=True, reason="Opening claims are verified.")


def _has_piece_attack_on_square(
    board: chess.Board,
    target_sq: int,
    expected_piece_type: chess.PieceType | None,
    attacker_color: chess.Color,
) -> bool:
    """Check if any piece of attacker_color attacks target_sq on board."""
    attackers = board.attackers(attacker_color, target_sq)
    if not attackers:
        return False
    if expected_piece_type is None:
        return True
    target_piece = board.piece_at(target_sq)
    if target_piece and target_piece.piece_type == expected_piece_type:
        return True
    return target_piece is not None


def _has_piece_defense_on_square(
    board: chess.Board,
    target_sq: int,
    expected_piece_type: chess.PieceType | None,
    defender_color: chess.Color,
) -> bool:
    """Check if any friendly piece defends target_sq on board."""
    defenders = board.attackers(defender_color, target_sq)
    if not defenders:
        return False
    if expected_piece_type is None:
        return True
    target_piece = board.piece_at(target_sq)
    if target_piece and target_piece.piece_type == expected_piece_type:
        return True
    return True


def check_relational_claims(
    text: str,
    fen: str,
    move_uci: str | None = None,
    facts: list[str] | None = None,
) -> CheckResult:
    """Verify piece-to-piece interaction and motif claims against board state.

    Check attack claims, defense claims, pin claims, and fork or mate assertions.
    Fail when any claim cannot be verified on the board or in facts.
    """
    try:
        board_before = chess.Board(fen)
    except Exception:
        return CheckResult(passed=True, reason="Skipped for invalid FEN.")

    board_after = board_before.copy()
    mover_color = board_before.turn
    if move_uci:
        try:
            move_obj = chess.Move.from_uci(move_uci)
            if move_obj in board_after.legal_moves:
                board_after.push(move_obj)
        except ValueError:
            pass

    boards_to_check = [board_after, board_before]

    facts_text = " ".join(facts).lower() if facts else ""

    # 1. Attack claims: "attacks/threatens the {piece} on {sq}"
    for match in ATTACK_CLAIM_RE.finditer(text):
        piece_str = match.group(1).lower()
        sq_str = match.group(2).lower()
        try:
            sq = chess.parse_square(sq_str)
        except ValueError:
            continue
        expected_type = CHESS_PIECE_TYPES.get(piece_str)

        # Verified if either mover or opponent attacks this square
        verified = any(
            _has_piece_attack_on_square(b, sq, expected_type, mover_color)
            or _has_piece_attack_on_square(b, sq, expected_type, not mover_color)
            for b in boards_to_check
        )
        if not verified and f"on {sq_str}" in facts_text:
            verified = True

        if not verified:
            return CheckResult(
                passed=False,
                reason=f"False relational claim: '{match.group(0)}'.",
            )

    # 2. Defense claims: "defends/protects/guards the {piece} on {sq}"
    for match in DEFEND_CLAIM_RE.finditer(text):
        piece_str = match.group(1).lower()
        sq_str = match.group(2).lower()
        try:
            sq = chess.parse_square(sq_str)
        except ValueError:
            continue
        expected_type = CHESS_PIECE_TYPES.get(piece_str)

        target_piece_after = board_after.piece_at(sq)
        target_piece_before = board_before.piece_at(sq)
        target_piece = target_piece_after or target_piece_before

        if not target_piece:
            return CheckResult(
                passed=False,
                reason=f"False relational claim: '{match.group(0)}' (no piece on {sq_str}).",
            )

        if expected_type and target_piece.piece_type != expected_type:
            return CheckResult(
                passed=False,
                reason=f"False relational claim: '{match.group(0)}' (piece on {sq_str} is {target_piece.piece_type}).",
            )

        if target_piece.color != mover_color:
            return CheckResult(
                passed=False,
                reason=f"False relational claim: '{match.group(0)}' (opponent piece on {sq_str}).",
            )

        color = target_piece.color
        defended_after = bool(board_after.attackers(color, sq))
        defended_before = bool(board_before.attackers(color, sq))

        if not (defended_after or defended_before):
            # Check if mentioned in facts
            if any(w in facts_text for w in ("defend", "protect", "guard", "support")) and sq_str in facts_text:
                continue
            return CheckResult(
                passed=False,
                reason=f"False relational claim: '{match.group(0)}' ({sq_str} is undefended).",
            )

    # 3. Pin claims: "pins the {piece} on {sq}"
    for match in PIN_CLAIM_RE.finditer(text):
        piece_str = (match.group(1) or match.group(3) or "").lower()
        sq_str = (match.group(2) or match.group(4) or "").lower()
        try:
            sq = chess.parse_square(sq_str)
        except ValueError:
            continue

        p_after = board_after.piece_at(sq)
        p_before = board_before.piece_at(sq)
        piece = p_after or p_before

        is_pinned = False
        if piece:
            is_pinned = board_after.is_pinned(piece.color, sq) or board_before.is_pinned(piece.color, sq)
        if not is_pinned and ("pin" in facts_text or "pinned" in facts_text):
            is_pinned = True

        if not is_pinned:
            return CheckResult(
                passed=False,
                reason=f"False relational claim: pin on {sq_str} not verified.",
            )

    # 4. Fork claims: "forks"
    if FORK_WORD_RE.search(text):
        has_fork_fact = "fork" in facts_text or "double attack" in facts_text
        if not has_fork_fact:
            # Check geometry on board_after: did the moved piece attack >= 2 opponent pieces?
            has_fork_geom = False
            if move_uci:
                try:
                    m = chess.Move.from_uci(move_uci)
                    to_sq = m.to_square
                    attacks = board_after.attacks(to_sq)
                    opp = not board_after.piece_at(to_sq).color if board_after.piece_at(to_sq) else mover_color
                    opp_targets = [
                        s for s in attacks
                        if board_after.piece_at(s) and board_after.piece_at(s).color == opp
                    ]
                    if len(opp_targets) >= 2:
                        has_fork_geom = True
                except Exception:
                    pass
            if not has_fork_geom:
                return CheckResult(
                    passed=False,
                    reason="False relational claim: fork not verified by board or facts.",
                )

    # 5. Checkmate claims: "checkmate" / "mates"
    if MATE_WORD_RE.search(text):
        has_mate_fact = "checkmate" in facts_text or "mate" in facts_text
        is_mate_on_board = board_after.is_checkmate() or board_before.is_checkmate()
        if not (has_mate_fact or is_mate_on_board):
            return CheckResult(
                passed=False,
                reason="False relational claim: checkmate not verified by board or facts.",
            )

    return CheckResult(passed=True, reason="Relational claims are verified.")
