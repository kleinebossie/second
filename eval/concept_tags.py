"""Deterministic concept tier tagger for chess evaluation items.

This module labels chess items with one of five concept tiers.
The tiers follow the ChessQA curriculum framework:
1. tier_1_mechanics: checks, direct captures, hanging pieces.
2. tier_2_motifs: forks, pins, skewers, discovered attacks.
3. tier_3_calculation: multi-ply tactical lines and mate threats.
4. tier_4_positional: king safety, piece development, castling, quiet middlegames.
5. tier_5_endgame: endgame positions and pawn promotion.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Final

import chess

from core.python.facts import compute_phase_fact
from rewards.claims import _has_opponent_tactical_punishment, _has_tactical_threat_fact

CONCEPT_TIERS = (
    "tier_1_mechanics",
    "tier_2_motifs",
    "tier_3_calculation",
    "tier_4_positional",
    "tier_5_endgame",
)

MOTIF_KEYWORDS = re.compile(
    r"\b(fork|forks|forking|pin|pins|pinned|pinning|skewer|skewers|skewering|"
    r"discovered attack|discovered check|double attack)\b",
    re.IGNORECASE,
)

CALCULATION_KEYWORDS = re.compile(
    r"\b(checkmate|mate in|forced mate|threatens mate|threatens checkmate|"
    r"wins the (?:queen|rook|bishop|knight|pawn)|wins material|wins a pawn|"
    r"avoids losing material|avoids losing a pawn)\b",
    re.IGNORECASE,
)

MECHANICS_KEYWORDS = re.compile(
    r"\b(check|checks|captures|capture|hanging|not defended|undefended|takes)\b",
    re.IGNORECASE,
)

POSITIONAL_KEYWORDS = re.compile(
    r"\b(castling|castled|castle|king safety|develop|development|outpost|space|pawn structure|"
    r"centraliz\w*|tension|activity|open file|half-open|recapture|central space)\b",
    re.IGNORECASE,
)

POSITIONAL_PRIMARY_MARKERS: Final[tuple[str, ...]] = (
    "centraliz",
    "central space",
    "central tension",
    "central pawn",
    "gains space",
    "increases ",
    "more active pieces",
    "open file",
    "half-open file",
    "-file",
    "develops the",
    "castles to",
    "challenges the center",
    "releases central tension",
    "central pawn tension",
    "more central space",
    "leads in piece development",
)


def _is_positional_primary(
    facts: list[str],
    eval_before: dict[str, Any] | None,
    eval_after: dict[str, Any] | None,
) -> bool:
    """Return True when positional facts dominate and no decisive tactical motif applies."""
    if not facts or has_motif_in_facts(facts) or has_calculation_sequence(eval_before, eval_after, facts):
        return False
    if _has_opponent_tactical_punishment(facts) or _has_tactical_threat_fact(facts):
        return False

    for fact in facts:
        f_lower = fact.lower()
        if any(p in f_lower for p in POSITIONAL_PRIMARY_MARKERS):
            return True

    return False


def has_motif_in_facts(facts: list[str]) -> bool:
    """Check if any fact text mentions a tactical motif.

    Return True when a motif keyword occurs in facts.
    """
    for fact in facts:
        if MOTIF_KEYWORDS.search(fact):
            return True
    return False


def has_board_motif(board: chess.Board, move: chess.Move | None = None) -> bool:
    """Detect fork, pin, or discovered attack from board state.

    Return True when a tactical motif exists on the board.
    """
    opp = not board.turn

    # 1. Absolute pins on opponent non-pawn pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == opp and piece.piece_type not in (chess.KING, chess.PAWN):
            if board.is_pinned(opp, sq):
                return True

    # 2. Fork or double attack after move
    if move is not None:
        target_sq = move.to_square
        piece = board.piece_at(target_sq)
        if piece:
            attacks = board.attacks(target_sq)
            valuable_targets: list[int] = []
            for a_sq in attacks:
                target_p = board.piece_at(a_sq)
                if target_p and target_p.color == opp:
                    if target_p.piece_type in (chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.KING):
                        valuable_targets.append(a_sq)
            if len(valuable_targets) >= 2:
                return True

    return False


def is_endgame_position(board: chess.Board, phase: str | None = None) -> bool:
    """Determine if a position belongs to the endgame tier.

    Return True if phase is endgame or both queens are absent.
    """
    if phase == "endgame":
        return True

    phase_str = compute_phase_fact(board).lower()
    if "endgame" in phase_str:
        return True

    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    if white_queens == 0 and black_queens == 0:
        return True

    return False


def has_calculation_sequence(
    eval_before: dict[str, Any] | None,
    eval_after: dict[str, Any] | None,
    facts: list[str],
) -> bool:
    """Check if position involves a forced mate or tactical sequence.

    Return True if evaluation contains forced mate or PV has multiple forcing captures.
    """
    for fact in facts:
        if CALCULATION_KEYWORDS.search(fact):
            return True

    for ev in (eval_before, eval_after):
        if not ev:
            continue
        pvs = ev.get("pvs", [])
        if not pvs:
            continue
        top = pvs[0]
        score = top.get("score", {})
        if score.get("type") == "mate":
            return True

    return False


def classify_concept_tier(item: dict[str, Any]) -> str:
    """Assign one of five concept tiers to an evaluation record.

    Evaluate endgame, motif, calculation, mechanics, and positional rules.
    Return the concept tier string.
    """
    fen = str(item.get("fen", chess.STARTING_FEN))
    try:
        board = chess.Board(fen)
    except Exception:
        board = chess.Board()

    move: chess.Move | None = None
    move_uci = item.get("move_uci")
    if move_uci:
        try:
            move = chess.Move.from_uci(str(move_uci))
        except ValueError:
            move = None

    facts = item.get("facts", [])
    if not isinstance(facts, list):
        facts = []

    phase = item.get("phase")
    eval_before = item.get("eval_before")
    eval_after = item.get("eval_after")

    # 1. Endgame tier
    if is_endgame_position(board, phase=phase):
        return "tier_5_endgame"

    # 2. Positional-primary positions without explicit tactical motifs or hanging material
    if _is_positional_primary(facts, eval_before, eval_after):
        return "tier_4_positional"

    # 3. Tactical motifs (forks, pins, skewers, discovered attacks)
    if has_motif_in_facts(facts) or has_board_motif(board, move=move):
        return "tier_2_motifs"

    # 3. Calculation lines (forced mates, multi-ply refutations)
    if has_calculation_sequence(eval_before, eval_after, facts):
        return "tier_3_calculation"

    # 4. Mechanics: checks, direct captures, hanging pieces
    has_mechanics = False
    for fact in facts:
        if " can capture the " in fact.lower():
            continue
        if MECHANICS_KEYWORDS.search(fact):
            has_mechanics = True
            break
    if (
        has_mechanics
        or board.is_check()
        or _has_tactical_threat_fact(facts)
        or _has_opponent_tactical_punishment(facts)
    ):
        return "tier_1_mechanics"

    # 5. Positional
    return "tier_4_positional"


def tag_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign concept_tier attribute to each record in a list.

    Return the list of modified records.
    """
    tagged: list[dict[str, Any]] = []
    for rec in records:
        updated = dict(rec)
        updated["concept_tier"] = classify_concept_tier(rec)
        tagged.append(updated)
    return tagged


def tag_file(input_path: Path | str, output_path: Path | str | None = None) -> int:
    """Read a JSONL file, tag records with concept_tier, and write back.

    If output_path is None, overwrite input_path.
    Return the total count of tagged records.
    """
    in_p = Path(input_path)
    out_p = Path(output_path) if output_path else in_p

    records: list[dict[str, Any]] = []
    with open(in_p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    tagged = tag_records(records)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in tagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(tagged)
