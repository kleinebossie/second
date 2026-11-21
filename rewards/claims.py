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
from .parser import SQUARE_RE

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


# Vague boilerplate patterns for v1.1 quality checks
COORDINATION_BOILERPLATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:blocks?|disrupts?|hurts?|harms?|damages?|weakens?|ruins?|breaks?|reduces?|hinders?|impairs?|"
    r"interferes?\s+with|improves?|maintains?|keeps?|preserves?|lacks?|limits?|undermines?)\s+"
    r"(?:(?:your|their|the|white's|black's|piece|pieces'|knight|bishop|rook|queen|pawn)\s+)*coordination\b"
    r"|\b(?:poor|bad|weak|awkward|lack\s+of|without)\s+(?:(?:your|their|the|piece|knight|bishop|rook|queen|pawn)\s+)?coordination\b"
    r"|\bcoordination\s+(?:suffers?|breaks?\s+down|falls?\s+apart|is\s+(?:poor|bad|weak|lacking))\b"
    r"|\bcoordination\s+of\s+(?:(?:the|your|their|white's|black's)\s+)?\w+",
    re.IGNORECASE,
)

HARMONY_OVERALL_BOILERPLATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:maintains?|improves?|preserves?|keeps?|creates?|disrupts?|hurts?|breaks?|weakens?|ruins?)\s+"
    r"(?:(?:a|the|your|their|piece)\s+)*(?:harmony|harmonious\s+position)\b"
    r"|\b(?:creates?|causes?)\s+disharmony\b"
    r"|\bin\s+harmony\b"
    r"|\b(?:more\s+)?harmonious\s+(?:position|setup|development|structure|pieces?)\b"
    r"|\b(?:better|stronger|weaker|worse|solid|overall)\s+(?:piece|board\s+)?overall\s+position\b"
    r"|\boverall\s+(?:piece|board)?\s*position\b"
    r"|\bposition\s+overall\b"
    r"|\boverall\s+(?:structure|setup|play)\b"
    r"|\b(?:better|stronger|weaker|worse|solid)\s+overall\s+position\b"
    r"|\b(?:improves?|weakens?|hurts?|helps?|helps\s+with)\s+(?:(?:a|the|your|their|white's|black's)\s+)+overall\s+position\b",
    re.IGNORECASE,
)

UNNECESSARY_WEAKNESS_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:unnecessary|structural)\s+weakness(?:es)?\b",
    re.IGNORECASE,
)

_TRADE_CLAUSE_SQUARE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:trading|capturing|taking|exchanging)\s+(?:(?:the\s+)?(?:pawn|knight|bishop|rook|queen|piece)\s+)?on\s+[a-h][1-8]\b",
    re.IGNORECASE,
)

GENERIC_FILLER_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:does\s+not\s+(?:improve|help)|improves?)\s+(?:the|your|white's|black's)\s+position\b"
    r"|\bplays?\s+a\s+(?:useful|quiet|solid)\s+waiting\s+move\b"
    r"|\bis\s+a\s+(?:passive|slow|inaccurate|weak|solid|good|strong)\s+(?:positional\s+)?move\b"
    r"|\b(?:gives|keeps?|leaves?)\s+(?:a|an)\s+(?:better|stronger|worse|solid|flexible)\s+position\b",
    re.IGNORECASE,
)

# Concrete tactical markers that indicate an immediate tactical punishment or motif
DIRECT_TACTICAL_MARKERS: Final[tuple[str, ...]] = (
    "is not defended",
    "has a check:",
    "king is in check",
    "forks",
    "pins",
    "skewers",
    "discovered attack",
    "checkmate",
    "wins the ",
    "wins a pawn",
    "wins material",
    "avoids losing material",
    "avoids losing a pawn",
    "together with the ",
)

NON_PAWN_CAPTURE_MARKERS: Final[tuple[str, ...]] = (
    "captures the knight",
    "captures the bishop",
    "captures the rook",
    "captures the queen",
)

BEST_MOVE_TACTICAL_MARKERS: Final[tuple[str, ...]] = (
    " captures the ",
    " defends the ",
    " moves the attacked ",
)

TACTICAL_THREAT_RE: Final[re.Pattern[str]] = re.compile(
    r"\bthreatens\s+(?:checkmate|\S*[x#])",
    re.IGNORECASE,
)

POSITIONAL_REFUTATION_MARKERS: Final[tuple[str, ...]] = (
    "'s recapture ",
    "keeps central space",
    "gains space",
    "gains central space",
    "claims central space",
    "maintains central tension",
    "centralizes the",
    "places the rook on",
    "increases ",
    "prevents ",
    "keeps stronger central pawn control",
    "better king safety",
    "more active pieces",
    "castles to keep the king safe",
    "prepares castling",
)

# Category concept patterns for positional grounding checks (beginner-safe terms)
POSITIONAL_CATEGORY_CONCEPT_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "centralization": re.compile(
        r"\b(?:centraliz\w*|centralis\w*|"
        r"central\s+(?:square|squares|outpost|outposts|post|posts|control|file|files|queen|knight|rook|bishop|piece|pieces)|"
        r"active\s+(?:center|centre|central|square|squares|piece|pieces|queen|rook|knight|bishop)|"
        r"strong\s+central\s+squares?|(?:to|toward|towards|in|into)\s+the\s+cent(?:er|re))\b",
        re.IGNORECASE,
    ),
    "space_tension": re.compile(
        r"\b(?:space|tension|"
        r"central\s+(?:space|tension|control|pawn|pawns|square|squares|presence|break|breaks|push|pushes|advance|advances)|"
        r"pawn\s+(?:push|pushes|advance|advances|structure|chain|break|breaks)|"
        r"push(?:es|ed|ing)?\s+(?:the\s+)?(?:[a-h]-?)?pawn|advanc(?:es|ed|ing)?\s+(?:the\s+)?(?:[a-h]-?)?pawn|"
        r"claim(?:s|ed|ing)?\s+(?:the\s+)?(?:center|centre|central|space)|"
        r"control(?:s|led|ing)?\s+(?:the\s+)?(?:center|centre|central|[a-h][1-8])|"
        r"challeng(?:es|ed|ing)?\s+(?:the\s+)?(?:center|centre)|"
        r"instead\s+of\s+(?:trading|capturing|exchanging))\b",
        re.IGNORECASE,
    ),
    "activity_dev": re.compile(
        r"\b(?:develop\w*|activ\w*|mobil\w*|scope|more\s+squares|open\s+lines?|"
        r"brings?\s+(?:(?:the|your|their|white's|black's)\s+)?(?:knight|bishop|rook|queen|piece)?\s*(?:out|into|toward|to)|"
        r"into\s+the\s+game)\b",
        re.IGNORECASE,
    ),
    "king_safety": re.compile(
        r"\b(?:castl\w*|king(?:'s)?\s+(?:is\s+)?safe(?:ty)?|safe(?:r)?\s+king|"
        r"safeguard\w*\s+(?:the\s+)?king|king\s+shelter|shelter(?:s|ed|ing)?\s+(?:the\s+)?king|"
        r"tuck(?:s|ed|ing)?\s+(?:the\s+)?king|king\s+is\s+tucked)\b",
        re.IGNORECASE,
    ),
    "open_file": re.compile(
        r"\b(?:[a-h]-file|files?|open\s+lines?|open\s+columns?|semi-open|half-open)\b",
        re.IGNORECASE,
    ),
}


_EXPLICIT_SQUARE_BEFORE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"on|onto|at|via|around|near|"
    r"the|this|that|"
    r"squares?|outposts?|weakens?|weakened|weakening|"
    r"guards?|guarded|guarding|targets?|targeted|targeting|"
    r"attacks?|attacked|attacking|defends?|defended|defending|"
    r"protects?|protected|protecting|controls?|controlled|controlling|"
    r"pressures?|pressured|pressuring|covers?|covered|covering|blocks?|blocking|"
    r"pawn|pawns|knight|knights|bishop|bishops|rook|rooks|queen|queens|king|kings"
    r")\s+$",
    re.IGNORECASE,
)

_EXPLICIT_SQUARE_AFTER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*,?\s*(?:"
    r"(?:square|squares|outpost|outposts|weakness|weaknesses|holes?|pawn|pawns|knight|knights|bishop|bishops|rook|rooks|queen|queens|king|kings)\b|"
    r"(?:is|are)\s+(?:weak|hanging|undefended|exposed|vulnerable)\b"
    r")",
    re.IGNORECASE,
)


def _parse_explicit_squares(text: str) -> list[str]:
    """Find square names used with explicit square anchors, excluding movement-verb pawn pushes."""
    squares: list[str] = []
    seen: set[str] = set()
    for match in SQUARE_RE.finditer(text.lower()):
        before = text[: match.start()]
        after = text[match.end() :]
        if _EXPLICIT_SQUARE_BEFORE_RE.search(before) or _EXPLICIT_SQUARE_AFTER_RE.search(after):
            sq = match.group(0)
            if sq not in seen:
                seen.add(sq)
                squares.append(sq)
    return squares


def _extract_positional_categories(facts: list[str] | None) -> set[str]:
    """Identify active positional fact categories from a prompt facts list."""
    categories: set[str] = set()
    if not facts:
        return categories

    for fact in facts:
        f_stripped = fact.strip()
        f_lower = f_stripped.lower()
        if (
            f_stripped.startswith(("Phase:", "Material:", "Opening:", "Top candidate moves:"))
            or "in material" in f_lower
            or "'s best reply is " in f_lower
            or " is up a " in f_lower
            or " is up two " in f_lower
            or " is up the exchange" in f_lower
        ):
            continue

        if "centraliz" in f_lower:
            categories.add("centralization")
        if any(
            kw in f_lower
            for kw in (
                "central space",
                "central tension",
                "tension",
                "claims central",
                "gains space",
                "challenges the center",
                "central pawn",
                "instead of trading",
                "more central space",
            )
        ):
            categories.add("space_tension")
        if any(
            kw in f_lower
            for kw in ("activity", "active pieces", "develops", "piece development")
        ):
            categories.add("activity_dev")
        if any(
            kw in f_lower
            for kw in (
                "castles to keep the king safe",
                "castles to safeguard the king",
                "better king safety",
                "prepares castling",
            )
        ):
            categories.add("king_safety")
        if "-file" in f_lower or "open file" in f_lower or "half-open file" in f_lower:
            categories.add("open_file")

    return categories


def _has_opponent_tactical_punishment(facts: list[str] | None) -> bool:
    """Return true when facts describe a concrete tactical motif or immediate material punishment."""
    if not facts:
        return False
    facts_text = " ".join(facts).lower()
    has_recap = "'s recapture " in facts_text or "releases central tension" in facts_text
    for fact in facts:
        f_lower = fact.lower()
        if any(marker in f_lower for marker in DIRECT_TACTICAL_MARKERS):
            return True
        if f_lower.startswith("best move ") and any(marker in f_lower for marker in BEST_MOVE_TACTICAL_MARKERS):
            return True
        if any(marker in f_lower for marker in NON_PAWN_CAPTURE_MARKERS):
            return True
        if "captures the pawn on " in f_lower and not has_recap:
            return True
    return False


def _has_tactical_threat_fact(facts: list[str] | None) -> bool:
    """Return true when facts include a checkmate or capture threat."""
    if not facts:
        return False
    return any(TACTICAL_THREAT_RE.search(fact) for fact in facts)


def _has_positional_refutation_fact(facts: list[str] | None) -> bool:
    """Return true when facts include a positional refutation, recapture, or comparison fact."""
    if not facts:
        return False
    facts_text = " ".join(facts).lower()
    return any(marker in facts_text for marker in POSITIONAL_REFUTATION_MARKERS)


def _parse_explicit_weakness_squares(sentence: str) -> list[str]:
    """Find explicit weakened square names outside trade or capture clauses."""
    stripped = _TRADE_CLAUSE_SQUARE_RE.sub("", sentence)
    return _parse_explicit_squares(stripped)


def check_vague_boilerplate(
    text: str,
    facts: list[str] | None = None,
) -> CheckResult:
    """Reject unsupported coordination, harmony, or empty generic boilerplate claims.

    Fail when text claims coordination or harmony issues absent from facts.
    Fail when generic filler appears with no concrete square, alternative move, or fact match.
    """
    facts_text = " ".join(facts).lower() if facts else ""

    # 1. Unsupported coordination cliches
    if "coordination" not in facts_text:
        coord_match = COORDINATION_BOILERPLATE_RE.search(text)
        if coord_match:
            return CheckResult(
                passed=False,
                reason=f"Unsupported boilerplate claim: '{coord_match.group(0)}'.",
            )

    # 2. Unsupported harmony or overall-position cliches
    harmony_match = HARMONY_OVERALL_BOILERPLATE_RE.search(text)
    if harmony_match:
        matched_str = harmony_match.group(0)
        matched_lower = matched_str.lower()
        supported = (
            ("harmon" in matched_lower and ("harmony" in facts_text or "harmonious" in facts_text))
            or ("overall" in matched_lower and "overall" in facts_text)
        )
        if not supported:
            return CheckResult(
                passed=False,
                reason=f"Unsupported boilerplate claim: '{matched_str}'.",
            )

    # 3. Unsupported "unnecessary weakness" without an explicit square in the same sentence
    if "weakness" not in facts_text:
        for sentence in re.split(r"[.!?\n]+", text):
            weak_match = UNNECESSARY_WEAKNESS_RE.search(sentence)
            if weak_match and not _parse_explicit_weakness_squares(sentence):
                return CheckResult(
                    passed=False,
                    reason=f"Unsupported vague weakness claim: '{weak_match.group(0)}'.",
                )

    # 4. Pure generic filler when the whole explanation lacks concrete grounding
    filler_match = GENERIC_FILLER_RE.search(text)
    if filler_match:
        has_square = bool(_parse_explicit_squares(text))
        categories = _extract_positional_categories(facts)
        has_pos_match = any(
            POSITIONAL_CATEGORY_CONCEPT_PATTERNS[cat].search(text)
            for cat in categories
        )
        if not (has_square or has_pos_match):
            return CheckResult(
                passed=False,
                reason=f"Generic filler without concrete grounding: '{filler_match.group(0)}'.",
            )

    return CheckResult(passed=True, reason="No unsupported boilerplate claims found.")


def check_positional_fact_grounding(
    text: str,
    facts: list[str] | None = None,
    verdict: str | None = None,
) -> CheckResult:
    """Require reference to positional facts when no tactic dominates the facts.

    If facts contain direct tactical motifs or hanging material, pass immediately.
    If facts contain a capture threat without positional refutation on a bad move, pass immediately.
    When positional facts apply without overriding tactics, require a matching positional concept.
    """
    if not facts:
        return CheckResult(passed=True, reason="No facts provided.")
    if _has_opponent_tactical_punishment(facts):
        return CheckResult(passed=True, reason="Tactical facts present.")
    is_bad = bool(verdict and verdict.strip().lower() in {"inaccuracy", "mistake", "blunder"})
    if _has_tactical_threat_fact(facts) and not (is_bad and _has_positional_refutation_fact(facts)):
        return CheckResult(passed=True, reason="Tactical threat present.")

    categories = _extract_positional_categories(facts)
    if not categories:
        return CheckResult(passed=True, reason="No positional facts to ground.")

    for cat in categories:
        pattern = POSITIONAL_CATEGORY_CONCEPT_PATTERNS[cat]
        if pattern.search(text):
            return CheckResult(
                passed=True,
                reason=f"Explanation references positional concept ({cat}).",
            )

    cat_list = ", ".join(sorted(categories))
    return CheckResult(
        passed=False,
        reason=f"Missing positional fact concept for non-tactical position ({cat_list}).",
    )
