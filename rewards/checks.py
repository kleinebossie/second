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

try:
    from langdetect import DetectorFactory, detect

    # Set deterministic seed for language detection
    DetectorFactory.seed = 0
except ImportError:
    DetectorFactory = None  # type: ignore
    detect = None  # type: ignore

from .parser import (
    SAN_MOVE_RE,
    clean_move_token,
    find_tone_words,
    parse_moves,
    parse_piece_square_pairs,
)

# Piece name mapping to python-chess PieceType
CHESS_PIECE_TYPES: Final[dict[str, chess.PieceType]] = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}

# Verdict class groupings
BAD_VERDICTS: Final[tuple[str, ...]] = ("Inaccuracy", "Mistake", "Blunder")
GOOD_VERDICTS: Final[tuple[str, ...]] = ("Best", "Good", "Only move")

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

# Negation, past-turn, and opponent context regex for negative tone words
NEGATED_OR_OPPONENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b(?:avoid(?:s|ed|ing)?|prevent(?:s|ed|ing)?|stop(?:s|ped|ping)?|"
    r"punish(?:es|ed|ing)?|exploit(?:s|ed|ing)?|take(?:s|ing)?\s+advantage\s+of|"
    r"capitaliz(?:e|es|ed|ing)\s+on|without|not|instead\s+of|judged|considered|"
    r"since\s+(?:i|we|white|black|he|she|they)|after\s+(?:i|we|white|black|he|she|they)|"
    r"had\s+already)\s+(?:a|an|the|any)?\s*|"
    r"\b(?:opponent(?:'s)?|black(?:'s)?|white(?:'s)?)\s+)$",
    re.IGNORECASE,
)

# Alternative move and opponent context regex for positive tone words
ALTERNATIVE_MOVE_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b(?:"
    r"instead\s+of|"
    r"better\s+(?:is|was|would\s+be|move)|"
    r"best\s+(?:is|was|move\s+was|move\s+is|move)|"
    r"should\s+have\s+played|should\s+play|"
    r"gives\s+(?:black|white|opponent)(?:'s)?(?:\s+(?:a|an))?|"
    r"allows?\s+(?:(?:black|white|opponent)(?:'s)?\s+)?(?:a|an)?|"
    r"for\s+(?:black|white|opponent)|"
    r"(?:black|white|opponent)\s+(?:has|had|gets|finds|can|could)(?:\s+(?:a|an))?"
    r")\b)",
    re.IGNORECASE,
)

# Regex to match alternative move context before positive tone words
ALT_MOVE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b(?:instead\s+of|better|best|alternative|"
    r"[a-h][1-8]|[KQRBN][a-h1-8]?x?[a-h][1-8])\b[^.!?\n]*?\b(?:had|was|would\s+be|is)\s+(?:a|an)?\s*$)",
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


from .claims import (
    check_coach_voice,
    check_opening_claims,
    check_relational_claims,
)


def is_bad_verdict(verdict: str | None) -> bool:
    """Return true when the verdict represents an inaccuracy, mistake, or blunder."""
    if not verdict:
        return False
    return verdict.strip().capitalize() in BAD_VERDICTS


def is_good_verdict(verdict: str | None) -> bool:
    """Return true when the verdict represents a good, best, or only move."""
    if not verdict:
        return False
    return verdict.strip().capitalize() in GOOD_VERDICTS


def strip_san_suffix(san: str) -> str:
    """Remove check, mate, and annotation suffixes from a SAN move."""
    return san.rstrip("+#?!")


def _find_unallowed_move(
    named_moves: list[str],
    allowed_sans: set[str],
    allowed_bases: set[str],
) -> str | None:
    """Return the first named move not present in the allowed sets."""
    for move in named_moves:
        base = strip_san_suffix(move)
        if move not in allowed_sans and base not in allowed_bases:
            return move
    return None


def _collect_line_moves(
    start_boards: list[chess.Board],
    fact_line: str,
    allowed_sans: set[str],
    allowed_bases: set[str],
) -> None:
    """Trace legal SAN moves at their place in a fact line and record them."""
    candidates = parse_moves(fact_line)
    if not candidates:
        return

    # Accept single-move threat facts (null-move threats if the side could move again)
    if len(candidates) == 1 and "threatens" in fact_line.lower():
        m_str = candidates[0]
        allowed_sans.add(m_str)
        allowed_bases.add(strip_san_suffix(m_str))
        return

    for start_board in start_boards:
        sim = start_board.copy()
        for idx, move_str in enumerate(candidates):
            parsed_move: chess.Move | None = None
            try:
                parsed_move = sim.parse_san(move_str)
            except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                if idx > 0:
                    try:
                        sim_reset = start_board.copy()
                        parsed_move = sim_reset.parse_san(move_str)
                        sim = sim_reset
                    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                        parsed_move = None
            if parsed_move is not None and parsed_move in sim.legal_moves:
                canonical = sim.san(parsed_move)
                allowed_sans.add(canonical)
                allowed_bases.add(strip_san_suffix(canonical))
                allowed_sans.add(move_str)
                allowed_bases.add(strip_san_suffix(move_str))
                sim.push(parsed_move)


def check_legal_moves(
    text: str,
    fen: str,
    move_uci: str | None = None,
    facts: list[str] | None = None,
) -> CheckResult:
    """Verify that all moves named in text are legal in position or fact lines.

    If a move is legal before the move, accept the move.
    If a move is legal after the played move, accept the move.
    If a move is legal at its place in a line from the facts, accept the move.
    If any named move is illegal, fail the check.
    """
    named_moves = parse_moves(text)
    if not named_moves:
        return CheckResult(passed=True, reason="No moves named in text.")

    board = chess.Board(fen)
    allowed_sans: set[str] = set()
    allowed_bases: set[str] = set()
    start_boards: list[chess.Board] = [board]

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
                start_boards.append(board_after)
                for m in board_after.legal_moves:
                    san = board_after.san(m)
                    allowed_sans.add(san)
                    allowed_bases.add(strip_san_suffix(san))
        except ValueError:
            pass

    # Add moves that are legal at their place in a line from the facts
    if facts:
        for f in facts:
            _collect_line_moves(start_boards, f, allowed_sans, allowed_bases)

    # Check if text specifies that a move is the initial first move despite not being legal in board
    board_legal_bases = {strip_san_suffix(board.san(m)) for m in board.legal_moves}
    if move_uci:
        try:
            pm = chess.Move.from_uci(move_uci)
            if pm in board.legal_moves:
                board_after_tmp = board.copy()
                board_after_tmp.push(pm)
                for m in board_after_tmp.legal_moves:
                    board_legal_bases.add(strip_san_suffix(board_after_tmp.san(m)))
        except ValueError:
            pass

    for move in named_moves:
        base = strip_san_suffix(move)
        if base not in board_legal_bases:
            first_move_claim = re.search(
                rf"\b(?:begin|start|open)s?\s+with\s+{re.escape(move)}\b",
                text,
                re.IGNORECASE,
            )
            if first_move_claim:
                return CheckResult(
                    passed=False,
                    reason=f"Named move is not legal in position: {move}.",
                )

    bad_move = _find_unallowed_move(named_moves, allowed_sans, allowed_bases)
    if bad_move is not None:
        return CheckResult(
            passed=False,
            reason=f"Named move is not legal in position: {bad_move}.",
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

    bad_move = _find_unallowed_move(named_moves, grounded_moves, grounded_bases)
    if bad_move is not None:
        return CheckResult(
            passed=False,
            reason=f"Move is not grounded in facts: {bad_move}.",
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


def check_verdict_tone(text: str, verdict: str | None = None) -> CheckResult:
    """Verify that explanation tone matches the move verdict.

    If the verdict is empty or missing, pass the check.
    If verdict is bad, fail on positive tone directed at the played move.
    If verdict is good, fail on unnegated negative move verdict terms.
    """
    if not verdict or not verdict.strip():
        return CheckResult(passed=True, reason="No move verdict to check.")

    bad_verdict = is_bad_verdict(verdict)
    good_verdict = is_good_verdict(verdict)

    tone_words = find_tone_words(text)
    positive_words = tone_words["positive"]
    negative_words = tone_words["negative"]

    if bad_verdict and positive_words:
        # Check if positive words describe an alternative or opponent instead of played move
        for p in positive_words:
            pattern = rf"\b{re.escape(p)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                sentence_before = re.split(r"[.!?\n]", text[: match.start()])[-1]
                if ALTERNATIVE_MOVE_CONTEXT_RE.search(sentence_before):
                    continue
                if ALT_MOVE_PREFIX_RE.search(sentence_before):
                    continue
                following = text[match.end() :]
                if re.match(
                    r"^\s*(?:here\s*)?(?:was|is|would\s+be|\?)",
                    following,
                    re.IGNORECASE,
                ):
                    continue
                if re.match(r"^\s+for\s+(?:black|white|opponent)\b", following, re.IGNORECASE):
                    continue
                return CheckResult(
                    passed=False,
                    reason=f"Positive tone used for {verdict}: '{p}'.",
                )

    if good_verdict and negative_words:
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
    Fail when text contains Markdown headings, lists, blockquotes, emphasis, code, or tables.
    Fail when text is not English.
    Fail when text repeats prompt headers or facts verbatim.
    """
    clean_text = text.strip()
    if not clean_text:
        return CheckResult(passed=False, reason="Explanation is empty.")

    # Check ATX and Setext Markdown headings
    if re.search(r"^\s*#{1,6}\s+", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown heading detected.")
    if re.search(r"^.+\n\s*(?:={3,}|-{3,})\s*$", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown heading detected.")

    # Check Markdown bullet lists
    if re.search(r"^\s*[\*\-]\s+", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown bullet list detected.")

    # Check Markdown numbered lists
    numbered_matches = list(re.finditer(r"^\s*(\d+)\.\s+(\S+)", text, re.MULTILINE))
    if len(numbered_matches) >= 2:
        return CheckResult(passed=False, reason="Markdown numbered list detected.")
    if len(numbered_matches) == 1:
        num_str = numbered_matches[0].group(1)
        first_word = clean_move_token(numbered_matches[0].group(2))
        is_san = bool(SAN_MOVE_RE.match(first_word) or re.match(r"^[a-h][1-8]$", first_word))
        if num_str == "1" and not is_san:
            return CheckResult(passed=False, reason="Markdown numbered list detected.")

    # Check Markdown blockquotes
    if re.search(r"^\s*>\s+", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown blockquote detected.")

    # Check Markdown tables
    if re.search(r"^\s*\|.+\|\s*$", text, re.MULTILINE):
        return CheckResult(passed=False, reason="Markdown table detected.")

    # Check bold and italic formatting
    if "**" in text or "__" in text:
        return CheckResult(passed=False, reason="Markdown bold formatting detected.")
    if re.search(r"(?<!\*)\*[A-Za-z][^*\n]*\*(?!\*)", text):
        return CheckResult(passed=False, reason="Markdown italic formatting detected.")

    # Check code blocks and inline code
    if "```" in text or "`" in text:
        return CheckResult(passed=False, reason="Markdown code formatting detected.")

    # Check English language for sentences with sufficient length
    words = clean_text.split()
    if len(words) >= 4 and detect is not None:
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
