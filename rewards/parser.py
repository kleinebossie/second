"""Text parsing utilities for chess explanations.

This module extracts chess moves, squares, piece locations, and tone words.
It parses standard algebraic notation and detects chess entity references.
"""

from __future__ import annotations

import re
from typing import Final

# Regex pattern for castling moves
CASTLING_PATTERN: Final[str] = r"(?:O-O-O|O-O|0-0-0|0-0)[+#]?"

# Regex pattern for piece moves (starts with uppercase piece letter K, Q, R, B, N)
PIECE_MOVE_PATTERN: Final[str] = (
    r"[KQRBN][a-h1-8]?[a-h1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
)

# Regex pattern for pawn captures
PAWN_CAPTURE_PATTERN: Final[str] = r"[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?"

# Regex pattern for pawn promotion
PAWN_PROMOTION_PATTERN: Final[str] = r"[a-h][1-8]=[QRBN][+#]?"

# Regex pattern for pawn move with check, mate, or punctuation
PAWN_SPECIAL_PATTERN: Final[str] = r"[a-h][1-8][+#]"

# Regex pattern for bare pawn moves
BARE_PAWN_PATTERN: Final[str] = r"[a-h][1-8]"

# Full SAN move pattern
SAN_MOVE_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?:{CASTLING_PATTERN}|{PIECE_MOVE_PATTERN}|{PAWN_CAPTURE_PATTERN}|{PAWN_PROMOTION_PATTERN}|{PAWN_SPECIAL_PATTERN})$"
)

# Valid chess squares
SQUARE_RE: Final[re.Pattern[str]] = re.compile(r"\b[a-h][1-8]\b")

# Piece names mapping to canonical singular names
PIECE_NAME_MAP: Final[dict[str, str]] = {
    "pawn": "pawn",
    "pawns": "pawn",
    "knight": "knight",
    "knights": "knight",
    "bishop": "bishop",
    "bishops": "bishop",
    "rook": "rook",
    "rooks": "rook",
    "queen": "queen",
    "queens": "queen",
    "king": "king",
    "kings": "king",
}

# Regex to find piece followed by square (e.g. "the knight on e5", "bishop at c4")
PIECE_THEN_SQUARE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:the\s+|a\s+|an\s+)?(?:white\s+|black\s+)?(pawn|knight|bishop|rook|queen|king)s?\s+"
    r"(?:on|at)\s+(?:the\s+)?(?:square\s+)?([a-h][1-8])\b",
    re.IGNORECASE,
)

# Regex to find square followed by piece (e.g. "the e5 knight", "c4 bishop", "e5-knight")
SQUARE_THEN_PIECE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:the\s+|a\s+|an\s+)?(?:white\s+|black\s+)?([a-h][1-8])\s+"
    r"(pawn|knight|bishop|rook|queen|king)s?\b",
    re.IGNORECASE,
)

# Words indicating a square context rather than a pawn move
SQUARE_CONTEXT_BEFORE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"\b(?:"
    r"on|onto|at|from|into|towards?|in|of|via|around|near|through|across|"
    r"the|an?|this|that|its|"
    r"squares?|outposts?|pawn\s+in|"
    r"reach(?:es|ed)?|occup(?:y|ies|ied)|"
    r"aim(?:s|ed|ing)?\s+at|target(?:s|ed|ing)?|"
    r"weakens?|weakened|weakening|"
    r"guards?|guarded|guarding|"
    r"surrenders?|surrendered|surrendering|"
    r"hits?|hitting|"
    r"abandons?|abandoned|abandoning|"
    r"blocks?|blocked|blocking|"
    r"eyes?|eyeing|"
    r"covers?|covered|covering|"
    r"supports?|supported|supporting|"
    r"pressures?|pressured|pressuring|"
    r"control(?:s|led|ling)?|"
    r"attack(?:s|ed|ing)?|"
    r"defend(?:s|ed|ing)?|"
    r"protect(?:s|ed|ing)?|"
    r"vacat(?:es|ed|ing)|"
    r"head(?:s|ed|ing)?\s+for|"
    r"man(?:eu|oeu)ver(?:s|ed|ing)?\s+(?:a\s+)?(?:pawn\s+)?to|"
    r"way\s+to\s+get\s+(?:a\s+)?(?:pawn\s+)?to"
    r")\s+|"
    r"(?<!\bpawn )(?<!pawns )\bto\s+|"
    r"\b(?:pawn|pawns)\s+|"
    r"\b(?:knight|bishop|rook|queen|king)s?\s+(?:to\s+)?|"
    r"\b[a-h][1-8]\s*(?:,\s*|\s+and\s+|\s+or\s+)"
    r")$",
    re.IGNORECASE,
)

SQUARE_CONTEXT_AFTER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*,?\s*(?:"
    r"(?:square|squares|pawn|knight|bishop|rook|queen|king)s?\b|"
    r"outposts?\b|"
    r"diagonals?\b|"
    r"weakness(?:es)?\b|"
    r"holes?\b|"
    r"blockade\b|"
    r"(?:is|are)\s+weak\b|"
    r"(?:(?:is|are)\s+)?under\s+control\b|"
    r"(?:[a-h][1-8]\s*,?\s*)?(?:and|or)\s+[a-h][1-8]\s+(?:square|squares|outposts?|pawn|pawns|diagonals?|weakness(?:es)?|(?:is|are)\s+weak)\b"
    r")",
    re.IGNORECASE,
)

# Tone phrases used for verdict tone contradiction checks
POSITIVE_TONE_PHRASES: Final[list[str]] = [
    "great move",
    "excellent move",
    "excellent",
    "brilliant move",
    "brilliant",
    "best move",
    "good move",
    "strong move",
    "winning move",
    "accurate move",
    "accurate",
    "well played",
    "nice move",
    "fine move",
    "sharp move",
    "sound move",
]

NEGATIVE_TONE_PHRASES: Final[list[str]] = [
    "fatal mistake",
    "fatal blunder",
    "throws away",
    "terrible move",
    "terrible",
    "bad move",
    "weak move",
    "poor move",
    "horrible move",
    "blunder",
    "blunders",
    "blundered",
    "mistake",
    "mistakes",
    "inaccuracy",
    "inaccuracies",
    "inaccurate",
    "error",
    "errors",
    "inferior",
    "suboptimal",
    "questionable",
    "dubious",
    "losing",
    "loses",
    "ruins",
]

NEUTRAL_TONE_PHRASES: Final[list[str]] = [
    "equal",
    "equality",
    "balanced",
    "drawn",
    "drawish",
    "winning",
]


def clean_move_token(token: str) -> str:
    """Clean punctuation and move numbers from a token.

    Remove leading move numbers like 14. or 14...
    Remove leading and trailing brackets and punctuation.
    Normalize numeric castling notation to uppercase O-O or O-O-O.
    Preserve check and mate characters at the end of the move.
    """
    token = token.strip()
    # Remove leading move numbers
    token = re.sub(r"^\d+\.+", "", token)
    # Strip leading enclosing marks
    token = token.lstrip("([{\"'")

    # Strip trailing punctuation except check and mate
    while token and token[-1] in ".,;:!?'\")]}":
        token = token[:-1]

    # Normalize numeric castling notation to standard SAN O-O / O-O-O
    token = re.sub(r"^0-0-0(?=[+#]?$)", "O-O-O", token)
    token = re.sub(r"^0-0(?=[+#]?$)", "O-O", token)

    return token


def parse_moves(text: str) -> list[str]:
    """Find move-like tokens in text.

    If tokens match standard algebraic notation, collect them.
    Do not classify plain English words as moves.
    Do not classify board square references as pawn moves.
    Return an ordered list of unique SAN move strings.
    """
    moves: list[str] = []
    seen: set[str] = set()

    # Split text into tokens with their start and end positions
    token_matches = list(re.finditer(r"\S+", text))

    for idx, match in enumerate(token_matches):
        raw_token = match.group(0)
        cleaned = clean_move_token(raw_token)
        if not cleaned:
            continue

        # Check standard unambiguous SAN patterns
        if SAN_MOVE_RE.match(cleaned):
            if cleaned not in seen:
                seen.add(cleaned)
                moves.append(cleaned)
            continue

        # Check bare pawn moves (e.g. e4, d5)
        if re.match(r"^[a-h][1-8]$", cleaned):
            start_pos = match.start()
            end_pos = match.end()
            text_before = text[:start_pos]
            text_after = text[end_pos:]

            # If square context precedes or follows, reject as a move
            if SQUARE_CONTEXT_BEFORE_RE.search(text_before):
                continue
            if SQUARE_CONTEXT_AFTER_RE.search(text_after):
                continue

            # Check if move context precedes or if it stands independently
            if cleaned not in seen:
                seen.add(cleaned)
                moves.append(cleaned)

    return moves


def parse_squares(text: str) -> list[str]:
    """Find valid chess squares in text.

    Return an ordered list of unique square names.
    """
    squares: list[str] = []
    seen: set[str] = set()
    for match in SQUARE_RE.finditer(text.lower()):
        sq = match.group(0)
        if sq not in seen:
            seen.add(sq)
            squares.append(sq)
    return squares


def parse_piece_square_pairs(text: str) -> list[tuple[str, str]]:
    """Find piece and square pairs in text.

    Match expressions like 'knight on e5' or 'e5 knight'.
    Return an ordered list of (piece_type, square) tuples.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # Search piece followed by square
    for match in PIECE_THEN_SQUARE_RE.finditer(text):
        piece_raw = match.group(1).lower()
        sq = match.group(2).lower()
        piece = PIECE_NAME_MAP.get(piece_raw, piece_raw)
        pair = (piece, sq)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    # Search square followed by piece
    for match in SQUARE_THEN_PIECE_RE.finditer(text):
        sq = match.group(1).lower()
        piece_raw = match.group(2).lower()
        piece = PIECE_NAME_MAP.get(piece_raw, piece_raw)
        pair = (piece, sq)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return pairs


def _match_phrase_list(text_lower: str, phrases: list[str]) -> list[str]:
    """Find all matching phrases from a candidate list in lowercase text."""
    found: list[str] = []
    for phrase in phrases:
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, text_lower):
            found.append(phrase)
    return found


def find_tone_words(text: str) -> dict[str, list[str]]:
    """Find positive, negative, and neutral tone words in text.

    Return a dictionary with positive, negative, and neutral matches.
    """
    text_lower = text.lower()
    return {
        "positive": _match_phrase_list(text_lower, POSITIVE_TONE_PHRASES),
        "negative": _match_phrase_list(text_lower, NEGATIVE_TONE_PHRASES),
        "neutral": _match_phrase_list(text_lower, NEUTRAL_TONE_PHRASES),
    }


def parse_verdict_words(text: str) -> list[str]:
    """Find all verdict and tone words in text.

    Collect positive, negative, and neutral tone words.
    Return an ordered list of unique verdict terms found in text.
    """
    tone_dict = find_tone_words(text)
    combined: list[str] = []
    seen: set[str] = set()
    for category in ("positive", "negative", "neutral"):
        for word in tone_dict[category]:
            if word not in seen:
                seen.add(word)
                combined.append(word)
    return combined
