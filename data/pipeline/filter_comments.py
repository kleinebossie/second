"""Comment filtering pipeline for chess training data.

This module cleans raw chess comments and removes low quality text.
It evaluates seven deterministic rejection filters in strict order.
It splits comments into short and long detail tiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

import chess
from langdetect import DetectorFactory, detect

from rewards.checks import (
    check_legal_moves,
    check_piece_on_square,
    check_verdict_tone,
)
from rewards.parser import parse_moves, parse_squares

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Regular expression to remove PGN engine and board tags
PGN_TAG_RE: Final[re.Pattern[str]] = re.compile(r"\[%[a-zA-Z]+[^\]]*\]")

# Regular expression for leading move numbers and move prefixes
LEADING_MOVE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*\d+\.+(?:\s*[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)?\s*",
    re.IGNORECASE,
)

# Regular expression for standalone annotation glyphs
STANDALONE_ANNOTATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|\s+)[!?]{1,2}(?=\s+|$)"
)

# URL and link patterns
LINK_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|io|edu|gov|chess)\b)",
    re.IGNORECASE,
)

# Username mention pattern
HANDLE_RE: Final[re.Pattern[str]] = re.compile(r"@\w+")

# Offensive and profane terms list
OFFENSIVE_WORDS: Final[set[str]] = {
    "fuck",
    "fucking",
    "fucked",
    "fucker",
    "shit",
    "bullshit",
    "shitty",
    "bitch",
    "cunt",
    "asshole",
    "assholes",
    "nigger",
    "nigga",
    "faggot",
    "retard",
    "bastard",
    "dick",
    "pussy",
    "cock",
}

# Standard chess vocabulary
CHESS_TERMS: Final[set[str]] = {
    "check",
    "checkmate",
    "mate",
    "castle",
    "castling",
    "castled",
    "fork",
    "forked",
    "forking",
    "pin",
    "pinned",
    "pinning",
    "skewer",
    "skewered",
    "capture",
    "captured",
    "captures",
    "capturing",
    "attack",
    "attacks",
    "attacking",
    "attacked",
    "attacker",
    "defend",
    "defends",
    "defending",
    "defended",
    "defender",
    "defense",
    "defence",
    "threat",
    "threats",
    "threaten",
    "threatens",
    "threatening",
    "opening",
    "endgame",
    "middlegame",
    "center",
    "centre",
    "file",
    "rank",
    "diagonal",
    "exchange",
    "sacrifice",
    "sacrificed",
    "sacrifices",
    "sacrificing",
    "tempo",
    "tempi",
    "gambit",
    "tactic",
    "tactics",
    "tactical",
    "blunder",
    "blundered",
    "blunders",
    "mistake",
    "mistakes",
    "inaccuracy",
    "inaccuracies",
    "advantage",
    "winning",
    "losing",
    "draw",
    "equal",
    "equality",
    "development",
    "develop",
    "develops",
    "developing",
    "space",
    "piece",
    "pieces",
    "initiative",
    "counterplay",
    "compensation",
    "zugzwang",
    "fianchetto",
    "outpost",
}

PIECE_NAMES: Final[set[str]] = {
    "pawn",
    "pawns",
    "knight",
    "knights",
    "bishop",
    "bishops",
    "rook",
    "rooks",
    "queen",
    "queens",
    "king",
    "kings",
}


@dataclass
class FilterStats:
    """Statistics tracker for comment filtering stage.

    Count total input items and removals per filter reason.
    Count total kept items across short and long detail tiers.
    """

    total_input: int = 0
    cleaned_empty_or_short: int = 0
    not_english: int = 0
    no_chess_content: int = 0
    illegal_move_or_wrong_piece: int = 0
    tone_disagrees_with_verdict: int = 0
    links_usernames_offensive: int = 0
    duplicate: int = 0
    kept: int = 0
    kept_short: int = 0
    kept_long: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert counters to a standard dictionary."""
        return {
            "total_input": self.total_input,
            "cleaned_empty_or_short": self.cleaned_empty_or_short,
            "not_english": self.not_english,
            "no_chess_content": self.no_chess_content,
            "illegal_move_or_wrong_piece": self.illegal_move_or_wrong_piece,
            "tone_disagrees_with_verdict": self.tone_disagrees_with_verdict,
            "links_usernames_offensive": self.links_usernames_offensive,
            "duplicate": self.duplicate,
            "kept": self.kept,
            "kept_short": self.kept_short,
            "kept_long": self.kept_long,
        }


def clean_comment(text: str) -> str:
    """Clean PGN annotations, leading move numbers, and extra spacing.

    Remove PGN tags like [%eval 0.2] and [%clk 0:05].
    Remove move numbers at the start of text.
    Remove standalone annotation symbols.
    Normalize internal whitespace.
    """
    # 1. Strip PGN tags
    cleaned = PGN_TAG_RE.sub(" ", text)

    # 2. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 3. Remove leading move number prefixes
    cleaned = LEADING_MOVE_PREFIX_RE.sub("", cleaned).strip()

    # 4. Remove standalone annotation symbols
    cleaned = STANDALONE_ANNOTATION_RE.sub(" ", cleaned)

    # 5. Final whitespace normalization
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def has_chess_content(text: str) -> bool:
    """Check whether text mentions pieces, squares, moves, or chess terms.

    Return true if any piece name is present.
    Return true if any board square or SAN move is present.
    Return true if any standard chess term is present.
    """
    words = [w.lower().strip(".,;:!?()[]\"'") for w in text.split()]
    for w in words:
        if w in PIECE_NAMES:
            return True
        if w in CHESS_TERMS:
            return True

    # Check for square references
    if parse_squares(text):
        return True

    # Check for SAN moves
    if parse_moves(text):
        return True

    # Check for multi-word chess terms
    text_lower = text.lower()
    for phrase in ("pawn structure", "passed pawn", "isolated pawn", "doubled pawn"):
        if phrase in text_lower:
            return True

    return False


def has_offensive_content(text: str) -> bool:
    """Check whether text contains profanity or offensive terms.

    Return true if any word matches the offensive vocabulary.
    """
    text_lower = text.lower()
    for word in OFFENSIVE_WORDS:
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, text_lower):
            return True
    return False


def compute_fingerprint(text: str) -> str:
    """Compute a normalized alphanumeric string for deduplication."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def compute_shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """Compute n-gram word shingles from cleaned text.

    Extract lowercase alphanumeric words.
    If text has fewer than n words, return a single tuple of all words.
    Return set of n-gram word tuples.
    """
    words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return set()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def is_near_duplicate(
    cleaned: str,
    seen_fingerprints: set[str],
    seen_shingles: list[set[tuple[str, ...]]] | None = None,
    threshold: float = 0.80,
) -> bool:
    """Check if text is an exact or near duplicate of seen comments.

    Check exact match with normalized alphanumeric fingerprint.
    If seen_shingles is provided, check Jaccard similarity on 5-gram shingles.
    Return true if exact match or Jaccard similarity meets threshold.
    """
    fingerprint = compute_fingerprint(cleaned)
    if fingerprint in seen_fingerprints:
        return True

    if seen_shingles is not None:
        shingles = compute_shingles(cleaned, n=5)
        if not shingles:
            return False
        for seen in seen_shingles:
            intersection = len(shingles & seen)
            union = len(shingles | seen)
            if union > 0 and (intersection / union) >= threshold:
                return True

    return False


def truncate_to_sentences(
    text: str,
    min_sentences: int = 2,
    max_sentences: int = 4,
    max_words: int = 80,
) -> str:
    """Cut long text down to designated sentence and word bounds.

    Split text by sentence punctuation.
    Select between two and four sentences.
    Keep total word count under the maximum limit.
    """
    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
    ]
    if not raw_sentences:
        return text

    selected: list[str] = []
    total_words = 0

    for s in raw_sentences:
        s_words = len(s.split())
        if selected and (total_words + s_words > max_words):
            if len(selected) >= min_sentences:
                break
        selected.append(s)
        total_words += s_words
        if len(selected) >= max_sentences:
            break

    result = " ".join(selected)
    words = result.split()
    if len(words) > max_words:
        result = " ".join(words[:max_words])
    return result


def filter_comment(
    text: str,
    fen: str,
    move_uci: str | None = None,
    verdict: str = "Good",
    facts: list[str] | None = None,
    seen_fingerprints: set[str] | None = None,
    seen_shingles: list[set[tuple[str, ...]]] | None = None,
    stats: FilterStats | None = None,
) -> tuple[bool, str | None, str]:
    """Filter a single chess comment according to Phase 2.3 rules.

    Evaluate filters in deterministic sequence:
    1. cleaned_empty_or_short (< 8 words after cleaning)
    2. not_english (language detection)
    3. no_chess_content (missing chess entities)
    4. illegal_move_or_wrong_piece (Phase 3 checks)
    5. tone_disagrees_with_verdict (Phase 3 tone check)
    6. links_usernames_offensive (links, handles, or profanity)
    7. duplicate (exact or near-duplicate)

    Return a tuple of (kept_flag, reject_reason, cleaned_text).
    """
    if stats:
        stats.total_input += 1

    # 1. Clean comment and check length
    cleaned = clean_comment(text)
    words = cleaned.split()
    if len(words) < 8:
        if stats:
            stats.cleaned_empty_or_short += 1
        return False, "cleaned_empty_or_short", cleaned

    # 2. Check English language
    try:
        detected_lang = detect(cleaned)
        if detected_lang != "en":
            if stats:
                stats.not_english += 1
            return False, "not_english", cleaned
    except Exception:
        if stats:
            stats.not_english += 1
        return False, "not_english", cleaned

    # 3. Check chess content
    if not has_chess_content(cleaned):
        if stats:
            stats.no_chess_content += 1
        return False, "no_chess_content", cleaned

    # 4. Check illegal moves and wrong piece on square
    legal_res = check_legal_moves(
        text=cleaned,
        fen=fen,
        move_uci=move_uci,
        facts=facts,
    )
    if not legal_res.passed:
        if stats:
            stats.illegal_move_or_wrong_piece += 1
        return False, "illegal_move_or_wrong_piece", cleaned

    piece_res = check_piece_on_square(
        text=cleaned,
        fen=fen,
        move_uci=move_uci,
    )
    if not piece_res.passed:
        if stats:
            stats.illegal_move_or_wrong_piece += 1
        return False, "illegal_move_or_wrong_piece", cleaned

    # 5. Check tone consistency with verdict
    tone_res = check_verdict_tone(text=cleaned, verdict=verdict)
    if not tone_res.passed:
        if stats:
            stats.tone_disagrees_with_verdict += 1
        return False, "tone_disagrees_with_verdict", cleaned

    # 6. Check links, usernames, and offensive language
    if LINK_RE.search(cleaned) or HANDLE_RE.search(cleaned) or has_offensive_content(cleaned):
        if stats:
            stats.links_usernames_offensive += 1
        return False, "links_usernames_offensive", cleaned

    # 7. Check duplicates
    if seen_fingerprints is not None:
        if is_near_duplicate(
            cleaned=cleaned,
            seen_fingerprints=seen_fingerprints,
            seen_shingles=seen_shingles,
        ):
            if stats:
                stats.duplicate += 1
            return False, "duplicate", cleaned
        seen_fingerprints.add(compute_fingerprint(cleaned))
        if seen_shingles is not None:
            seen_shingles.append(compute_shingles(cleaned, n=5))

    if stats:
        stats.kept += 1

    return True, None, cleaned


def pre_filter_comment(
    text: str,
    fen: str,
    move_uci: str | None = None,
    seen_fingerprints: set[str] | None = None,
    seen_shingles: list[set[tuple[str, ...]]] | None = None,
    stats: FilterStats | None = None,
) -> tuple[bool, str | None, str]:
    """Pre-filter a raw comment before engine analysis.

    Evaluate deterministic filters without engine evaluation.
    Check comment length, English language, chess content, and board validity.
    Check piece on square, offensive words, and duplicate text.
    Return a tuple of kept status, rejection reason, and cleaned text.
    """
    if stats:
        stats.total_input += 1

    # 1. Clean comment and check length
    cleaned = clean_comment(text)
    words = cleaned.split()
    if len(words) < 8:
        if stats:
            stats.cleaned_empty_or_short += 1
        return False, "cleaned_empty_or_short", cleaned

    # 2. Check English language
    try:
        detected_lang = detect(cleaned)
        if detected_lang != "en":
            if stats:
                stats.not_english += 1
            return False, "not_english", cleaned
    except Exception:
        if stats:
            stats.not_english += 1
        return False, "not_english", cleaned

    # 3. Check chess content
    if not has_chess_content(cleaned):
        if stats:
            stats.no_chess_content += 1
        return False, "no_chess_content", cleaned

    # 4. Check board validity, move legality, and piece on square
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            if stats:
                stats.illegal_move_or_wrong_piece += 1
            return False, "illegal_move_or_wrong_piece", cleaned
        if move_uci:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                if stats:
                    stats.illegal_move_or_wrong_piece += 1
                return False, "illegal_move_or_wrong_piece", cleaned
    except Exception:
        if stats:
            stats.illegal_move_or_wrong_piece += 1
        return False, "illegal_move_or_wrong_piece", cleaned

    piece_res = check_piece_on_square(
        text=cleaned,
        fen=fen,
        move_uci=move_uci,
    )
    if not piece_res.passed:
        if stats:
            stats.illegal_move_or_wrong_piece += 1
        return False, "illegal_move_or_wrong_piece", cleaned

    # 5. Check links, usernames, and offensive language
    if LINK_RE.search(cleaned) or HANDLE_RE.search(cleaned) or has_offensive_content(cleaned):
        if stats:
            stats.links_usernames_offensive += 1
        return False, "links_usernames_offensive", cleaned

    # 6. Check duplicates
    if seen_fingerprints is not None:
        if is_near_duplicate(
            cleaned=cleaned,
            seen_fingerprints=seen_fingerprints,
            seen_shingles=seen_shingles,
        ):
            if stats:
                stats.duplicate += 1
            return False, "duplicate", cleaned
        seen_fingerprints.add(compute_fingerprint(cleaned))
        if seen_shingles is not None:
            seen_shingles.append(compute_shingles(cleaned, n=5))

    return True, None, cleaned


def assign_detail_and_truncate(text: str) -> list[dict[str, str]]:
    """Assign detail tiers and truncate long comments.

    If word count is between 8 and 80 words, assign short detail.
    If word count is between 81 and 200 words and prose is good, assign long detail.
    If prose is not good or text exceeds 200 words, cut to short detail.
    Return at most one record to avoid duplicate positions in training data.
    """
    words = text.split()
    word_count = len(words)

    if word_count < 8:
        return []

    if word_count <= 80:
        return [{"text": text, "detail": "short"}]

    if word_count <= 200:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0

        if sentence_count >= 3 and avg_sentence_len <= 25.0:
            return [{"text": text, "detail": "long"}]

        short_version = truncate_to_sentences(
            text, min_sentences=2, max_sentences=4, max_words=80
        )
        words_short = short_version.split()
        if len(words_short) > 80:
            short_version = " ".join(words_short[:80])
        elif len(words_short) < 8:
            short_version = " ".join(words[:80])
        return [{"text": short_version, "detail": "short"}]

    # When word count exceeds 200 words
    short_version = truncate_to_sentences(
        text, min_sentences=2, max_sentences=4, max_words=80
    )
    words_short = short_version.split()
    if len(words_short) > 80:
        short_version = " ".join(words_short[:80])
    elif len(words_short) < 8:
        short_version = " ".join(words[:80])
    return [{"text": short_version, "detail": "short"}]
