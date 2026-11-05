"""Soft scoring and composite reward calculation for chess explanations.

This module scores explanations across key points, move suggestions, length, and style.
It combines hard checks and soft scores into a single reinforcement learning reward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import chess

from .checks import (
    CheckResult,
    check_beginner_jargon,
    check_format,
    check_grounded_moves,
    check_legal_moves,
    check_piece_on_square,
    check_verdict_tone,
    is_bad_verdict,
    strip_san_suffix,
)
from .parser import (
    clean_move_token,
    parse_moves,
    parse_piece_square_pairs,
    parse_squares,
)

# Jargon words prohibited for beginner level fit
CHESS_JARGON_WORDS: Final[list[str]] = [
    "prophylaxis",
    "zwischenzug",
    "outpost",
    "minority attack",
    "centipawn",
    "zugzwang",
    "fianchetto",
    "desperado",
    "initiative",
    "tempo",
    "compensation",
    "hypermodern",
    "opposition",
    "triangulation",
    "liquidation",
    "overloading",
    "deflection",
    "decoy",
    "intermezzo",
    "en prise",
    "luft",
    "priyome",
    "tabiya",
]

# Soft score weights summing to 1.0
WEIGHT_KEY_POINT: Final[float] = 0.35
WEIGHT_BETTER_MOVE: Final[float] = 0.15
WEIGHT_LENGTH: Final[float] = 0.15
WEIGHT_LEVEL_FIT: Final[float] = 0.15
WEIGHT_NOT_COPIED: Final[float] = 0.10
WEIGHT_NO_REPEATS: Final[float] = 0.10


@dataclass(frozen=True)
class RewardResult:
    """Composite reward outcome for an explanation.

    The reward field holds the final scalar reward in range -1.0 to 1.0.
    The passed_all_hard field indicates whether all hard checks passed.
    The hard_checks field maps check names to CheckResult instances.
    The soft_scores field maps metric names to scores in range 0.0 to 1.0.
    """

    reward: float
    passed_all_hard: bool
    hard_checks: dict[str, CheckResult]
    soft_scores: dict[str, float]


def _tokenize_words(text: str) -> list[str]:
    """Split text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentence strings."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


# Stopwords excluded when matching fact keywords in explanations
FACT_STOPWORDS: Final[set[str]] = {
    "after",
    "before",
    "white",
    "black",
    "best",
    "reply",
    "move",
    "moves",
    "with",
    "from",
    "into",
    "that",
    "this",
    "there",
    "their",
    "have",
    "has",
    "does",
    "not",
    "phase",
    "material",
    "equal",
    "opening",
    "middlegame",
    "endgame",
    "together",
}

# Problem and consequence keywords for bad move explanations (excludes verdict words)
PROBLEM_KEYWORDS: Final[list[str]] = [
    "threat",
    "threatens",
    "threatening",
    "loses",
    "losing",
    "lost",
    "loss",
    "hangs",
    "hanging",
    "undefended",
    "unprotected",
    "unanswered",
    "fork",
    "forks",
    "forked",
    "pin",
    "pins",
    "pinned",
    "skewer",
    "skewers",
    "skewered",
    "checkmate",
    "mate",
    "trapped",
    "trap",
    "traps",
    "drop",
    "drops",
    "dropped",
    "weak",
    "weaken",
    "weakens",
    "weakening",
    "weakness",
    "expose",
    "exposes",
    "exposed",
    "exposing",
    "vulnerable",
    "dangerous",
    "danger",
    "passive",
    "waste",
    "wastes",
    "wasted",
    "wasting",
    "miss",
    "misses",
    "missed",
    "missing",
    "overlook",
    "overlooks",
    "overlooked",
    "problem",
    "trouble",
    "punish",
    "punishes",
    "punished",
]

# Alternative move indicator phrases
ALT_MOVE_PHRASES: Final[list[str]] = [
    "better was",
    "better is",
    "better move",
    "better option",
    "better choice",
    "would be better",
    "would have been better",
    "might have been better",
    "could have been better",
    "was better",
    "is better",
    "should play",
    "should have played",
    "could play",
    "could have played",
    "instead of",
    "best move",
    "best was",
    "best is",
    "best option",
    "correct move",
    "right move",
    "alternative was",
    "alternative is",
    "alternative move",
]

# Chess piece names excluded from fact keyword matching
PIECE_NAMES: Final[set[str]] = {
    "king",
    "kings",
    "queen",
    "queens",
    "rook",
    "rooks",
    "bishop",
    "bishops",
    "knight",
    "knights",
    "pawn",
    "pawns",
    "piece",
    "pieces",
}

# Generic words excluded from fact keyword matching
GENERIC_WORDS: Final[set[str]] = {
    "time",
    "give",
    "gives",
    "gave",
    "giving",
    "leave",
    "leaves",
    "left",
    "leaving",
    "allow",
    "allows",
    "allowed",
    "allowing",
    "center",
    "centre",
    "central",
    "development",
}

CONTENT_WORDS_STOPLIST: Final[set[str]] = (
    FACT_STOPWORDS | PIECE_NAMES | GENERIC_WORDS
)


def _match_fact(
    fact: str,
    text: str,
    text_lower: str,
    text_words: set[str],
    text_moves: set[str],
    text_squares: set[str],
    text_pairs: set[tuple[str, str]],
) -> bool:
    """Check if explanation text matches a specific fact.

    Do not match generic phase or material facts.
    Match when the text includes a square from the fact.
    Match when the text includes a SAN move from the fact.
    Match when the text includes a piece and square pair from the fact.
    Match when the text includes at least two non-generic content words.
    Do not count piece names alone as content words.
    """
    fact_stripped = fact.strip()
    if fact_stripped.startswith(("Phase:", "Material:")):
        return False

    # Check square anchor
    fact_squares = set(parse_squares(fact))
    if fact_squares and any(sq in text_squares for sq in fact_squares):
        return True

    # Check SAN move anchor
    fact_moves = set(parse_moves(fact))
    fact_move_bases = {strip_san_suffix(m) for m in fact_moves}
    if fact_moves:
        for m in text_moves:
            if m in fact_moves or strip_san_suffix(m) in fact_move_bases:
                return True

    # Check piece and square pair anchor
    fact_pairs = set(parse_piece_square_pairs(fact))
    if fact_pairs and any(pair in text_pairs for pair in fact_pairs):
        return True

    # Check non-generic content words
    fact_words = [
        w for w in _tokenize_words(fact)
        if len(w) > 3 and w not in CONTENT_WORDS_STOPLIST
    ]
    matched_words = sum(
        1 for w in set(fact_words) if w in text_words or w in text_lower
    )
    return matched_words >= 2


def _extract_prompt_field(prompt_text: str, field_prefix: str) -> str:
    """Extract a header field value from a formatted prompt."""
    if not prompt_text:
        return ""
    pattern = rf"^{re.escape(field_prefix)}\s*([^\s.]+)"
    match = re.search(pattern, prompt_text, re.MULTILINE)
    if match:
        return strip_san_suffix(match.group(1).strip())
    return ""


def _names_best_move(text: str, best_move_san: str) -> bool:
    """Return true when the text names the best move SAN."""
    if not best_move_san:
        return False
    base_best = strip_san_suffix(best_move_san)
    named_moves = parse_moves(text)
    named_bases = [strip_san_suffix(m) for m in named_moves]
    return best_move_san in named_moves or base_best in named_bases


def _has_alternative_idea(text_lower: str, text: str, best_move_san: str) -> bool:
    """Return true when the text names the best move or an alternative phrase."""
    if _names_best_move(text, best_move_san):
        return True
    return any(phrase in text_lower for phrase in ALT_MOVE_PHRASES)


def score_key_point(
    text: str,
    verdict: str | None = "Good",
    facts: list[str] | None = None,
    best_move_san: str = "",
) -> float:
    """Score whether explanation captures the essential positional truth.

    When the move is a mistake, score identification of threat and better move.
    When the move is good, score identification of the main positional plan.
    Cap fact matching at two facts so listing all facts gives no extra reward.
    Return a score between 0.0 and 1.0.
    """
    is_bad = is_bad_verdict(verdict)
    text_lower = text.lower()
    text_words = set(_tokenize_words(text))
    text_moves = set(parse_moves(text))
    text_squares = set(parse_squares(text))
    text_pairs = set(parse_piece_square_pairs(text))

    # Count matched facts across non-generic facts (anti-hacking cap at 2 facts)
    matched_facts = 0
    if facts:
        for fact in facts:
            if _match_fact(
                fact,
                text,
                text_lower,
                text_words,
                text_moves,
                text_squares,
                text_pairs,
            ):
                matched_facts += 1

    if is_bad:
        # Check problem identification (threat, lost material, or specific tactical fact)
        has_problem = matched_facts >= 1
        if not has_problem:
            if any(
                (kw in text_words if " " not in kw else kw in text_lower)
                for kw in PROBLEM_KEYWORDS
            ):
                has_problem = True

        # Check alternative move or idea identification
        has_better = _has_alternative_idea(text_lower, text, best_move_san)

        if has_problem and has_better:
            return 1.0
        if has_problem or has_better:
            return 0.5
        return 0.0

    # For good moves, best moves, or position explanations: check alignment with facts
    if not facts:
        return 1.0

    if matched_facts >= 2:
        return 1.0
    if matched_facts == 1:
        return 0.7

    # Check if explanation discusses core positional ideas even when phrasing differs
    positional_keywords = {
        "develop",
        "develops",
        "developing",
        "development",
        "center",
        "centre",
        "central",
        "castle",
        "castles",
        "castling",
        "castled",
        "attack",
        "attacks",
        "attacking",
        "defend",
        "defends",
        "defending",
        "defense",
        "protect",
        "protects",
        "protecting",
        "protected",
        "control",
        "controls",
        "controlling",
        "active",
        "activity",
        "space",
        "pressure",
        "threat",
        "threatens",
        "capture",
        "captures",
        "exchange",
        "trade",
        "trades",
        "open",
        "opens",
        "file",
        "diagonal",
        "rank",
        "king",
        "queen",
        "rook",
        "bishop",
        "knight",
        "pawn",
        "pawns",
        "piece",
        "pieces",
        "endgame",
        "opening",
        "middlegame",
    }
    if len(text_words.intersection(positional_keywords)) >= 2:
        return 0.7

    return 0.4


def score_better_move(
    text: str,
    verdict: str | None = "Good",
    best_move_san: str = "",
) -> float:
    """Score whether explanation identifies the superior move.

    When the verdict is Best, Good, Only move, or missing, return 1.0.
    When the verdict is bad and text names the best move, return 1.0.
    When the verdict is bad and text mentions an alternative idea, return 0.5.
    Otherwise return 0.0.
    """
    if not is_bad_verdict(verdict):
        return 1.0

    if _names_best_move(text, best_move_san):
        return 1.0

    if any(phrase in text.lower() for phrase in ALT_MOVE_PHRASES):
        return 0.5

    return 0.0


def score_length(text: str, detail: str) -> float:
    """Score whether explanation length conforms to detail target band.

    For short detail, target word band is 25 to 80 words.
    For long detail, target word band is 80 to 200 words.
    Penalize explanations that contain mostly move notation without prose.
    Score smoothly decreases outside the designated band.
    """
    words = text.split()
    count = len(words)
    if count == 0:
        return 0.0

    # Anti-hacking guard: penalize outputs that list moves without prose words
    moves_set = set(parse_moves(text))
    move_tokens = sum(1 for w in words if clean_move_token(w) in moves_set)
    if (move_tokens / count) > 0.5:
        return 0.0

    if detail.lower() == "long":
        if 80 <= count <= 200:
            return 1.0
        if count < 80:
            return max(0.0, count / 80.0)
        return max(0.0, 1.0 - (count - 200) / 100.0)

    # Short detail mode
    if 25 <= count <= 80:
        return 1.0
    if count < 25:
        return max(0.0, count / 25.0)
    return max(0.0, 1.0 - (count - 80) / 40.0)


def score_level_fit(text: str, level: str) -> float:
    """Score whether explanation vocabulary matches player skill level.

    When level is club, return 1.0.
    When level is beginner, require short sentences and penalize jargon.
    """
    if level.lower() != "beginner":
        return 1.0

    sentences = _split_sentences(text)
    if not sentences:
        return 0.0

    # Calculate average sentence length in words
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_len = sum(sentence_lengths) / len(sentence_lengths)

    if avg_len <= 15.0:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (avg_len - 15.0) / 15.0)

    # Check chess jargon words
    text_lower = text.lower()
    has_jargon = any(
        re.search(rf"\b{re.escape(w)}\b", text_lower)
        for w in CHESS_JARGON_WORDS
    )
    jargon_score = 0.0 if has_jargon else 1.0

    return 0.5 * length_score + 0.5 * jargon_score


def score_not_copied(text: str, facts: list[str] | None = None) -> float:
    """Score whether explanation avoids verbatim fact overlap.

    Check both 4-gram overlap and verbatim short fact copying.
    Penalize higher overlap to promote original phrasing.
    """
    if not facts:
        return 1.0

    text_lower = re.sub(r"\s+", " ", text.lower()).strip()
    if not text_lower:
        return 1.0

    # Check verbatim copying of short fact lines (< 4 words)
    short_fact_matches = 0
    for f in facts:
        f_clean = re.sub(r"\s+", " ", f.lower()).strip().rstrip(".")
        if f_clean and f_clean in text_lower:
            short_fact_matches += 1

    text_words = _tokenize_words(text)
    if len(text_words) < 4:
        return 0.0 if short_fact_matches > 0 else 1.0

    # Build 4-grams from text
    text_4grams = set(
        tuple(text_words[i : i + 4]) for i in range(len(text_words) - 3)
    )

    # Build 4-grams from facts
    fact_4grams: set[tuple[str, ...]] = set()
    for f in facts:
        f_words = _tokenize_words(f)
        for i in range(len(f_words) - 3):
            fact_4grams.add(tuple(f_words[i : i + 4]))

    overlap = len(text_4grams.intersection(fact_4grams)) + short_fact_matches
    overlap_rate = overlap / len(text_4grams)

    # Overlap rate 0.0 gives 1.0; overlap rate >= 0.5 gives 0.0
    return max(0.0, min(1.0, 1.0 - (overlap_rate / 0.5)))


def score_no_repeats(text: str) -> float:
    """Score whether explanation avoids repetitive sentences or phrases.

    Penalize duplicated full sentences and repeated 4-gram sequences.
    """
    sentences = _split_sentences(text)
    norm_sentences = [re.sub(r"\s+", " ", s.lower()) for s in sentences]

    sentence_penalty = 0.0
    if len(norm_sentences) != len(set(norm_sentences)):
        sentence_penalty = 0.5

    words = _tokenize_words(text)
    phrase_penalty = 0.0
    if len(words) >= 8:
        four_grams: list[tuple[str, ...]] = [
            tuple(words[i : i + 4]) for i in range(len(words) - 3)
        ]
        if len(four_grams) != len(set(four_grams)):
            phrase_penalty = 0.5

    total_penalty = sentence_penalty + phrase_penalty
    return max(0.0, 1.0 - total_penalty)


def compute_reward(
    text: str,
    fen: str,
    level: str = "club",
    detail: str = "short",
    verdict: str | None = "Good",
    move_uci: str | None = None,
    best_move_san: str = "",
    facts: list[str] | None = None,
    prompt_text: str = "",
) -> RewardResult:
    """Compute the total scalar reward and component diagnostics.

    Evaluate all hard rule checks first.
    If any hard check fails, assign scalar reward of -1.0.
    If all hard checks pass, compute weighted sum of soft scores.
    """
    resolved_best_san = best_move_san or _extract_prompt_field(prompt_text, "Best move:")

    # Combine prompt_text, facts, played move, and best move for grounding
    grounding_parts: list[str] = []
    if prompt_text:
        grounding_parts.append(prompt_text)
    if facts:
        grounding_parts.extend(facts)
    if resolved_best_san:
        grounding_parts.append(f"Best move: {resolved_best_san}.")
    if move_uci and fen:
        try:
            board = chess.Board(fen)
            played_move = chess.Move.from_uci(move_uci)
            if played_move in board.legal_moves:
                grounding_parts.append(f"Move played: {board.san(played_move)}.")
        except ValueError:
            pass
    grounding_source = "\n".join(grounding_parts)

    # 1. Run hard checks
    hard_results: dict[str, CheckResult] = {
        "legal_moves": check_legal_moves(
            text=text,
            fen=fen,
            move_uci=move_uci,
            facts=facts,
        ),
        "grounded_moves": check_grounded_moves(
            text=text,
            prompt_or_facts=grounding_source,
        ),
        "piece_on_square": check_piece_on_square(
            text=text,
            fen=fen,
            move_uci=move_uci,
        ),
        "verdict_tone": check_verdict_tone(
            text=text,
            verdict=verdict,
        ),
        "format": check_format(
            text=text,
            prompt_text=prompt_text,
        ),
        "beginner_jargon": check_beginner_jargon(
            text=text,
            level=level,
        ),
    }

    passed_all_hard = all(res.passed for res in hard_results.values())

    # 2. Run soft scores
    soft_results: dict[str, float] = {
        "key_point": score_key_point(
            text=text,
            verdict=verdict,
            facts=facts,
            best_move_san=resolved_best_san,
        ),
        "better_move": score_better_move(
            text=text,
            verdict=verdict,
            best_move_san=resolved_best_san,
        ),
        "length": score_length(
            text=text,
            detail=detail,
        ),
        "level_fit": score_level_fit(
            text=text,
            level=level,
        ),
        "not_copied": score_not_copied(
            text=text,
            facts=facts,
        ),
        "no_repeats": score_no_repeats(text=text),
    }

    if not passed_all_hard:
        final_reward = -1.0
    else:
        final_reward = (
            WEIGHT_KEY_POINT * soft_results["key_point"]
            + WEIGHT_BETTER_MOVE * soft_results["better_move"]
            + WEIGHT_LENGTH * soft_results["length"]
            + WEIGHT_LEVEL_FIT * soft_results["level_fit"]
            + WEIGHT_NOT_COPIED * soft_results["not_copied"]
            + WEIGHT_NO_REPEATS * soft_results["no_repeats"]
        )
        final_reward = round(max(0.0, min(1.0, final_reward)), 4)

    return RewardResult(
        reward=final_reward,
        passed_all_hard=passed_all_hard,
        hard_checks=hard_results,
        soft_scores=soft_results,
    )
