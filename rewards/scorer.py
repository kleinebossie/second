"""Soft scoring and composite reward calculation for chess explanations.

This module scores explanations across key points, move suggestions, length, and style.
It combines hard checks and soft scores into a single reinforcement learning reward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from .checks import (
    CheckResult,
    check_beginner_jargon,
    check_format,
    check_grounded_moves,
    check_legal_moves,
    check_piece_on_square,
    check_verdict_tone,
    strip_san_suffix,
)
from .parser import parse_moves

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


def score_key_point(
    text: str,
    verdict: str,
    facts: list[str] | None = None,
    best_move_san: str = "",
) -> float:
    """Score whether explanation captures the essential positional truth.

    When the move is a mistake, score identification of threat and better move.
    When the move is good, score identification of the main positional plan.
    Return a score between 0.0 and 1.0.
    """
    verdict_clean = verdict.strip().capitalize()
    is_bad = verdict_clean in ("Inaccuracy", "Mistake", "Blunder")
    text_lower = text.lower()

    if is_bad:
        # Check problem identification (threat, lost material, capture, or reply)
        has_problem = False
        problem_keywords = [
            "threat",
            "threatens",
            "loses",
            "hangs",
            "hanging",
            "capture",
            "captures",
            "fork",
            "pin",
            "defend",
            "reply",
            "blunder",
            "mistake",
        ]
        if any(kw in text_lower for kw in problem_keywords):
            has_problem = True

        # Check alternative move identification
        has_better = False
        if best_move_san:
            base_best = strip_san_suffix(best_move_san)
            if best_move_san in text or base_best in text:
                has_better = True
        if not has_better:
            alt_keywords = ["better was", "should play", "instead of", "better move"]
            if any(kw in text_lower for kw in alt_keywords):
                has_better = True

        if has_problem and has_better:
            return 1.0
        if has_problem or has_better:
            return 0.5
        return 0.0

    # For good or best moves: check alignment with key facts
    if not facts:
        return 1.0

    # Extract keywords from the first three facts
    matched_facts = 0
    facts_to_check = facts[:3]
    for fact in facts_to_check:
        fact_words = [w for w in _tokenize_words(fact) if len(w) > 3]
        if any(w in text_lower for w in fact_words):
            matched_facts += 1

    if matched_facts >= 2:
        return 1.0
    if matched_facts == 1:
        return 0.7
    return 0.4


def score_better_move(
    text: str,
    verdict: str,
    best_move_san: str = "",
) -> float:
    """Score whether explanation identifies the superior move.

    When the verdict is Best or Good, return 1.0.
    When the verdict is bad and text names the best move, return 1.0.
    When the verdict is bad and text mentions an alternative idea, return 0.5.
    Otherwise return 0.0.
    """
    verdict_clean = verdict.strip().capitalize()
    if verdict_clean in ("Best", "Good", "Only move"):
        return 1.0

    if best_move_san:
        base_best = strip_san_suffix(best_move_san)
        named_moves = parse_moves(text)
        named_bases = [strip_san_suffix(m) for m in named_moves]
        if best_move_san in named_moves or base_best in named_bases:
            return 1.0

    text_lower = text.lower()
    alt_indicators = [
        "better was",
        "better is",
        "instead of",
        "best move",
        "should have played",
        "should play",
    ]
    if any(phrase in text_lower for phrase in alt_indicators):
        return 0.5

    return 0.0


def score_length(text: str, detail: str) -> float:
    """Score whether explanation length conforms to detail target band.

    For short detail, target word band is 25 to 80 words.
    For long detail, target word band is 80 to 200 words.
    Score smoothly decreases outside the designated band.
    """
    words = text.split()
    count = len(words)

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
    """Score whether explanation avoids verbatim fact n-gram overlap.

    Extract 4-grams and 5-grams from explanation and facts.
    Penalize higher overlap to promote original phrasing.
    """
    if not facts:
        return 1.0

    text_words = _tokenize_words(text)
    if len(text_words) < 4:
        return 1.0

    # Build n-grams from text
    text_4grams = set(
        tuple(text_words[i : i + 4]) for i in range(len(text_words) - 3)
    )

    # Build n-grams from facts
    fact_4grams: set[tuple[str, ...]] = set()
    for f in facts:
        f_words = _tokenize_words(f)
        for i in range(len(f_words) - 3):
            fact_4grams.add(tuple(f_words[i : i + 4]))

    if not text_4grams or not fact_4grams:
        return 1.0

    overlap = len(text_4grams.intersection(fact_4grams))
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
    verdict: str = "Good",
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
            prompt_or_facts=prompt_text or (facts or []),
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
            best_move_san=best_move_san,
        ),
        "better_move": score_better_move(
            text=text,
            verdict=verdict,
            best_move_san=best_move_san,
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
