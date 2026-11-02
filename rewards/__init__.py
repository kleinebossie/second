"""Rewards package for chess explanation evaluation.

This package provides rule checks and scoring functions for reinforcement learning.
It checks move legality, grounding, piece squares, tone, formatting, and jargon.
"""

from __future__ import annotations

from .checks import (
    CheckResult,
    check_beginner_jargon,
    check_format,
    check_grounded_moves,
    check_legal_moves,
    check_piece_on_square,
    check_verdict_tone,
)
from .parser import (
    find_tone_words,
    parse_moves,
    parse_piece_square_pairs,
    parse_squares,
    parse_verdict_words,
)
from .scorer import (
    RewardResult,
    compute_reward,
    score_better_move,
    score_key_point,
    score_length,
    score_level_fit,
    score_no_repeats,
    score_not_copied,
)

__all__ = [
    "CheckResult",
    "RewardResult",
    "check_beginner_jargon",
    "check_format",
    "check_grounded_moves",
    "check_legal_moves",
    "check_piece_on_square",
    "check_verdict_tone",
    "compute_reward",
    "find_tone_words",
    "parse_moves",
    "parse_piece_square_pairs",
    "parse_squares",
    "parse_verdict_words",
    "score_better_move",
    "score_key_point",
    "score_length",
    "score_level_fit",
    "score_no_repeats",
    "score_not_copied",
]
