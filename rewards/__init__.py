"""Rewards package for chess explanation evaluation.

This package provides rule checks and scoring functions for reinforcement learning.
It checks move legality, grounding, piece squares, tone, formatting, and jargon.
"""

from __future__ import annotations

from .api import (
    PromptInfo,
    grpo_reward_fn,
    parse_prompt,
    score_completion,
    verdict_accuracy,
)
from .checks import (
    CheckResult,
    check_beginner_jargon,
    check_coach_voice,
    check_format,
    check_grounded_moves,
    check_legal_moves,
    check_opening_claims,
    check_piece_on_square,
    check_positional_fact_grounding,
    check_relational_claims,
    check_vague_boilerplate,
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
    "PromptInfo",
    "RewardResult",
    "check_beginner_jargon",
    "check_coach_voice",
    "check_format",
    "check_grounded_moves",
    "check_legal_moves",
    "check_opening_claims",
    "check_piece_on_square",
    "check_positional_fact_grounding",
    "check_relational_claims",
    "check_vague_boilerplate",
    "check_verdict_tone",
    "compute_reward",
    "find_tone_words",
    "grpo_reward_fn",
    "parse_moves",
    "parse_piece_square_pairs",
    "parse_prompt",
    "parse_squares",
    "parse_verdict_words",
    "score_better_move",
    "score_completion",
    "score_key_point",
    "score_length",
    "score_level_fit",
    "score_no_repeats",
    "score_not_copied",
    "verdict_accuracy",
]
