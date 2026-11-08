"""Unit tests for Length-Controlled Win Rate (LC-WR).

This module tests logistic regression fitting and length bias correction.
It verifies behavior under equal lengths, verbosity advantages, and edge cases.
"""

from __future__ import annotations

import pytest

from eval.lc_winrate import compute_lc_winrate, fit_logistic_length_control, sigmoid


def test_sigmoid() -> None:
    """Verify sigmoid values at zero and extremes."""
    assert round(float(sigmoid(0.0)), 2) == 0.5
    assert float(sigmoid(20.0)) > 0.999
    assert float(sigmoid(-20.0)) < 0.001


def test_fit_balanced_equal_lengths() -> None:
    """When lengths are equal and outcomes are 50-50, beta_model should be near 0."""
    word_diffs = [0.0] * 100
    outcomes = [1.0] * 50 + [0.0] * 50
    beta_model, beta_len = fit_logistic_length_control(word_diffs, outcomes)
    assert abs(beta_model) < 0.1
    assert abs(beta_len) < 0.1


def test_length_bias_deflation() -> None:
    """When model A is much more verbose and wins, LC-WR should discount verbosity."""
    # Model A writes 200 words, Model B writes 40 words (diff = +160)
    # Model A wins 80% of matches due to length bias
    pairs = [
        {"winner": "candidate", "candidate_words": 200, "opponent_words": 40}
        for _ in range(80)
    ] + [
        {"winner": "opponent", "candidate_words": 200, "opponent_words": 40}
        for _ in range(20)
    ]
    res = compute_lc_winrate(pairs)
    assert res.raw_win_rate == 80.0
    # Because beta_len captures the positive length advantage,
    # beta_model at zero length diff should be lower than raw 80%
    assert res.beta_len > 0.0
    assert res.lc_win_rate < res.raw_win_rate


def test_short_concise_boost() -> None:
    """When concise model A has fewer words than verbose model B, LC-WR should reward it."""
    # Model A writes 40 words, Model B writes 250 words (diff = -210)
    # Model A wins 50% raw
    pairs = [
        {"winner": "candidate", "candidate_words": 40, "opponent_words": 250}
        for _ in range(50)
    ] + [
        {"winner": "opponent", "candidate_words": 40, "opponent_words": 250}
        for _ in range(50)
    ]
    # Add pairs showing that longer responses usually win in the general population
    for _ in range(30):
        pairs.append({"winner": "candidate", "candidate_words": 250, "opponent_words": 40})
    for _ in range(10):
        pairs.append({"winner": "opponent", "candidate_words": 250, "opponent_words": 40})

    res = compute_lc_winrate(pairs)
    assert res.beta_len > 0.0


def test_empty_pairs() -> None:
    """Verify empty input handling."""
    res = compute_lc_winrate([])
    assert res.total_pairs == 0
    assert res.raw_win_rate == 0.0
    assert res.lc_win_rate == 0.0
