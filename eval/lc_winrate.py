"""Length-controlled win rate module based on AlpacaEval 2.0.

This module fits a logistic regression model over pairwise match outcomes.
It isolates explanation quality from response verbosity.
It controls for word count differentials between candidate and opponent outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LCWinRateResult:
    """Length-controlled win rate evaluation result."""

    total_pairs: int
    raw_win_rate: float
    lc_win_rate: float
    beta_model: float
    beta_len: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


def sigmoid(z: float | np.ndarray) -> Any:
    """Compute standard sigmoid function with numerical clipping."""
    z_clipped = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def fit_logistic_length_control(
    word_diffs: Sequence[float],
    outcomes: Sequence[float],
    l2_reg: float = 1e-4,
    max_iters: int = 25,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Fit two-parameter logistic regression using Newton-Raphson.

    word_diffs contains words(A) - words(B).
    outcomes contains pairwise match labels where 1.0 is A win, 0.0 is B win, 0.5 is tie.
    Return tuple of (beta_model, beta_len).
    """
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0

    y = np.asarray(outcomes, dtype=np.float64)
    diffs = np.asarray(word_diffs, dtype=np.float64)
    x1 = np.tanh(diffs / 100.0)

    # Design matrix: column 0 is bias (beta_model), column 1 is tanh length feature
    X = np.column_stack([np.ones(n, dtype=np.float64), x1])

    # Initial weights: start from empirical log-odds of mean outcome
    mean_y = float(np.clip(np.mean(y), 0.01, 0.99))
    beta = np.array([math.log(mean_y / (1.0 - mean_y)), 0.0], dtype=np.float64)

    for _ in range(max_iters):
        logits = X @ beta
        p = sigmoid(logits)
        p = np.clip(p, 1e-7, 1.0 - 1e-7)

        # Gradient with L2 regularization
        grad = (X.T @ (p - y)) / n + l2_reg * beta

        # Hessian: X^T * W * X / n + l2_reg * I
        w = p * (1.0 - p)
        H = (X.T * w) @ X / n + np.eye(2, dtype=np.float64) * l2_reg

        # Newton step
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = grad * 0.1

        beta -= step
        if float(np.max(np.abs(step))) < tol:
            break

    return float(beta[0]), float(beta[1])


def compute_lc_winrate(
    pairs: Sequence[dict[str, Any]],
    l2_reg: float = 1e-4,
) -> LCWinRateResult:
    """Compute length-controlled win rate from a collection of evaluated pairs.

    Each pair must contain:
    - 'winner': 'candidate', 'opponent', or 'tie'
    - 'candidate_text' or 'candidate_words'
    - 'opponent_text' or 'opponent_words'
    Return an LCWinRateResult instance.
    """
    if not pairs:
        return LCWinRateResult(
            total_pairs=0,
            raw_win_rate=0.0,
            lc_win_rate=0.0,
            beta_model=0.0,
            beta_len=0.0,
        )

    word_diffs: list[float] = []
    outcomes: list[float] = []
    cand_wins = 0

    for p in pairs:
        winner = str(p.get("winner", "tie")).lower()
        if winner in ("candidate", "a", "1"):
            y = 1.0
            cand_wins += 1
        elif winner in ("opponent", "b", "2"):
            y = 0.0
        else:
            y = 0.5

        if "candidate_words" in p and "opponent_words" in p:
            w_a = float(p["candidate_words"])
            w_b = float(p["opponent_words"])
        else:
            t_a = str(p.get("candidate_text") or p.get("candidate") or "")
            t_b = str(p.get("opponent_text") or p.get("opponent") or "")
            w_a = float(len(t_a.split()))
            w_b = float(len(t_b.split()))

        word_diffs.append(w_a - w_b)
        outcomes.append(y)

    total = len(pairs)
    raw_wr = round(100.0 * cand_wins / total, 2)

    beta_model, beta_len = fit_logistic_length_control(
        word_diffs=word_diffs,
        outcomes=outcomes,
        l2_reg=l2_reg,
    )

    # LC-WR sets word differential to zero: P(A > B | word_diff = 0) = sigmoid(beta_model)
    lc_wr = round(float(sigmoid(beta_model)) * 100.0, 2)

    return LCWinRateResult(
        total_pairs=total,
        raw_win_rate=raw_wr,
        lc_win_rate=lc_wr,
        beta_model=round(beta_model, 4),
        beta_len=round(beta_len, 4),
    )
