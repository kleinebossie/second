"""Evaluation package for the Second chess assistant.

This package provides evaluation metrics, baseline generators, and judge harnesses.
It implements Phase 4 evaluation requirements from the Second project plan.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AggregateMetrics",
    "EvalMetrics",
    "GeminiBaselineGenerator",
    "GeminiClient",
    "GeminiJudge",
    "ItemMetric",
    "JudgeCache",
    "OpenAIBaselineGenerator",
    "PairwiseComparison",
    "PairwiseSummary",
    "RubricScore",
    "RubricSummary",
    "RuleBaselineGenerator",
    "TransformersBaselineGenerator",
    "clean_completion_text",
    "count_words",
    "evaluate_completions",
    "evaluate_file",
    "evaluate_item",
    "generate_baseline",
    "generate_markdown_report",
    "is_in_length_band",
    "load_baseline",
    "run_eval",
    "run_judge_eval",
]


def __getattr__(name: str) -> Any:
    """Load evaluation exports on demand.

    Lazy loading avoids runpy circular import warnings during CLI execution.
    Return requested attribute from submodules.
    """
    if name in (
        "AggregateMetrics",
        "EvalMetrics",
        "ItemMetric",
        "clean_completion_text",
        "count_words",
        "evaluate_completions",
        "evaluate_file",
        "evaluate_item",
        "is_in_length_band",
    ):
        from . import metrics

        return getattr(metrics, name)

    if name in (
        "GeminiClient",
        "GeminiJudge",
        "JudgeCache",
        "PairwiseComparison",
        "PairwiseSummary",
        "RubricScore",
        "RubricSummary",
        "run_judge_eval",
    ):
        from . import judge

        return getattr(judge, name)

    if name in (
        "GeminiBaselineGenerator",
        "OpenAIBaselineGenerator",
        "RuleBaselineGenerator",
        "TransformersBaselineGenerator",
        "generate_baseline",
        "load_baseline",
    ):
        from . import baselines

        return getattr(baselines, name)

    if name in ("generate_markdown_report", "run_eval"):
        from . import run_eval

        return getattr(run_eval, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
