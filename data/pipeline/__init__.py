"""Pipeline modules for the Second chess assistant data pipeline.

This package provides tools to process chess data:
- Stockfish position evaluation and Parquet caching.
- Game-level dataset splitting.
- Reinforcement learning prompt set creation.
- Dataset export and synchronization.
- End-to-end data pipeline orchestration.
"""

from __future__ import annotations

from .analyse import (
    StockfishAnalyzer,
    StockfishCache,
    StockfishPool,
    analyse_records,
    find_stockfish_binary,
)
from .deglue import (
    count_unknown_tokens,
    deglue_comment,
    deglue_token,
    has_too_many_unknown_tokens,
    remove_reader_questions,
)
from .eval_pool import (
    build_eval_pool_from_evaluated,
    generate_eval_pool,
    save_eval_pool,
)
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from .filter_comments import (
        FilterStats,
        assign_detail_and_truncate,
        clean_comment,
        compute_shingles,
        filter_comment,
        is_near_duplicate,
        pre_filter_comment,
    )
    from .synthesize_teacher import (
        TEACHER_SYSTEM_PROMPT,
        build_teacher_messages,
        extract_think_trace,
        parse_teacher_completion,
        split_thinking_record,
    )
from .label_levels import (
    label_level,
)
from .publish import (
    prepare_hf_export,
    sync_to_drive,
    upload_to_hf,
)
from .rl_prompts import (
    build_rl_prompts_from_evaluated,
    generate_rl_prompts,
    load_excluded_keys,
    save_rl_prompts,
)
from .run_pipeline import (
    run_pipeline,
)
from .splits import (
    normalize_fen,
    save_splits,
    split_by_game,
)

_FILTER_COMMENT_SYMBOLS: Final[set[str]] = {
    "FilterStats",
    "assign_detail_and_truncate",
    "clean_comment",
    "compute_shingles",
    "filter_comment",
    "is_near_duplicate",
    "pre_filter_comment",
}

_TEACHER_SYMBOLS: Final[set[str]] = {
    "TEACHER_SYSTEM_PROMPT",
    "build_teacher_messages",
    "extract_think_trace",
    "parse_teacher_completion",
    "split_thinking_record",
}


def __getattr__(name: str) -> Any:
    """Import pipeline attributes on demand.

    If name belongs to filter_comments, import and return the attribute.
    If name belongs to synthesize_teacher, import and return the attribute.
    If name is unknown, raise an AttributeError.
    """
    if name in _FILTER_COMMENT_SYMBOLS:
        from . import filter_comments

        val = getattr(filter_comments, name)
        globals()[name] = val
        return val
    if name in _TEACHER_SYMBOLS:
        from . import synthesize_teacher

        val = getattr(synthesize_teacher, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all public module attribute names."""
    return sorted(
        list(globals().keys())
        + list(_FILTER_COMMENT_SYMBOLS)
        + list(_TEACHER_SYMBOLS)
    )


__all__ = [
    "StockfishAnalyzer",
    "StockfishCache",
    "StockfishPool",
    "analyse_records",
    "find_stockfish_binary",
    "normalize_fen",
    "split_by_game",
    "save_splits",
    "build_eval_pool_from_evaluated",
    "generate_eval_pool",
    "save_eval_pool",
    "build_rl_prompts_from_evaluated",
    "generate_rl_prompts",
    "load_excluded_keys",
    "save_rl_prompts",
    "prepare_hf_export",
    "upload_to_hf",
    "sync_to_drive",
    "run_pipeline",
    "filter_comment",
    "pre_filter_comment",
    "clean_comment",
    "compute_shingles",
    "is_near_duplicate",
    "assign_detail_and_truncate",
    "FilterStats",
    "label_level",
    "deglue_comment",
    "deglue_token",
    "remove_reader_questions",
    "has_too_many_unknown_tokens",
    "count_unknown_tokens",
    "TEACHER_SYSTEM_PROMPT",
    "build_teacher_messages",
    "extract_think_trace",
    "parse_teacher_completion",
    "split_thinking_record",
]

