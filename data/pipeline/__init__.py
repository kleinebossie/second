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
from .filter_comments import (
    FilterStats,
    assign_detail_and_truncate,
    clean_comment,
    compute_shingles,
    filter_comment,
    is_near_duplicate,
    pre_filter_comment,
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

__all__ = [
    "StockfishAnalyzer",
    "StockfishCache",
    "StockfishPool",
    "analyse_records",
    "find_stockfish_binary",
    "normalize_fen",
    "split_by_game",
    "save_splits",
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
]
