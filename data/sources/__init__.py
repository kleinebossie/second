"""Raw chess data source collection modules.

This package provides collectors and parsers for chess games and commentary.
It exports shared record definitions and functions for each data source.
"""

from __future__ import annotations

from .gameknot import (
    collect_gameknot,
    fetch_gameknot_direct,
    parse_gameknot_html,
    parse_gameknot_pgn,
    stream_waterhorse_records,
)
from .lichess_games import (
    collect_lichess_games,
    is_acceptable_rating,
    sample_game_positions,
    stream_lichess_games_zst,
)
from .lichess_studies import (
    collect_lichess_studies,
    detect_study_license,
    fetch_lichess_study,
    parse_study_pgn,
    strip_lichess_tags,
)
from .mate import (
    collect_mate,
    download_mate_zip,
    parse_mate_entry,
    stream_mate_from_zip,
)
from .schema import (
    RECORD_KEYS,
    ChessRecord,
    format_moves_up_to,
    make_record,
    parse_elo,
    validate_record,
)

__all__ = [
    "ChessRecord",
    "RECORD_KEYS",
    "make_record",
    "validate_record",
    "parse_elo",
    "format_moves_up_to",
    "collect_gameknot",
    "parse_gameknot_pgn",
    "parse_gameknot_html",
    "fetch_gameknot_direct",
    "stream_waterhorse_records",
    "collect_lichess_studies",
    "parse_study_pgn",
    "fetch_lichess_study",
    "strip_lichess_tags",
    "detect_study_license",
    "collect_mate",
    "parse_mate_entry",
    "stream_mate_from_zip",
    "download_mate_zip",
    "collect_lichess_games",
    "stream_lichess_games_zst",
    "sample_game_positions",
    "is_acceptable_rating",
]
