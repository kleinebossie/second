"""Lichess game database sampler.

This module streams and samples rated games from the CC0 Lichess database.
It decompresses zstandard streams on the fly without saving large files.
It filters games by player ratings and samples diverse positions for RL prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import requests
import zstandard

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from data.sources.schema import (
        ChessRecord,
        format_moves_up_to,
        make_record,
        parse_elo,
    )
else:
    from .schema import (
        ChessRecord,
        format_moves_up_to,
        make_record,
        parse_elo,
    )

DEFAULT_LICHESS_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
)


def extract_game_id(headers: chess.pgn.Headers) -> str:
    """Extract a unique identifier from game headers.

    If the Site header contains a Lichess game URL, extract the game code.
    When no Site header matches, build a deterministic hash from headers.
    Return the identifier string.
    """
    site = headers.get("Site", "")
    match = re.search(r"lichess\.org/([A-Za-z0-9]+)", site)
    if match:
        return match.group(1)

    white = headers.get("White", "unknown")
    black = headers.get("Black", "unknown")
    date = headers.get("UTCDate", headers.get("Date", "????.??.??"))
    time_val = headers.get("UTCTime", "")
    fingerprint = f"lichess_{white}_{black}_{date}_{time_val}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def is_acceptable_rating(
    white_elo: int | None,
    black_elo: int | None,
    count_focus: int,
    count_other: int,
    min_elo: int = 600,
    max_elo: int = 2600,
    focus_min: int = 800,
    focus_max: int = 2000,
    focus_ratio: float = 0.80,
) -> bool:
    """Check if game ratings match the sampling criteria.

    If either rating is missing, reject the game.
    Compute the average rating between players.
    If the average rating is outside [min_elo, max_elo], reject the game.
    If the average rating is in the focus range, accept the game.
    When the average rating is outside the focus range, check the quota ratio.
    Return True if accepted, False otherwise.
    """
    if white_elo is None or black_elo is None:
        return False

    avg_elo = (white_elo + black_elo) / 2.0
    if avg_elo < min_elo or avg_elo > max_elo:
        return False

    if focus_min <= avg_elo <= focus_max:
        return True

    total = count_focus + count_other
    if total == 0:
        return False

    other_ratio = (count_other + 1) / (total + 1)
    return other_ratio <= (1.0 - focus_ratio)


def sample_game_positions(
    game: chess.pgn.Game,
    rng: random.Random,
    max_positions_per_game: int = 3,
) -> list[ChessRecord]:
    """Sample diverse positions from a single chess game.

    Skip the first three plies of the game.
    Sample up to three positions across opening, middlegame, and endgame stages.
    Format the move history up to each sampled position.
    Return the list of sampled records.
    """
    moves = list(game.mainline_moves())
    total_plies = len(moves)
    if total_plies < 4:
        return []

    stages: list[tuple[int, int]] = []
    # Opening stage: plies 3 to 16
    stages.append((3, min(16, total_plies - 1)))
    # Middlegame stage: plies 17 to 40
    if total_plies > 17:
        stages.append((17, min(40, total_plies - 1)))
    # Endgame stage: plies 41 and above
    if total_plies > 41:
        stages.append((41, total_plies - 1))

    sampled_indices: list[int] = []
    for low, high in stages:
        if low <= high:
            sampled_indices.append(rng.randint(low, high))

    target_indices = sorted(set(sampled_indices))[:max_positions_per_game]

    game_id = extract_game_id(game.headers)
    white_elo = parse_elo(game.headers.get("WhiteElo"))
    black_elo = parse_elo(game.headers.get("BlackElo"))

    initial_board = game.board()
    board = game.board()
    moves_history: list[chess.Move] = []
    records: list[ChessRecord] = []

    for idx, move in enumerate(moves):
        if idx in target_indices:
            fen_before = board.fen()
            uci = move.uci()
            san = board.san(move)
            pgn_text = format_moves_up_to(initial_board, moves_history + [move])

            records.append(
                make_record(
                    source="lichess_games",
                    license_name="CC0-1.0",
                    redistributable=True,
                    game_id=game_id,
                    pgn=pgn_text,
                    fen=fen_before,
                    move_uci=uci,
                    move_san=san,
                    comment="",
                    white_elo=white_elo,
                    black_elo=black_elo,
                )
            )

        board.push(move)
        moves_history.append(move)

    return records


def stream_games_from_text_stream(text_stream: Any) -> Iterator[chess.pgn.Game]:
    """Yield parsed chess games from an open text stream.

    Read games sequentially until the stream ends.
    Yield each valid game.
    """
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        yield game


def stream_lichess_games_zst(
    source_url: str = DEFAULT_LICHESS_URL,
    local_path: Path | None = None,
    timeout: int = 60,
) -> Iterator[chess.pgn.Game]:
    """Stream and decompress games from a zstandard archive.

    If a local path is provided, read directly from the local file.
    Otherwise, open an HTTP stream to the remote database URL.
    Decompress the byte stream on the fly.
    Yield parsed games one by one.
    """
    dctx = zstandard.ZstdDecompressor()

    if local_path and local_path.exists():
        if local_path.suffix == ".zst":
            with local_path.open("rb") as raw_file:
                reader = dctx.stream_reader(raw_file)
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield from stream_games_from_text_stream(text_stream)
        else:
            with local_path.open("r", encoding="utf-8", errors="replace") as text_file:
                yield from stream_games_from_text_stream(text_file)
        return

    with requests.get(source_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        reader = dctx.stream_reader(response.raw)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        yield from stream_games_from_text_stream(text_stream)


def collect_lichess_games(
    output_path: Path,
    source_url: str = DEFAULT_LICHESS_URL,
    local_path: Path | None = None,
    max_games: int | None = None,
    max_positions: int | None = None,
    seed: int = 42,
    min_elo: int = 600,
    max_elo: int = 2600,
    focus_min: int = 800,
    focus_max: int = 2000,
    focus_ratio: float = 0.80,
) -> int:
    """Collect sampled game positions and write them to disk.

    Stream games from the Lichess database.
    Filter games to maintain target rating distributions.
    Sample 1 to 3 positions per accepted game.
    Write records to the target JSONL file.
    Return the total number of written position records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    count_focus = 0
    count_other = 0
    games_accepted = 0
    positions_written = 0

    games_iter = stream_lichess_games_zst(source_url=source_url, local_path=local_path)

    with output_path.open("w", encoding="utf-8") as writer:
        for game in games_iter:
            white_elo = parse_elo(game.headers.get("WhiteElo"))
            black_elo = parse_elo(game.headers.get("BlackElo"))

            acceptable = is_acceptable_rating(
                white_elo=white_elo,
                black_elo=black_elo,
                count_focus=count_focus,
                count_other=count_other,
                min_elo=min_elo,
                max_elo=max_elo,
                focus_min=focus_min,
                focus_max=focus_max,
                focus_ratio=focus_ratio,
            )
            if not acceptable:
                continue

            positions = sample_game_positions(game=game, rng=rng)
            if not positions:
                continue

            avg_elo = (white_elo + black_elo) / 2.0  # type: ignore[operator]
            if focus_min <= avg_elo <= focus_max:
                count_focus += 1
            else:
                count_other += 1

            games_accepted += 1

            for pos in positions:
                writer.write(json.dumps(pos) + "\n")
                positions_written += 1
                if max_positions is not None and positions_written >= max_positions:
                    return positions_written

            if max_games is not None and games_accepted >= max_games:
                break

    return positions_written


def main() -> None:
    """Run the Lichess game sampler script from the command line."""
    parser = argparse.ArgumentParser(description="Sample positions from Lichess games.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/lichess_games.jsonl"),
        help="Path for output JSONL file.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_LICHESS_URL,
        help="URL to the Lichess .pgn.zst database dump.",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=None,
        help="Local .pgn.zst or .pgn file to read.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to accept.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum number of position records to emit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling positions.",
    )

    args = parser.parse_args()
    total = collect_lichess_games(
        output_path=args.output,
        source_url=args.url,
        local_path=args.local_path,
        max_games=args.max_games,
        max_positions=args.max_positions,
        seed=args.seed,
    )
    print(f"Sampled {total} positions into {args.output}")


if __name__ == "__main__":
    main()
