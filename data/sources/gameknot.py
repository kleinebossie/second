"""GameKnot annotated games collector.

This module downloads and parses GameKnot annotated chess games.
It extracts board positions, moves, and human commentary into standard records.
The raw data is proprietary and must remain local.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import urllib.parse
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import requests

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

WATERHORSE_URLS: list[str] = [
    (
        "https://huggingface.co/datasets/Waterhorse/chess_data/resolve/main/"
        "chessgpt_data/annotated_pgn/annotated_pgn-data.jsonl-00000-of-00002"
    ),
    (
        "https://huggingface.co/datasets/Waterhorse/chess_data/resolve/main/"
        "chessgpt_data/annotated_pgn/annotated_pgn-data.jsonl-00001-of-00002"
    ),
]


def extract_game_id(game: chess.pgn.Game, fallback_key: str = "") -> str:
    """Extract a unique identifier for a GameKnot game.

    If the fallback key contains an identifier, use it.
    If the Site or Event headers contain a game number, extract that number.
    When no explicit identifier exists, build a deterministic hash.
    """
    if fallback_key:
        match = re.search(r"gameknot_game_(\w+)", fallback_key)
        if match:
            return match.group(1)

    site = game.headers.get("Site", "")
    match_site = re.search(r"[?&](?:gm|id)=(\d+)", site)
    if match_site:
        return match_site.group(1)

    event = game.headers.get("Event", "")
    match_event = re.search(r"#\s*(\d+)", event)
    if match_event:
        return match_event.group(1)

    white = game.headers.get("White", "unknown")
    black = game.headers.get("Black", "unknown")
    date = game.headers.get("Date", "????.??.??")
    fingerprint = f"gameknot_{white}_{black}_{event}_{date}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def extract_records_from_game(
    game: chess.pgn.Game,
    game_id: str,
    include_variations: bool = True,
) -> list[ChessRecord]:
    """Extract move records and comments from a game tree.

    Walk the mainline and variations of the game.
    For every node with a comment after a move, build a record.
    Return the list of records.
    """
    initial_board = game.board()
    current_board = game.board()
    white_elo = parse_elo(game.headers.get("WhiteElo"))
    black_elo = parse_elo(game.headers.get("BlackElo"))

    records: list[ChessRecord] = []

    def walk(node: chess.pgn.GameNode, moves_history: list[chess.Move]) -> None:
        variations = node.variations if include_variations else (
            [node.variation(0)] if node.variations else []
        )
        for child in variations:
            move = child.move
            fen_before = current_board.fen()
            uci = move.uci()
            san = current_board.san(move)
            comment = child.comment.strip()
            new_history = moves_history + [move]

            if comment:
                pgn_text = format_moves_up_to(initial_board, new_history)
                records.append(
                    make_record(
                        source="gameknot",
                        license_name="proprietary",
                        redistributable=False,
                        game_id=game_id,
                        pgn=pgn_text,
                        fen=fen_before,
                        move_uci=uci,
                        move_san=san,
                        comment=comment,
                        white_elo=white_elo,
                        black_elo=black_elo,
                    )
                )

            current_board.push(move)
            walk(child, new_history)
            current_board.pop()

    walk(game, [])
    return records


def parse_gameknot_pgn(
    pgn_text: str,
    fallback_id: str = "",
    include_variations: bool = True,
) -> list[ChessRecord]:
    """Parse GameKnot PGN text into records.

    If the PGN text is empty, return an empty list.
    Read each game from the stream and extract comment records.
    Return all parsed records.
    """
    stream = io.StringIO(pgn_text)
    records: list[ChessRecord] = []

    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        game_id = extract_game_id(game, fallback_id)
        records.extend(
            extract_records_from_game(
                game=game,
                game_id=game_id,
                include_variations=include_variations,
            )
        )

    return records


def parse_gameknot_html(html_text: str, default_id: str = "") -> list[ChessRecord]:
    """Parse GameKnot HTML content into records.

    Check for embedded script variables or HTML table structures.
    If script variables exist, decode the move list and notes.
    If table rows exist, extract move notations and comments.
    Return all parsed records.
    """
    records: list[ChessRecord] = []

    id_match = re.search(r"game_id\s*=\s*(\d+)", html_text)
    game_id = id_match.group(1) if id_match else default_id
    if not game_id:
        game_id = hashlib.sha256(html_text[:500].encode("utf-8")).hexdigest()[:16]

    white_match = re.search(r"game_player_w\s*=\s*decode_str\([\"'](.*?)[\"']\)", html_text)
    black_match = re.search(r"game_player_b\s*=\s*decode_str\([\"'](.*?)[\"']\)", html_text)
    _white = white_match.group(1) if white_match else ""
    _black = black_match.group(1) if black_match else ""

    movelist_match = re.search(r"game_movelist\s*=\s*[\"']([^\"']*)[\"']", html_text)
    notes_match = re.search(r"game_notes\s*=\s*\[(.*?)\];", html_text, re.DOTALL)

    if movelist_match and notes_match:
        moves_raw = movelist_match.group(1).strip("-")
        uci_moves = [m for m in moves_raw.split("-") if m]
        notes_raw = notes_match.group(1)
        notes = [
            urllib.parse.unquote(n.strip().strip("\"'"))
            for n in re.findall(r"\"(.*?)\"", notes_raw)
        ]

        board = chess.Board()
        initial_board = chess.Board()
        history: list[chess.Move] = []

        for idx, uci in enumerate(uci_moves):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break

            note_index = idx + 1
            comment = notes[note_index].strip() if note_index < len(notes) else ""
            fen_before = board.fen()
            san = board.san(move)
            history.append(move)

            if comment:
                pgn_text = format_moves_up_to(initial_board, history)
                records.append(
                    make_record(
                        source="gameknot",
                        license_name="proprietary",
                        redistributable=False,
                        game_id=game_id,
                        pgn=pgn_text,
                        fen=fen_before,
                        move_uci=uci,
                        move_san=san,
                        comment=comment,
                        white_elo=None,
                        black_elo=None,
                    )
                )
            board.push(move)

        return records

    table_match = re.findall(
        r"<tr[^>]*>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>\s*</tr>",
        html_text,
        re.DOTALL | re.IGNORECASE,
    )
    if table_match:
        board = chess.Board()
        initial_board = chess.Board()
        history = []
        for cell1, cell2 in table_match:
            move_text = re.sub(r"<[^>]+>", " ", cell1).replace("&nbsp;", " ").strip()
            comment_text = re.sub(r"<[^>]+>", " ", cell2).replace("&nbsp;", " ").strip()
            token_match = re.search(
                r"(?:[O0]-[O0](?:-[O0])?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#]?)",
                move_text,
            )
            if not token_match:
                continue
            san_token = token_match.group(0)
            try:
                move = board.parse_san(san_token)
            except ValueError:
                continue

            fen_before = board.fen()
            uci = move.uci()
            san = board.san(move)
            history.append(move)

            if comment_text:
                pgn_text = format_moves_up_to(initial_board, history)
                records.append(
                    make_record(
                        source="gameknot",
                        license_name="proprietary",
                        redistributable=False,
                        game_id=game_id,
                        pgn=pgn_text,
                        fen=fen_before,
                        move_uci=uci,
                        move_san=san,
                        comment=comment_text,
                        white_elo=None,
                        black_elo=None,
                    )
                )
            board.push(move)

    return records


def fetch_gameknot_direct(game_id: str | int, timeout: int = 10) -> list[ChessRecord]:
    """Fetch and parse a game from GameKnot by ID.

    Send a request to the annotate endpoint.
    If the request succeeds, parse the response HTML into records.
    When the request fails, return an empty list.
    """
    url = f"https://gameknot.com/annotate.pl?id={game_id}"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return []
        return parse_gameknot_html(response.text, default_id=str(game_id))
    except requests.RequestException:
        return []


def _iter_waterhorse_lines(lines: Iterable[str | bytes]) -> Iterator[tuple[str, str]]:
    """Yield PGN text and key pairs from Waterhorse lines.

    If a line contains bytes, decode it as UTF-8.
    Skip empty lines and invalid JSON records.
    Check if the entry comes from GameKnot.
    If the entry is valid GameKnot content, yield the text and pipeline key.
    """
    for raw in lines:
        if isinstance(raw, bytes):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
        else:
            line = raw

        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = payload.get("text", "")
        meta = payload.get("metadata") or {}
        key = str(payload.get("pipeline_key") or meta.get("pipeline_key", ""))

        is_gameknot = (
            key.startswith("gameknot_game_")
            or "gameknot" in key.lower()
            or "gameknot.com" in text.lower()
        )
        if is_gameknot and text:
            yield text, key


def stream_waterhorse_records(
    urls: list[str] | None = None,
    max_games: int | None = None,
    max_items: int | None = None,
    cache_file: Path | None = None,
) -> Iterator[ChessRecord]:
    """Stream GameKnot records from the Waterhorse corpus.

    Read entries line by line from the remote dataset or local cache.
    Filter for games that originate from GameKnot.
    Yield each extracted record.
    """
    if urls is None:
        urls = WATERHORSE_URLS

    def _generate_pairs() -> Iterator[tuple[str, str]]:
        if cache_file and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as file:
                yield from _iter_waterhorse_lines(file)
            return

        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=30) as response:
                    if response.status_code != 200:
                        continue
                    yield from _iter_waterhorse_lines(response.iter_lines())
            except requests.RequestException:
                continue

    games_processed = 0
    items_yielded = 0

    for text, key in _generate_pairs():
        records = parse_gameknot_pgn(text, fallback_id=key)
        if not records:
            continue

        games_processed += 1
        for record in records:
            yield record
            items_yielded += 1
            if max_items is not None and items_yielded >= max_items:
                return
        if max_games is not None and games_processed >= max_games:
            return


def collect_gameknot(
    output_path: Path,
    max_games: int | None = None,
    max_items: int | None = None,
    direct_web: bool = False,
    game_ids: list[str] | None = None,
    cache_file: Path | None = None,
) -> int:
    """Collect GameKnot records and write them to disk.

    If direct web mode is active, fetch specified game IDs.
    Otherwise, stream records from the Waterhorse dataset.
    Write every record to the target JSONL file.
    Return the total number of written records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as writer:
        if direct_web:
            ids = game_ids or []
            for gid in ids:
                records = fetch_gameknot_direct(gid)
                for record in records:
                    writer.write(json.dumps(record) + "\n")
                    count += 1
                    if max_items is not None and count >= max_items:
                        return count
                if max_games is not None and count >= max_games:
                    break
        else:
            for record in stream_waterhorse_records(
                max_games=max_games,
                max_items=max_items,
                cache_file=cache_file,
            ):
                writer.write(json.dumps(record) + "\n")
                count += 1
                if max_items is not None and count >= max_items:
                    break

    return count


def main() -> None:
    """Run the GameKnot collection script from the command line."""
    parser = argparse.ArgumentParser(description="Collect GameKnot annotated games.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/gameknot.jsonl"),
        help="Path for output JSONL file.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to parse.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of records to emit.",
    )
    parser.add_argument(
        "--direct-web",
        action="store_true",
        help="Fetch games directly from GameKnot web pages.",
    )
    parser.add_argument(
        "--game-ids",
        nargs="*",
        default=[],
        help="GameKnot IDs to fetch when direct web mode is set.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Local cached JSONL file to use instead of remote download.",
    )

    args = parser.parse_args()
    total = collect_gameknot(
        output_path=args.output,
        max_games=args.max_games,
        max_items=args.max_items,
        direct_web=args.direct_web,
        game_ids=args.game_ids,
        cache_file=args.cache_file,
    )
    print(f"Collected {total} GameKnot records into {args.output}")


if __name__ == "__main__":
    main()
