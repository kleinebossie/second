"""Lichess studies commentary collector.

This module extracts annotated positions and comments from Lichess studies.
It strips visual markup tags and detects open license terms.
It exports standard records for training or local rebuilding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections.abc import Iterator
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

ICANNOS_CSV_URL = (
    "https://huggingface.co/datasets/Icannos/chess_studies/resolve/main/lichess_studies.csv"
)

OPEN_LICENSE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcc0\b", re.IGNORECASE), "CC0-1.0"),
    (re.compile(r"\bpublic\s+domain\b", re.IGNORECASE), "Public Domain"),
    (re.compile(r"\bcc-by(?:-sa)?(?:\s+\d+\.\d+)?\b", re.IGNORECASE), "CC-BY"),
    (re.compile(r"\bcreative\s+commons\b", re.IGNORECASE), "CC-BY"),
    (re.compile(r"\bmit\s+license\b", re.IGNORECASE), "MIT"),
]


def strip_lichess_tags(comment: str) -> str:
    """Remove Lichess visual markup tags and normalize whitespace.

    Remove all tags with the syntax [%tag value].
    Normalize extra spaces and line breaks.
    Return the cleaned comment string.
    """
    cleaned = re.sub(r"\[%[a-zA-Z]+\s*[^\]]*\]", "", comment)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def detect_study_license(
    headers: chess.pgn.Headers,
    intro_comment: str = "",
    source_hint: str = "",
) -> tuple[str, bool]:
    """Detect license type and redistribution rights for a study.

    Check headers, introductory comments, and source hints.
    If restrictive license terms match, return standard Lichess terms and False.
    If an explicit open license matches, return the license and True.
    When no open license matches, return standard Lichess terms and False.
    """
    if "pgnlib_" in source_hint.lower():
        return "Public Domain", True

    combined_text = " ".join(
        [
            headers.get("License", ""),
            headers.get("Event", ""),
            headers.get("Annotator", ""),
            headers.get("Remark", ""),
            intro_comment,
            source_hint,
        ]
    )

    if re.search(r"\bcc-by-(?:nc|nd)\b|\bnon-?commercial\b", combined_text, re.IGNORECASE):
        return "Lichess Terms of Service", False

    for pattern, name in OPEN_LICENSE_PATTERNS:
        if pattern.search(combined_text):
            return name, True

    return "Lichess Terms of Service", False


def extract_study_id(headers: chess.pgn.Headers, fallback: str = "") -> str:
    """Extract a study identifier from PGN headers.

    If the Site header contains a Lichess study URL, return the study ID.
    When no URL matches, use the fallback key or create a hash.
    Return the identifier string.
    """
    site = headers.get("Site", "")
    match = re.search(r"lichess\.org/study/([A-Za-z0-9]+)(?:/([A-Za-z0-9]+))?", site)
    if match:
        return match.group(1)

    if fallback:
        return fallback

    event = headers.get("Event", "study")
    date = headers.get("Date", "????.??.??")
    fingerprint = f"study_{event}_{date}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def parse_study_pgn(
    pgn_text: str,
    source_hint: str = "",
    require_open_license: bool = False,
    include_variations: bool = True,
) -> list[ChessRecord]:
    """Parse Lichess study PGN text into records.

    If the text is empty, return an empty list.
    Read each chapter game from the PGN stream.
    Strip visual markup from every move comment.
    Return all valid records.
    """
    stream = io.StringIO(pgn_text)
    records: list[ChessRecord] = []

    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break

        intro = strip_lichess_tags(game.comment)
        license_name, redistributable = detect_study_license(
            headers=game.headers,
            intro_comment=intro,
            source_hint=source_hint,
        )

        if require_open_license and not redistributable:
            continue

        game_id = extract_study_id(game.headers, fallback=source_hint)
        initial_board = game.board()
        current_board = game.board()
        white_elo = parse_elo(game.headers.get("WhiteElo"))
        black_elo = parse_elo(game.headers.get("BlackElo"))

        def walk(node: chess.pgn.GameNode, moves_history: list[chess.Move]) -> None:
            variations = node.variations if include_variations else (
                [node.variation(0)] if node.variations else []
            )
            for child in variations:
                move = child.move
                fen_before = current_board.fen()
                uci = move.uci()
                san = current_board.san(move)
                raw_comment = child.comment
                cleaned_comment = strip_lichess_tags(raw_comment)
                new_history = moves_history + [move]

                if cleaned_comment:
                    pgn_text_item = format_moves_up_to(initial_board, new_history)
                    records.append(
                        make_record(
                            source="lichess_studies",
                            license_name=license_name,
                            redistributable=redistributable,
                            game_id=game_id,
                            pgn=pgn_text_item,
                            fen=fen_before,
                            move_uci=uci,
                            move_san=san,
                            comment=cleaned_comment,
                            white_elo=white_elo,
                            black_elo=black_elo,
                        )
                    )

                current_board.push(move)
                walk(child, new_history)
                current_board.pop()

        walk(game, [])

    return records


def fetch_lichess_study(
    study_id: str,
    require_open_license: bool = False,
    timeout: int = 15,
) -> list[ChessRecord]:
    """Fetch study chapters using the Lichess Study API.

    Send an HTTP request to the study PGN export endpoint.
    If the study exists, parse the PGN content into records.
    When the request fails, return an empty list.
    """
    clean_id = study_id.strip().split("/")[0]
    url = f"https://lichess.org/api/study/{clean_id}.pgn"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return []
        return parse_study_pgn(
            pgn_text=response.text,
            source_hint=f"study_{clean_id}",
            require_open_license=require_open_license,
        )
    except requests.RequestException:
        return []


def _iter_csv_rows(raw_bytes: bytes) -> Iterator[list[str]]:
    """Yield data rows from CSV bytes.

    If the magic bytes match gzip, decompress the payload.
    Set the field size limit to support long PGN entries.
    Skip the header row.
    Yield each non-empty row.
    """
    import gzip

    if raw_bytes[:2] == b"\x1f\x8b":
        text_data = gzip.decompress(raw_bytes).decode("utf-8", errors="replace")
    else:
        text_data = raw_bytes.decode("utf-8", errors="replace")

    csv.field_size_limit(10 * 1024 * 1024)
    reader = csv.reader(io.StringIO(text_data))
    _header = next(reader, None)
    for row in reader:
        if row:
            yield row


def stream_icannos_csv(
    csv_url: str = ICANNOS_CSV_URL,
    local_csv: Path | None = None,
    max_studies: int | None = None,
    max_items: int | None = None,
    require_open_license: bool = False,
) -> Iterator[ChessRecord]:
    """Stream study records from the Icannos CSV dataset.

    Read lines from the local file or remote CSV stream.
    Extract PGN text and parse comments for each study.
    Yield each extracted record.
    """
    if local_csv and local_csv.exists():
        raw_bytes = local_csv.read_bytes()
    else:
        try:
            with requests.get(csv_url, stream=True, timeout=30) as response:
                if response.status_code != 200:
                    return
                raw_bytes = response.content
        except requests.RequestException:
            return

    studies_processed = 0
    items_yielded = 0

    for row in _iter_csv_rows(raw_bytes):
        pgn_content = row[0]
        if pgn_content.strip() == "text":
            continue
        records = parse_study_pgn(
            pgn_text=pgn_content,
            require_open_license=require_open_license,
        )
        if not records:
            continue

        studies_processed += 1
        for record in records:
            yield record
            items_yielded += 1
            if max_items is not None and items_yielded >= max_items:
                return
        if max_studies is not None and studies_processed >= max_studies:
            return


def collect_lichess_studies(
    output_path: Path,
    study_ids: list[str] | None = None,
    csv_file: Path | None = None,
    max_studies: int | None = None,
    max_items: int | None = None,
    require_open_license: bool = False,
) -> int:
    """Collect study records and write them to disk.

    If study IDs are provided, fetch them via the Lichess API.
    Otherwise, stream records from the Icannos CSV dataset.
    Write every record to the target JSONL file.
    Return the total number of written records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as writer:
        if study_ids:
            for sid in study_ids:
                records = fetch_lichess_study(
                    study_id=sid,
                    require_open_license=require_open_license,
                )
                for record in records:
                    writer.write(json.dumps(record) + "\n")
                    count += 1
                    if max_items is not None and count >= max_items:
                        return count
                if max_studies is not None and count >= max_studies:
                    break
        else:
            for record in stream_icannos_csv(
                local_csv=csv_file,
                max_studies=max_studies,
                max_items=max_items,
                require_open_license=require_open_license,
            ):
                writer.write(json.dumps(record) + "\n")
                count += 1
                if max_items is not None and count >= max_items:
                    break

    return count


def main() -> None:
    """Run the Lichess studies collection script from the command line."""
    parser = argparse.ArgumentParser(description="Collect Lichess study comments.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/lichess_studies.jsonl"),
        help="Path for output JSONL file.",
    )
    parser.add_argument(
        "--study-ids",
        nargs="*",
        default=[],
        help="Lichess study IDs to fetch from the API.",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=None,
        help="Local Icannos CSV file path.",
    )
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Maximum number of studies to parse.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of records to emit.",
    )
    parser.add_argument(
        "--require-open-license",
        action="store_true",
        help="Export only studies with explicit open license terms.",
    )
    parser.add_argument(
        "--include-all-studies",
        action="store_true",
        help="Export all studies regardless of open license.",
    )

    args = parser.parse_args()
    require_open = args.require_open_license and not args.include_all_studies

    total = collect_lichess_studies(
        output_path=args.output,
        study_ids=args.study_ids,
        csv_file=args.csv_file,
        max_studies=args.max_studies,
        max_items=args.max_items,
        require_open_license=require_open,
    )
    print(f"Collected {total} Lichess study records into {args.output}")


if __name__ == "__main__":
    main()
