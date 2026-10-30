"""MATE dataset collector.

This module downloads and parses the MIT-licensed MATE dataset from Hugging Face.
It extracts chess positions, winning moves, and strategic or tactical explanations.
It verifies board legality and exports standard records.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chess
import requests

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from data.sources.schema import (
        ChessRecord,
        make_record,
    )
else:
    from .schema import (
        ChessRecord,
        make_record,
    )

MATE_TESTSET_URL = (
    "https://huggingface.co/datasets/OutFlankShu/MATE_DATASET/resolve/main/testset.zip"
)


def parse_mate_entry(entry: dict[str, Any]) -> ChessRecord | None:
    """Parse a single MATE dataset dictionary into a chess record.

    Extract the FEN string from the input prompt.
    Find candidate moves and their explanations.
    Identify the winning candidate from the output field.
    Verify that the winning move is legal on the board.
    When validation passes, return a ChessRecord.
    If parsing or legality fails, return None.
    """
    input_text = str(entry.get("input", ""))
    output_text = str(entry.get("output", "")).strip()

    fen_match = re.search(r'The FEN of the given chess board is "([^"]+)"', input_text)
    if not fen_match:
        return None
    fen = fen_match.group(1).strip()

    idx_a = input_text.find("MoveA:")
    idx_b = input_text.find("MoveB:")
    if idx_a == -1 or idx_b == -1 or idx_b <= idx_a:
        return None

    text_a = input_text[idx_a + len("MoveA:") : idx_b].strip()
    text_b = input_text[idx_b + len("MoveB:") :].strip()

    pattern = re.compile(
        r"^([a-h][1-8][a-h][1-8][qrbn]?)[,:\s]+(.*)$",
        re.DOTALL | re.IGNORECASE,
    )
    match_a = pattern.match(text_a)
    match_b = pattern.match(text_b)
    if not match_a or not match_b:
        return None

    uci_a = match_a.group(1).lower()
    exp_a = match_a.group(2).strip()
    uci_b = match_b.group(1).lower()
    exp_b = match_b.group(2).strip()

    out_lower = output_text.lower()
    if "movea" in out_lower or uci_a in out_lower:
        win_uci = uci_a
        win_exp = exp_a
    elif "moveb" in out_lower or uci_b in out_lower:
        win_uci = uci_b
        win_exp = exp_b
    else:
        return None

    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    try:
        move = chess.Move.from_uci(win_uci)
    except ValueError:
        return None

    if move not in board.legal_moves:
        return None

    san = board.san(move)
    # Clean raw UCI tactic move lists if present in MATE explanations
    cleaned_exp = re.sub(
        r"\b(?:tactic|Tactic[AB]):?\s*(?:[a-h][1-8][a-h][1-8][qrbn]?\s*)+",
        " ",
        win_exp,
    )
    cleaned_exp = re.sub(r"\s+", " ", cleaned_exp).strip()
    if not cleaned_exp:
        return None

    fingerprint = f"{fen}_{win_uci}"
    game_id = f"mate_{hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:16]}"

    return make_record(
        source="mate",
        license_name="MIT",
        redistributable=True,
        game_id=game_id,
        pgn="",
        fen=fen,
        move_uci=win_uci,
        move_san=san,
        comment=cleaned_exp,
        white_elo=None,
        black_elo=None,
    )


def stream_mate_from_zip(
    zip_bytes_or_path: bytes | Path,
    max_items: int | None = None,
) -> Iterator[ChessRecord]:
    """Stream parsed records from a MATE zip archive.

    Open the zip archive in memory or from disk.
    Read all JSONL files inside the archive.
    Parse each line and yield valid records.
    """
    if isinstance(zip_bytes_or_path, bytes):
        archive: zipfile.ZipFile = zipfile.ZipFile(io.BytesIO(zip_bytes_or_path))
    else:
        archive = zipfile.ZipFile(zip_bytes_or_path)

    items_yielded = 0
    with archive:
        names = [
            n
            for n in archive.namelist()
            if n.endswith(".jsonl")
            and "__MACOSX" not in n
            and "noexplain" not in n.lower()
            and "no_explain" not in n.lower()
        ]
        # Prefer richer combined and strategy explanations over bare tactic templates
        names.sort(
            key=lambda n: (
                0 if "both" in n.lower() else (1 if "strategy" in n.lower() else 2),
                n,
            )
        )
        for name in names:
            with archive.open(name, "r") as file:
                for raw_line in file:
                    if not raw_line.strip():
                        continue
                    try:
                        entry = json.loads(raw_line.decode("utf-8", errors="replace"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    record = parse_mate_entry(entry)
                    if record is not None:
                        yield record
                        items_yielded += 1
                        if max_items is not None and items_yielded >= max_items:
                            return


def stream_mate_from_jsonl(
    jsonl_path: Path,
    max_items: int | None = None,
) -> Iterator[ChessRecord]:
    """Stream parsed records from a local JSONL file.

    Read entries line by line from disk.
    Parse each entry and yield valid records.
    """
    items_yielded = 0
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            record = parse_mate_entry(entry)
            if record is not None:
                yield record
                items_yielded += 1
                if max_items is not None and items_yielded >= max_items:
                    return


def download_mate_zip(
    url: str = MATE_TESTSET_URL,
    target_path: Path | None = None,
    timeout: int = 60,
) -> Path:
    """Download the MATE dataset zip file from Hugging Face.

    If target path is not provided, use a default cache path.
    Download the file using streaming chunks.
    Return the path to the downloaded zip file.
    """
    if target_path is None:
        target_path = Path("data/cache/mate/testset.zip")

    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp")

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temp_target.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1048576):
                if chunk:
                    file.write(chunk)

    temp_target.replace(target_path)
    return target_path


def collect_mate(
    output_path: Path,
    zip_path: Path | None = None,
    jsonl_path: Path | None = None,
    download_url: str = MATE_TESTSET_URL,
    max_items: int | None = None,
) -> int:
    """Collect MATE records and write them to disk.

    If a local JSONL file is provided, read records directly from it.
    If a zip archive is provided or downloaded, read records from the zip.
    Write every record to the target JSONL file.
    Return the total number of written records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    if jsonl_path and jsonl_path.exists():
        records_iter = stream_mate_from_jsonl(jsonl_path=jsonl_path, max_items=max_items)
    else:
        archive_path = zip_path if zip_path and zip_path.exists() else download_mate_zip(download_url)
        records_iter = stream_mate_from_zip(archive_path, max_items=max_items)

    with output_path.open("w", encoding="utf-8") as writer:
        for record in records_iter:
            writer.write(json.dumps(record) + "\n")
            count += 1
            if max_items is not None and count >= max_items:
                break

    return count


def main() -> None:
    """Run the MATE dataset collection script from the command line."""
    parser = argparse.ArgumentParser(description="Collect MATE dataset records.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/mate.jsonl"),
        help="Path for output JSONL file.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Path to an existing MATE zip file.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=None,
        help="Path to an existing MATE JSONL file.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=MATE_TESTSET_URL,
        help="URL to download the dataset zip from.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of records to emit.",
    )

    args = parser.parse_args()
    total = collect_mate(
        output_path=args.output,
        zip_path=args.zip_path,
        jsonl_path=args.jsonl_path,
        download_url=args.url,
        max_items=args.max_items,
    )
    print(f"Collected {total} MATE records into {args.output}")


if __name__ == "__main__":
    main()
