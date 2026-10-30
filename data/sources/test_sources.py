"""Unit tests for chess data source collectors.

This module tests all data parsers and samplers using offline fixtures.
It verifies schema validity, license assignments, and move extraction accuracy.
"""

from __future__ import annotations

import io
import json
import random
import tempfile
import zipfile
from pathlib import Path

import chess
import chess.pgn
import pytest
import zstandard

from data.sources import (
    ChessRecord,
    collect_gameknot,
    collect_lichess_games,
    collect_lichess_studies,
    detect_study_license,
    format_moves_up_to,
    is_acceptable_rating,
    make_record,
    parse_elo,
    parse_gameknot_html,
    parse_gameknot_pgn,
    parse_mate_entry,
    parse_study_pgn,
    sample_game_positions,
    stream_mate_from_zip,
    strip_lichess_tags,
    validate_record,
)


def test_schema_make_and_validate() -> None:
    """Test record creation and validation with valid data.

    Create a complete record.
    Verify that validate_record returns True.
    """
    record = make_record(
        source="gameknot",
        license_name="proprietary",
        redistributable=False,
        game_id="gk_123",
        pgn="1. e4 e5",
        fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        move_uci="g1f3",
        move_san="Nf3",
        comment="Knight attacks the e5 pawn.",
        white_elo=1500,
        black_elo=1450,
    )
    assert validate_record(record) is True


def test_schema_validate_invalid_keys() -> None:
    """Test validation when required keys are missing or invalid.

    Provide an incomplete dictionary.
    Verify that validate_record returns False.
    """
    incomplete: dict[str, object] = {
        "source": "gameknot",
        "license": "proprietary",
    }
    assert validate_record(incomplete) is False  # type: ignore[arg-type]


def test_parse_elo() -> None:
    """Test Elo parsing from various input representations.

    Test valid integers, digit strings, and missing values.
    Verify the expected parsed output.
    """
    assert parse_elo(1800) == 1800
    assert parse_elo("2100") == 2100
    assert parse_elo("?") is None
    assert parse_elo(None) is None
    assert parse_elo("approx 1650 elo") == 1650


def test_format_moves_up_to() -> None:
    """Test standard PGN move formatting from initial board.

    Replay moves on a standard board.
    Verify that the formatted text matches standard chess notation.
    """
    board = chess.Board()
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]
    formatted = format_moves_up_to(board, moves)
    assert formatted == "1. e4 e5 2. Nf3"


def test_gameknot_pgn_parser() -> None:
    """Test parsing GameKnot PGN text into records.

    Provide a fixture with mainline comments and headers.
    Verify that extracted records contain the expected moves and comments.
    """
    pgn_sample = """[Event "Tournament Round 1"]
[Site "http://gameknot.com/?gm=55555"]
[White "MasterA"]
[Black "MasterB"]
[WhiteElo "1950"]
[BlackElo "1920"]

1. e4 {Strong king pawn opening} e5 2. Nf3 Nc6 3. Bc4 {Italian Game setup} *
"""
    records = parse_gameknot_pgn(pgn_sample)
    assert len(records) == 2

    first = records[0]
    assert first["source"] == "gameknot"
    assert first["license"] == "proprietary"
    assert first["redistributable"] is False
    assert first["game_id"] == "55555"
    assert first["move_san"] == "e4"
    assert first["move_uci"] == "e2e4"
    assert first["comment"] == "Strong king pawn opening"
    assert first["white_elo"] == 1950
    assert first["black_elo"] == 1920
    assert validate_record(first) is True

    second = records[1]
    assert second["move_san"] == "Bc4"
    assert second["move_uci"] == "f1c4"
    assert second["comment"] == "Italian Game setup"
    assert second["pgn"] == "1. e4 e5 2. Nf3 Nc6 3. Bc4"


def test_gameknot_html_script_parser() -> None:
    """Test parsing GameKnot HTML with embedded script variables.

    Provide HTML containing game_movelist and game_notes.
    Verify that decoded moves and notes produce valid records.
    """
    html_sample = """
    <html>
    <head><title>Annotation</title></head>
    <body>
    <script>
    game_id = 99999;
    game_movelist = 'e2e4-e7e5-g1f3-b8c6-';
    game_notes = [ "", "Pawn advances", "", "Knight develops", "" ];
    </script>
    </body>
    </html>
    """
    records = parse_gameknot_html(html_sample)
    assert len(records) == 2

    assert records[0]["game_id"] == "99999"
    assert records[0]["move_san"] == "e4"
    assert records[0]["comment"] == "Pawn advances"

    assert records[1]["move_san"] == "Nf3"
    assert records[1]["comment"] == "Knight develops"


def test_gameknot_collect_offline(tmp_path: Path) -> None:
    """Test GameKnot collection with an offline cached JSONL file.

    Create a mock cache file with one GameKnot game and one other game.
    Run collect_gameknot and verify output records.
    """
    cache_path = tmp_path / "cache.jsonl"
    output_path = tmp_path / "gameknot_out.jsonl"

    item_gk = {
        "metadata": {"pipeline_key": "gameknot_game_123"},
        "text": '[Event "GK Game"]\n[Site "http://gameknot.com/"]\n\n1. d4 {Queen pawn} d5 *',
    }
    item_other = {
        "metadata": {"pipeline_key": "random_456"},
        "text": '[Event "Other"]\n1. c4 c5 *',
    }

    with cache_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(item_gk) + "\n")
        file.write(json.dumps(item_other) + "\n")

    count = collect_gameknot(output_path=output_path, cache_file=cache_path)
    assert count == 1
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        data = json.loads(file.readline())
        assert data["source"] == "gameknot"
        assert data["move_san"] == "d4"
        assert data["comment"] == "Queen pawn"


def test_strip_lichess_tags() -> None:
    """Test stripping visual markup from Lichess comments.

    Provide comments with drawing, square, and eval annotations.
    Verify that all markup tags are removed cleanly.
    """
    raw = "Great move! [%cal Gc2c4,Re1e8] [%csl Gc4] [%eval 0.45] Keeps pressure."
    cleaned = strip_lichess_tags(raw)
    assert cleaned == "Great move! Keeps pressure."


def test_detect_study_license() -> None:
    """Test license detection logic for Lichess studies.

    Verify that open license keywords set redistributable to True.
    Verify that standard studies receive Lichess Terms of Service.
    """
    headers_cc0 = chess.pgn.Headers({"License": "Creative Commons CC0 1.0 Universal"})
    lic, redist = detect_study_license(headers_cc0)
    assert lic == "CC0-1.0"
    assert redist is True

    headers_tos = chess.pgn.Headers({"Event": "My private study"})
    lic_tos, redist_tos = detect_study_license(headers_tos)
    assert lic_tos == "Lichess Terms of Service"
    assert redist_tos is False

    headers_empty = chess.pgn.Headers({})
    lic_pd, redist_pd = detect_study_license(
        headers_empty,
        source_hint="pgnlib_famous_games",
    )
    assert lic_pd == "Public Domain"
    assert redist_pd is True


def test_parse_study_pgn() -> None:
    """Test parsing Lichess study PGN content.

    Provide a study chapter with markup tags in comments.
    Verify that records are extracted with tags removed.
    """
    study_pgn = """[Event "Traxler Counter-Attack: Chapter 1"]
[Site "https://lichess.org/study/WLyfoXTJ/xdA6LWme"]
[White "?"]
[Black "?"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 {Fried Liver line [%cal Gg5f7] [%eval 0.8]} 4... Bc5 {Traxler invitation} *
"""
    records = parse_study_pgn(study_pgn)
    assert len(records) == 2

    r0 = records[0]
    assert r0["source"] == "lichess_studies"
    assert r0["game_id"] == "WLyfoXTJ_xdA6LWme"
    assert r0["move_san"] == "Ng5"
    assert r0["comment"] == "Fried Liver line"
    assert r0["license"] == "Lichess Terms of Service"
    assert r0["redistributable"] is False

    r1 = records[1]
    assert r1["move_san"] == "Bc5"
    assert r1["comment"] == "Traxler invitation"


def test_parse_study_require_open_license() -> None:
    """Test filtering studies when open license is required.

    Verify that studies under standard ToS are skipped.
    Verify that CC0 studies are kept.
    """
    study_pgn = """[Event "Open Study"]
[Site "https://lichess.org/study/ABC12345"]
[License "CC0"]

1. e4 {King pawn} e5 *
"""
    records = parse_study_pgn(study_pgn, require_open_license=True)
    assert len(records) == 1
    assert records[0]["redistributable"] is True


def test_mate_parse_entry_tactic() -> None:
    """Test parsing a tactic record from the MATE dataset.

    Verify that candidate moves and winner are parsed.
    Verify move legality on the board.
    """
    entry = {
        "instruction": "Expert chess player.",
        "input": (
            'The FEN of the given chess board is "8/8/8/7R/8/1BP3kP/Pr6/6K1 b - - 0 36". '
            "Which move is better? "
            "MoveA:b2b1,tactic b2b1 b3d1 b1d1 Checkmate!  "
            "MoveB:g3f4,tactic g3f4 h5d5 b2b1 g1f2 b1h1 d5d3  "
        ),
        "output": "MoveA:b2b1",
    }
    record = parse_mate_entry(entry)
    assert record is not None
    assert record["source"] == "mate"
    assert record["license"] == "MIT"
    assert record["redistributable"] is True
    assert record["fen"] == "8/8/8/7R/8/1BP3kP/Pr6/6K1 b - - 0 36"
    assert record["move_uci"] == "b2b1"
    assert record["move_san"] == "Rb1+"
    assert "Checkmate!" in record["comment"]
    assert validate_record(record) is True


def test_mate_parse_entry_strategy() -> None:
    """Test parsing a strategy record from the MATE dataset.

    Verify that the winning strategy explanation is extracted.
    """
    entry = {
        "instruction": "Expert chess player.",
        "input": (
            'The FEN of the given chess board is "r3k2r/pb2bpBp/1p1np3/8/8/P2B4/1PPN1PPP/R4RK1 b kq - 0 15". '
            "Which move is better? "
            "MoveA:a7a5 Increase control of the flank. "
            "MoveB:h8g8 Relocate rook to the open g-file with tempo. "
        ),
        "output": "MoveB:h8g8",
    }
    record = parse_mate_entry(entry)
    assert record is not None
    assert record["move_uci"] == "h8g8"
    assert record["move_san"] == "Rg8"
    assert "open g-file" in record["comment"]


def test_mate_parse_illegal_move() -> None:
    """Test that illegal moves in MATE entries are rejected.

    Provide an impossible move for the board position.
    Verify that parse_mate_entry returns None.
    """
    entry = {
        "instruction": "Expert chess player.",
        "input": (
            'The FEN of the given chess board is "8/8/8/8/8/8/4K3/4k3 w - - 0 1". '
            "Which move is better? MoveA:e2e8 King jumps across board. MoveB:e2e3 King moves forward. "
        ),
        "output": "MoveA:e2e8",
    }
    assert parse_mate_entry(entry) is None


def test_mate_stream_from_zip() -> None:
    """Test streaming records from a zip archive in memory.

    Create a zip archive containing a mock JSONL file.
    Stream records and verify count.
    """
    entry = {
        "instruction": "Test",
        "input": (
            'The FEN of the given chess board is "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1". '
            "Which move is better? MoveA:e2e4 Open the center. MoveB:h2h4 Flank pawn. "
        ),
        "output": "MoveA:e2e4",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data/sample.jsonl", json.dumps(entry) + "\n")

    records = list(stream_mate_from_zip(buf.getvalue()))
    assert len(records) == 1
    assert records[0]["move_san"] == "e4"


def test_lichess_games_rating_filter() -> None:
    """Test player rating filtering criteria.

    Verify that missing ratings are rejected.
    Verify that ratings in the focus range are accepted.
    Verify that outside ratings respect the quota ratio.
    """
    assert is_acceptable_rating(None, 1500, count_focus=10, count_other=1) is False
    assert is_acceptable_rating(1400, 1600, count_focus=10, count_other=1) is True
    # At the start with zero focus games, outside range should wait
    assert is_acceptable_rating(2200, 2200, count_focus=0, count_other=0) is False
    # With 10 focus games and 1 outside game, outside range is under 20%
    assert is_acceptable_rating(2200, 2200, count_focus=10, count_other=1) is True


def test_lichess_games_sample_positions() -> None:
    """Test position sampling from a mainline game.

    Verify that positions are sampled from opening and middlegame stages.
    Verify that the first three plies are skipped.
    """
    pgn_sample = """[Event "Rated Classical game"]
[Site "https://lichess.org/abc12345"]
[White "PlayerA"]
[Black "PlayerB"]
[WhiteElo "1500"]
[BlackElo "1550"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 9. exd5 Nxd5 10. O-O O-O *
"""
    game = chess.pgn.read_game(io.StringIO(pgn_sample))
    assert game is not None

    rng = random.Random(42)
    positions = sample_game_positions(game, rng=rng)
    assert 1 <= len(positions) <= 3

    for pos in positions:
        assert pos["source"] == "lichess_games"
        assert pos["license"] == "CC0-1.0"
        assert pos["redistributable"] is True
        assert pos["comment"] == ""
        assert pos["white_elo"] == 1500
        assert pos["black_elo"] == 1550
        assert validate_record(pos) is True


def test_lichess_games_collect_offline(tmp_path: Path) -> None:
    """Test Lichess games collection with a compressed local fixture.

    Create a small zstandard-compressed PGN file.
    Run collect_lichess_games and verify output records.
    """
    pgn_content = b"""[Event "Rated Blitz game"]
[Site "https://lichess.org/testgame1"]
[WhiteElo "1600"]
[BlackElo "1650"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ *
"""
    cctx = zstandard.ZstdCompressor()
    compressed = cctx.compress(pgn_content)

    zst_path = tmp_path / "sample.pgn.zst"
    output_path = tmp_path / "lichess_games_out.jsonl"
    zst_path.write_bytes(compressed)

    count = collect_lichess_games(
        output_path=output_path,
        local_path=zst_path,
        max_positions=5,
    )
    assert count >= 1
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        record = json.loads(file.readline())
        assert record["source"] == "lichess_games"
        assert record["white_elo"] == 1600
        assert record["black_elo"] == 1650
