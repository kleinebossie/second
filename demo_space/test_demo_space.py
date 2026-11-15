"""Unit tests for the Second Hugging Face demo Space."""

from __future__ import annotations

from pathlib import Path
import pytest

from demo_space.app import (
    POSITION_ONLY_LABEL,
    clean_model_output,
    find_stockfish_binary,
    parse_chess_input,
    render_demo_board_svg,
    analyze_with_stockfish,
)
from demo_space.prepare_hf_space import stage_demo_space


def test_parse_pgn_extracts_all_moves() -> None:
    """Verify PGN input returns every mainline ply in order."""
    parsed = parse_chess_input("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6")
    assert parsed.mode == "pgn"
    assert len(parsed.entries) == 6
    assert parsed.entries[0].move_san == "e4"
    assert parsed.entries[-1].move_san == "a6"
    assert parsed.default_label == parsed.entries[-1].label


def test_parse_fen_extracts_position_overview_and_legal_moves() -> None:
    """Verify FEN input returns Position Overview default plus legal move choices."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4"
    parsed = parse_chess_input(fen)
    assert parsed.mode == "fen"
    assert parsed.base_fen == fen
    assert parsed.default_label == POSITION_ONLY_LABEL
    assert any(e.move_uci == "d7d5" for e in parsed.entries)


def test_parse_invalid_input_raises() -> None:
    """Verify invalid text raises a clear ValueError."""
    with pytest.raises(ValueError):
        parse_chess_input("   ")
    with pytest.raises(ValueError):
        parse_chess_input("not a fen and not a valid pgn @@@")


def test_clean_model_output_strips_think_tags() -> None:
    """Verify stray thinking tags are removed cleanly."""
    assert clean_model_output("reason</think>\n\nPlain answer.") == "Plain answer."
    assert clean_model_output("<think>a</think>  Hello world. ") == "Hello world."


def test_render_demo_board_svg() -> None:
    """Verify SVG rendering succeeds for both position and move arrow."""
    svg = render_demo_board_svg(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        move_uci="e7e5",
        best_move_uci="c7c5",
    )
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_stockfish_integration_when_available() -> None:
    """Verify Stockfish analysis includes fen key and multi-pv lines."""
    if not find_stockfish_binary():
        pytest.skip("Stockfish binary not available")
    eval_before, eval_after = analyze_with_stockfish(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        move_uci="e7e5",
        depth=8,
    )
    assert "fen" in eval_before
    assert len(eval_before["pvs"]) >= 1
    assert eval_after is not None and "fen" in eval_after


def test_stage_demo_space(tmp_path: Path) -> None:
    """Verify prepare_hf_space copies all files needed for standalone Space deployment."""
    staged = stage_demo_space(tmp_path / "space")
    for required in ("README.md", "app.py", "requirements.txt", "packages.txt", "core/python/facts.py"):
        assert (staged / required).is_file()
