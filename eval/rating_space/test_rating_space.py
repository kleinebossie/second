"""Unit tests for the Second human rating arena.

This module tests board rendering, vote persistence, blind randomization,
and rating session state management.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import chess
import pytest

from eval.rating_space.app import (
    ActiveItem,
    ComparisonPair,
    RatingSession,
    create_ui,
    filter_pairs,
    load_pairs,
    prepare_blind_item,
    record_vote,
    resolve_winner,
)
from eval.rating_space.board import (
    extract_position_metadata,
    parse_move,
    render_board_svg,
)
from eval.rating_space.storage import (
    Vote,
    compute_summary_stats,
    format_summary_markdown,
    load_local_votes,
    save_vote_locally,
    upload_votes_to_hub,
)


# ============================================================================
# Board Rendering Tests (board.py)
# ============================================================================


def test_parse_move_valid_san() -> None:
    """Test parsing a legal move with SAN notation."""
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move = parse_move(board, "Ng5")
    assert move is not None
    assert move.uci() == "f3g5"


def test_parse_move_valid_uci() -> None:
    """Test parsing a legal move with UCI notation."""
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move = parse_move(board, "f3g5")
    assert move is not None
    assert move.uci() == "f3g5"


def test_parse_move_invalid_and_empty() -> None:
    """Test parsing invalid move strings and empty inputs."""
    board = chess.Board()
    assert parse_move(board, None) is None
    assert parse_move(board, "") is None
    assert parse_move(board, "   ") is None
    assert parse_move(board, "invalid") is None
    assert parse_move(board, "e5") is None


def test_render_board_svg_valid() -> None:
    """Test SVG generation for a legal chess position."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    svg = render_board_svg(fen, move_san="Ng5")
    assert "<svg" in svg
    assert "</svg>" in svg


def test_render_board_svg_black_orientation() -> None:
    """Test board orientation defaults to Black when Black moves."""
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR b KQkq - 1 2"
    svg = render_board_svg(fen, move_san="Nf6")
    assert "<svg" in svg


def test_render_board_svg_post_move() -> None:
    """Test rendering the position after applying the played move."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    svg = render_board_svg(fen, move_san="Ng5", show_post_move=True)
    assert "<svg" in svg


def test_render_board_svg_without_move() -> None:
    """Test rendering a position when no move is provided."""
    svg = render_board_svg(chess.STARTING_FEN)
    assert "<svg" in svg


def test_render_board_svg_invalid_fen() -> None:
    """Test SVG error banner generation for invalid FEN strings."""
    svg = render_board_svg("invalid_fen_string")
    assert "<svg" in svg
    assert "Invalid chess position" in svg


def test_extract_position_metadata() -> None:
    """Test extracting game metadata from a valid position."""
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    meta = extract_position_metadata(fen, move_san="Ng5")
    assert meta["valid"] is True
    assert meta["side_to_move"] == "White"
    assert meta["move_number"] == 4
    assert meta["is_check"] is False
    assert meta["move_san"] == "Ng5"


def test_extract_position_metadata_in_check() -> None:
    """Test check status detection in position metadata."""
    # Black king on e8 is checked by White queen on e7.
    fen = "rnb1kbnr/pppp1Qpp/8/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"
    meta = extract_position_metadata(fen, move_san="Kxf7")
    assert meta["valid"] is True
    assert meta["is_check"] is True


def test_extract_position_metadata_invalid_fen() -> None:
    """Test metadata extraction on an invalid FEN string."""
    meta = extract_position_metadata("not_a_fen")
    assert meta["valid"] is False
    assert meta["side_to_move"] == "Unknown"


# ============================================================================
# Storage Tests (storage.py)
# ============================================================================


def test_vote_serialization() -> None:
    """Test serialization and deserialization of Vote instances."""
    vote = Vote(
        timestamp="2026-11-05T12:00:00Z",
        session_id="session_123",
        rater_id="test_rater",
        item_id="pair_001",
        comparison_set="second_vs_base_qwen",
        fen="fen_string",
        move_san="Ng5",
        level="club",
        detail="short",
        model_a="Second-1.0-2B",
        model_b="Base Qwen 3.5 2B",
        choice="A",
        winner_model="Second-1.0-2B",
    )
    raw_dict = vote.to_dict()
    assert raw_dict["item_id"] == "pair_001"
    assert raw_dict["winner_model"] == "Second-1.0-2B"

    restored = Vote.from_dict(raw_dict)
    assert restored.vote_id == vote.vote_id
    assert restored.rater_id == "test_rater"
    assert restored.choice == "A"


def test_local_storage_lifecycle(tmp_path: Path) -> None:
    """Test saving and loading votes from a local JSONL file."""
    votes_file = tmp_path / "subfolder" / "votes.jsonl"

    assert load_local_votes(votes_file) == []

    v1 = Vote(
        timestamp="2026-11-05T12:00:00Z",
        session_id="s1",
        rater_id="rater_1",
        item_id="pair_001",
        comparison_set="second_vs_base_qwen",
        fen="fen_1",
        move_san="Ng5",
        level="club",
        detail="short",
        model_a="Second-1.0-2B",
        model_b="Base Qwen 3.5 2B",
        choice="A",
        winner_model="Second-1.0-2B",
    )
    v2 = Vote(
        timestamp="2026-11-05T12:05:00Z",
        session_id="s1",
        rater_id="rater_1",
        item_id="pair_002",
        comparison_set="second_vs_gemini_flash",
        fen="fen_2",
        move_san="Qe5+",
        level="beginner",
        detail="short",
        model_a="Gemini Flash",
        model_b="Second-1.0-2B",
        choice="same",
        winner_model="tie",
    )

    save_vote_locally(v1, filepath=votes_file)
    save_vote_locally(v2, filepath=votes_file)

    loaded = load_local_votes(votes_file)
    assert len(loaded) == 2
    assert loaded[0].item_id == "pair_001"
    assert loaded[1].winner_model == "tie"


def test_upload_votes_to_hub_missing_credentials(tmp_path: Path) -> None:
    """Test Hub upload returns False when credentials are not configured."""
    votes_file = tmp_path / "votes.jsonl"
    votes_file.write_text("{}\n", encoding="utf-8")

    result = upload_votes_to_hub(
        filepath=votes_file,
        repo_id=None,
        token=None,
    )
    assert result is False


def test_upload_votes_to_hub_missing_file(tmp_path: Path) -> None:
    """Test Hub upload returns False when the file does not exist."""
    missing_file = tmp_path / "missing.jsonl"
    result = upload_votes_to_hub(
        filepath=missing_file,
        repo_id="owner/repo",
        token="hf_test_token",
    )
    assert result is False


def test_upload_votes_to_hub_success(tmp_path: Path) -> None:
    """Test successful Hub upload with a mocked Hugging Face client."""
    votes_file = tmp_path / "votes.jsonl"
    votes_file.write_text('{"item_id": "test"}\n', encoding="utf-8")

    mock_client = MagicMock()
    result = upload_votes_to_hub(
        filepath=votes_file,
        repo_id="test_org/test_repo",
        token="hf_mock_token",
        api=mock_client,
    )

    assert result is True
    mock_client.upload_file.assert_called_once()
    kwargs = mock_client.upload_file.call_args[1]
    assert kwargs["path_in_repo"] == "votes.jsonl"
    assert kwargs["repo_id"] == "test_org/test_repo"
    assert kwargs["repo_type"] == "dataset"


def test_compute_summary_stats_and_markdown() -> None:
    """Test summary statistics calculation and markdown table generation."""
    v1 = Vote(
        timestamp="2026-11-05T12:00:00Z",
        session_id="s1",
        rater_id="rater_1",
        item_id="p1",
        comparison_set="second_vs_base_qwen",
        fen="fen",
        move_san="Ng5",
        level="club",
        detail="short",
        model_a="Second-1.0-2B",
        model_b="Base Qwen 3.5 2B",
        choice="A",
        winner_model="Second-1.0-2B",
    )
    v2 = Vote(
        timestamp="2026-11-05T12:01:00Z",
        session_id="s1",
        rater_id="rater_1",
        item_id="p2",
        comparison_set="second_vs_base_qwen",
        fen="fen",
        move_san="Ng5",
        level="club",
        detail="short",
        model_a="Second-1.0-2B",
        model_b="Base Qwen 3.5 2B",
        choice="B",
        winner_model="Base Qwen 3.5 2B",
    )
    v3 = Vote(
        timestamp="2026-11-05T12:02:00Z",
        session_id="s1",
        rater_id="rater_1",
        item_id="p3",
        comparison_set="second_vs_base_qwen",
        fen="fen",
        move_san="Ng5",
        level="club",
        detail="short",
        model_a="Second-1.0-2B",
        model_b="Base Qwen 3.5 2B",
        choice="same",
        winner_model="tie",
    )

    stats = compute_summary_stats([v1, v2, v3])
    set_stats = stats["by_set"]["second_vs_base_qwen"]
    assert set_stats["total_votes"] == 3
    assert set_stats["ties"] == 1
    assert set_stats["winner_counts"]["Second-1.0-2B"] == 1
    assert set_stats["winner_counts"]["Base Qwen 3.5 2B"] == 1
    assert set_stats["win_rates"]["Second-1.0-2B"] == pytest.approx(0.3333, abs=0.01)
    assert set_stats["decisive_win_rates"]["Second-1.0-2B"] == 0.5
    assert set_stats["tie_rate"] == pytest.approx(0.3333, abs=0.01)

    markdown = format_summary_markdown(stats)
    assert "second_vs_base_qwen" in markdown
    assert "Overall Total" in markdown


# ============================================================================
# Core Logic & App Tests (app.py)
# ============================================================================


def test_comparison_pair_from_dict() -> None:
    """Test creating ComparisonPair instances from various dictionary schemas."""
    data = {
        "id": "pair_test",
        "comparison_set": "second_vs_base_qwen",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "move_san": "Ng5",
        "move_uci": "f3g5",
        "level": "club",
        "detail": "short",
        "model_second_name": "Second-1.0-2B",
        "model_second_text": "Explanation from Second.",
        "model_competitor_name": "Base Qwen",
        "model_competitor_text": "Explanation from Base Qwen.",
    }
    pair = ComparisonPair.from_dict(data)
    assert pair.id == "pair_test"
    assert pair.model_1_name == "Second-1.0-2B"
    assert pair.model_2_name == "Base Qwen"
    assert pair.model_1_text == "Explanation from Second."


def test_load_sample_pairs() -> None:
    """Test loading comparison pairs from the bundled sample_pairs.jsonl file.

    Public sample_pairs.jsonl must contain only open comparison sets.
    Private second_vs_chesscom pairs load only from private files or Hub datasets.
    """
    pairs = load_pairs(include_private=False)
    assert len(pairs) >= 6
    sets = {p.comparison_set for p in pairs}
    assert "second_vs_base_qwen" in sets
    assert "second_vs_gemini_flash" in sets
    assert "second_vs_chesscom" not in sets


def test_filter_pairs() -> None:
    """Test filtering pairs by comparison set name."""
    pairs = load_pairs(include_private=False)
    all_pairs = filter_pairs(pairs, "All Comparison Sets")
    assert len(all_pairs) == len(pairs)

    qwen_pairs = filter_pairs(pairs, "second_vs_base_qwen")
    assert len(qwen_pairs) > 0
    assert all(p.comparison_set == "second_vs_base_qwen" for p in qwen_pairs)


def test_prepare_blind_item_randomization() -> None:
    """Test blind randomization of candidate explanation positions."""
    pair = ComparisonPair(
        id="p1",
        comparison_set="second_vs_base_qwen",
        fen=chess.STARTING_FEN,
        move_san="e4",
        model_1_name="Model_Alpha",
        model_1_text="Text Alpha",
        model_2_name="Model_Beta",
        model_2_text="Text Beta",
    )

    # With seed 0, verify consistent assignment.
    item_0 = prepare_blind_item(pair, seed=0)
    # With seed 1, verify opposing assignment.
    item_1 = prepare_blind_item(pair, seed=1)

    # Explanation text must match the model name assigned to that slot.
    for item in (item_0, item_1):
        if item.model_a_name == "Model_Alpha":
            assert item.explanation_a == "Text Alpha"
            assert item.model_b_name == "Model_Beta"
            assert item.explanation_b == "Text Beta"
        else:
            assert item.model_a_name == "Model_Beta"
            assert item.explanation_a == "Text Beta"
            assert item.model_b_name == "Model_Alpha"
            assert item.explanation_b == "Text Alpha"


def test_resolve_winner() -> None:
    """Test resolving winning models from evaluator choices."""
    model_a = "Second-1.0-2B"
    model_b = "Base Qwen 3.5 2B"

    assert resolve_winner("A", model_a, model_b) == model_a
    assert resolve_winner("B", model_a, model_b) == model_b
    assert resolve_winner("same", model_a, model_b) == "tie"
    assert resolve_winner("tie", model_a, model_b) == "tie"

    with pytest.raises(ValueError):
        resolve_winner("invalid_choice", model_a, model_b)


def test_record_vote(tmp_path: Path) -> None:
    """Test recording a vote and saving it to local JSONL."""
    pair = ComparisonPair(
        id="p100",
        comparison_set="second_vs_base_qwen",
        fen=chess.STARTING_FEN,
        move_san="e4",
        model_1_name="Second-1.0-2B",
        model_1_text="Explanation A",
        model_2_name="Base Qwen",
        model_2_text="Explanation B",
    )
    active = ActiveItem(
        pair=pair,
        model_a_name="Second-1.0-2B",
        model_b_name="Base Qwen",
        explanation_a="Explanation A",
        explanation_b="Explanation B",
    )

    votes_file = tmp_path / "test_votes.jsonl"
    vote = record_vote(
        active_item=active,
        choice="A",
        rater_id="joe_eval",
        session_id="session_01",
        votes_path=votes_file,
        sync_hf=False,
    )

    assert vote.item_id == "p100"
    assert vote.winner_model == "Second-1.0-2B"
    assert vote.rater_id == "joe_eval"

    saved = load_local_votes(votes_file)
    assert len(saved) == 1
    assert saved[0].winner_model == "Second-1.0-2B"


def test_rating_session_flow(tmp_path: Path) -> None:
    """Test full navigation flow of a RatingSession."""
    pairs = load_pairs(include_private=False)
    votes_file = tmp_path / "session_votes.jsonl"

    session = RatingSession(
        pairs=pairs,
        comparison_set="second_vs_base_qwen",
        rater_id="test_user",
        votes_path=votes_file,
        sync_hf=False,
        seed=42,
    )

    assert session.current_index == 0
    assert session.active_item is not None
    assert "Item 1 of" in session.get_progress_text()

    # Verify board SVG and metadata generation.
    board_svg = session.get_board_svg()
    assert "<svg" in board_svg
    meta_md = session.get_metadata_markdown()
    assert "Position Details" in meta_md

    # Record first vote.
    vote_1 = session.record_choice("A")
    assert vote_1 is not None
    assert session.votes_count == 1
    assert session.current_index == 1

    # Skip second item.
    session.skip()
    assert session.current_index == 2

    # Switch comparison set.
    session.set_comparison_set("second_vs_gemini_flash")
    assert session.current_index == 0
    assert session.comparison_set == "second_vs_gemini_flash"


def test_create_ui_guard() -> None:
    """Test that create_ui raises RuntimeError when Gradio is absent."""
    import eval.rating_space.app as rating_app

    if rating_app.gr is None:
        with pytest.raises(RuntimeError, match="Gradio is not installed"):
            create_ui()
    else:
        demo = create_ui()
        assert demo is not None
