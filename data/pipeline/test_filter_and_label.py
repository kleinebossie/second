"""Unit tests for comment filtering and level labeling pipelines.

This test module verifies comment cleaning, rejection filters, and level assignment.
It tests the deterministic filter sequence and detail tier splitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from data.pipeline.filter_comments import (
    FilterStats,
    assign_detail_and_truncate,
    clean_comment,
    compute_shingles,
    filter_comment,
    has_chess_content,
    has_offensive_content,
    is_near_duplicate,
    pre_filter_comment,
    truncate_to_sentences,
)
from data.pipeline.label_levels import (
    calculate_average_sentence_length,
    has_jargon_or_eval,
    label_level,
)

# Italian Game FEN
ITALIAN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


# --- 1. Level Labeling Tests ---


def test_calculate_average_sentence_length() -> None:
    """Verify calculation of words per sentence."""
    text = "White plays Ng5. Black defends f7 with d5."
    # 3 words + 5 words = 8 words / 2 sentences = 4.0
    avg = calculate_average_sentence_length(text)
    assert avg == 4.0


def test_has_jargon_or_eval() -> None:
    """Verify detection of chess jargon and evaluation numbers."""
    assert has_jargon_or_eval("White plays prophylaxis.") is True
    assert has_jargon_or_eval("The knight reaches an outpost.") is True
    assert has_jargon_or_eval("The engine evaluation is +1.5.") is True
    assert has_jargon_or_eval("White attacks the center pawn.") is False


def test_label_level_beginner() -> None:
    """Verify that simple short sentences receive beginner label."""
    text = "White attacks the pawn on f7. Black must defend with d5."
    assert label_level(text) == "beginner"


def test_label_level_club_from_jargon() -> None:
    """Verify that presence of jargon assigns club label."""
    text = "White plays prophylaxis to stop counterplay."
    assert label_level(text) == "club"


def test_label_level_club_from_sentence_length() -> None:
    """Verify that long sentences receive club label."""
    long_sentence = (
        "White chooses to advance the knight aggressively toward the enemy king territory "
        "because the unprotected state of the opposing f7 pawn invites an immediate two piece attack."
    )
    assert label_level(long_sentence) == "club"


# --- 2. Comment Cleaning Tests ---


def test_clean_comment_pgn_tags() -> None:
    """Verify removal of PGN engine and arrow tags."""
    raw = "Good move. [%eval 0.35] [%clk 0:03:12] [%cal Gf3e5] White controls the center."
    cleaned = clean_comment(raw)
    assert "[%eval" not in cleaned
    assert "[%clk" not in cleaned
    assert "[%cal" not in cleaned
    assert "Good move. White controls the center." == cleaned


def test_clean_comment_move_number_prefix() -> None:
    """Verify stripping of leading move number prefixes."""
    assert clean_comment("14. White attacks the center.") == "White attacks the center."
    assert clean_comment("14... d5 opens lines for Black.") == "opens lines for Black."
    assert clean_comment("14...Nxe5 captures the pawn.") == "captures the pawn."


def test_clean_comment_annotation_glyphs() -> None:
    """Verify removal of standalone annotation glyphs."""
    raw = "White plays d4 !? and opens the center."
    cleaned = clean_comment(raw)
    assert "!?" not in cleaned
    assert "White plays d4 and opens the center." == cleaned


# --- 3. Filter Sequence Tests ---


def test_filter_cleaned_empty_or_short() -> None:
    """Verify rejection of comments with fewer than 8 words."""
    stats = FilterStats()
    short_comment = "Good move here."
    kept, reason, _ = filter_comment(
        text=short_comment,
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept is False
    assert reason == "cleaned_empty_or_short"
    assert stats.cleaned_empty_or_short == 1


def test_filter_not_english() -> None:
    """Verify rejection of non-English commentary."""
    stats = FilterStats()
    french_text = "Les blancs developpent le cavalier pour attaquer le pion faible f7."
    kept, reason, _ = filter_comment(
        text=french_text,
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept is False
    assert reason == "not_english"
    assert stats.not_english == 1


def test_filter_no_chess_content() -> None:
    """Verify rejection of text without chess terminology."""
    stats = FilterStats()
    generic_text = "I really enjoyed eating my delicious lunch while reading this book."
    kept, reason, _ = filter_comment(
        text=generic_text,
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept is False
    assert reason == "no_chess_content"
    assert stats.no_chess_content == 1


def test_filter_illegal_move() -> None:
    """Verify rejection of comments naming illegal moves."""
    stats = FilterStats()
    illegal_text = "White should play Qxe5 and win the opponent piece immediately."
    kept, reason, _ = filter_comment(
        text=illegal_text,
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept is False
    assert reason == "illegal_move_or_wrong_piece"
    assert stats.illegal_move_or_wrong_piece == 1


def test_filter_wrong_piece_on_square() -> None:
    """Verify rejection of comments with wrong piece on square."""
    stats = FilterStats()
    wrong_piece_text = "The rook on e4 controls the central files effectively."
    kept, reason, _ = filter_comment(
        text=wrong_piece_text,
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept is False
    assert reason == "illegal_move_or_wrong_piece"
    assert stats.illegal_move_or_wrong_piece == 1


def test_filter_tone_disagrees_with_verdict() -> None:
    """Verify rejection when tone conflicts with engine verdict."""
    stats = FilterStats()
    praise_on_blunder = (
        "This great move wins the opponent queen through a brilliant tactical shot."
    )
    kept, reason, _ = filter_comment(
        text=praise_on_blunder,
        fen=ITALIAN_FEN,
        verdict="Blunder",
        stats=stats,
    )
    assert kept is False
    assert reason == "tone_disagrees_with_verdict"
    assert stats.tone_disagrees_with_verdict == 1


def test_filter_links_usernames_offensive() -> None:
    """Verify rejection of comments containing links, handles, or profanity."""
    stats = FilterStats()
    link_text = "Check out https://chess.com for more information about this opening."
    kept, reason, _ = filter_comment(
        text=link_text,
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept is False
    assert reason == "links_usernames_offensive"

    handle_text = "Good move as suggested by @grandmaster in the opening video."
    kept2, reason2, _ = filter_comment(
        text=handle_text,
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept2 is False
    assert reason2 == "links_usernames_offensive"

    profane_text = "This knight blunder is total bullshit and loses the game quickly."
    kept3, reason3, _ = filter_comment(
        text=profane_text,
        fen=ITALIAN_FEN,
        verdict="Blunder",
        stats=stats,
    )
    assert kept3 is False
    assert reason3 == "links_usernames_offensive"
    assert stats.links_usernames_offensive == 3


def test_filter_duplicate() -> None:
    """Verify rejection of duplicate comments."""
    seen: set[str] = set()
    stats = FilterStats()
    text = "White develops the bishop to c4 and attacks the weak pawn on f7."

    kept1, reason1, _ = filter_comment(
        text=text,
        fen=ITALIAN_FEN,
        seen_fingerprints=seen,
        stats=stats,
    )
    assert kept1 is True
    assert reason1 is None

    kept2, reason2, _ = filter_comment(
        text=text,
        fen=ITALIAN_FEN,
        seen_fingerprints=seen,
        stats=stats,
    )
    assert kept2 is False
    assert reason2 == "duplicate"
    assert stats.duplicate == 1
    assert stats.kept == 1


def test_filter_near_duplicate() -> None:
    """Verify rejection of near duplicate comments."""
    seen_fps: set[str] = set()
    seen_shingles: list[set[tuple[str, ...]]] = []
    stats = FilterStats()

    text1 = (
        "White develops the bishop to c4 aiming at the sensitive f7 square and preparing castling."
    )
    kept1, reason1, _ = filter_comment(
        text=text1,
        fen=ITALIAN_FEN,
        seen_fingerprints=seen_fps,
        seen_shingles=seen_shingles,
        stats=stats,
    )
    assert kept1 is True
    assert reason1 is None

    # Text2 adds only two words to text1 (Jaccard > 0.80)
    text2 = (
        "White develops the bishop to c4 aiming at the sensitive f7 square and preparing castling right now."
    )
    kept2, reason2, _ = filter_comment(
        text=text2,
        fen=ITALIAN_FEN,
        seen_fingerprints=seen_fps,
        seen_shingles=seen_shingles,
        stats=stats,
    )
    assert kept2 is False
    assert reason2 == "duplicate"
    assert stats.duplicate == 1


def test_pre_filter_comment() -> None:
    """Verify Stockfish-independent checks in pre_filter_comment."""
    seen_fps: set[str] = set()
    seen_shingles: list[set[tuple[str, ...]]] = []
    stats = FilterStats()

    # Short text fails
    kept_short, reason_short, _ = pre_filter_comment(
        text="Too short.",
        fen=ITALIAN_FEN,
        stats=stats,
    )
    assert kept_short is False
    assert reason_short == "cleaned_empty_or_short"

    # Valid text passes
    valid_text = "White develops the bishop to c4 and targets the weak pawn on f7."
    kept_valid, reason_valid, cleaned = pre_filter_comment(
        text=valid_text,
        fen=ITALIAN_FEN,
        move_uci="c4f7",
        seen_fingerprints=seen_fps,
        seen_shingles=seen_shingles,
        stats=stats,
    )
    assert kept_valid is True
    assert reason_valid is None
    assert len(cleaned.split()) >= 8

    # Duplicate text fails
    kept_dup, reason_dup, _ = pre_filter_comment(
        text=valid_text,
        fen=ITALIAN_FEN,
        move_uci="c4f7",
        seen_fingerprints=seen_fps,
        seen_shingles=seen_shingles,
        stats=stats,
    )
    assert kept_dup is False
    assert reason_dup == "duplicate"


def test_filter_valid_comment_passes() -> None:
    """Verify that a high quality chess comment passes all filters."""
    stats = FilterStats()
    text = (
        "White develops the bishop to c4, aiming at the sensitive f7 square. "
        "This move prepares castling and establishes active central control."
    )
    kept, reason, cleaned = filter_comment(
        text=text,
        fen=ITALIAN_FEN,
        verdict="Good",
        stats=stats,
    )
    assert kept is True
    assert reason is None
    assert len(cleaned.split()) >= 8
    assert stats.kept == 1


# --- 4. Detail Tiering and Truncation Tests ---


def test_assign_detail_and_truncate_short() -> None:
    """Verify that comments under 80 words receive short detail."""
    text = "White attacks the center pawn and prepares castling kingside."
    records = assign_detail_and_truncate(text)
    assert len(records) == 1
    assert records[0]["detail"] == "short"
    assert records[0]["text"] == text


def test_assign_detail_and_truncate_medium() -> None:
    """Verify that 80 to 200 word good comments produce a single long detail record."""
    sentence = "White develops pieces actively toward the center to prepare an attack. "
    # 11 words per sentence * 10 sentences = 110 words
    long_text = (sentence * 10).strip()
    records = assign_detail_and_truncate(long_text)
    assert len(records) == 1
    assert records[0]["detail"] == "long"
    assert records[0]["text"] == long_text


def test_assign_detail_and_truncate_medium_unstructured() -> None:
    """Verify that long unstructured comments cut down to short detail."""
    # 1 long sentence of 90 words without punctuation
    words = ["word"] * 90
    unstructured_text = "White plays " + " ".join(words) + " carefully."
    records = assign_detail_and_truncate(unstructured_text)
    assert len(records) == 1
    assert records[0]["detail"] == "short"
    assert len(records[0]["text"].split()) <= 80


def test_assign_detail_and_truncate_over_200() -> None:
    """Verify that comments over 200 words cut down to short detail."""
    sentence = "White develops pieces actively toward the center to prepare an attack. "
    # 11 words * 25 sentences = 275 words
    huge_text = (sentence * 25).strip()
    records = assign_detail_and_truncate(huge_text)
    assert len(records) == 1
    assert records[0]["detail"] == "short"
    assert len(records[0]["text"].split()) <= 80


def test_truncate_to_sentences() -> None:
    """Verify sentence truncation within word bounds."""
    text = (
        "Sentence one develops pieces. "
        "Sentence two controls center squares. "
        "Sentence three prepares kingside castling. "
        "Sentence four creates dangerous threats. "
        "Sentence five continues the long plan."
    )
    truncated = truncate_to_sentences(text, min_sentences=2, max_sentences=4, max_words=30)
    assert "Sentence one" in truncated
    assert "Sentence two" in truncated
    assert "Sentence five" not in truncated


def test_run_pipeline_with_filter_sft(tmp_path: Path) -> None:
    """Verify pipeline integration when filter_sft option is enabled."""
    import json
    from data.pipeline.analyse import find_stockfish_binary
    from data.pipeline.run_pipeline import run_pipeline

    sft_file = tmp_path / "raw_sft.jsonl"
    sft_records = [
        {
            "game_id": "game_1",
            "fen": ITALIAN_FEN,
            "move_uci": "c4f7",
            "task": "explain_move",
            "comment": "White attacks the weak pawn on f7 with two active pieces.",
            "source": "lichess",
            "redistributable": True,
        },
        {
            "game_id": "game_2",
            "fen": ITALIAN_FEN,
            "move_uci": "c4f7",
            "task": "explain_move",
            "comment": "Too short.",
            "source": "lichess",
            "redistributable": True,
        },
    ]
    with open(sft_file, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r) + "\n")

    sf_bin = find_stockfish_binary()
    out_dir = tmp_path / "processed"
    pub_dir = tmp_path / "public"
    cache_file = tmp_path / "cache.parquet"

    report = run_pipeline(
        sft_input_path=sft_file,
        rl_input_path=None,
        output_dir=out_dir,
        public_dir=pub_dir,
        cache_path=cache_file,
        stockfish_path=sf_bin,
        depth=6,
        num_workers=2,
        filter_sft=True,
    )

    assert report["sft_records_in"] == 2
    assert "filter_stats" in report
    assert report["filter_stats"]["cleaned_empty_or_short"] == 1
    assert report["filter_stats"]["kept"] == 1

