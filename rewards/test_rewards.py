"""Unit tests for the rewards package.

This test module verifies parsing, hard rule checks, and soft score calculations.
It checks move legality, grounding, piece squares, tone, formatting, and rewards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import pytest

from rewards.checks import (
    check_beginner_jargon,
    check_format,
    check_grounded_moves,
    check_legal_moves,
    check_piece_on_square,
    check_verdict_tone,
)
from rewards.parser import (
    clean_move_token,
    find_tone_words,
    parse_moves,
    parse_piece_square_pairs,
    parse_squares,
    parse_verdict_words,
)
from rewards.scorer import (
    compute_reward,
    score_better_move,
    score_key_point,
    score_length,
    score_level_fit,
    score_no_repeats,
    score_not_copied,
)

# Sample FEN: Italian Game position
# 1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6
ITALIAN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


# --- 1. Parser Tests ---


def test_clean_move_token() -> None:
    """Verify that move token cleaning removes numbers and normalizes castling."""
    assert clean_move_token("14.Nxf7") == "Nxf7"
    assert clean_move_token("14...Nxf7!") == "Nxf7"
    assert clean_move_token("(e4)") == "e4"
    assert clean_move_token("Qh5+,") == "Qh5+"
    assert clean_move_token("O-O-O.") == "O-O-O"
    assert clean_move_token("0-0.") == "O-O"
    assert clean_move_token("0-0-0+") == "O-O-O+"
    assert clean_move_token("Qxf7#") == "Qxf7#"


def test_parse_moves_standard() -> None:
    """Verify parsing of standard algebraic notation moves."""
    text = (
        "White played 4. Ng5, attacking f7. Black replied with 4... d5. "
        "White took 5. exd5 and later castled O-O while Black played O-O-O."
    )
    moves = parse_moves(text)
    assert "Ng5" in moves
    assert "d5" in moves
    assert "exd5" in moves
    assert "O-O" in moves
    assert "O-O-O" in moves


def test_parse_moves_promotions_and_checks() -> None:
    """Verify parsing of pawn promotions and checks."""
    text = "After e8=Q+ and fxg8=N#, the game ended immediately."
    moves = parse_moves(text)
    assert "e8=Q+" in moves
    assert "fxg8=N#" in moves


def test_parse_moves_rejects_plain_english() -> None:
    """Verify that plain English words are not parsed as moves."""
    text = "In fact, he said that Be careful and As for that, No one was there."
    moves = parse_moves(text)
    assert "In" not in moves
    assert "Be" not in moves
    assert "As" not in moves
    assert "No" not in moves
    assert len(moves) == 0


def test_parse_moves_distinguishes_squares() -> None:
    """Verify that square locations are not parsed as moves."""
    text = (
        "The knight on e5 attacks the pawn on f7. "
        "White guards g4 and h5, weakens h3, and surrenders d5. "
        "White pushes the pawn to d4."
    )
    moves = parse_moves(text)
    assert "e5" not in moves
    assert "f7" not in moves
    assert "g4" not in moves
    assert "h5" not in moves
    assert "h3" not in moves
    assert "d5" not in moves
    assert "d4" in moves


def test_parse_squares() -> None:
    """Verify square extraction from text."""
    text = "The rook moves from a1 to d1, while the bishop stays on c4."
    squares = parse_squares(text)
    assert squares == ["a1", "d1", "c4"]


def test_parse_piece_square_pairs() -> None:
    """Verify extraction of piece and square associations."""
    text = (
        "The knight on e5 and the c4 bishop control central squares. "
        "Meanwhile, Black has a pawn on f7 and an e8 king."
    )
    pairs = parse_piece_square_pairs(text)
    assert ("knight", "e5") in pairs
    assert ("bishop", "c4") in pairs
    assert ("pawn", "f7") in pairs
    assert ("king", "e8") in pairs

    # Verify that dynamic move destinations are not classified as static pieces
    move_text = "White develops the bishop to c4 and moves the knight to f3."
    move_pairs = parse_piece_square_pairs(move_text)
    assert ("bishop", "c4") not in move_pairs
    assert ("knight", "f3") not in move_pairs


def test_find_tone_words_and_verdict_words() -> None:
    """Verify detection of positive, negative, and neutral verdict words."""
    good_text = "This is a great move and an excellent strategic decision."
    bad_text = "This blunder is a terrible mistake and an inaccurate move."
    neutral_text = "The position remains equal and balanced."

    good_tones = find_tone_words(good_text)
    bad_tones = find_tone_words(bad_text)
    neutral_tones = find_tone_words(neutral_text)

    assert "great move" in good_tones["positive"]
    assert "excellent" in good_tones["positive"]
    assert "blunder" in bad_tones["negative"]
    assert "mistake" in bad_tones["negative"]
    assert "terrible" in bad_tones["negative"]
    assert "inaccurate" in bad_tones["negative"]
    assert "equal" in neutral_tones["neutral"]
    assert "balanced" in neutral_tones["neutral"]

    verdict_words = parse_verdict_words(
        "This mistake loses the equal position after a good move was missed."
    )
    assert "good move" in verdict_words
    assert "mistake" in verdict_words
    assert "loses" in verdict_words
    assert "equal" in verdict_words
    assert "winning" in parse_verdict_words("White is winning.")

    # Verify that standalone winning does not trigger positive praise
    eval_text = "White is winning after this blunder."
    eval_tones = find_tone_words(eval_text)
    assert "winning" not in eval_tones["positive"]
    assert "winning move" in find_tone_words("This is a winning move.")["positive"]


# --- 2. Hard Checks Tests ---


def test_check_legal_moves_valid() -> None:
    """Verify that legal moves in position pass the check."""
    # In Italian Game: Ng5 is legal, d3 is legal, O-O is legal
    text = "White can play Ng5 or d3 or castle O-O or 0-0."
    result = check_legal_moves(text, ITALIAN_FEN)
    assert result.passed is True


def test_check_legal_moves_after_played_move() -> None:
    """Verify that moves legal after the played move pass."""
    # Position: Italian Game. Played move: Ng5 (g1g5)
    # After Ng5, Black can play d5
    text = "After Ng5, Black responds with d5."
    result = check_legal_moves(text, ITALIAN_FEN, move_uci="f3g5")
    assert result.passed is True


def test_check_legal_moves_from_facts() -> None:
    """Verify that moves in facts pass at their legal place in variation."""
    facts = ["Best line: d3 Bc5 c3 d6."]
    text = "White should continue with c3 and d6."
    result = check_legal_moves(text, ITALIAN_FEN, facts=facts)
    assert result.passed is True

    wrong_turn = "White should begin with Bc5."
    assert check_legal_moves(wrong_turn, ITALIAN_FEN, facts=facts).passed is False


def test_check_legal_moves_illegal_fails() -> None:
    """Verify that illegal moves fail the check."""
    # Nxe5 is not legal in Italian position (e5 is protected, f3 knight cannot take e5 safely... wait,
    # Nxe5 IS legal in Italian if knight takes e5! But Qxe5 is NOT legal because queen cannot reach e5)
    text = "White should play Qxe5 and win."
    result = check_legal_moves(text, ITALIAN_FEN)
    assert result.passed is False
    assert "Illegal move" in result.reason or "not legal" in result.reason


def test_check_grounded_moves_success() -> None:
    """Verify that moves mentioned in facts pass the grounded check."""
    facts = ["Move played: Ng5.", "Best move: d3.", "Best line: d3 Bc5."]
    text = "White played Ng5, but d3 was better followed by Bc5."
    result = check_grounded_moves(text, facts)
    assert result.passed is True


def test_check_grounded_moves_unmentioned_fails() -> None:
    """Verify that moves omitted from facts fail the grounded check."""
    facts = ["Move played: Ng5.", "Best move: d3."]
    text = "White should try h4 to attack."
    result = check_grounded_moves(text, facts)
    assert result.passed is False
    assert "not grounded" in result.reason


def test_check_piece_on_square_success() -> None:
    """Verify that actual pieces on board pass the check."""
    # In Italian Game: bishop on c4, knight on f3, pawn on e5
    text = "The bishop on c4 and the knight on f3 attack Black."
    result = check_piece_on_square(text, ITALIAN_FEN)
    assert result.passed is True


def test_check_piece_on_square_after_move() -> None:
    """Verify that piece arrival after played move passes."""
    # Played move: Ng5 (knight lands on g5)
    text = "The knight on g5 attacks the pawn on f7."
    result = check_piece_on_square(text, ITALIAN_FEN, move_uci="f3g5")
    assert result.passed is True


def test_check_piece_on_square_wrong_piece_fails() -> None:
    """Verify that claiming wrong piece on square fails."""
    # There is no rook on e4 in the Italian Game
    text = "The rook on e4 controls the file."
    result = check_piece_on_square(text, ITALIAN_FEN)
    assert result.passed is False
    assert "No rook on square e4" in result.reason


def test_check_verdict_tone_praise_on_blunder_fails() -> None:
    """Verify that praise on a blunder fails."""
    text = "This is a great move that wins material."
    result = check_verdict_tone(text, "Blunder")
    assert result.passed is False
    assert "Positive tone used" in result.reason


def test_check_verdict_tone_criticism_on_good_fails() -> None:
    """Verify that unnegated criticism on a good move fails."""
    text = "This move is a terrible blunder."
    result = check_verdict_tone(text, "Good")
    assert result.passed is False
    assert "Negative tone used" in result.reason


def test_check_verdict_tone_negated_criticism_passes() -> None:
    """Verify that negated criticism on a good move passes."""
    text = "This move avoids a blunder and prevents a mistake."
    result = check_verdict_tone(text, "Good")
    assert result.passed is True


def test_check_verdict_tone_opponent_criticism_passes() -> None:
    """Verify that criticism directed at opponent passes."""
    text = "White punishes Black's blunder with strong central play."
    result = check_verdict_tone(text, "Best")
    assert result.passed is True


def test_check_verdict_tone_best_move_mention_on_blunder_passes() -> None:
    """Verify that naming the best move or opponent move on a blunder passes."""
    text = "This blunder loses a pawn. The best move was d3."
    result = check_verdict_tone(text, "Blunder")
    assert result.passed is True

    opp_text = "A blunder that gives Black a great move."
    assert check_verdict_tone(opp_text, "Blunder").passed is True

    bad_subject = "White's blunder is a great move."
    assert check_verdict_tone(bad_subject, "Blunder").passed is False

    none_res = check_verdict_tone("The position is equal.", None)
    assert none_res.passed is True


def test_check_format_clean_passes() -> None:
    """Verify that clean English prose passes format check."""
    text = "White develops the knight to g5 to attack the weak pawn on f7."
    result = check_format(text)
    assert result.passed is True

    short_text = "White wins."
    assert check_format(short_text).passed is True

    move_num_text = "1. e4 controls the center and opens lines for the bishop."
    assert check_format(move_num_text).passed is True


def test_check_format_markdown_fails() -> None:
    """Verify that markdown formatting fails the check."""
    assert check_format("# Header\nProse text.").passed is False
    assert check_format("Header\n======\nProse text.").passed is False
    assert check_format("- Single bullet item").passed is False
    assert check_format("- Item 1\n- Item 2").passed is False
    assert check_format("1. First item\n2. Second item").passed is False
    assert check_format("> Blockquote line").passed is False
    assert check_format("This is **very** important.").passed is False
    assert check_format("This is *italic* text.").passed is False
    assert check_format("Play `Ng5` immediately.").passed is False
    assert check_format("```python\nmove = 1\n```").passed is False


def test_check_format_empty_fails() -> None:
    """Verify that empty explanation fails."""
    assert check_format("   ").passed is False


def test_check_format_non_english_fails() -> None:
    """Verify that non-English text fails format check."""
    french_text = "Les blancs jouent le fou en c4 pour attaquer les noirs."
    assert check_format(french_text).passed is False

    short_french = "C'est un tres bon coup."
    assert check_format(short_french).passed is False


def test_check_format_repeats_prompt_fails() -> None:
    """Verify that repeating prompt headers fails."""
    text = "task: explain_move\nFEN: r1bqk\nWhite played Ng5."
    assert check_format(text).passed is False


def test_check_beginner_jargon() -> None:
    """Verify beginner jargon and eval numbers detection."""
    # Fails for beginner with engine jargon
    assert check_beginner_jargon("The engine gives +1.3 eval.", "beginner").passed is False
    assert check_beginner_jargon("Stockfish calculated depth 18.", "beginner").passed is False
    assert check_beginner_jargon("White is ahead by 1.5 pawns.", "beginner").passed is False

    # Passes for beginner without jargon
    clean = "White attacks the f7 pawn with two pieces."
    assert check_beginner_jargon(clean, "beginner").passed is True

    # Passes for club even with technical terms
    assert check_beginner_jargon("The engine eval is +1.3.", "club").passed is True


# --- 3. Soft Scores Tests ---


def test_score_key_point() -> None:
    """Verify key point scoring for bad moves and good moves."""
    # Blunder with threat and better move
    blunder_good = "This blunder allows Black to threaten checkmate. Better was d3."
    assert score_key_point(blunder_good, "Blunder", best_move_san="d3") == 1.0

    # Blunder with only problem
    blunder_half = "This move loses a piece to Black's capture."
    assert score_key_point(blunder_half, "Blunder") == 0.5

    # Minimal reward-hacking templates without real problem reason get at most 0.5
    minimal_verdict = "This is a mistake. Better was d3."
    assert score_key_point(minimal_verdict, "Mistake", best_move_san="d3") == 0.5

    minimal_material = "Material is equal; best move d3."
    assert (
        score_key_point(
            minimal_material,
            "Mistake",
            best_move_san="d3",
            facts=["Material: equal.", "Phase: opening."],
        )
        == 0.5
    )

    # Good move matching facts
    facts = ["Ng5 attacks the pawn on f7 together with the bishop on c4."]
    good_text = "Ng5 attacks the pawn on f7 with the bishop on c4."
    assert score_key_point(good_text, "Good", facts=facts) >= 0.7


def test_score_better_move() -> None:
    """Verify better move scoring."""
    # Good verdict always gives 1.0
    assert score_better_move("Solid move.", "Good") == 1.0

    # Mistake naming best move gives 1.0
    text_named = "A mistake. White should have played d3 instead."
    assert score_better_move(text_named, "Mistake", best_move_san="d3") == 1.0

    # Mistake with general comparison gives 0.5
    text_alt = "A mistake. Better was to develop another piece."
    assert score_better_move(text_alt, "Mistake", best_move_san="d3") == 0.5

    # Mistake with no alternative gives 0.0
    assert score_better_move("This move is bad.", "Mistake", best_move_san="d3") == 0.0


def test_score_length() -> None:
    """Verify length scoring inside and outside target bands."""
    # Short band: 25 to 80 words
    words_30 = "word " * 30
    assert score_length(words_30, "short") == 1.0

    # Short under 25 words
    words_10 = "word " * 10
    assert score_length(words_10, "short") == pytest.approx(10 / 25.0)

    # Long band: 80 to 200 words
    words_100 = "word " * 100
    assert score_length(words_100, "long") == 1.0

    # Move-list spam gets 0.0
    move_spam = "Ng5 d5 exd5 Nxd5 Bc4 Be6 Qe2 O-O d3 Nc6"
    assert score_length(move_spam, "short") == 0.0


def test_score_level_fit() -> None:
    """Verify level fit score for club and beginner."""
    assert score_level_fit("Complex text.", "club") == 1.0

    beginner_clean = "White attacks the pawn. Black defends the piece."
    assert score_level_fit(beginner_clean, "beginner") == 1.0

    beginner_jargon = "White plays prophylaxis to avoid zugzwang."
    assert score_level_fit(beginner_jargon, "beginner") < 1.0


def test_score_not_copied() -> None:
    """Verify fact overlap penalty in copying score."""
    facts = ["Ng5 attacks the pawn on f7 together with the bishop on c4."]
    exact_copy = "Ng5 attacks the pawn on f7 together with the bishop on c4."
    paraphrase = "White targets Black's king defense by jumping the knight forward."

    assert score_not_copied(exact_copy, facts) < 0.5
    assert score_not_copied(paraphrase, facts) == 1.0

    short_facts = ["Material: equal.", "Phase: opening."]
    short_copy = "Material: equal. Phase: opening. White develops the bishop."
    assert score_not_copied(short_copy, short_facts) < 1.0


def test_score_no_repeats() -> None:
    """Verify repetition penalty."""
    clean = "White attacks the center. Black defends the pawn."
    repeated = "White attacks the center. White attacks the center."

    assert score_no_repeats(clean) == 1.0
    assert score_no_repeats(repeated) <= 0.5


# --- 4. Composite Reward Tests ---


def test_compute_reward_hard_fail() -> None:
    """Verify that any hard check failure assigns -1.0 reward."""
    # Contains illegal move
    text = "White should play Qxe5 and win easily."
    res = compute_reward(text=text, fen=ITALIAN_FEN, verdict="Good")
    assert res.reward == -1.0
    assert res.passed_all_hard is False
    assert res.hard_checks["legal_moves"].passed is False


def test_compute_reward_success() -> None:
    """Verify that valid explanation receives a positive reward."""
    facts = [
        "Ng5 attacks the pawn on f7 together with the bishop on c4.",
        "After Ng5, Black's best reply is d5.",
    ]
    prompt_text = (
        "Move played: Ng5.\nVerdict: Good.\n"
        "Best move: d3.\nFacts:\n- Ng5 attacks the pawn on f7.\n- After Ng5, Black's best reply is d5."
    )
    explanation = (
        "White plays Ng5 to put immediate pressure on the vulnerable f7 square. "
        "The move coordinates with the bishop to create an active opening attack. "
        "Black must defend carefully by pushing d5 to block the line."
    )
    res = compute_reward(
        text=explanation,
        fen=ITALIAN_FEN,
        level="club",
        detail="short",
        verdict="Good",
        move_uci="f3g5",
        best_move_san="d3",
        facts=facts,
        prompt_text=prompt_text,
    )
    assert res.passed_all_hard is True
    assert 0.6 <= res.reward <= 1.0
    assert res.hard_checks["legal_moves"].passed is True
    assert res.hard_checks["grounded_moves"].passed is True
    assert res.hard_checks["format"].passed is True


def test_compute_reward_explain_position_verdict_none() -> None:
    """Verify that compute_reward handles explain_position records with verdict None."""
    facts = [
        "Top candidate moves: d3, Ng5, O-O.",
        "Material: equal.",
        "Phase: opening.",
    ]
    prompt_text = (
        "task: explain_position\nlevel: club\ndetail: short\n\n"
        f"FEN: {ITALIAN_FEN}\nWhite to move, move 4.\n"
        "White's winning chances: 54%.\nBest move: d3.\nBest line: d3 Bc5 c3 d6.\n"
        "Facts:\n- Top candidate moves: d3, Ng5, O-O.\n- Material: equal.\n- Phase: opening."
    )
    explanation = (
        "White has a solid opening position with equal material. "
        "The best continuation is d3 to support the center and open the diagonal for the bishop. "
        "White can also consider Ng5 or castling O-O to bring the king to safety."
    )
    res = compute_reward(
        text=explanation,
        fen=ITALIAN_FEN,
        level="club",
        detail="short",
        verdict=None,
        move_uci=None,
        facts=facts,
        prompt_text=prompt_text,
    )
    assert res.passed_all_hard is True
    assert res.reward >= 0.70
    assert res.soft_scores["better_move"] == 1.0


def test_human_commentary_reward_distribution() -> None:
    """Verify reward distribution on human commentary examples.

    Check compute_reward on grounded human explanations.
    If sft_train.jsonl exists, evaluate the first 500 records and 500 grounded records.
    Verify that hard checks produce zero false alarms on grounded human records.
    Verify that mean compute_reward across the 500 grounded records is at least 0.60.
    """
    human_samples = [
        (
            "White develops the pawn to d3 and supports the bishop on c4. Best move is d3.",
            ITALIAN_FEN,
            "club",
            "short",
            "Good",
            "d3",
            ["d3 supports the pawn on e4 and bishop on c4.", "White attacks f7."],
            "FEN: ... Move played: d3.\nVerdict: Good.\nBest move: d3.\nFacts:\n- d3 supports the pawn on e4.",
        ),
        (
            "White attacks the pawn on f7 with two active pieces. Black defends with d5.",
            ITALIAN_FEN,
            "beginner",
            "short",
            "Good",
            "Ng5",
            ["After Ng5, Black plays d5.", "White attacks f7."],
            "FEN: ... Move played: Ng5.\nVerdict: Good.\nBest move: Ng5.\nFacts:\n- After Ng5, Black plays d5.",
        ),
        (
            "This blunder leaves the f7 pawn undefended. Better was d3 to support the center.",
            ITALIAN_FEN,
            "club",
            "short",
            "Blunder",
            "d3",
            ["Best move: d3.", "White leaves f7 vulnerable."],
            "FEN: ... Move played: Ng5.\nVerdict: Blunder.\nBest move: d3.\nFacts:\n- Best move: d3.",
        ),
    ]

    for text, fen, level, detail, verdict, best_san, facts, prompt in human_samples:
        res = compute_reward(
            text=text,
            fen=fen,
            level=level,
            detail=detail,
            verdict=verdict,
            best_move_san=best_san,
            facts=facts,
            prompt_text=prompt,
        )
        assert res.passed_all_hard is True
        assert res.reward >= 0.60

    sft_train_path = Path("data/processed/sft_train.jsonl")
    if sft_train_path.exists():
        first_500: list[dict[str, Any]] = []
        grounded_500: list[dict[str, Any]] = []
        with open(sft_train_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if len(first_500) < 500:
                    first_500.append(record)
                prompt_and_facts = [record.get("user_message", "")] + list(
                    record.get("facts", [])
                )
                if (
                    len(grounded_500) < 500
                    and check_grounded_moves(
                        record.get("comment", ""), prompt_and_facts
                    ).passed
                ):
                    grounded_500.append(record)
                if len(first_500) >= 500 and len(grounded_500) >= 500:
                    break

        assert len(first_500) == 500
        assert len(grounded_500) == 500

        for r in first_500:
            res = compute_reward(
                text=r.get("comment", ""),
                fen=r.get("fen", ""),
                level=r.get("level", "club"),
                detail=r.get("detail", "short"),
                verdict=r.get("verdict"),
                move_uci=r.get("move_uci"),
                facts=r.get("facts", []),
                prompt_text=r.get("user_message", ""),
            )
            assert res.hard_checks["legal_moves"].passed is True
            assert res.hard_checks["piece_on_square"].passed is True
            assert res.hard_checks["verdict_tone"].passed is True
            assert res.hard_checks["format"].passed is True
            assert res.hard_checks["beginner_jargon"].passed is True

        grounded_rewards: list[float] = []
        for r in grounded_500:
            res = compute_reward(
                text=r.get("comment", ""),
                fen=r.get("fen", ""),
                level=r.get("level", "club"),
                detail=r.get("detail", "short"),
                verdict=r.get("verdict"),
                move_uci=r.get("move_uci"),
                facts=r.get("facts", []),
                prompt_text=r.get("user_message", ""),
            )
            assert res.passed_all_hard is True
            grounded_rewards.append(res.reward)

        assert (sum(grounded_rewards) / len(grounded_rewards)) >= 0.60
