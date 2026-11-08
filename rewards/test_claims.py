"""Unit tests for coaching voice, opening claims, and relational claim verifiers.

This module tests hard checks introduced in Stage T1:
- check_coach_voice
- check_opening_claims
- check_relational_claims
It validates positive coaching prose, rejects first-person player excuses,
and catches historical Phase 6 hallucinations (Petrov, London, Dragon, false protection).
"""

from __future__ import annotations

import pytest

from rewards.claims import (
    check_coach_voice,
    check_opening_claims,
    check_relational_claims,
)


# =====================================================================
# 1. Coach Voice Verification Tests
# =====================================================================

@pytest.mark.parametrize(
    "valid_text",
    [
        "White should play Qa5+ to check the king.",
        "Black has a tactical opportunity with Nf6.",
        "You should look for undefended pieces before moving.",
        "Your best plan is to castle and secure king safety.",
        "The bishop on c4 is undefended and vulnerable.",
        "Moving the knight improves piece activity.",
        "This move is an inaccuracy that drops a pawn.",
        "Initiative is critical in this open position.",
        "Pinning the knight prevents e5.",
        "Position evaluation favors White after castling.",
        "After Bxf7+, Black's king must step into the center.",
        "White's move threatens checkmate on the next turn.",
    ],
)
def test_coach_voice_valid(valid_text: str) -> None:
    """Objective coach explanations in second or third person must pass."""
    res = check_coach_voice(valid_text)
    assert res.passed, f"Expected pass, failed with: {res.reason}"


@pytest.mark.parametrize(
    "diary_text,expected_word",
    [
        ("I was trying to get my knight to a better square.", "I"),
        ("I thought the pawn was protected.", "I"),
        ("I should have seen the fork on d7.", "I"),
        ("I'm looking to attack the white king.", "I'm"),
        ("I've made a terrible mistake in the opening.", "I've"),
        ("I'd rather defend than attack here.", "I'd"),
        ("I'll try to find a better defense next time.", "I'll"),
        ("This blunder lost my queen immediately.", "my"),
        ("The opponent launched a fierce attack against me.", "me"),
        ("The winning position was mine to throw away.", "mine"),
        ("My plan was to develop the bishop first.", "My"),
        ("It seemed to me that Nf6 was safe.", "me"),
    ],
)
def test_coach_voice_first_person_fails(diary_text: str, expected_word: str) -> None:
    """First-person player diary voice must fail coach voice check."""
    res = check_coach_voice(diary_text)
    assert not res.passed
    assert expected_word.lower() in res.reason.lower()


# =====================================================================
# 2. Opening Claim Verification Tests
# =====================================================================

def test_opening_claims_valid_when_in_facts() -> None:
    """Opening mentioned in text must match Opening fact line."""
    facts = ["Opening: Italian Game.", "Material: equal."]
    text = "In this Italian Game, White develops the bishop to an active diagonal."
    res = check_opening_claims(text, facts)
    assert res.passed


def test_opening_claims_valid_when_no_opening_mentioned() -> None:
    """Text without opening names must pass regardless of facts."""
    facts = ["Opening: Sicilian Defense."]
    text = "White plays d4 to challenge the center and open lines for development."
    res = check_opening_claims(text, facts)
    assert res.passed


def test_opening_claims_valid_empty_facts_and_text() -> None:
    """Empty text or empty facts without opening claims must pass."""
    res = check_opening_claims("White plays Nf3 developing toward the center.", facts=[])
    assert res.passed


@pytest.mark.parametrize(
    "hallucinated_text,hallucinated_name,facts",
    [
        (
            "Black employs the Petrov Defense to challenge White's e4 pawn.",
            "Petrov",
            ["Opening: Italian Game."],
        ),
        (
            "White sets up the solid London System pawn pyramid.",
            "London",
            [],
        ),
        (
            "This sharp setup is typical of the Sicilian Dragon.",
            "Sicilian",
            ["Opening: French Defense."],
        ),
        (
            "White plays a classic Ruy Lopez maneuvering game.",
            "Ruy Lopez",
            [],
        ),
        (
            "Black uses the Caro-Kann to keep a solid pawn structure.",
            "Caro-Kann",
            ["Opening: Scandinavian Defense."],
        ),
        (
            "White offers the Queen's Gambit to control central squares.",
            "Queen's Gambit",
            [],
        ),
        (
            "Black chooses the King's Indian setup with g6 and Bg7.",
            "King's Indian",
            ["Opening: English Opening."],
        ),
        (
            "The Dutch defense creates immediate kingside imbalances.",
            "Dutch",
            [],
        ),
        (
            "White opens with the Scotch game.",
            "Scotch",
            ["Opening: Vienna Game."],
        ),
        (
            "Black relies on the Slav defense for central stability.",
            "Slav",
            [],
        ),
    ],
)
def test_opening_claims_hallucination_fails(
    hallucinated_text: str,
    hallucinated_name: str,
    facts: list[str],
) -> None:
    """Unverified opening claims must fail the check."""
    res = check_opening_claims(hallucinated_text, facts)
    assert not res.passed
    assert f"Unverified opening claim: '{hallucinated_name}'" in res.reason


# =====================================================================
# 3. Relational Claim Verification Tests
# =====================================================================

def test_relational_claim_valid_attack() -> None:
    """Valid attack claim must pass when piece attacks square."""
    # White queen on h5 attacks f7 and h7
    fen = "r1bqkbnr/1p1n1ppp/p2p4/2pPp1BQ/B3P3/8/PPP2PPP/RN2K1NR b KQkq - 1 7"
    text = "Nf6 is played. White attacks the pawn on f7."
    res = check_relational_claims(text, fen, move_uci="g8f6")
    assert res.passed


def test_relational_claim_false_attack() -> None:
    """False attack claim must fail when no piece attacks target square."""
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    text = "White attacks the rook on a8."
    res = check_relational_claims(text, fen, move_uci="d2d4")
    assert not res.passed
    assert "False relational claim" in res.reason


def test_relational_claim_false_protection_test_098() -> None:
    """Historical test_098: move does not protect pawn on e4 (no pawn on e4)."""
    fen = "rnb1kbnr/ppp1pppp/3p4/q7/3P4/4P1P1/PPP2P1P/RNBQKBNR w KQkq - 5 6"
    text = "This is the best move, because it forces the queen to move, and it also protects the pawn on e4."
    res = check_relational_claims(text, fen, move_uci="c1d2")
    assert not res.passed
    assert "False relational claim" in res.reason


def test_relational_claim_valid_defense() -> None:
    """Valid defense claim must pass when friendly piece attacks friendly square."""
    fen = "r1bqkb1r/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3"
    # White pawn on d4 is defended by White queen on d1 and knight on f3
    text = "White defends the pawn on d4 with multiple pieces."
    res = check_relational_claims(text, fen, move_uci="e5d4")
    assert res.passed


def test_relational_claim_valid_pin() -> None:
    """Absolute pin on board must pass pin check."""
    # White bishop on b5 pins Black knight on c6 to Black king on e8 (after ...d6)
    fen = "r1bqkbnr/ppp2ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
    text = "White pins the knight on c6 to the black king."
    res = check_relational_claims(text, fen)
    assert res.passed


def test_relational_claim_false_pin() -> None:
    """Hallucinated pin on unpinned piece must fail check."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    text = "White pins the knight on b8."
    res = check_relational_claims(text, fen)
    assert not res.passed
    assert "pin on b8 not verified" in res.reason


def test_relational_claim_valid_fork() -> None:
    """Fork mentioned in facts must pass."""
    fen = "r1bqkb1r/pppp1ppp/8/4n3/4P3/8/PPP2PPP/RNBQKB1R w KQkq - 0 6"
    facts = ["Nf3 forks the king on e8 and the rook on h8."]
    text = "White's knight forks the king and rook winning material."
    res = check_relational_claims(text, fen, facts=facts)
    assert res.passed


def test_relational_claim_false_fork() -> None:
    """Claiming a fork when no fork exists on board or facts must fail."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    text = "This move forks two major pieces."
    res = check_relational_claims(text, fen, move_uci="e7e5", facts=[])
    assert not res.passed
    assert "fork not verified" in res.reason


def test_relational_claim_valid_checkmate() -> None:
    """Checkmate on board or facts must pass."""
    # Scholar's mate
    fen = "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    text = "White delivers checkmate on f7."
    res = check_relational_claims(text, fen)
    assert res.passed


def test_relational_claim_false_checkmate() -> None:
    """Claiming checkmate when position is not checkmate must fail."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    text = "White's move delivers checkmate immediately."
    res = check_relational_claims(text, fen, facts=[])
    assert not res.passed
    assert "checkmate not verified" in res.reason


def test_relational_claim_qualified_piece_color() -> None:
    """Claims with color prefixes ('black queen', 'white pawn') must parse correctly."""
    fen = "r1bqkb1r/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3"
    # White pawn on d4 is defended by white queen on d1
    text_def = "White defends the white pawn on d4."
    res_def = check_relational_claims(text_def, fen, move_uci="e5d4")
    assert res_def.passed

    # Falsely claiming to defend black queen on d4 when piece on d4 is white pawn
    text_wrong = "White defends the black queen on d4."
    res_wrong = check_relational_claims(text_wrong, fen, move_uci="e5d4")
    assert not res_wrong.passed
    assert "False relational claim" in res_wrong.reason

