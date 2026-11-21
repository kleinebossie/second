"""Fact extractor for chess positions.

This module computes deterministic facts from chess positions and engine evaluations.
It exports functions to evaluate winning chances, classify moves, and extract facts.
"""

from __future__ import annotations

import math
from typing import Any

import chess

from .openings import lookup_opening

PIECE_NAMES: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

CENTRAL_SQUARES: frozenset[int] = frozenset({chess.D4, chess.E4, chess.D5, chess.E5})
EXTENDED_CENTER_SQUARES: frozenset[int] = frozenset(
    chess.square(f, r) for f in range(2, 6) for r in range(2, 6)
)
CENTER_45_SQUARES: frozenset[int] = frozenset(
    chess.square(f, r) for f in range(2, 6) for r in range(3, 5)
)
WING_CENTER_45_SQUARES: frozenset[int] = frozenset({chess.C4, chess.F4, chess.C5, chess.F5})
WHITE_START_MINORS: tuple[int, ...] = (chess.B1, chess.C1, chess.F1, chess.G1)
BLACK_START_MINORS: tuple[int, ...] = (chess.B8, chess.C8, chess.F8, chess.G8)
FILE_LETTERS: str = "abcdefgh"


def _piece_mobility(board: chess.Board, sq: int) -> int:
    """Count squares that the piece on sq attacks, excluding friendly pieces."""
    piece = board.piece_at(sq)
    if piece is None:
        return 0
    return sum(
        1
        for t_sq in board.attacks(sq)
        if (target := board.piece_at(t_sq)) is None or target.color != piece.color
    )


def _count_pawns_on_file(
    board: chess.Board,
    file_idx: int,
    color: chess.Color | None = None,
) -> int:
    """Count pawns on a file for one color or both colors."""
    if color is not None:
        return sum(
            1
            for sq in board.pieces(chess.PAWN, color)
            if chess.square_file(sq) == file_idx
        )
    return sum(
        1
        for c in (chess.WHITE, chess.BLACK)
        for sq in board.pieces(chess.PAWN, c)
        if chess.square_file(sq) == file_idx
    )


def _classify_rook_file_move(
    board_before: chess.Board,
    board_after: chess.Board,
    from_sq: int,
    to_sq: int,
    mover_color: chess.Color,
) -> str | None:
    """Classify whether a rook move claims an open or half-open file.

    If the destination file has friendly pawns or matches the source file, return None.
    When moving from a closed or half-open file to a file with no enemy pawns after, return open.
    When moving from a closed file to a file with enemy pawns remaining, return half-open.
    """
    from_file = chess.square_file(from_sq)
    to_file = chess.square_file(to_sq)
    if from_file == to_file:
        return None
    if _count_pawns_on_file(board_before, to_file, mover_color) > 0:
        return None

    opp_color = not mover_color
    from_friendly = _count_pawns_on_file(board_before, from_file, mover_color)
    from_enemy = _count_pawns_on_file(board_before, from_file, opp_color)
    to_enemy = _count_pawns_on_file(board_after, to_file, opp_color)

    if to_enemy == 0 and (from_friendly >= 1 or from_enemy >= 1):
        return "open"
    if to_enemy >= 1 and from_friendly >= 1:
        return "half-open"
    return None


def _is_pawn_takes_pawn(board_before: chess.Board, move: chess.Move) -> bool:
    """Return True when move is a pawn capturing an enemy pawn or en passant."""
    mover = board_before.piece_at(move.from_square)
    if mover is None or mover.piece_type != chess.PAWN:
        return False
    if not board_before.is_capture(move):
        return False
    if board_before.is_en_passant(move):
        return True
    target = board_before.piece_at(move.to_square)
    return target is not None and target.piece_type == chess.PAWN


def _is_king_takes_friendly_rook(board: chess.Board, move: chess.Move) -> bool:
    """Return True when move encodes castling as king onto a friendly piece square."""
    p = board.piece_at(move.from_square)
    t = board.piece_at(move.to_square)
    return bool(p is not None and p.piece_type == chess.KING and t is not None and t.color == p.color)


def _canonical_move_uci(board: chess.Board, uci: str) -> str:
    """Normalize Chess960 king-takes-friendly-rook UCI to standard castling UCI."""
    if not uci:
        return uci
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return uci
    if move in board.legal_moves and _is_king_takes_friendly_rook(board, move):
        return board.uci(move)
    return uci


def score_to_white_cp(score: dict[str, Any]) -> float:
    """Convert an engine score to White centipawns.

    If score type is cp, return the centipawn value.
    If score type is mate, compute a bounded centipawn value.
    When score type is unknown, return zero.
    """
    score_type = score.get("type", "cp")
    if score_type == "cp":
        return float(score.get("value", 0))
    if score_type == "mate":
        m = int(score.get("value", 0))
        if m > 0:
            return 10000.0 - m * 10.0
        if m < 0:
            return -10000.0 - m * 10.0
        return 0.0
    return 0.0


def win_chance_white(score: dict[str, Any]) -> float:
    """Calculate the White winning chance percentage.

    This function uses the standard Lichess winning chances formula.
    """
    cp = max(-10000.0, min(10000.0, score_to_white_cp(score)))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def win_chance(score: dict[str, Any], is_white: bool = True) -> float:
    """Calculate the winning chance percentage for one side.

    If is_white is true, return the White winning chance.
    Otherwise, return one hundred minus the White winning chance.
    """
    w = win_chance_white(score)
    return w if is_white else (100.0 - w)


win_chance_side = win_chance


def round_pct(val: float) -> int:
    """Round a percentage value to the nearest integer.

    This function rounds half values upward.
    """
    return int(math.floor(val + 0.5))


def classify_verdict(
    eval_before: dict[str, Any],
    eval_after: dict[str, Any] | None,
    move_uci: str,
) -> str:
    """Classify the quality of a played move.

    If the move is best and other moves lose win chance, return Only move.
    If the win chance drop is at least twenty percent, return Blunder.
    If the drop is at least ten percent, return Mistake.
    If the drop is at least five percent, return Inaccuracy.
    When the drop is small, return Best or Good.
    """
    board = chess.Board(eval_before["fen"])
    is_white = (board.turn == chess.WHITE)
    pvs = eval_before.get("pvs", [])
    best_move_uci = pvs[0]["moves"][0] if (pvs and pvs[0].get("moves")) else ""
    norm_move_uci = _canonical_move_uci(board, move_uci)
    norm_best_uci = _canonical_move_uci(board, best_move_uci)
    wc_before = win_chance_side(pvs[0]["score"], is_white) if (pvs and "score" in pvs[0]) else 50.0

    if eval_after and eval_after.get("pvs") and "score" in eval_after["pvs"][0]:
        wc_after = win_chance_side(eval_after["pvs"][0]["score"], is_white)
    else:
        wc_after = wc_before

    if norm_move_uci == norm_best_uci:
        if board.legal_moves.count() == 1:
            return "Only move"
        if len(pvs) >= 2 and "score" in pvs[1]:
            wc_second = win_chance_side(pvs[1]["score"], is_white)
            if (wc_before - wc_second) >= 15.0:
                return "Only move"

    drop = wc_before - wc_after
    if norm_move_uci == norm_best_uci and drop <= 2.0:
        return "Best"
    if drop >= 20.0:
        return "Blunder"
    if drop >= 10.0:
        return "Mistake"
    if drop >= 5.0:
        return "Inaccuracy"
    return "Good"


def uci_line_to_san(fen: str, uci_moves: list[str], max_plies: int = 6) -> list[str]:
    """Convert a sequence of UCI moves to SAN strings.

    If a move is illegal on the board, stop processing.
    Return at most max_plies SAN strings.
    """
    board = chess.Board(fen)
    san_list: list[str] = []
    for uci in uci_moves[:max_plies]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        san_list.append(board.san(move))
        board.push(move)
    return san_list


def compute_material_fact(board: chess.Board) -> str:
    """Compute the material balance fact.

    When material is equal, return material equal.
    When one side has more material, name the advantage.
    """
    white_pts = sum(
        PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values()
        if p.color == chess.WHITE and p.piece_type != chess.KING
    )
    black_pts = sum(
        PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values()
        if p.color == chess.BLACK and p.piece_type != chess.KING
    )

    diff = white_pts - black_pts
    if diff == 0:
        return "Material: equal."

    winner_name = "White" if diff > 0 else "Black"
    winner_color = chess.WHITE if diff > 0 else chess.BLACK
    loser_color = chess.BLACK if diff > 0 else chess.WHITE
    adv = abs(diff)

    def count_piece(c: chess.Color, pt: chess.PieceType) -> int:
        return len(board.pieces(pt, c))

    net_p = max(0, count_piece(winner_color, chess.PAWN) - count_piece(loser_color, chess.PAWN))
    net_n = max(0, count_piece(winner_color, chess.KNIGHT) - count_piece(loser_color, chess.KNIGHT))
    net_b = max(0, count_piece(winner_color, chess.BISHOP) - count_piece(loser_color, chess.BISHOP))
    net_r = max(0, count_piece(winner_color, chess.ROOK) - count_piece(loser_color, chess.ROOK))
    net_q = max(0, count_piece(winner_color, chess.QUEEN) - count_piece(loser_color, chess.QUEEN))

    loser_net_p = max(0, count_piece(loser_color, chess.PAWN) - count_piece(winner_color, chess.PAWN))
    loser_net_n = max(0, count_piece(loser_color, chess.KNIGHT) - count_piece(winner_color, chess.KNIGHT))
    loser_net_b = max(0, count_piece(loser_color, chess.BISHOP) - count_piece(winner_color, chess.BISHOP))
    loser_net_r = max(0, count_piece(loser_color, chess.ROOK) - count_piece(winner_color, chess.ROOK))
    loser_net_q = max(0, count_piece(loser_color, chess.QUEEN) - count_piece(winner_color, chess.QUEEN))

    # Single-type clean advantages
    if (
        loser_net_p == 0
        and loser_net_n == 0
        and loser_net_b == 0
        and loser_net_r == 0
        and loser_net_q == 0
    ):
        if adv == 1 and net_p == 1:
            return f"{winner_name} is up a pawn."
        if adv == 2 and net_p == 2:
            return f"{winner_name} is up two pawns."
        if adv == 3 and net_n == 1:
            return f"{winner_name} is up a knight."
        if adv == 3 and net_b == 1:
            return f"{winner_name} is up a bishop."
        if adv == 5 and net_r == 1:
            return f"{winner_name} is up a rook."
        if adv == 9 and net_q == 1:
            return f"{winner_name} is up a queen."

    # Exchange advantage
    if (
        adv == 2
        and net_r == 1
        and (loser_net_n + loser_net_b == 1)
        and net_p == 0
        and loser_net_p == 0
        and net_q == 0
        and loser_net_q == 0
    ):
        return f"{winner_name} is up the exchange."

    # Fallback descriptions
    if adv == 1:
        return f"{winner_name} is up a pawn in material."
    if adv == 2:
        return f"{winner_name} is up two pawns in material."
    if adv == 3:
        return f"{winner_name} is up a minor piece in material."
    if adv == 4:
        return f"{winner_name} is up a minor piece and a pawn in material."
    if adv == 5:
        return f"{winner_name} is up a rook in material."
    if adv == 6:
        return f"{winner_name} is up two minor pieces in material."
    if adv == 7:
        return f"{winner_name} is up a rook and two pawns in material."
    if adv == 8:
        return f"{winner_name} is up a rook and a minor piece in material."
    return f"{winner_name} is up a queen in material."


def compute_phase_fact(board: chess.Board) -> str:
    """Compute the current game phase fact.

    When piece count is low, return endgame.
    When early moves have undeveloped back ranks, return opening.
    Otherwise, return middlegame.
    """
    major_minor = 0
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            major_minor += len(board.pieces(piece_type, color))

    initial_minors = [
        (chess.B1, chess.Piece(chess.KNIGHT, chess.WHITE)),
        (chess.C1, chess.Piece(chess.BISHOP, chess.WHITE)),
        (chess.F1, chess.Piece(chess.BISHOP, chess.WHITE)),
        (chess.G1, chess.Piece(chess.KNIGHT, chess.WHITE)),
        (chess.B8, chess.Piece(chess.KNIGHT, chess.BLACK)),
        (chess.C8, chess.Piece(chess.BISHOP, chess.BLACK)),
        (chess.F8, chess.Piece(chess.BISHOP, chess.BLACK)),
        (chess.G8, chess.Piece(chess.KNIGHT, chess.BLACK)),
    ]
    backrank_developed = 0
    for sq, p in initial_minors:
        if board.piece_at(sq) != p:
            backrank_developed += 1

    if major_minor <= 6:
        return "Phase: endgame."
    if board.fullmove_number <= 10 and major_minor >= 12 and backrank_developed < 6:
        return "Phase: opening."
    return "Phase: middlegame."


def compute_king_safety_facts(board: chess.Board) -> list[str]:
    """Compute king safety facts for both kings.

    If a king is in check, return a check fact.
    If a king has not castled after move nine, return a castling fact.
    """
    facts: list[str] = []
    for color, side_name, start_sq in [
        (chess.WHITE, "White", chess.E1),
        (chess.BLACK, "Black", chess.E8),
    ]:
        ksq = board.king(color)
        if board.turn == color and board.is_check():
            if ksq is not None:
                facts.append(f"{side_name} king is in check on {chess.square_name(ksq)}.")
        elif board.fullmove_number >= 10:
            if ksq == start_sq and board.has_castling_rights(color):
                facts.append(f"{side_name} has not castled.")
    return facts


def compute_hanging_facts(board: chess.Board) -> list[str]:
    """Find undefended pieces that enemy pieces attack.

    Return at most two facts for hanging pieces.
    """
    facts: list[str] = []
    for sq in range(64):
        p = board.piece_at(sq)
        if p is None or p.piece_type in (chess.PAWN, chess.KING):
            continue
        enemy = not p.color
        friendly = p.color
        if board.is_attacked_by(enemy, sq) and not board.is_attacked_by(friendly, sq):
            p_name = PIECE_NAMES.get(p.piece_type, "piece")
            facts.append(f"The {p_name} on {chess.square_name(sq)} is not defended.")
    return facts[:2]


def _get_qualifying_pv_check(
    board: chess.Board,
    engine_eval: dict[str, Any] | None,
) -> chess.Move | None:
    """Return the top Stockfish PV check if within one hundred centipawns of PV zero."""
    if not engine_eval:
        return None
    pvs = engine_eval.get("pvs", [])
    if not pvs:
        return None

    side_sign = 1.0 if board.turn == chess.WHITE else -1.0
    pv0_score = pvs[0].get("score")
    pv0_cp = score_to_white_cp(pv0_score) * side_sign if pv0_score is not None else None

    for idx, pv in enumerate(pvs[:3]):
        moves = pv.get("moves") or []
        if not moves:
            continue
        if idx > 0:
            pv_score = pv.get("score")
            if pv0_cp is None or pv_score is None:
                continue
            pv_cp = score_to_white_cp(pv_score) * side_sign
            if (pv0_cp - pv_cp) > 100.0:
                continue
        try:
            pv_move = chess.Move.from_uci(moves[0])
        except ValueError:
            continue
        if pv_move in board.legal_moves and board.gives_check(pv_move):
            return pv_move
    return None


def _is_forcing_check(board: chess.Board, move: chess.Move) -> tuple[bool, bool]:
    """Check if a legal check forces mate, wins material, or wins a clear advantage.

    Return a tuple of (is_mate, qualifies_tactically).
    """
    mover_piece = board.piece_at(move.from_square)
    if mover_piece is None:
        return (False, False)

    side = board.turn
    opp = not side
    orig_mover_val = PIECE_VALUES.get(mover_piece.piece_type, 0)
    effective_mover = move.promotion or mover_piece.piece_type
    mover_val = PIECE_VALUES.get(effective_mover, 0)

    is_cap = board.is_capture(move)
    target_val = 0
    if is_cap:
        if board.is_en_passant(move):
            target_val = PIECE_VALUES[chess.PAWN]
        else:
            target = board.piece_at(move.to_square)
            if target is not None:
                target_val = PIECE_VALUES.get(target.piece_type, 0)

    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return (True, True)

    opp_legal = list(after.legal_moves)
    recaptures = [r for r in opp_legal if r.to_square == move.to_square]

    # Priority 2: captures material safely or wins an exchange or piece
    if is_cap and (not recaptures or target_val > orig_mover_val):
        return (False, True)

    # Confirm the checking piece is safe before evaluating non-winning-capture forcing checks
    if not recaptures:
        is_safe = True
    else:
        defended = len(after.attackers(side, move.to_square)) >= len(recaptures)
        all_ge = True
        for r in recaptures:
            rp = after.piece_at(r.from_square)
            if rp is None or PIECE_VALUES.get(rp.piece_type, 0) < mover_val:
                all_ge = False
                break
        is_safe = defended and all_ge

    if not is_safe:
        return (False, False)

    # Priority 3a: royal fork or double attack against an undefended or higher-value piece
    recapturer_from_sqs = {r.from_square for r in recaptures}
    for t_sq in after.attacks(move.to_square):
        t_piece = after.piece_at(t_sq)
        if t_piece is None or t_piece.color != opp:
            continue
        if t_piece.piece_type in (chess.PAWN, chess.KING):
            continue
        if t_sq in recapturer_from_sqs:
            continue
        is_undefended = not after.is_attacked_by(opp, t_sq)
        is_higher = PIECE_VALUES.get(t_piece.piece_type, 0) > mover_val
        if is_undefended or is_higher:
            return (False, True)

    # Priority 3b: safe check strips castling rights by forcing a king move
    if after.has_castling_rights(opp) and opp_legal:
        if all(
            (rp := after.piece_at(r.from_square)) is not None and rp.piece_type == chess.KING
            for r in opp_legal
        ):
            return (False, True)

    return (False, False)


def compute_checks_and_captures_facts(
    board: chess.Board,
    engine_eval: dict[str, Any] | None = None,
) -> list[str]:
    """Find Stockfish-supported or forcing checks and high value captures.

    If Stockfish top lines include a check within 100 centipawns, report that check.
    If no PV check qualifies, report a check only when it wins an advantage.
    If a capture of a non-pawn piece is legal, record the top capture.
    """
    side_name = "White" if board.turn == chess.WHITE else "Black"
    facts: list[str] = []
    check_move_san: str | None = None
    check_move_from_to: tuple[int, int] | None = None

    pv_check = _get_qualifying_pv_check(board, engine_eval)
    if pv_check is not None:
        check_move_san = board.san(pv_check)
        check_move_from_to = (pv_check.from_square, pv_check.to_square)
        facts.append(f"{side_name} has a check: {check_move_san}.")
    else:
        checks = [m for m in board.legal_moves if board.gives_check(m)]
        checks.sort(key=lambda m: (m.from_square, m.to_square, -(m.promotion or 0)))
        first_forcing: chess.Move | None = None
        mate_move: chess.Move | None = None
        for m in checks:
            is_mate, qualifies = _is_forcing_check(board, m)
            if is_mate:
                mate_move = m
                break
            if qualifies and first_forcing is None:
                first_forcing = m
        chosen_check = mate_move or first_forcing
        if chosen_check is not None:
            check_move_san = board.san(chosen_check)
            check_move_from_to = (chosen_check.from_square, chosen_check.to_square)
            facts.append(f"{side_name} has a check: {check_move_san}.")

    captures: list[tuple[int, int, int, int, chess.Move, chess.Piece]] = []
    for m in board.legal_moves:
        if board.is_capture(m):
            target = board.piece_at(m.to_square)
            if target is not None and target.piece_type != chess.PAWN:
                target_val = PIECE_VALUES.get(target.piece_type, 0)
                captures.append((
                    -target_val,
                    m.from_square,
                    m.to_square,
                    -(m.promotion or 0),
                    m,
                    target,
                ))

    captures.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    for item in captures:
        m = item[4]
        target = item[5]
        if check_move_from_to is not None and (m.from_square, m.to_square) == check_move_from_to:
            continue
        san = board.san(m)
        if san != check_move_san:
            cap_name = PIECE_NAMES.get(target.piece_type, "piece")
            sq_name = chess.square_name(m.to_square)
            facts.append(f"{side_name} can capture the {cap_name} on {sq_name} with {san}.")
            break

    return facts


def compute_threat_fact(board_state: chess.Board, color: chess.Color) -> str | None:
    """Find the most dangerous threat for the specified color.

    If the position has check, return None.
    If the setup is invalid, return None.
    Check for checkmate threats, winning captures, and safe checks.
    """
    if board_state.is_check():
        return None

    temp = board_state.copy(stack=False)
    temp.turn = color
    temp.ep_square = None
    if not temp.is_valid():
        return None

    color_name = "White" if color == chess.WHITE else "Black"
    enemy_color = not color

    # Category A: Checkmate
    mate_moves: list[chess.Move] = []
    for m in temp.legal_moves:
        temp.push(m)
        is_mate = temp.is_checkmate()
        temp.pop()
        if is_mate:
            mate_moves.append(m)

    if mate_moves:
        mate_moves.sort(key=lambda m: (m.from_square, m.to_square, -(m.promotion or 0)))
        san = temp.san(mate_moves[0])
        return f"{color_name} threatens {san}."

    # Category B: Winning capture
    winning_captures: list[tuple[int, int, int, int, int, chess.Move]] = []
    for m in temp.legal_moves:
        if temp.is_capture(m):
            target = temp.piece_at(m.to_square)
            attacker = temp.piece_at(m.from_square)
            if target is not None and attacker is not None:
                target_val = PIECE_VALUES.get(target.piece_type, 0)
                attacker_val = PIECE_VALUES.get(attacker.piece_type, 0)
                is_safe_capture = target_val >= 3 and not temp.is_attacked_by(enemy_color, m.to_square)
                is_higher_value = target_val > attacker_val
                if is_safe_capture or is_higher_value:
                    winning_captures.append((
                        -target_val,
                        attacker_val,
                        m.from_square,
                        m.to_square,
                        -(m.promotion or 0),
                        m,
                    ))

    if winning_captures:
        winning_captures.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        san = temp.san(winning_captures[0][5])
        return f"{color_name} threatens {san}."

    # Category C: Safe check
    safe_checks: list[chess.Move] = []
    for m in temp.legal_moves:
        if temp.gives_check(m):
            after_m = temp.copy(stack=False)
            after_m.push(m)
            is_safe = (not after_m.is_attacked_by(enemy_color, m.to_square)) or after_m.is_attacked_by(color, m.to_square)
            if is_safe:
                safe_checks.append(m)

    if safe_checks:
        safe_checks.sort(key=lambda m: (m.from_square, m.to_square, -(m.promotion or 0)))
        san = temp.san(safe_checks[0])
        return f"{color_name} threatens {san}."

    return None


def compute_moved_piece_attack_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
) -> str | None:
    """Find enemy pieces that the moved piece attacks.

    If another friendly piece supports the attack, mention that piece.
    When no attacked piece exists, return None.
    """
    if board_after.is_check():
        return None

    mover_piece = board_before.piece_at(move.from_square)
    if mover_piece is not None and mover_piece.piece_type == chess.KING:
        return None

    mover_color = board_before.turn
    to_sq = move.to_square
    move_san = board_before.san(move)
    attacks = board_after.attacks(to_sq)

    candidates: list[tuple[int, int, str]] = []
    for sq in attacks:
        target = board_after.piece_at(sq)
        if target is not None and target.color != mover_color and target.piece_type != chess.KING:
            target_val = PIECE_VALUES.get(target.piece_type, 0)
            other_attackers = [s for s in board_after.attackers(mover_color, sq) if s != to_sq]
            other_non_pawns = [
                s for s in other_attackers
                if board_after.piece_at(s)
                and board_after.piece_at(s).piece_type not in (chess.PAWN, chess.KING)
            ]
            other_non_pawns.sort()
            if other_non_pawns:
                other_sq = other_non_pawns[0]
                other_p = board_after.piece_at(other_sq)
                target_name = PIECE_NAMES.get(target.piece_type, "piece")
                other_name = PIECE_NAMES.get(other_p.piece_type, "piece")
                text = f"{move_san} attacks the {target_name} on {chess.square_name(sq)} together with the {other_name} on {chess.square_name(other_sq)}."
                candidates.append((-target_val, sq, text))
            elif target.piece_type != chess.PAWN:
                target_name = PIECE_NAMES.get(target.piece_type, "piece")
                text = f"{move_san} attacks the {target_name} on {chess.square_name(sq)}."
                candidates.append((-target_val, sq, text))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def compute_reply_and_material_change_facts(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_after: dict[str, Any] | None,
) -> list[str]:
    """Compute opponent reply and line material changes.

    Identify the opponent best reply from engine variations.
    Trace the variation to detect material gain.
    """
    if not eval_after or not eval_after.get("pvs"):
        return []
    pvs = eval_after["pvs"]
    if not pvs or not pvs[0].get("moves"):
        return []

    pvs_moves: list[str] = pvs[0]["moves"]
    reply_uci = pvs_moves[0]
    try:
        reply_move = chess.Move.from_uci(reply_uci)
    except ValueError:
        return []

    if reply_move not in board_after.legal_moves:
        return []

    opp_color = board_after.turn
    opp_name = "White" if opp_color == chess.WHITE else "Black"
    move_san = board_before.san(move)
    reply_san = board_after.san(reply_move)

    facts: list[str] = [
        f"After {move_san}, {opp_name}'s best reply is {reply_san}."
    ]

    def get_net_pts(b: chess.Board) -> int:
        opp_pts = sum(
            PIECE_VALUES[p.piece_type]
            for p in b.piece_map().values()
            if p.color == opp_color and p.piece_type != chess.KING
        )
        mover_pts = sum(
            PIECE_VALUES[p.piece_type]
            for p in b.piece_map().values()
            if p.color != opp_color and p.piece_type != chess.KING
        )
        return opp_pts - mover_pts

    initial_net = get_net_pts(board_after)

    sim = board_after.copy(stack=False)
    cap = sim.piece_at(reply_move.to_square)
    cap_sq_idx = reply_move.to_square
    if cap is None and board_after.is_en_passant(reply_move):
        cap_sq_idx = reply_move.to_square + (-8 if opp_color == chess.WHITE else 8)
        cap = sim.piece_at(cap_sq_idx)
    sim.push(reply_move)

    for uci in pvs_moves[1:3]:
        try:
            m = chess.Move.from_uci(uci)
            if m in sim.legal_moves:
                sim.push(m)
            else:
                break
        except ValueError:
            break

    # Case 1: reply_move captures a piece
    if cap is not None and cap.color != opp_color and cap.piece_type != chess.KING:
        cap_val = PIECE_VALUES.get(cap.piece_type, 0)
        net_gain = get_net_pts(sim) - initial_net
        if net_gain >= cap_val:
            cap_name = PIECE_NAMES.get(cap.piece_type, "piece")
            cap_sq = chess.square_name(cap_sq_idx)
            facts.append(f"After {move_san}, {opp_name} captures the {cap_name} on {cap_sq} with {reply_san}.")
            return facts

    # Case 2: reply_move does not capture a non-pawn piece
    if (cap is None or cap.piece_type == chess.PAWN) and len(pvs_moves) >= 3:
        sim2 = board_after.copy(stack=False)
        sim2.push(reply_move)
        try:
            m1 = chess.Move.from_uci(pvs_moves[1])
            if m1 in sim2.legal_moves:
                sim2.push(m1)
                m2 = chess.Move.from_uci(pvs_moves[2])
                if m2 in sim2.legal_moves:
                    cap3 = sim2.piece_at(m2.to_square)
                    if cap3 is not None and cap3.color != opp_color and cap3.piece_type not in (chess.PAWN, chess.KING):
                        sim2.push(m2)
                        if len(pvs_moves) >= 4:
                            try:
                                m3 = chess.Move.from_uci(pvs_moves[3])
                                if m3 in sim2.legal_moves:
                                    sim2.push(m3)
                            except ValueError:
                                pass
                        net_gain = get_net_pts(sim2) - initial_net
                        if net_gain >= 2:
                            cap_name = PIECE_NAMES.get(cap3.piece_type, "piece")
                            sq_name = chess.square_name(m2.to_square)
                            p_before = board_before.piece_at(m2.to_square)
                            p_after = board_after.piece_at(m2.to_square)
                            stood_on_sq = (
                                (
                                    p_before is not None
                                    and p_before.piece_type == cap3.piece_type
                                    and p_before.color == cap3.color
                                )
                                or (
                                    p_after is not None
                                    and p_after.piece_type == cap3.piece_type
                                    and p_after.color == cap3.color
                                )
                            )
                            if stood_on_sq:
                                facts.append(f"After {reply_san}, {opp_name} wins the {cap_name} on {sq_name}.")
                            else:
                                facts.append(f"After {reply_san}, {opp_name} wins the {cap_name}.")
        except ValueError:
            pass

    return facts


def is_between(sq_mid: int, sq1: int, sq2: int) -> bool:
    """Check if a square lies between two squares on a line.

    If the three squares align, check the intermediate square.
    Return true when the square lies between the endpoints.
    """
    r1, c1 = divmod(sq1, 8)
    rm, cm = divmod(sq_mid, 8)
    r2, c2 = divmod(sq2, 8)
    dr, dc = r2 - r1, c2 - c1

    if dr == 0 and dc != 0:
        return r1 == rm and min(c1, c2) < cm < max(c1, c2)
    if dc == 0 and dr != 0:
        return c1 == cm and min(r1, r2) < rm < max(r1, r2)
    if abs(dr) == abs(dc) and dr != 0:
        drm, dcm = rm - r1, cm - c1
        if abs(drm) != abs(dcm):
            return False
        if (dr > 0) != (drm > 0) or (dc > 0) != (dcm > 0):
            return False
        return abs(drm) < abs(dr)
    return False


def _get_ray_directions(piece_type: chess.PieceType) -> list[tuple[int, int]]:
    """Return direction vectors for sliding pieces.

    Bishops move diagonally.
    Rooks move orthogonally.
    Queens move in all eight directions.
    """
    diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    orth = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if piece_type == chess.BISHOP:
        return diag
    if piece_type == chess.ROOK:
        return orth
    if piece_type == chess.QUEEN:
        return diag + orth
    return []


def _scan_ray(
    board: chess.Board,
    start_sq: int,
    dr: int,
    dc: int,
) -> tuple[int | None, chess.Piece | None, int | None, chess.Piece | None]:
    """Scan along a direction vector for the first two pieces.

    Trace squares step by step.
    Record the first two pieces encountered on the ray.
    """
    r = start_sq // 8 + dr
    c = start_sq % 8 + dc
    first_sq: int | None = None
    first_p: chess.Piece | None = None
    second_sq: int | None = None
    second_p: chess.Piece | None = None

    while 0 <= r < 8 and 0 <= c < 8:
        sq = r * 8 + c
        p = board.piece_at(sq)
        if p is not None:
            if first_sq is None:
                first_sq = sq
                first_p = p
            elif second_sq is None:
                second_sq = sq
                second_p = p
                break
        r += dr
        c += dc

    return first_sq, first_p, second_sq, second_p


def _check_fork_on_board(
    board: chess.Board,
    attacker_sq: int,
    attacker_color: chess.Color,
) -> tuple[tuple[chess.Piece, int], tuple[chess.Piece, int]] | None:
    """Find the top two enemy pieces forked by the piece on attacker_sq."""
    attacker_piece = board.piece_at(attacker_sq)
    if attacker_piece is None or attacker_piece.piece_type == chess.KING:
        return None
    attacker_val = PIECE_VALUES.get(attacker_piece.piece_type, 0)
    opp_color = not attacker_color
    attacks = board.attacks(attacker_sq)

    candidates: list[tuple[chess.Piece, int]] = []
    for sq in attacks:
        target = board.piece_at(sq)
        if target is None or target.color != opp_color:
            continue
        if target.piece_type == chess.KING:
            candidates.append((target, sq))
        else:
            target_val = PIECE_VALUES.get(target.piece_type, 0)
            is_undefended = not board.is_attacked_by(opp_color, sq)
            is_higher_val = target_val > attacker_val
            if is_undefended or is_higher_val:
                candidates.append((target, sq))

    if not candidates:
        return None

    has_king = any(t[0].piece_type == chess.KING for t in candidates)
    if not has_king:
        candidates = [t for t in candidates if PIECE_VALUES.get(t[0].piece_type, 0) >= 3]

    if len(candidates) < 2:
        return None

    candidates.sort(
        key=lambda item: (
            -(100 if item[0].piece_type == chess.KING else PIECE_VALUES.get(item[0].piece_type, 0)),
            item[1],
        )
    )
    return candidates[0], candidates[1]


def compute_fork_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_after: dict[str, Any] | None,
) -> str | None:
    """Detect a fork by the moved piece or the best reply piece.

    A fork attacks two or more high value or undefended pieces.
    If the moved piece forks, return the fork fact.
    If the best reply forks, return the reply fork fact.
    """
    mover_color = board_before.turn
    pair = _check_fork_on_board(board_after, move.to_square, mover_color)
    if pair is not None:
        move_san = board_before.san(move)
        p1_name = PIECE_NAMES.get(pair[0][0].piece_type, "piece")
        sq1_name = chess.square_name(pair[0][1])
        p2_name = PIECE_NAMES.get(pair[1][0].piece_type, "piece")
        sq2_name = chess.square_name(pair[1][1])
        return f"{move_san} forks the {p1_name} on {sq1_name} and the {p2_name} on {sq2_name}."

    if eval_after and eval_after.get("pvs") and eval_after["pvs"][0].get("moves"):
        reply_uci = eval_after["pvs"][0]["moves"][0]
        try:
            reply_move = chess.Move.from_uci(reply_uci)
            if reply_move in board_after.legal_moves:
                sim = board_after.copy(stack=False)
                sim.push(reply_move)
                pair_reply = _check_fork_on_board(sim, reply_move.to_square, not mover_color)
                if pair_reply is not None:
                    reply_san = board_after.san(reply_move)
                    p1_name = PIECE_NAMES.get(pair_reply[0][0].piece_type, "piece")
                    sq1_name = chess.square_name(pair_reply[0][1])
                    p2_name = PIECE_NAMES.get(pair_reply[1][0].piece_type, "piece")
                    sq2_name = chess.square_name(pair_reply[1][1])
                    return f"{reply_san} forks the {p1_name} on {sq1_name} and the {p2_name} on {sq2_name}."
        except ValueError:
            pass

    return None


def _check_pin_on_board(
    board: chess.Board,
    slider_sq: int,
    slider_color: chess.Color,
) -> tuple[int, chess.Piece, int, chess.Piece] | None:
    """Find the highest priority enemy piece pinned along a ray from slider_sq."""
    slider = board.piece_at(slider_sq)
    if slider is None or slider.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None

    opp_color = not slider_color
    directions = _get_ray_directions(slider.piece_type)
    candidates: list[tuple[int, int, chess.Piece, int, chess.Piece]] = []

    for dr, dc in directions:
        f_sq, f_p, s_sq, s_p = _scan_ray(board, slider_sq, dr, dc)
        if f_sq is None or f_p is None or f_p.color != opp_color:
            continue
        if f_p.piece_type == chess.KING:
            continue
        if s_sq is None or s_p is None or s_p.color != opp_color:
            continue

        f_val = PIECE_VALUES.get(f_p.piece_type, 0)
        s_val = PIECE_VALUES.get(s_p.piece_type, 0)

        is_absolute = (s_p.piece_type == chess.KING)
        is_relative = (s_p.piece_type in (chess.QUEEN, chess.ROOK) and s_val > f_val)

        if is_absolute or is_relative:
            target_rank = 100 if is_absolute else s_val
            candidates.append((-target_rank, f_sq, f_p, s_sq, s_p))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    top = candidates[0]
    return top[1], top[2], top[3], top[4]


def compute_pin_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_after: dict[str, Any] | None,
) -> str | None:
    """Detect a pin by the moved piece or the best reply piece.

    A pin holds a piece in front of a more valuable piece.
    If the moved piece pins, return the pin fact.
    If the best reply pins, return the reply pin fact.
    """
    mover_color = board_before.turn
    res = _check_pin_on_board(board_after, move.to_square, mover_color)
    if res is not None:
        move_san = board_before.san(move)
        f_sq, f_p, s_sq, s_p = res
        p_name = PIECE_NAMES.get(f_p.piece_type, "piece")
        sq_name = chess.square_name(f_sq)
        tgt_name = PIECE_NAMES.get(s_p.piece_type, "piece")
        tgt_sq = chess.square_name(s_sq)
        return f"{move_san} pins the {p_name} on {sq_name} to the {tgt_name} on {tgt_sq}."

    if eval_after and eval_after.get("pvs") and eval_after["pvs"][0].get("moves"):
        reply_uci = eval_after["pvs"][0]["moves"][0]
        try:
            reply_move = chess.Move.from_uci(reply_uci)
            if reply_move in board_after.legal_moves:
                sim = board_after.copy(stack=False)
                sim.push(reply_move)
                res_reply = _check_pin_on_board(sim, reply_move.to_square, not mover_color)
                if res_reply is not None:
                    reply_san = board_after.san(reply_move)
                    f_sq, f_p, s_sq, s_p = res_reply
                    p_name = PIECE_NAMES.get(f_p.piece_type, "piece")
                    sq_name = chess.square_name(f_sq)
                    tgt_name = PIECE_NAMES.get(s_p.piece_type, "piece")
                    tgt_sq = chess.square_name(s_sq)
                    return f"{reply_san} pins the {p_name} on {sq_name} to the {tgt_name} on {tgt_sq}."
        except ValueError:
            pass

    return None


def _check_skewer_on_board(
    board: chess.Board,
    slider_sq: int,
    slider_color: chess.Color,
) -> tuple[int, chess.Piece, int, chess.Piece] | None:
    """Find the highest priority enemy piece skewered along a ray from slider_sq."""
    slider = board.piece_at(slider_sq)
    if slider is None or slider.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None

    opp_color = not slider_color
    directions = _get_ray_directions(slider.piece_type)
    candidates: list[tuple[int, int, int, chess.Piece, int, chess.Piece]] = []

    for dr, dc in directions:
        f_sq, f_p, s_sq, s_p = _scan_ray(board, slider_sq, dr, dc)
        if f_sq is None or f_p is None or f_p.color != opp_color:
            continue
        if f_p.piece_type not in (chess.KING, chess.QUEEN, chess.ROOK):
            continue
        if s_sq is None or s_p is None or s_p.color != opp_color:
            continue
        if s_p.piece_type == chess.KING:
            continue

        f_val = 100 if f_p.piece_type == chess.KING else PIECE_VALUES.get(f_p.piece_type, 0)
        s_val = PIECE_VALUES.get(s_p.piece_type, 0)

        if s_val <= f_val and s_val >= 1:
            candidates.append((-f_val, -s_val, f_sq, f_p, s_sq, s_p))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    top = candidates[0]
    return top[2], top[3], top[4], top[5]


def compute_skewer_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_after: dict[str, Any] | None,
) -> str | None:
    """Detect a skewer by the moved piece or the best reply piece.

    A skewer attacks a high value piece with a piece behind it.
    If the moved piece skewers, return the skewer fact.
    If the best reply skewers, return the reply skewer fact.
    """
    mover_color = board_before.turn
    res = _check_skewer_on_board(board_after, move.to_square, mover_color)
    if res is not None:
        move_san = board_before.san(move)
        f_sq, f_p, s_sq, s_p = res
        p1_name = PIECE_NAMES.get(f_p.piece_type, "piece")
        sq1_name = chess.square_name(f_sq)
        p2_name = PIECE_NAMES.get(s_p.piece_type, "piece")
        sq2_name = chess.square_name(s_sq)
        return f"{move_san} skewers the {p1_name} on {sq1_name} and the {p2_name} on {sq2_name}."

    if eval_after and eval_after.get("pvs") and eval_after["pvs"][0].get("moves"):
        reply_uci = eval_after["pvs"][0]["moves"][0]
        try:
            reply_move = chess.Move.from_uci(reply_uci)
            if reply_move in board_after.legal_moves:
                sim = board_after.copy(stack=False)
                sim.push(reply_move)
                res_reply = _check_skewer_on_board(sim, reply_move.to_square, not mover_color)
                if res_reply is not None:
                    reply_san = board_after.san(reply_move)
                    f_sq, f_p, s_sq, s_p = res_reply
                    p1_name = PIECE_NAMES.get(f_p.piece_type, "piece")
                    sq1_name = chess.square_name(f_sq)
                    p2_name = PIECE_NAMES.get(s_p.piece_type, "piece")
                    sq2_name = chess.square_name(s_sq)
                    return f"{reply_san} skewers the {p1_name} on {sq1_name} and the {p2_name} on {sq2_name}."
        except ValueError:
            pass

    return None


def compute_discovered_attack_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
) -> str | None:
    """Detect a discovered attack created by moving a piece.

    Moving a piece unblocks a friendly sliding piece attack.
    If an attack opens against a king, queen, rook, or undefended minor piece, return the fact.
    """
    mover_color = board_before.turn
    opp_color = not mover_color
    from_sq = move.from_square
    to_sq = move.to_square

    candidates: list[tuple[int, int, chess.Piece, int, chess.Piece]] = []

    for sq1 in board_after.piece_map():
        slider = board_after.piece_at(sq1)
        if slider is None or slider.color != mover_color:
            continue
        if slider.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        if sq1 == to_sq:
            continue

        for sq2 in board_after.attacks(sq1):
            target = board_after.piece_at(sq2)
            if target is None or target.color != opp_color:
                continue

            if target.piece_type == chess.KING:
                target_val = 100
            elif target.piece_type in (chess.QUEEN, chess.ROOK):
                target_val = PIECE_VALUES.get(target.piece_type, 0)
            elif target.piece_type in (chess.BISHOP, chess.KNIGHT):
                if not board_after.is_attacked_by(opp_color, sq2):
                    target_val = PIECE_VALUES.get(target.piece_type, 0)
                else:
                    continue
            else:
                continue

            if is_between(from_sq, sq1, sq2):
                if sq2 not in board_before.attacks(sq1):
                    candidates.append((-target_val, sq1, slider, sq2, target))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    top = candidates[0]
    move_san = board_before.san(move)
    slider_name = PIECE_NAMES.get(top[2].piece_type, "piece")
    sq1_name = chess.square_name(top[1])
    target_name = PIECE_NAMES.get(top[4].piece_type, "piece")
    sq2_name = chess.square_name(top[3])
    return f"{move_san} opens a discovered attack from the {slider_name} on {sq1_name} against the {target_name} on {sq2_name}."


def compute_mate_threat_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_after: dict[str, Any] | None,
) -> str | None:
    """Detect a forced checkmate or checkmate threat from engine evaluations.

    If the mover has forced mate, report the moves to mate.
    If the opponent has forced mate, report the checkmate threat.
    """
    if not eval_after or not eval_after.get("pvs"):
        return None
    pvs = eval_after["pvs"]
    if not pvs or "score" not in pvs[0]:
        return None
    score = pvs[0]["score"]
    if score.get("type") != "mate":
        return None
    val = int(score.get("value", 0))
    if val == 0:
        return None
    moves = pvs[0].get("moves", [])
    if not moves:
        return None
    try:
        reply_move = chess.Move.from_uci(moves[0])
        if reply_move not in board_after.legal_moves:
            return None
    except ValueError:
        return None

    reply_san = board_after.san(reply_move)
    mover_color = board_before.turn
    mover_name = "White" if mover_color == chess.WHITE else "Black"
    opp_name = "Black" if mover_color == chess.WHITE else "White"
    mover_has_mate = (val > 0 if mover_color == chess.WHITE else val < 0)
    n = abs(val)
    moves_str = "move" if n == 1 else "moves"

    if mover_has_mate:
        return f"{mover_name} has a forced checkmate in {n} {moves_str} starting with {reply_san}."
    return f"{opp_name} threatens checkmate with {reply_san}."


def compute_recapture_activity_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_after: dict[str, Any] | None,
) -> str | None:
    """Detect whether an opponent recapture centralizes or activates a piece.

    Check whether the played move is a capture and the top reply recaptures there.
    When that recapture enters the center or gains activity, return a fact.
    """
    if not board_before.is_capture(move):
        return None
    if not eval_after or not eval_after.get("pvs"):
        return None
    pvs = eval_after["pvs"]
    if not pvs or not pvs[0].get("moves"):
        return None

    try:
        reply_move = chess.Move.from_uci(pvs[0]["moves"][0])
    except ValueError:
        return None
    if reply_move not in board_after.legal_moves:
        return None
    if reply_move.to_square != move.to_square or not board_after.is_capture(reply_move):
        return None

    recap_piece = board_after.piece_at(reply_move.from_square)
    if recap_piece is None or recap_piece.piece_type not in (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ):
        return None

    move_san = board_before.san(move)
    reply_san = board_after.san(reply_move)
    opp_color = board_after.turn
    opp_name = "White" if opp_color == chess.WHITE else "Black"
    piece_name = PIECE_NAMES.get(recap_piece.piece_type, "piece")

    if (
        reply_move.to_square in CENTRAL_SQUARES
        and reply_move.from_square not in CENTRAL_SQUARES
    ):
        return f"After {move_san}, {opp_name}'s recapture {reply_san} centralizes the {piece_name}."

    mobility_before = _piece_mobility(board_after, reply_move.from_square)
    sim_reply = board_after.copy(stack=False)
    sim_reply.push(reply_move)
    mobility_after = _piece_mobility(sim_reply, reply_move.to_square)

    back_rank = 0 if opp_color == chess.WHITE else 7
    developed_from_back = (
        chess.square_rank(reply_move.from_square) == back_rank
        and reply_move.to_square in EXTENDED_CENTER_SQUARES
    )
    if developed_from_back or mobility_after >= mobility_before + 2:
        return f"After {move_san}, {opp_name}'s recapture {reply_san} increases {piece_name} activity."

    return None


def compute_refutation_reason_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_before: dict[str, Any],
    eval_after: dict[str, Any] | None,
) -> str | None:
    """Explain what the best move accomplishes when the played move is bad.

    When the verdict is Inaccuracy, Mistake, or Blunder, check the best move.
    Report tactical gains, central pawn space or tension, king safety, or activity.
    """
    verdict = classify_verdict(eval_before, eval_after, move.uci())
    _ = board_after
    if verdict not in ("Inaccuracy", "Mistake", "Blunder"):
        return None

    pvs = eval_before.get("pvs", [])
    if not pvs or not pvs[0].get("moves"):
        return None
    best_uci = pvs[0]["moves"][0]
    if best_uci == move.uci():
        return None

    try:
        best_move = chess.Move.from_uci(best_uci)
        if best_move not in board_before.legal_moves:
            return None
    except ValueError:
        return None

    best_san = board_before.san(best_move)
    if best_san == board_before.san(move):
        return None
    mover_color = board_before.turn
    opp_color = not mover_color
    best_to = best_move.to_square
    best_piece = board_before.piece_at(best_move.from_square)

    # 1. Best move captures a piece
    if board_before.is_capture(best_move):
        cap_sq = best_to
        target = board_before.piece_at(best_to)
        if target is None and board_before.is_en_passant(best_move):
            cap_sq = best_to + (-8 if mover_color == chess.WHITE else 8)
            target = board_before.piece_at(cap_sq)
        if target is not None:
            target_name = PIECE_NAMES.get(target.piece_type, "piece")
            sq_name = chess.square_name(cap_sq)
            return f"Best move {best_san} captures the {target_name} on {sq_name}."

    # 2. Best move defends a hanging piece
    board_best = board_before.copy(stack=False)
    board_best.push(best_move)
    defended_candidates: list[tuple[int, int, chess.Piece]] = []
    for sq in board_best.attacks(best_to):
        if sq != best_to:
            p = board_best.piece_at(sq)
            if p and p.color == mover_color and p.piece_type != chess.KING:
                if board_before.is_attacked_by(opp_color, sq) and not board_before.is_attacked_by(mover_color, sq):
                    defended_candidates.append((PIECE_VALUES.get(p.piece_type, 0), sq, p))

    if defended_candidates:
        defended_candidates.sort(key=lambda item: (-item[0], item[1]))
        top_p = defended_candidates[0][2]
        top_sq = defended_candidates[0][1]
        piece_name = PIECE_NAMES.get(top_p.piece_type, "piece")
        sq_name = chess.square_name(top_sq)
        return f"Best move {best_san} defends the {piece_name} on {sq_name}."

    # 3. Best move develops a piece and prepares castling
    if board_before.has_castling_rights(mover_color) and board_before.fullmove_number <= 12:
        if best_piece and best_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            start_sqs = WHITE_START_MINORS if mover_color == chess.WHITE else BLACK_START_MINORS
            if best_move.from_square in start_sqs:
                piece_name = PIECE_NAMES.get(best_piece.piece_type, "piece")
                return f"Best move {best_san} develops the {piece_name} and prepares castling."

    # 4. Central pawn advance: space, tension, or pawn attack on a piece
    if best_piece and best_piece.piece_type == chess.PAWN:
        played_is_pawn_capture = _is_pawn_takes_pawn(board_before, move)
        played_to = move.to_square
        played_to_sq = chess.square_name(played_to)

        # 4a. Central pawn advance attacking an enemy non-pawn, non-king piece
        if best_to in EXTENDED_CENTER_SQUARES:
            attacked_targets: list[tuple[int, int, chess.Piece]] = []
            for t_sq in board_best.attacks(best_to):
                target = board_best.piece_at(t_sq)
                if (
                    target is not None
                    and target.color == opp_color
                    and target.piece_type not in (chess.PAWN, chess.KING)
                    and board_before.piece_at(t_sq) is not None
                ):
                    attacked_targets.append((PIECE_VALUES.get(target.piece_type, 0), t_sq, target))
            if attacked_targets:
                attacked_targets.sort(key=lambda item: (-item[0], item[1]))
                top_target = attacked_targets[0][2]
                t_sq_name = chess.square_name(attacked_targets[0][1])
                target_name = PIECE_NAMES.get(top_target.piece_type, "piece")
                if best_to in CENTER_45_SQUARES:
                    if played_is_pawn_capture and played_to in CENTRAL_SQUARES:
                        return (
                            f"Best move {best_san} gains space and attacks the "
                            f"{target_name} on {t_sq_name} instead of trading in the center."
                        )
                    return f"Best move {best_san} gains central space and attacks the {target_name} on {t_sq_name}."
                return f"Best move {best_san} attacks the {target_name} on {t_sq_name}."

        # 4b. Space-gaining central or 4th/5th-rank pawn push
        if best_to in CENTER_45_SQUARES:
            if played_is_pawn_capture and played_to in EXTENDED_CENTER_SQUARES:
                return f"Best move {best_san} keeps central space and tension instead of trading on {played_to_sq}."
            if best_to in CENTRAL_SQUARES:
                return f"Best move {best_san} claims central space on {chess.square_name(best_to)}."
            return f"Best move {best_san} gains space and challenges the center."

        # 4c. Solid tension-holding pawn move
        if best_to in EXTENDED_CENTER_SQUARES and played_is_pawn_capture and played_to in CENTRAL_SQUARES:
            return f"Best move {best_san} maintains central tension instead of trading on {played_to_sq}."

    # 5. Best move castles to keep the king safe
    if board_before.is_castling(best_move):
        return f"Best move {best_san} castles to keep the king safe."

    # 6. Best move saves an attacked piece
    if best_piece and best_piece.piece_type in (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ):
        piece_name = PIECE_NAMES.get(best_piece.piece_type, "piece")
        if board_before.is_attacked_by(opp_color, best_move.from_square) and not board_best.is_attacked_by(
            opp_color, best_to
        ):
            return f"Best move {best_san} moves the attacked {piece_name} to safety."

        # 7. Piece centralization, open file, or activity gain
        if best_to in CENTRAL_SQUARES and best_move.from_square not in CENTRAL_SQUARES:
            return f"Best move {best_san} centralizes the {piece_name}."

        if best_piece.piece_type == chess.ROOK:
            rook_file_kind = _classify_rook_file_move(
                board_before, board_best, best_move.from_square, best_to, mover_color
            )
            if rook_file_kind == "open":
                return f"Best move {best_san} places the rook on an open file."
            if rook_file_kind == "half-open":
                return f"Best move {best_san} places the rook on a half-open file."

        mobility_before = _piece_mobility(board_before, best_move.from_square)
        mobility_after = _piece_mobility(board_best, best_to)
        if mobility_after >= mobility_before + 2:
            return f"Best move {best_san} increases {piece_name} activity."

    return None


def _mover_net_material(board: chess.Board, mover_color: chess.Color) -> int:
    """Compute material point difference from mover_color perspective."""
    net = 0
    for p in board.piece_map().values():
        if p.piece_type == chess.KING:
            continue
        val = PIECE_VALUES.get(p.piece_type, 0)
        net += val if p.color == mover_color else -val
    return net


def _count_opp_central_pieces(board: chess.Board, opp_color: chess.Color) -> int:
    """Count opponent non-pawn, non-king pieces on the four center squares."""
    count = 0
    for sq in CENTRAL_SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.color == opp_color and p.piece_type not in (chess.PAWN, chess.KING):
            count += 1
    return count


def _count_mover_central_pawns(board: chess.Board, mover_color: chess.Color) -> int:
    """Count mover pawns on the center squares and fifth-rank extended center."""
    rank_5 = 4 if mover_color == chess.WHITE else 3
    count = 0
    for sq in board.pieces(chess.PAWN, mover_color):
        if sq in CENTRAL_SQUARES or (
            chess.square_rank(sq) == rank_5 and 2 <= chess.square_file(sq) <= 5
        ):
            count += 1
    return count


def _net_piece_mobility(board: chess.Board, mover_color: chess.Color) -> int:
    """Compute net non-pawn, non-king mobility difference for mover_color."""
    opp_color = not mover_color
    mover_mob = 0
    opp_mob = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, mover_color):
            mover_mob += _piece_mobility(board, sq)
        for sq in board.pieces(pt, opp_color):
            opp_mob += _piece_mobility(board, sq)
    return mover_mob - opp_mob


def _settled_material(
    states_meta: list[tuple[chess.Board, bool, int]],
    pv_moves: list[str],
    common_plies: int,
    mover_color: chess.Color,
    next_pv_idx: int,
) -> int:
    """Settle one recapture ply when the horizon ends in a capture."""
    eq_board, was_cap, cap_to_sq = states_meta[common_plies - 1]
    if was_cap and 0 <= next_pv_idx < len(pv_moves):
        try:
            next_move = chess.Move.from_uci(pv_moves[next_pv_idx])
        except ValueError:
            next_move = None
        if (
            next_move is not None
            and next_move in eq_board.legal_moves
            and not _is_king_takes_friendly_rook(eq_board, next_move)
            and eq_board.is_capture(next_move)
            and next_move.to_square == cap_to_sq
        ):
            settled = eq_board.copy(stack=False)
            settled.push(next_move)
            return _mover_net_material(settled, mover_color)
    return _mover_net_material(eq_board, mover_color)


def compute_line_comparison_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_before: dict[str, Any],
    eval_after: dict[str, Any] | None,
) -> str | None:
    """Compare key positional or material differences between played and best lines.

    Simulate up to four plies from board_before for both lines on bad moves.
    Report the highest priority concrete difference.
    """
    verdict = classify_verdict(eval_before, eval_after, move.uci())
    if verdict not in ("Inaccuracy", "Mistake", "Blunder"):
        return None
    if not eval_after:
        return None

    pvs_before = eval_before.get("pvs", [])
    pvs_after = eval_after.get("pvs", [])
    if not pvs_before or not pvs_before[0].get("moves"):
        return None
    if not pvs_after or not pvs_after[0].get("moves"):
        return None

    best_uci = pvs_before[0]["moves"][0]
    if best_uci == move.uci():
        return None
    try:
        best_move = chess.Move.from_uci(best_uci)
        if best_move not in board_before.legal_moves:
            return None
    except ValueError:
        return None

    mover_color = board_before.turn
    opp_color = not mover_color
    mover_name = "White" if mover_color == chess.WHITE else "Black"
    opp_name = "Black" if mover_color == chess.WHITE else "White"
    move_san = board_before.san(move)
    best_san = board_before.san(best_move)
    if best_san == move_san:
        return None

    played_pv_moves = list(pvs_after[0]["moves"])
    played_sim = board_after.copy(stack=False)
    mover_castled_played = board_before.is_castling(move)
    played_states_meta: list[tuple[chess.Board, bool, int]] = [
        (played_sim.copy(stack=False), board_before.is_capture(move), move.to_square)
    ]
    played_castled_at_ply: list[bool] = [mover_castled_played]
    for uci in played_pv_moves[:3]:
        try:
            m = chess.Move.from_uci(uci)
        except ValueError:
            break
        if m not in played_sim.legal_moves:
            break
        if played_sim.turn == mover_color and played_sim.is_castling(m):
            mover_castled_played = True
        was_cap = played_sim.is_capture(m)
        played_sim.push(m)
        played_states_meta.append((played_sim.copy(stack=False), was_cap, m.to_square))
        played_castled_at_ply.append(mover_castled_played)

    best_pv_moves = list(pvs_before[0]["moves"])
    best_sim = board_before.copy(stack=False)
    mover_castled_best = False
    best_states_meta: list[tuple[chess.Board, bool, int]] = []
    best_castled_at_ply: list[bool] = []
    for uci in best_pv_moves[:4]:
        try:
            m = chess.Move.from_uci(uci)
        except ValueError:
            break
        if m not in best_sim.legal_moves:
            break
        if best_sim.turn == mover_color and best_sim.is_castling(m):
            mover_castled_best = True
        was_cap = best_sim.is_capture(m)
        best_sim.push(m)
        best_states_meta.append((best_sim.copy(stack=False), was_cap, m.to_square))
        best_castled_at_ply.append(mover_castled_best)

    if not best_states_meta:
        return None

    # 0. Checkmate delivered or avoided
    best_final = best_states_meta[-1][0]
    played_final = played_states_meta[-1][0]
    best_mate_by_mover = best_final.is_checkmate() and best_final.turn == opp_color
    played_mate_by_mover = played_final.is_checkmate() and played_final.turn == opp_color
    played_mate_by_opp = played_final.is_checkmate() and played_final.turn == mover_color
    best_mate_by_opp = best_final.is_checkmate() and best_final.turn == mover_color

    if best_mate_by_mover and not played_mate_by_mover:
        return f"Unlike {move_san}, the best line with {best_san} leads to checkmate."
    if played_mate_by_opp and not best_mate_by_opp:
        return f"Unlike {move_san}, {best_san} avoids checkmate."

    # Equalize ply count across simulated lines before comparing state metrics
    common_plies = min(len(played_states_meta), len(best_states_meta))
    if common_plies % 2 == 1:
        common_plies -= 1
    if common_plies < 2:
        return None

    played_eq = played_states_meta[common_plies - 1][0]
    best_eq = best_states_meta[common_plies - 1][0]
    mover_castled_played = played_castled_at_ply[common_plies - 1]
    mover_castled_best = best_castled_at_ply[common_plies - 1]

    # 1. Material difference (settle 1 immediate recapture ply when horizon ends in a capture)
    mat_before = _mover_net_material(board_before, mover_color)
    mat_played = _settled_material(
        played_states_meta, played_pv_moves, common_plies, mover_color, common_plies - 1
    )
    mat_best = _settled_material(
        best_states_meta, best_pv_moves, common_plies, mover_color, common_plies
    )
    mat_diff = mat_best - mat_played
    if mat_diff >= 1:
        if mat_diff >= 2:
            if mat_played < mat_before:
                return f"Unlike {move_san}, the best line with {best_san} avoids losing material."
            return f"Unlike {move_san}, the best line with {best_san} wins material."
        if mat_played < mat_before:
            return f"Unlike {move_san}, the best line with {best_san} avoids losing a pawn."
        return f"Unlike {move_san}, the best line with {best_san} wins a pawn."

    # 2. Opponent centralization
    if _count_opp_central_pieces(played_eq, opp_color) > _count_opp_central_pieces(best_eq, opp_color):
        return f"Unlike {move_san}, {best_san} prevents {opp_name} from centralizing pieces."

    # 3. Mover central pawn control
    if _count_mover_central_pawns(best_eq, mover_color) > _count_mover_central_pawns(played_eq, mover_color):
        return f"Unlike {move_san}, {best_san} keeps stronger central pawn control."

    # 4. King safety
    played_in_check = played_eq.turn == mover_color and played_eq.is_check()
    best_in_check = best_eq.turn == mover_color and best_eq.is_check()
    if (mover_castled_best and not mover_castled_played) or (played_in_check and not best_in_check):
        return f"Unlike {move_san}, {best_san} gives {mover_name} better king safety."

    # 5. Piece activity
    if _net_piece_mobility(best_eq, mover_color) - _net_piece_mobility(played_eq, mover_color) >= 4:
        return f"Unlike {move_san}, {best_san} leaves {mover_name} with more active pieces."

    return None


def compute_move_positional_fact(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
) -> str | None:
    """Compute a positional fact describing what a quiet or strategic move does."""
    move_san = board_before.san(move)
    mover_color = board_before.turn
    piece = board_before.piece_at(move.from_square)
    if piece is None:
        return None

    if board_before.is_castling(move):
        return f"{move_san} castles to safeguard the king."

    to_sq = move.to_square
    to_sq_name = chess.square_name(to_sq)

    if piece.piece_type == chess.PAWN:
        is_cap = board_before.is_capture(move)
        if _is_pawn_takes_pawn(board_before, move) and to_sq in CENTRAL_SQUARES:
            return f"{move_san} releases central tension on {to_sq_name}."
        if not is_cap and to_sq in CENTRAL_SQUARES:
            return f"{move_san} claims central space on {to_sq_name}."
        if not is_cap and to_sq in WING_CENTER_45_SQUARES:
            return f"{move_san} gains space and challenges the center."
        return None

    piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
    if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        start_sqs = WHITE_START_MINORS if mover_color == chess.WHITE else BLACK_START_MINORS
        if move.from_square in start_sqs and to_sq in EXTENDED_CENTER_SQUARES:
            return f"{move_san} develops the {piece_name} toward the center."

    if piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if to_sq in CENTRAL_SQUARES and move.from_square not in CENTRAL_SQUARES:
            return f"{move_san} centralizes the {piece_name}."

    if piece.piece_type == chess.ROOK:
        rook_file_kind = _classify_rook_file_move(board_before, board_after, move.from_square, to_sq, mover_color)
        if rook_file_kind is not None:
            file_letter = FILE_LETTERS[chess.square_file(to_sq)]
            if rook_file_kind == "open":
                return f"{move_san} places the rook on the open {file_letter}-file."
            return f"{move_san} places the rook on the half-open {file_letter}-file."

    return None


def compute_position_structure_fact(board: chess.Board) -> str | None:
    """Compute one structural or positional fact about the current board."""
    # 1. Central pawn tension
    white_c4_f4 = (chess.C4, chess.D4, chess.E4, chess.F4)
    black_c5_f5 = (chess.C5, chess.D5, chess.E5, chess.F5)
    for w_sq in white_c4_f4:
        wp = board.piece_at(w_sq)
        if wp is not None and wp.color == chess.WHITE and wp.piece_type == chess.PAWN:
            w_file = chess.square_file(w_sq)
            for b_sq in black_c5_f5:
                bp = board.piece_at(b_sq)
                if bp is not None and bp.color == chess.BLACK and bp.piece_type == chess.PAWN:
                    if abs(w_file - chess.square_file(b_sq)) == 1:
                        return (
                            f"Central pawn tension exists between "
                            f"{chess.square_name(w_sq)} and {chess.square_name(b_sq)}."
                        )

    # 2. Central space advantage
    w_space = sum(
        1
        for sq in board.pieces(chess.PAWN, chess.WHITE)
        if 2 <= chess.square_file(sq) <= 5 and 3 <= chess.square_rank(sq) <= 5
    )
    b_space = sum(
        1
        for sq in board.pieces(chess.PAWN, chess.BLACK)
        if 2 <= chess.square_file(sq) <= 5 and 2 <= chess.square_rank(sq) <= 4
    )
    if abs(w_space - b_space) >= 2:
        leader = "White" if w_space > b_space else "Black"
        return f"{leader} has more central space."

    # 3. Open central file (d-file, then e-file)
    for file_idx in (3, 4):
        if _count_pawns_on_file(board, file_idx) == 0:
            return f"The {FILE_LETTERS[file_idx]}-file is open."

    # 4. Opening development lead
    if board.fullmove_number <= 12:
        w_dev = sum(
            1
            for sq in board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.BISHOP, chess.WHITE)
            if sq not in WHITE_START_MINORS
        )
        b_dev = sum(
            1
            for sq in board.pieces(chess.KNIGHT, chess.BLACK) | board.pieces(chess.BISHOP, chess.BLACK)
            if sq not in BLACK_START_MINORS
        )
        if abs(w_dev - b_dev) >= 2:
            leader = "White" if w_dev > b_dev else "Black"
            return f"{leader} leads in piece development."

    return None


def _append_position_facts(
    board: chess.Board,
    fen: str,
    move_uci: str | None,
    raw_facts: list[str | None],
    engine_eval: dict[str, Any] | None = None,
) -> None:
    """Append shared position facts to raw_facts.

    Extract hanging pieces, checks, captures, threats, and king safety.
    Extract position structure, material balance, phase, and opening facts.
    """
    board_before = chess.Board(fen)

    for f in compute_hanging_facts(board):
        raw_facts.append(f)

    for f in compute_checks_and_captures_facts(board, engine_eval=engine_eval):
        raw_facts.append(f)

    raw_facts.append(compute_threat_fact(board, board_before.turn))

    for f in compute_king_safety_facts(board):
        raw_facts.append(f)

    raw_facts.append(compute_position_structure_fact(board))
    raw_facts.append(compute_material_fact(board))
    raw_facts.append(compute_phase_fact(board_before))

    opening = lookup_opening(fen, move_uci)
    if opening:
        raw_facts.append(f"Opening: {opening}.")


def extract_facts(
    task: str,
    fen: str,
    eval_before: dict[str, Any],
    eval_after: dict[str, Any] | None = None,
    move_uci: str | None = None,
) -> list[str]:
    """Extract ordered facts for a position or move task.

    If the task is explain_move, extract move-specific facts.
    If the task is explain_position, extract candidate moves and position facts.
    Remove duplicate facts while preserving order.
    """
    board_before = chess.Board(fen)
    raw_facts: list[str | None] = []

    if task == "explain_move":
        if not move_uci:
            raise ValueError("move_uci is required for task explain_move")
        move = chess.Move.from_uci(move_uci)
        if _is_king_takes_friendly_rook(board_before, move) or move not in board_before.legal_moves:
            raise ValueError(f"Illegal move_uci: {move_uci}")
        board_after = board_before.copy()
        board_after.push(move)

        # 1. Moved-piece attack fact
        raw_facts.append(compute_moved_piece_attack_fact(board_before, move, board_after))

        # 2 & 3. Reply and material change in line
        for f in compute_reply_and_material_change_facts(board_before, move, board_after, eval_after):
            raw_facts.append(f)

        # 4. Opponent recapture centralization or activity gain
        raw_facts.append(compute_recapture_activity_fact(board_before, move, board_after, eval_after))

        # 5. Tactical motifs
        raw_facts.append(compute_fork_fact(board_before, move, board_after, eval_after))
        raw_facts.append(compute_pin_fact(board_before, move, board_after, eval_after))
        raw_facts.append(compute_skewer_fact(board_before, move, board_after, eval_after))
        raw_facts.append(compute_discovered_attack_fact(board_before, move, board_after))
        raw_facts.append(compute_mate_threat_fact(board_before, move, board_after, eval_after))

        # 6. Refutation reason and line comparison
        raw_facts.append(compute_refutation_reason_fact(board_before, move, board_after, eval_before, eval_after))
        raw_facts.append(compute_line_comparison_fact(board_before, move, board_after, eval_before, eval_after))

        # 7. Move positional fact
        raw_facts.append(compute_move_positional_fact(board_before, move, board_after))

        # 8. Shared position facts
        _append_position_facts(board_after, fen, move_uci, raw_facts, engine_eval=eval_after)

    elif task == "explain_position":
        # 1. Candidate moves fact
        pvs = eval_before.get("pvs", [])
        if len(pvs) >= 2:
            c_sans: list[str] = []
            for pv in pvs[:3]:
                moves = pv.get("moves", [])
                if moves:
                    try:
                        m = chess.Move.from_uci(moves[0])
                        if m in board_before.legal_moves:
                            san = board_before.san(m)
                            if san not in c_sans:
                                c_sans.append(san)
                    except ValueError:
                        pass
            if len(c_sans) >= 2:
                raw_facts.append(f"Top candidate moves: {', '.join(c_sans)}.")

        # 2-9. Shared position facts
        _append_position_facts(board_before, fen, None, raw_facts, engine_eval=eval_before)

    else:
        raise ValueError(f"Unknown task: {task}")

    unique_facts: list[str] = []
    seen: set[str] = set()
    for f in raw_facts:
        if f and f not in seen:
            seen.add(f)
            unique_facts.append(f)

    return unique_facts
