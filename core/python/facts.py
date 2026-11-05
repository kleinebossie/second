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
    wc_before = win_chance_side(pvs[0]["score"], is_white) if (pvs and "score" in pvs[0]) else 50.0

    if eval_after and eval_after.get("pvs") and "score" in eval_after["pvs"][0]:
        wc_after = win_chance_side(eval_after["pvs"][0]["score"], is_white)
    else:
        wc_after = wc_before

    if move_uci == best_move_uci:
        if board.legal_moves.count() == 1:
            return "Only move"
        if len(pvs) >= 2 and "score" in pvs[1]:
            wc_second = win_chance_side(pvs[1]["score"], is_white)
            if (wc_before - wc_second) >= 15.0:
                return "Only move"

    drop = wc_before - wc_after
    if move_uci == best_move_uci and drop <= 2.0:
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


def compute_checks_and_captures_facts(board: chess.Board) -> list[str]:
    """Find available checks and high value captures.

    If a check is legal, record the first check.
    If a capture of a non-pawn piece is legal, record the top capture.
    """
    side_name = "White" if board.turn == chess.WHITE else "Black"
    facts: list[str] = []
    check_move_san: str | None = None

    checks: list[chess.Move] = []
    for m in board.legal_moves:
        if board.gives_check(m):
            checks.append(m)

    if checks:
        checks.sort(key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
        check_move = checks[0]
        check_move_san = board.san(check_move)
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
                    m.promotion or 0,
                    m,
                    target,
                ))

    captures.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    for item in captures:
        m = item[4]
        target = item[5]
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
        mate_moves.sort(key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
        san = temp.san(mate_moves[0])
        return f"{color_name} threatens {san}."

    # Category B: Winning capture
    winning_captures: list[tuple[int, int, int, int, chess.Move]] = []
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
                        m.promotion or 0,
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
        safe_checks.sort(key=lambda m: (m.from_square, m.to_square, m.promotion or 0))
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
    if cap is None and board_after.is_en_passant(reply_move):
        cap = sim.piece_at(reply_move.to_square + (-8 if opp_color == chess.WHITE else 8))
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
            cap_sq = chess.square_name(reply_move.to_square)
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
                            facts.append(f"After {reply_san}, {opp_name} wins the {cap_name} on {sq_name}.")
        except ValueError:
            pass

    return facts


def _append_position_facts(
    board: chess.Board,
    fen: str,
    move_uci: str | None,
    raw_facts: list[str | None],
) -> None:
    """Append shared position facts to raw_facts.

    Extract hanging pieces, checks, captures, threats, and king safety.
    Extract material balance, phase, and opening facts.
    """
    board_before = chess.Board(fen)

    for f in compute_hanging_facts(board):
        raw_facts.append(f)

    for f in compute_checks_and_captures_facts(board):
        raw_facts.append(f)

    raw_facts.append(compute_threat_fact(board, board_before.turn))

    for f in compute_king_safety_facts(board):
        raw_facts.append(f)

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
        board_after = board_before.copy()
        board_after.push(move)

        # 1. Moved-piece attack fact
        raw_facts.append(compute_moved_piece_attack_fact(board_before, move, board_after))

        # 2 & 3. Reply and material change in line
        for f in compute_reply_and_material_change_facts(board_before, move, board_after, eval_after):
            raw_facts.append(f)

        # 4-10. Shared position facts
        _append_position_facts(board_after, fen, move_uci, raw_facts)

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

        # 2-8. Shared position facts
        _append_position_facts(board_before, fen, None, raw_facts)

    else:
        raise ValueError(f"Unknown task: {task}")

    unique_facts: list[str] = []
    seen: set[str] = set()
    for f in raw_facts:
        if f and f not in seen:
            seen.add(f)
            unique_facts.append(f)

    return unique_facts
