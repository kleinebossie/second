"""Opening book lookup for chess positions.

This module stores standard opening positions.
It matches positions by their first three FEN fields.
"""

from __future__ import annotations

import chess

OPENING_BOOK: dict[str, str] = {
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq": "King's Pawn Game",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": "Open Game",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq": "King's Knight Opening",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq": "King's Knight Opening, Normal Variation",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq": "Italian Game",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq": "Italian Game, Giuoco Piano",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq": "Italian Game, Two Knights Defense",
    "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq": "Italian Game, Two Knights Defense",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq": "Ruy Lopez",
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq": "Ruy Lopez, Morphy Defense",
    "r1bqk2r/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq": "Ruy Lopez",
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq": "Four Knights Game, Italian Variation",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq": "Scotch Game",
    "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq": "Bishop's Opening",
    "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq": "King's Gambit",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": "Sicilian Defense",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq": "Sicilian Defense, Open",
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq": "Sicilian Defense",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq": "Sicilian Defense, Open",
    "rnbqk2r/pp2bppp/3p1n2/4p3/3NP3/2N5/PPP1BPPP/R1BQK2R w KQkq": "Sicilian Defense, Najdorf Variation",
    "rnbqkb1r/pp3ppp/3p1n2/2p1p3/2P1P3/2N2N2/PP1P1PPP/R1BQKB1R w KQkq": "Sicilian Defense",
    "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq": "Sicilian Defense, Old Sicilian",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": "French Defense",
    "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq": "French Defense",
    "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq": "French Defense",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": "Caro-Kann Defense",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq": "Caro-Kann Defense",
    "rnbqkbnr/pp2pppp/2p5/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq": "Caro-Kann Defense, Two Knights Attack",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": "Scandinavian Defense",
    "rnbqkb1r/pppppp1p/5np1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": "Modern Defense",
    "rnbqkbnr/pppppp1p/8/6p1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq": "Borg Defense",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq": "Queen's Pawn Game",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq": "Queen's Pawn Game",
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq": "Queen's Gambit",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq": "Queen's Gambit Declined",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq": "Queen's Gambit Declined",
    "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq": "Queen's Gambit Accepted",
    "rnbqkb1r/ppp2ppp/4pn2/8/2pP4/4PN2/PP3PPP/RNBQKB1R w KQkq": "Queen's Gambit Accepted",
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq": "Slav Defense",
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq": "Slav Defense",
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq": "King's Indian Defense",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ": "King's Indian Defense, Orthodox Variation",
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq": "Grunfeld Defense",
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq": "Indian Defense",
    "r1bqk1nr/pppp1ppp/2n1p3/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq": "Nimzo-Indian Defense",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq": "English Opening",
    "rnbqkb1r/pp1ppppp/5n2/2p5/2PP4/8/PP2PPPP/RNBQKBNR w KQkq": "Benoni Defense",
    "r1bqkbnr/pp1ppppp/2n5/2p5/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq": "English Opening, Symmetrical Variation",
    "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq": "English Opening",
}


def lookup_opening(fen: str, move_uci: str | None = None) -> str | None:
    """Find the opening name for a position.

    If move_uci is set, test the position after the move.
    If the move is not in the book, test the current position.
    When no match exists, return None.
    """
    if move_uci:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                after_key = " ".join(board.fen().split(" ")[:3])
                if after_key in OPENING_BOOK:
                    return OPENING_BOOK[after_key]
        except ValueError:
            pass

    key = " ".join(fen.split(" ")[:3])
    return OPENING_BOOK.get(key)
