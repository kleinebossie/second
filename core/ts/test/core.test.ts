/**
 * Unit tests for the Second chess shared core in TypeScript.
 * This test module verifies fact extraction, verdict classification, and prompt generation.
 * It tests winning chance calculations and opening book lookups.
 * It tests golden file parity when golden files exist.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Chess } from "chessops";
import { parseFen } from "chessops/fen";
import { NormalMove } from "chessops/types";
import { parseUci } from "chessops/util";
import { describe, expect, it } from "vitest";
import {
  buildPromptMessages,
  buildUserMessage,
  classifyVerdict,
  computeChecksAndCapturesFacts,
  computeHangingFacts,
  computeKingSafetyFacts,
  computeLineComparisonFact,
  computeMaterialFact,
  computeMovePositionalFact,
  computeMovedPieceAttackFact,
  computePhaseFact,
  computePositionStructureFact,
  computeRecaptureActivityFact,
  computeRefutationReasonFact,
  computeReplyAndMaterialChangeFacts,
  computeThreatFact,
  EngineEval,
  EngineScore,
  extractFacts,
  formatQwenChatPrompt,
  lookupOpening,
  OPENING_BOOK,
  roundPct,
  scoreToWhiteCp,
  SYSTEM_MESSAGE,
  winChance,
  winChanceWhite,
} from "../src/index.js";

function setupBoard(fen: string): Chess {
  return Chess.fromSetup(parseFen(fen).unwrap()).unwrap();
}

describe("Win chances and score conversion", () => {
  it("converts centipawn scores", () => {
    const scorePos: EngineScore = { type: "cp", value: 150 };
    expect(scoreToWhiteCp(scorePos)).toBe(150.0);

    const scoreNeg: EngineScore = { type: "cp", value: -75 };
    expect(scoreToWhiteCp(scoreNeg)).toBe(-75.0);
  });

  it("converts mate scores to bounded centipawns", () => {
    const mateInOne: EngineScore = { type: "mate", value: 1 };
    expect(scoreToWhiteCp(mateInOne)).toBe(9990.0);

    const mateInThree: EngineScore = { type: "mate", value: 3 };
    expect(scoreToWhiteCp(mateInThree)).toBe(9970.0);

    const matedInOne: EngineScore = { type: "mate", value: -1 };
    expect(scoreToWhiteCp(matedInOne)).toBe(-9990.0);

    const mateZero: EngineScore = { type: "mate", value: 0 };
    expect(scoreToWhiteCp(mateZero)).toBe(0.0);
  });

  it("maintains symmetry between White and Black win chances", () => {
    const score: EngineScore = { type: "cp", value: 65 };
    const w = winChance(score, true);
    const b = winChance(score, false);
    expect(Math.abs(w + b - 100.0)).toBeLessThan(1e-6);
  });

  it("calculates win chances at extremes", () => {
    const mateScore: EngineScore = { type: "mate", value: 1 };
    expect(winChanceWhite(mateScore)).toBeGreaterThan(99.9);

    const matedScore: EngineScore = { type: "mate", value: -1 };
    expect(winChanceWhite(matedScore)).toBeLessThan(0.1);

    const equalScore: EngineScore = { type: "cp", value: 0 };
    expect(Math.abs(winChanceWhite(equalScore) - 50.0)).toBeLessThan(1e-6);
  });

  it("rounds percentage values to nearest integer", () => {
    expect(roundPct(50.4)).toBe(50);
    expect(roundPct(50.5)).toBe(51);
    expect(roundPct(50.6)).toBe(51);
    expect(roundPct(0.0)).toBe(0);
    expect(roundPct(100.0)).toBe(100);
  });
});

describe("Move quality verdict classification", () => {
  it("classifies Only move when alternative drops winning chance", () => {
    const evalBefore: EngineEval = {
      fen: "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      pvs: [
        { score: { type: "cp", value: 100 }, moves: ["e2e4"] },
        { score: { type: "cp", value: -200 }, moves: ["d2d4"] },
      ],
    };
    const evalAfter: EngineEval = {
      fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: 95 }, moves: ["e7e5"] }],
    };
    expect(classifyVerdict(evalBefore, evalAfter, "e2e4")).toBe("Only move");
  });

  it("classifies Only move when the board has only one legal move", () => {
    const evalBefore: EngineEval = {
      fen: "7K/r7/8/8/8/8/8/7k w - - 0 1",
      pvs: [{ score: { type: "cp", value: -1000 }, moves: ["h8g8"] }],
    };
    expect(classifyVerdict(evalBefore, null, "h8g8")).toBe("Only move");
  });

  it("classifies Best when move matches top PV with low drop", () => {
    const evalBefore: EngineEval = {
      fen: "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      pvs: [
        { score: { type: "cp", value: 30 }, moves: ["e2e4"] },
        { score: { type: "cp", value: 25 }, moves: ["d2d4"] },
      ],
    };
    const evalAfter: EngineEval = {
      fen: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: 28 }, moves: ["e7e5"] }],
    };
    expect(classifyVerdict(evalBefore, evalAfter, "e2e4")).toBe("Best");
  });

  it("classifies Good for alternative moves with low drop", () => {
    const evalBefore: EngineEval = {
      fen: "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      pvs: [
        { score: { type: "cp", value: 30 }, moves: ["e2e4"] },
        { score: { type: "cp", value: 25 }, moves: ["d2d4"] },
      ],
    };
    const evalAfter: EngineEval = {
      fen: "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: 24 }, moves: ["d7d5"] }],
    };
    expect(classifyVerdict(evalBefore, evalAfter, "d2d4")).toBe("Good");
  });

  it("classifies Inaccuracy for drops between 5 and 10 percent", () => {
    const evalBefore: EngineEval = {
      fen: "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: 0 }, moves: ["e2e4"] }],
    };
    const evalAfter: EngineEval = {
      fen: "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
      pvs: [{ score: { type: "cp", value: -80 }, moves: ["d7d5"] }],
    };
    expect(classifyVerdict(evalBefore, evalAfter, "g1f3")).toBe("Inaccuracy");
  });

  it("classifies Mistake for drops between 10 and 20 percent", () => {
    const evalBefore: EngineEval = {
      fen: "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: 0 }, moves: ["e2e4"] }],
    };
    const evalAfter: EngineEval = {
      fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: -150 }, moves: ["e7e5"] }],
    };
    expect(classifyVerdict(evalBefore, evalAfter, "a2a3")).toBe("Mistake");
  });

  it("classifies Blunder for drops of at least 20 percent", () => {
    const evalBefore: EngineEval = {
      fen: "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: 0 }, moves: ["e2e4"] }],
    };
    const evalAfter: EngineEval = {
      fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
      pvs: [{ score: { type: "cp", value: -300 }, moves: ["e7e5"] }],
    };
    expect(classifyVerdict(evalBefore, evalAfter, "f2f3")).toBe("Blunder");
  });
});

describe("Material balance facts", () => {
  it("describes equal material", () => {
    const pos = setupBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    expect(computeMaterialFact(pos)).toBe("Material: equal.");
  });

  it("describes single piece advantages", () => {
    const b1 = setupBoard("rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    expect(computeMaterialFact(b1)).toBe("White is up a pawn.");

    const b2 = setupBoard("rnbqkbnr/pppppppp/8/8/8/8/PP1PP1PP/RNBQKBNR w KQkq - 0 1");
    expect(computeMaterialFact(b2)).toBe("Black is up two pawns.");

    const b3 = setupBoard("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    expect(computeMaterialFact(b3)).toBe("White is up a knight.");

    const b4 = setupBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1");
    expect(computeMaterialFact(b4)).toBe("Black is up a bishop.");

    const b5 = setupBoard("1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    expect(computeMaterialFact(b5)).toBe("White is up a rook.");

    const b6 = setupBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
    expect(computeMaterialFact(b6)).toBe("Black is up a queen.");
  });

  it("describes exchange advantages", () => {
    const b = setupBoard("4k1n1/8/8/8/8/8/8/4K1R1 w - - 0 1");
    expect(computeMaterialFact(b)).toBe("White is up the exchange.");
  });

  it("describes composite material advantages with fallbacks", () => {
    const b4 = setupBoard("4k3/8/8/8/8/8/4P3/4KN2 w - - 0 1");
    expect(computeMaterialFact(b4)).toBe("White is up a minor piece and a pawn in material.");

    const b6 = setupBoard("4k3/8/8/8/8/8/8/4KBB1 w - - 0 1");
    expect(computeMaterialFact(b6)).toBe("White is up two minor pieces in material.");

    const b7 = setupBoard("4k3/8/8/8/8/8/3PP3/4K1R1 w - - 0 1");
    expect(computeMaterialFact(b7)).toBe("White is up a rook and two pawns in material.");

    const b8 = setupBoard("4k3/8/8/8/8/8/8/4K1RN w - - 0 1");
    expect(computeMaterialFact(b8)).toBe("White is up a rook and a minor piece in material.");
  });
});

describe("Game phase facts", () => {
  it("identifies opening, middlegame, and endgame", () => {
    const bOpen = setupBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    expect(computePhaseFact(bOpen)).toBe("Phase: opening.");

    const bEnd = setupBoard("4k3/8/8/8/8/8/8/4K3 w - - 0 50");
    expect(computePhaseFact(bEnd)).toBe("Phase: endgame.");

    const bMid = setupBoard("r1b1k2r/pp1n1ppp/2p1pn2/q2p4/2PP4/2N1PN2/PP1Q1PPP/R3KB1R w KQkq - 1 12");
    expect(computePhaseFact(bMid)).toBe("Phase: middlegame.");
  });
});

describe("King safety facts", () => {
  it("reports check and uncastled kings", () => {
    const bChk = setupBoard("rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3");
    expect(computeKingSafetyFacts(bChk)).toEqual(["Black king is in check on e8."]);

    const bNocastle = setupBoard("r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 10 10");
    expect(computeKingSafetyFacts(bNocastle)).toEqual([
      "White has not castled.",
      "Black has not castled.",
    ]);
  });
});

describe("Hanging pieces facts", () => {
  it("detects undefended attacked pieces", () => {
    const b = setupBoard("rnbqkb1r/ppp1pppp/3p4/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1");
    expect(computeHangingFacts(b)).toEqual(["The knight on e5 is not defended."]);
  });
});

describe("Checks and captures facts", () => {
  it("filters noisy checks while keeping high value captures", () => {
    const b = setupBoard("r1bqkb1r/pppp1ppp/8/4p3/2Bn4/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5");
    const facts = computeChecksAndCapturesFacts(b);
    expect(facts).not.toContain("White has a check: Bxf7+.");
    expect(facts).toContain("White can capture the knight on d4 with Nxd4.");
  });

  it("keeps Stockfish PV checks within 100 cp and drops weaker PV checks", () => {
    const fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5";
    const b = setupBoard(fen);
    const evClose: EngineEval = {
      fen,
      pvs: [
        { score: { type: "cp", value: 25 }, moves: ["c1g5", "h7h6"] },
        { score: { type: "cp", value: -40 }, moves: ["c4f7", "e8f7"] },
      ],
    };
    expect(computeChecksAndCapturesFacts(b, evClose)).toContain("White has a check: Bxf7+.");

    const evFar: EngineEval = {
      fen,
      pvs: [
        { score: { type: "cp", value: 30 }, moves: ["c1g5", "h7h6"] },
        { score: { type: "cp", value: -120 }, moves: ["c4f7", "e8f7"] },
      ],
    };
    expect(computeChecksAndCapturesFacts(b, evFar)).not.toContain("White has a check: Bxf7+.");
  });

  it("keeps mate, material-winning, royal-fork, and castling-strip checks", () => {
    const bMate = setupBoard("6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1");
    expect(computeChecksAndCapturesFacts(bMate)).toContain("White has a check: Rd8#.");

    const bWinCap = setupBoard("8/4k3/3n4/4P3/8/8/8/4K3 w - - 0 1");
    expect(computeChecksAndCapturesFacts(bWinCap)).toContain("White has a check: exd6+.");

    const bFork = setupBoard("r3k2r/ppp2ppp/8/3N4/8/8/PPP2PPP/R1B1K2R w KQkq - 0 10");
    expect(computeChecksAndCapturesFacts(bFork)).toContain("White has a check: Nxc7+.");

    const bCastleStrip = setupBoard("4k2r/4pppp/8/8/2B5/8/8/4K3 w k - 0 15");
    expect(computeChecksAndCapturesFacts(bCastleStrip)).toContain("White has a check: Bb5+.");
  });
});

describe("Threats facts", () => {
  it("identifies checkmate threats", () => {
    const b = setupBoard("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3");
    expect(computeThreatFact(b, "white")).toBe("White threatens Qxf7#.");
  });

  it("identifies winning capture threats", () => {
    const b = setupBoard("4k3/1ppppppp/8/n7/8/8/1PPPPPPP/R3KBNR b - - 0 1");
    expect(computeThreatFact(b, "white")).toBe("White threatens Rxa5.");
  });

  it("identifies safe check threats", () => {
    const b = setupBoard("r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4");
    expect(computeThreatFact(b, "white")).toBe("White threatens Bxf7+.");
  });
});

describe("Moved piece attack facts", () => {
  it("identifies joint attacks after a move", () => {
    const bBefore = setupBoard("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
    const move = parseUci("f3g5")! as NormalMove;
    const bAfter = bBefore.clone();
    bAfter.play(move);

    const fact = computeMovedPieceAttackFact(bBefore, move, bAfter);
    expect(fact).toBe("Ng5 attacks the pawn on f7 together with the bishop on c4.");
  });

  it("returns null when the moved piece is a king", () => {
    const bBefore = setupBoard("4k3/8/8/8/8/4n3/8/4K3 w - - 0 1");
    const move = parseUci("e1f2")! as NormalMove;
    const bAfter = bBefore.clone();
    bAfter.play(move);

    const fact = computeMovedPieceAttackFact(bBefore, move, bAfter);
    expect(fact).toBeNull();
  });
});

describe("Reply and material change facts", () => {
  it("identifies opponent reply and subsequent material win", () => {
    const bBefore = setupBoard("r1bqkb1r/pp1p1ppp/2n5/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 5");
    const move = parseUci("f3e5")! as NormalMove;
    const bAfter = bBefore.clone();
    bAfter.play(move);

    const evalAfter: EngineEval = {
      fen: "",
      pvs: [
        {
          score: { type: "cp", value: -150 },
          moves: ["d8a5", "b1c3", "a5e5"],
        },
      ],
    };
    const facts = computeReplyAndMaterialChangeFacts(bBefore, move, bAfter, evalAfter);
    expect(facts.length).toBe(2);
    expect(facts[0]).toBe("After Nxe5, Black's best reply is Qa5.");
    expect(facts[1]).toBe("After Qa5, Black wins the knight on e5.");
  });

  it("handles en passant capture in Case 1", () => {
    const bBefore = setupBoard("rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2");
    const move = parseUci("d7d5")! as NormalMove;
    const bAfter = bBefore.clone();
    bAfter.play(move);

    const evalAfter: EngineEval = {
      fen: "",
      pvs: [
        {
          score: { type: "cp", value: 150 },
          moves: ["e5d6"],
        },
      ],
    };
    const facts = computeReplyAndMaterialChangeFacts(bBefore, move, bAfter, evalAfter);
    expect(facts.length).toBe(2);
    expect(facts[0]).toBe("After d5, White's best reply is exd6.");
    expect(facts[1]).toBe("After d5, White captures the pawn on d5 with exd6.");
  });
});

describe("Opening book lookup", () => {
  it("matches positions and moves to opening names", () => {
    expect(Object.keys(OPENING_BOOK).length).toBe(49);

    const fenE4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    expect(lookupOpening(fenE4)).toBe("King's Pawn Game");

    const fenStart = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    expect(lookupOpening(fenStart, "c7c5")).toBe("Sicilian Defense");

    expect(lookupOpening("8/8/8/8/8/8/8/4K2k w - - 0 1")).toBeNull();
  });
});

describe("Full prompt formatting", () => {
  it("formats explain_move user message", () => {
    const fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    const evalBefore: EngineEval = {
      fen,
      pvs: [
        { score: { type: "cp", value: 45 }, moves: ["d2d3", "f8c5", "c2c3", "d7d6"] },
        { score: { type: "cp", value: 40 }, moves: ["b1c3"] },
      ],
    };
    const evalAfter: EngineEval = {
      fen: "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4",
      pvs: [
        { score: { type: "cp", value: 35 }, moves: ["d7d5", "e4d5", "c6a5"] },
      ],
    };
    const msg = buildUserMessage("explain_move", "club", "short", fen, evalBefore, evalAfter, "f3g5");
    const lines = msg.split("\n");

    expect(lines[0]).toBe("task: explain_move");
    expect(lines[1]).toBe("level: club");
    expect(lines[2]).toBe("detail: short");
    expect(lines[3]).toBe("");
    expect(lines[4]).toBe(`FEN: ${fen}`);
    expect(lines[5]).toBe("White to move, move 4.");
    expect(lines[6]).toBe("Move played: Ng5.");
    expect(lines[7]).toBe("Verdict: Good.");
    expect(lines[8]).toBe("White's winning chances: 54% → 53%.");
    expect(lines[9]).toBe("Best move: d3.");
    expect(lines[10]).toBe("Best line: d3 Bc5 c3 d6.");
    expect(lines[11]).toBe("Facts:");
    expect(lines).toContain("- Ng5 attacks the pawn on f7 together with the bishop on c4.");
    expect(lines).toContain("- After Ng5, Black's best reply is d5.");
    expect(lines).toContain("- Material: equal.");
    expect(lines).toContain("- Phase: opening.");
    expect(lines).toContain("- Opening: Italian Game, Two Knights Defense.");
  });

  it("formats explain_move user message when move captures piece", () => {
    const fen = "r4rk1/4ppbp/2p3p1/1p1Q4/p2P4/2P2N2/PPK2PPP/3R3R b - - 0 19";
    const evalBefore: EngineEval = {
      fen,
      pvs: [{ score: { type: "cp", value: 0 }, moves: ["c6d5", "f3e5"] }],
    };
    const evalAfter: EngineEval = {
      fen: "r4rk1/4ppbp/6p1/1p1p4/p2P4/2P2N2/PPK2PPP/3R3R w - - 0 20",
      pvs: [{ score: { type: "cp", value: 0 }, moves: ["f3e5"] }],
    };
    const msg = buildUserMessage("explain_move", "club", "short", fen, evalBefore, evalAfter, "c6d5");
    const lines = msg.split("\n");

    expect(lines).toContain("Move played: cxd5.");
    expect(lines).toContain("- Material: equal.");
    expect(lines).not.toContain("- White is up a queen in material.");
  });

  it("formats explain_position user message", () => {
    const fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    const evalBefore: EngineEval = {
      fen,
      pvs: [
        { score: { type: "cp", value: 45 }, moves: ["d2d3", "f8c5"] },
        { score: { type: "cp", value: 40 }, moves: ["b1c3", "f8c5"] },
      ],
    };
    const msg = buildUserMessage("explain_position", "beginner", "long", fen, evalBefore);
    const lines = msg.split("\n");

    expect(lines[0]).toBe("task: explain_position");
    expect(lines[1]).toBe("level: beginner");
    expect(lines[2]).toBe("detail: long");
    expect(lines[3]).toBe("");
    expect(lines[4]).toBe(`FEN: ${fen}`);
    expect(lines[5]).toBe("White to move, move 4.");
    expect(lines[6]).toBe("White's winning chances: 54%.");
    expect(lines[7]).toBe("Best move: d3.");
    expect(lines[8]).toBe("Best line: d3 Bc5.");
    expect(lines[9]).toBe("Facts:");
    expect(lines).toContain("- Top candidate moves: d3, Nc3.");
    expect(lines).toContain("- Material: equal.");
  });

  it("formats chat messages and Qwen prompt template", () => {
    const fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    const evalBefore: EngineEval = {
      fen,
      pvs: [{ score: { type: "cp", value: 15 }, moves: ["c7c5"] }],
    };
    const messages = buildPromptMessages("explain_position", "club", "short", fen, evalBefore);
    expect(messages.length).toBe(2);
    expect(messages[0].role).toBe("system");
    expect(messages[0].content).toBe(SYSTEM_MESSAGE);
    expect(messages[1].role).toBe("user");

    const chatPrompt = formatQwenChatPrompt(messages[1].content);
    expect(chatPrompt.startsWith(`<|im_start|>system\n${SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n`)).toBe(true);
    expect(chatPrompt.endsWith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")).toBe(true);
  });
});

interface GoldenCase {
  task: string;
  level: string;
  detail: string;
  fen: string;
  eval_before: EngineEval;
  eval_after?: EngineEval | null;
  move_uci?: string | null;
  expected_user_message: string;
  expected_chat_prompt: string;
}

function assertGoldenFile(filename: string, expectedCount: number): void {
  const goldenPath = resolve(import.meta.dirname, `../../golden/${filename}`);
  expect(existsSync(goldenPath)).toBe(true);

  const data = JSON.parse(readFileSync(goldenPath, "utf-8")) as GoldenCase[];
  expect(data.length).toBe(expectedCount);

  for (const c of data) {
    const actualUserMsg = buildUserMessage(
      c.task,
      c.level,
      c.detail,
      c.fen,
      c.eval_before,
      c.eval_after,
      c.move_uci
    );
    expect(actualUserMsg).toBe(c.expected_user_message);

    const actualChatPrompt = formatQwenChatPrompt(actualUserMsg);
    expect(actualChatPrompt).toBe(c.expected_chat_prompt);
  }
}

describe("Golden parity tests", () => {
  it("verifies byte-for-byte parity against golden file", () => {
    assertGoldenFile("golden_200.json", 200);
  });

  it("verifies byte-for-byte parity against tactical golden file", () => {
    assertGoldenFile("golden_tactical_30.json", 30);
  });

  it("verifies byte-for-byte parity against positional golden file", () => {
    assertGoldenFile("golden_positional_30.json", 30);
  });
});

describe("Positional and Phase 11 facts", () => {
  it("detects opponent recapture centralization and activity gain", () => {
    const bBefore = setupBoard("rnb1k2r/ppppqppp/5n2/4p3/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R b KQkq - 0 6");
    const move = parseUci("e5d4")! as NormalMove;
    const bAfter = bBefore.clone();
    bAfter.play(move);
    const evalAfter: EngineEval = {
      fen: "",
      pvs: [{ score: { type: "cp", value: 74 }, moves: ["f3d4", "e8g8"] }],
    };
    expect(computeRecaptureActivityFact(bBefore, move, bAfter, evalAfter)).toBe(
      "After exd4, White's recapture Nxd4 centralizes the knight."
    );

    const bAct = setupBoard("r2qk2r/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 0 7");
    const mAct = parseUci("g4f3")! as NormalMove;
    const baAct = bAct.clone();
    baAct.play(mAct);
    const evAct: EngineEval = {
      fen: "",
      pvs: [{ score: { type: "cp", value: 70 }, moves: ["d1f3", "c6d4"] }],
    };
    expect(computeRecaptureActivityFact(bAct, mAct, baAct, evAct)).toBe(
      "After Bxf3, White's recapture Qxf3 increases queen activity."
    );
  });

  it("extracts positional refutation and line comparison facts", () => {
    const fen = "rnb1k2r/ppppqppp/5n2/4p3/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R b KQkq - 0 6";
    const b = setupBoard(fen);
    const m = parseUci("e5d4")! as NormalMove;
    const ba = b.clone();
    ba.play(m);
    const eb: EngineEval = {
      fen,
      pvs: [{ score: { type: "cp", value: 16 }, moves: ["e5e4", "f3g1", "b4c3", "d2c3"] }],
    };
    const ea: EngineEval = {
      fen: "",
      pvs: [{ score: { type: "cp", value: 74 }, moves: ["f3d4", "e8g8", "f1d3"] }],
    };
    expect(computeRefutationReasonFact(b, m, ba, eb, ea)).toBe(
      "Best move e4 gains space and attacks the knight on f3 instead of trading in the center."
    );
    expect(computeLineComparisonFact(b, m, ba, eb, ea)).toBe(
      "Unlike exd4, e4 prevents White from centralizing pieces."
    );
  });

  it("extracts quiet move positional and position structure facts", () => {
    const bCastle = setupBoard("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
    const mCastle = parseUci("e1g1")! as NormalMove;
    const baCastle = bCastle.clone();
    baCastle.play(mCastle);
    expect(computeMovePositionalFact(bCastle, mCastle, baCastle)).toBe(
      "O-O castles to safeguard the king."
    );

    const bTension = setupBoard("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3");
    expect(computePositionStructureFact(bTension)).toBe(
      "Central pawn tension exists between d4 and e5."
    );

    const bSpace = setupBoard("rnbqkbnr/pp3ppp/2p1p3/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 5");
    expect(computePositionStructureFact(bSpace)).toBe("White has more central space.");
  });

  it("handles en passant check, defended promo check, O-O+, rook files, mate comparison, and illegal moves", () => {
    const bEp = setupBoard("8/4k3/8/2Pp4/2N5/8/8/6K1 w - d6 0 1");
    expect(computeChecksAndCapturesFacts(bEp)).toEqual(["White has a check: cxd6+."]);

    const bPromoUndef = setupBoard("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1");
    expect(computeChecksAndCapturesFacts(bPromoUndef)).toEqual([
      "White has a check: exd8=Q+.",
    ]);
    const bPromoDef = setupBoard("q2r3k/4P3/8/8/8/8/8/4K3 w - - 0 1");
    expect(computeChecksAndCapturesFacts(bPromoDef)).toEqual([
      "White has a check: exd8=Q+.",
    ]);

    const bCas = setupBoard("5k2/8/8/8/8/8/8/4K2R w K - 0 1");
    expect(
      computeChecksAndCapturesFacts(bCas, {
        fen: "",
        pvs: [{ score: { type: "cp", value: 120 }, moves: ["e1g1"] }],
      })
    ).toEqual(["White has a check: O-O+."]);

    const fenStart = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const evalDummy: EngineEval = {
      fen: fenStart,
      pvs: [{ score: { type: "cp", value: 20 }, moves: ["e2e4"] }],
    };
    expect(() => extractFacts("explain_move", fenStart, evalDummy, evalDummy, "e2e5")).toThrow();
    expect(() =>
      buildUserMessage("explain_move", "club", "short", fenStart, evalDummy, evalDummy, "e2e5")
    ).toThrow();

    const fenRk = "3k4/p4ppp/8/8/8/8/PP3PPP/R5K1 w - - 0 20";
    const bRk = setupBoard(fenRk);
    const mBad = parseUci("a2a3")! as NormalMove;
    const baBad = bRk.clone();
    baBad.play(mBad);
    expect(
      computeRefutationReasonFact(
        bRk,
        mBad,
        baBad,
        { fen: fenRk, pvs: [{ score: { type: "cp", value: 60 }, moves: ["a1d1"] }] },
        { fen: "", pvs: [{ score: { type: "cp", value: -20 }, moves: ["d8c7"] }] }
      )
    ).toBe("Best move Rd1+ places the rook on an open file.");

    const mAd1 = parseUci("a1d1")! as NormalMove;
    const baAd1 = bRk.clone();
    baAd1.play(mAd1);
    expect(computeMovePositionalFact(bRk, mAd1, baAd1)).toBe("Rd1+ places the rook on the open d-file.");

    const fenMate = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 25";
    const bMate = setupBoard(fenMate);
    const mh3 = parseUci("h2h3")! as NormalMove;
    const bah3 = bMate.clone();
    bah3.play(mh3);
    expect(
      computeLineComparisonFact(
        bMate,
        mh3,
        bah3,
        { fen: fenMate, pvs: [{ score: { type: "mate", value: 1 }, moves: ["d1d8"] }] },
        { fen: "", pvs: [{ score: { type: "cp", value: 0 }, moves: ["h7h6", "g1h2"] }] }
      )
    ).toBe("Unlike h3, the best line with Rd8# leads to checkmate.");

    const fenAv = "4r1k1/5ppp/8/8/8/8/PR3PPP/6K1 w - - 0 25";
    const bAv = setupBoard(fenAv);
    const ma3 = parseUci("a2a3")! as NormalMove;
    const baa3 = bAv.clone();
    baa3.play(ma3);
    expect(
      computeLineComparisonFact(
        bAv,
        ma3,
        baa3,
        { fen: fenAv, pvs: [{ score: { type: "cp", value: 0 }, moves: ["g2g3", "e8e1", "g1g2", "g7g6"] }] },
        { fen: "", pvs: [{ score: { type: "mate", value: -1 }, moves: ["e8e1"] }] }
      )
    ).toBe("Unlike a3, g3 avoids checkmate.");

    const fenEx = "3r2k1/pp3ppp/8/3N4/8/8/PP3PPP/3R2K1 b - - 0 22";
    const bEx = setupBoard(fenEx);
    const mrxd5 = parseUci("d8d5")! as NormalMove;
    const barxd5 = bEx.clone();
    barxd5.play(mrxd5);
    expect(
      computeLineComparisonFact(
        bEx,
        mrxd5,
        barxd5,
        { fen: fenEx, pvs: [{ score: { type: "cp", value: 0 }, moves: ["g8f8", "d5e3", "g7g6", "d1c1"] }] },
        { fen: "", pvs: [{ score: { type: "cp", value: 100 }, moves: ["d1d5", "g8f8", "g1f1"] }] }
      )
    ).toBe("Unlike Rxd5, the best line with Kf8 avoids losing material.");

    const fenBxc6 = "r1bqkb1r/pp1p1ppp/2n1pn2/1Bp5/3PP3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 5";
    const bBxc6 = setupBoard(fenBxc6);
    const mBxc6 = parseUci("b5c6")! as NormalMove;
    const baBxc6 = bBxc6.clone();
    baBxc6.play(mBxc6);
    const refBxc6 = computeRefutationReasonFact(
      bBxc6,
      mBxc6,
      baBxc6,
      { fen: fenBxc6, pvs: [{ score: { type: "cp", value: 75 }, moves: ["c2c4"] }] },
      { fen: "", pvs: [{ score: { type: "cp", value: -20 }, moves: ["d7c6"] }] }
    );
    expect(refBxc6).not.toBeNull();
    expect(refBxc6).not.toContain("instead of trading");

    const fenF3 = "rnbqkb1r/pppp1ppp/8/4p3/4P1n1/8/PPPP1PPP/RNBQKBNR w KQkq - 2 4";
    const bF3 = setupBoard(fenF3);
    const mf3Bad = parseUci("a2a3")! as NormalMove;
    const baf3Bad = bF3.clone();
    baf3Bad.play(mf3Bad);
    expect(
      computeRefutationReasonFact(
        bF3,
        mf3Bad,
        baf3Bad,
        { fen: fenF3, pvs: [{ score: { type: "cp", value: 60 }, moves: ["f2f3"] }] },
        { fen: "", pvs: [{ score: { type: "cp", value: -30 }, moves: ["g4f6"] }] }
      )
    ).toBe("Best move f3 attacks the knight on g4.");

    const fenOdd = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
    const bOdd = setupBoard(fenOdd);
    const ma3Odd = parseUci("a2a3")! as NormalMove;
    const baa3Odd = bOdd.clone();
    baa3Odd.play(ma3Odd);
    expect(
      computeLineComparisonFact(
        bOdd,
        ma3Odd,
        baa3Odd,
        { fen: fenOdd, pvs: [{ score: { type: "cp", value: 40 }, moves: ["f1b5", "c8d7", "b5c6"] }] },
        { fen: "", pvs: [{ score: { type: "cp", value: -10 }, moves: ["d7d6", "f1c4"] }] }
      )
    ).toBeNull();

    const fenPtp = "4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1";
    const bPtp = setupBoard(fenPtp);
    const mDxe5N = parseUci("d4e5")! as NormalMove;
    const baDxe5N = bPtp.clone();
    baDxe5N.play(mDxe5N);
    expect(computeMovePositionalFact(bPtp, mDxe5N, baDxe5N)).toBeNull();

    const fenPtpPawn = "4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1";
    const bPtpPawn = setupBoard(fenPtpPawn);
    const mDxe5P = parseUci("d4e5")! as NormalMove;
    const baDxe5P = bPtpPawn.clone();
    baDxe5P.play(mDxe5P);
    expect(computeMovePositionalFact(bPtpPawn, mDxe5P, baDxe5P)).toBe("dxe5 releases central tension on e5.");

    const fenRopen = "6k1/R3p3/8/8/8/8/P7/6K1 w - - 0 1";
    const bRopen = setupBoard(fenRopen);
    const mRxe7 = parseUci("a7e7")! as NormalMove;
    const baRxe7 = bRopen.clone();
    baRxe7.play(mRxe7);
    expect(computeMovePositionalFact(bRopen, mRxe7, baRxe7)).toBe("Rxe7 places the rook on the open e-file.");

    const fenC960 = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    const evC960: EngineEval = {
      fen: fenC960,
      pvs: [{ score: { type: "cp", value: 0 }, moves: ["e1g1"] }],
    };
    expect(extractFacts("explain_move", fenC960, evC960, evC960, "e1g1").length).toBeGreaterThan(0);
    expect(() => extractFacts("explain_move", fenC960, evC960, evC960, "e1h1")).toThrow();
    expect(() => buildUserMessage("explain_move", "club", "short", fenC960, evC960, evC960, "e1h1")).toThrow();

    const fenDev152 = "r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQ1RK1 w kq - 2 7";
    const bDev152 = setupBoard(fenDev152);
    const mC4 = parseUci("c2c4")! as NormalMove;
    const baC4 = bDev152.clone();
    baC4.play(mC4);
    const lcDev152 = computeLineComparisonFact(
      bDev152,
      mC4,
      baC4,
      {
        fen: fenDev152,
        pvs: [{ score: { type: "cp", value: 30 }, moves: ["f1e1", "h7h6", "h2h3", "c6a5", "b3c2", "c7c5"] }],
      },
      {
        fen: "",
        pvs: [{ score: { type: "cp", value: -34 }, moves: ["c6a5", "c1d2", "a5b3", "a2b3", "c8e6"] }],
      }
    );
    expect(lcDev152 === null || !lcDev152.includes("avoids losing material")).toBe(true);

    const fenDownP = "4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1";
    const bDownP = setupBoard(fenDownP);
    const mKf1 = parseUci("e1f1")! as NormalMove;
    const baKf1 = bDownP.clone();
    baKf1.play(mKf1);
    expect(
      computeLineComparisonFact(
        bDownP,
        mKf1,
        baKf1,
        { fen: fenDownP, pvs: [{ score: { type: "cp", value: 0 }, moves: ["f3e5", "e8e7"] }] },
        { fen: "", pvs: [{ score: { type: "cp", value: -120 }, moves: ["e8e7"] }] }
      )
    ).toBe("Unlike Kf1, the best line with Nxe5 wins a pawn.");
  });
});
