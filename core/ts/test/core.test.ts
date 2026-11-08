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
import { Color, NormalMove } from "chessops/types";
import { parseUci } from "chessops/util";
import { describe, expect, it } from "vitest";
import {
  buildPromptMessages,
  buildUserMessage,
  classifyVerdict,
  computeChecksAndCapturesFacts,
  computeHangingFacts,
  computeKingSafetyFacts,
  computeMaterialFact,
  computeMovedPieceAttackFact,
  computePhaseFact,
  computeReplyAndMaterialChangeFacts,
  computeThreatFact,
  EngineEval,
  EngineScore,
  formatQwenChatPrompt,
  lookupOpening,
  OPENING_BOOK,
  roundPct,
  scoreToWhiteCp,
  SYSTEM_MESSAGE,
  winChance,
  winChanceSide,
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
  it("detects legal checks and high value captures", () => {
    const b = setupBoard("r1bqkb1r/pppp1ppp/8/4p3/2Bn4/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5");
    const facts = computeChecksAndCapturesFacts(b);
    expect(facts).toContain("White has a check: Bxf7+.");
    expect(facts).toContain("White can capture the knight on d4 with Nxd4.");
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
    expect(facts[1]).toBe("After d5, White captures the pawn on d6 with exd6.");
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

describe("Golden parity tests", () => {
  it("verifies byte-for-byte parity against golden file", () => {
    const goldenPath = resolve(import.meta.dirname, "../../golden/golden_200.json");
    expect(existsSync(goldenPath)).toBe(true);

    const data = JSON.parse(readFileSync(goldenPath, "utf-8"));
    expect(data.length).toBe(200);
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

    for (let idx = 0; idx < data.length; idx++) {
      const c = data[idx] as GoldenCase;
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
  });

  it("verifies byte-for-byte parity against tactical golden file", () => {
    const tacticalPath = resolve(import.meta.dirname, "../../golden/golden_tactical_30.json");
    expect(existsSync(tacticalPath)).toBe(true);

    const data = JSON.parse(readFileSync(tacticalPath, "utf-8"));
    expect(data.length).toBe(30);
    interface TacticalCase {
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

    for (let idx = 0; idx < data.length; idx++) {
      const c = data[idx] as TacticalCase;
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
  });
});
