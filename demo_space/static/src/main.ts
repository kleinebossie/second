/**
 * Static in-browser demo for Second-1.0-2B (Hugging Face Static Space).
 */

import { Chess } from '../../../extension/node_modules/chessops/dist/esm/chess.js';
import { makeFen, parseFen } from '../../../extension/node_modules/chessops/dist/esm/fen.js';
import { parsePgn, startingPosition } from '../../../extension/node_modules/chessops/dist/esm/pgn.js';
import { makeSan, parseSan } from '../../../extension/node_modules/chessops/dist/esm/san.js';
import { makeUci } from '../../../extension/node_modules/chessops/dist/esm/util.js';
import {
  SYSTEM_MESSAGE,
  buildUserMessage,
  classifyVerdict,
  formatQwenChatPrompt,
} from '../../../core/ts/src/index.js';
import { StockfishEngine } from '../../../extension/src/engine/stockfish.js';
import { CloudEvalClient } from '../../../extension/src/engine/cloudEval.js';
import { ModelRunner } from '../../../extension/src/runtime/modelRunner.js';

interface SelectableMove {
  label: string;
  fenBefore: string;
  fenAfter: string;
  moveUci: string;
  moveSan: string;
  fromSq: number;
  toSq: number;
}

const POSITION_ONLY = '__POSITION__';

const PRESETS: Record<string, { text: string; targetMoveIndex?: number }> = {
  'Opening Blunder (4... Nxe4?)': {
    text: '1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 Nxe4',
    targetMoveIndex: 7,
  },
  'Tactical Fork FEN (After 4. Ng5)': {
    text: 'r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4',
  },
  'Quiet Opening Position (Italian Game)': {
    text: 'r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4',
  },
};

const UNICODE_PIECES: Record<string, string> = {
  wk: '♔', wq: '♕', wr: '♖', wb: '♗', wn: '♘', wp: '♙',
  bk: '♚', bq: '♛', br: '♜', bb: '♝', bn: '♞', bp: '♟',
};

const presetEl = document.getElementById('preset') as HTMLSelectElement;
const inputEl = document.getElementById('chess-input') as HTMLTextAreaElement;
const parseBtn = document.getElementById('btn-parse') as HTMLButtonElement;
const downloadBtn = document.getElementById('btn-download') as HTMLButtonElement;
const explainBtn = document.getElementById('btn-explain') as HTMLButtonElement;
const moveSelect = document.getElementById('move-select') as HTMLSelectElement;
const levelSelect = document.getElementById('level-select') as HTMLSelectElement;
const detailSelect = document.getElementById('detail-select') as HTMLSelectElement;
const statusEl = document.getElementById('status') as HTMLDivElement;
const progressBar = document.getElementById('download-progress') as HTMLDivElement;
const progressFill = document.getElementById('download-fill') as HTMLDivElement;
const boardEl = document.getElementById('board') as HTMLDivElement;
const verdictEl = document.getElementById('verdict') as HTMLDivElement;
const outputEl = document.getElementById('output') as HTMLDivElement;
const promptDebugEl = document.getElementById('prompt-debug') as HTMLPreElement;

const stockfish = new StockfishEngine({
  workerScriptUrl: './engine/stockfish.wasm.js',
  defaultDepth: 15,
  defaultTimeMs: 1800,
});
const cloudEval = new CloudEvalClient({ enabled: true });
const runner = new ModelRunner();

let currentMoves: SelectableMove[] = [];
let currentBaseFen = makeFen(Chess.default().toSetup());
let isDownloading = false;
let isBusy = false;

function renderBoard(fen: string, highlights: number[] = []): void {
  const setup = parseFen(fen).unwrap();
  const pos = Chess.fromSetup(setup).unwrap();
  boardEl.innerHTML = '';

  for (let rank = 7; rank >= 0; rank--) {
    for (let file = 0; file < 8; file++) {
      const sq = rank * 8 + file;
      const div = document.createElement('div');
      const isLight = (rank + file) % 2 === 1;
      div.className = `sq ${isLight ? 'light' : 'dark'}${highlights.includes(sq) ? ' highlight' : ''}`;
      const piece = pos.board.get(sq);
      if (piece) {
        const key = (piece.color === 'white' ? 'w' : 'b') +
          (piece.role === 'knight' ? 'n' : piece.role[0]);
        const span = document.createElement('span');
        span.className = piece.color === 'white' ? 'piece-w' : 'piece-b';
        span.textContent = UNICODE_PIECES[key] || '';
        div.appendChild(span);
      }
      boardEl.appendChild(div);
    }
  }
}

function parseInput(raw: string): { baseFen: string; moves: SelectableMove[] } {
  const text = raw.trim();
  if (!text) throw new Error('Paste a PGN or FEN first.');

  // Try direct FEN parsing first.
  const fenRes = parseFen(text);
  if (fenRes.isOk) {
    const posRes = Chess.fromSetup(fenRes.value);
    if (posRes.isOk) {
      return { baseFen: makeFen(posRes.value.toSetup()), moves: [] };
    }
  }

  // Otherwise parse PGN mainline.
  const games = parsePgn(text);
  if (!games.length) throw new Error('Could not parse PGN or FEN.');
  const game = games[0];
  const startRes = startingPosition(game.headers);
  if (startRes.isErr) throw new Error('Invalid starting position in PGN.');

  const pos = startRes.value;
  const moves: SelectableMove[] = [];
  let node = game.moves;
  while (node.children.length > 0) {
    const child = node.children[0];
    const san = child.data.san;
    const move = parseSan(pos, san);
    if (!move) break;

    const fenBefore = makeFen(pos.toSetup());
    const moveSan = makeSan(pos, move);
    const moveUci = makeUci(move);
    const prefix = pos.turn === 'white' ? `${pos.fullmoves}.` : `${pos.fullmoves}...`;

    pos.play(move);
    const fenAfter = makeFen(pos.toSetup());
    const fromSq = 'from' in move ? move.from : -1;
    const toSq = move.to;

    moves.push({
      label: `${prefix} ${moveSan}`,
      fenBefore,
      fenAfter,
      moveUci,
      moveSan,
      fromSq,
      toSq,
    });
    node = child;
  }

  if (moves.length === 0) {
    return { baseFen: makeFen(pos.toSetup()), moves: [] };
  }
  return { baseFen: moves[moves.length - 1].fenAfter, moves };
}

function updateMoveDropdown(selectIdx?: number): void {
  moveSelect.innerHTML = '';
  const posOpt = document.createElement('option');
  posOpt.value = POSITION_ONLY;
  posOpt.textContent = 'Position Overview (current FEN)';
  moveSelect.appendChild(posOpt);

  currentMoves.forEach((m, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = m.label;
    moveSelect.appendChild(opt);
  });

  if (typeof selectIdx === 'number' && currentMoves[selectIdx]) {
    moveSelect.value = String(selectIdx);
  } else if (currentMoves.length > 0) {
    moveSelect.value = String(currentMoves.length - 1);
  } else {
    moveSelect.value = POSITION_ONLY;
  }
  syncBoardToSelection();
}

function syncBoardToSelection(): void {
  const val = moveSelect.value;
  if (val === POSITION_ONLY) {
    renderBoard(currentBaseFen);
  } else {
    const entry = currentMoves[Number(val)];
    if (entry) {
      const hl = [entry.fromSq, entry.toSq].filter((s) => s >= 0);
      renderBoard(entry.fenAfter, hl);
    }
  }
}

async function evaluateFen(fen: string) {
  try {
    const cloud = await cloudEval.getEvaluation(fen);
    if (cloud && cloud.pvs && cloud.pvs.length > 0) return cloud;
  } catch {
    // Fall back to local Stockfish WASM.
  }
  await stockfish.init();
  return stockfish.evaluate(fen);
}

async function ensureModelReady(): Promise<boolean> {
  if (runner.isModelLoaded()) return true;
  const cached = await runner.isModelCached('2B');
  if (!cached) return false;
  statusEl.textContent = 'Loading cached Second-1.0-2B model...';
  await runner.loadModel('2B', (p) => {
    statusEl.textContent = p.message;
  });
  downloadBtn.textContent = 'Second-1.0-2B Ready ✓';
  downloadBtn.disabled = true;
  return true;
}

downloadBtn.addEventListener('click', async () => {
  if (isDownloading) return;
  isDownloading = true;
  downloadBtn.disabled = true;
  progressBar.style.display = 'block';
  try {
    await runner.loadModel('2B', (p) => {
      progressFill.style.width = `${Math.max(4, p.percent)}%`;
      statusEl.textContent = p.message;
    });
    downloadBtn.textContent = 'Second-1.0-2B Ready ✓';
    statusEl.textContent = 'Model loaded cleanly in browser.';
  } catch (err) {
    downloadBtn.disabled = false;
    statusEl.textContent = `Download error: ${err instanceof Error ? err.message : String(err)}`;
  } finally {
    isDownloading = false;
  }
});

parseBtn.addEventListener('click', () => {
  try {
    const parsed = parseInput(inputEl.value);
    currentBaseFen = parsed.baseFen;
    currentMoves = parsed.moves;
    updateMoveDropdown();
    statusEl.textContent = currentMoves.length
      ? `Loaded ${currentMoves.length} moves.`
      : 'Loaded FEN position.';
  } catch (err) {
    statusEl.textContent = err instanceof Error ? err.message : String(err);
  }
});

moveSelect.addEventListener('change', syncBoardToSelection);

explainBtn.addEventListener('click', async () => {
  if (isBusy) return;
  isBusy = true;
  explainBtn.disabled = true;
  verdictEl.textContent = '';

  try {
    const ready = await ensureModelReady();
    if (!ready) {
      statusEl.textContent = 'Click "Download Second-1.0-2B (~1.2 GB)" first.';
      return;
    }

    const level = levelSelect.value === 'beginner' ? 'beginner' : 'club';
    const detail = detailSelect.value === 'long' ? 'long' : 'short';
    const sel = moveSelect.value;

    statusEl.textContent = 'Running Stockfish evaluation...';
    let userMessage = '';

    if (sel === POSITION_ONLY) {
      const evalBefore = await evaluateFen(currentBaseFen);
      verdictEl.textContent = 'Position Overview';
      userMessage = buildUserMessage('explain_position', level, detail, currentBaseFen, evalBefore);
    } else {
      const entry = currentMoves[Number(sel)];
      const evalBefore = await evaluateFen(entry.fenBefore);
      const evalAfter = await evaluateFen(entry.fenAfter);
      const verdict = classifyVerdict(evalBefore, evalAfter, entry.moveUci);
      verdictEl.textContent = `${entry.label} · ${verdict}`;
      userMessage = buildUserMessage(
        'explain_move',
        level,
        detail,
        entry.fenBefore,
        evalBefore,
        evalAfter,
        entry.moveUci
      );
    }

    const fullPrompt = formatQwenChatPrompt(userMessage, SYSTEM_MESSAGE);
    promptDebugEl.textContent = userMessage;
    outputEl.textContent = '';
    statusEl.textContent = 'Generating explanation...';

    await runner.generateStream(
      fullPrompt,
      { detail, maxTokens: detail === 'long' ? 320 : 180 },
      (token) => {
        outputEl.textContent += token;
      }
    );
    statusEl.textContent = 'Done.';
  } catch (err) {
    statusEl.textContent = `Error: ${err instanceof Error ? err.message : String(err)}`;
  } finally {
    isBusy = false;
    explainBtn.disabled = false;
  }
});

Object.keys(PRESETS).forEach((name) => {
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = name;
  presetEl.appendChild(opt);
});

presetEl.addEventListener('change', () => {
  const item = PRESETS[presetEl.value];
  if (!item) return;
  inputEl.value = item.text;
  const parsed = parseInput(item.text);
  currentBaseFen = parsed.baseFen;
  currentMoves = parsed.moves;
  updateMoveDropdown(item.targetMoveIndex);
});

// Initialize default view
const initKey = Object.keys(PRESETS)[0];
presetEl.value = initKey;
inputEl.value = PRESETS[initKey].text;
const initParsed = parseInput(inputEl.value);
currentBaseFen = initParsed.baseFen;
currentMoves = initParsed.moves;
updateMoveDropdown(PRESETS[initKey].targetMoveIndex);
void runner.isModelCached('2B').then((cached) => {
  if (cached) {
    downloadBtn.textContent = 'Cached in Browser ✓';
  }
});
