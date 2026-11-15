/**
 * Content Script for Second Chrome Extension (injected on lichess.org).
 *
 * Implements Lichess page detection and board state extraction per PLAN.md Section 9.2:
 * 1. Detects page type from URL and DOM.
 * 2. Reads game ID, current FEN, and move list (prefers API, falls back to DOM).
 * 3. Observes move updates and communicates with extension side panel.
 */

import {
  detectPageType,
  checkGameStatus,
  getCachedFinishedGameData,
  LOCK_OFF_MESSAGE,
  PUZZLE_OFF_MESSAGE,
  OTHER_PAGE_MESSAGE,
} from './lock.js';
import type { ExtensionMessage, LichessGameState, PageType } from './types.js';
import {
  fetchLichessGameApi,
  clearApiGameCache,
  type ApiGameData,
} from './api/lichessApi.js';

export type { LichessGameState, PageType, ApiGameData };
export { fetchLichessGameApi, clearApiGameCache };

/**
 * Validates that a candidate string looks like a genuine FEN rather than a share URL.
 */
export function isLikelyFen(candidate: string): boolean {
  const trimmed = candidate.trim();
  if (!trimmed || /^https?:\/\//i.test(trimmed)) return false;
  return /^[prnbqkPRNBQK1-8]{1,8}(?:\/[prnbqkPRNBQK1-8]{1,8}){7}\s+[wb]\s+(?:-|[KQkq]{1,4})\s+(?:-|[a-h][36])(?:\s+\d+\s+\d+)?$/.test(
    trimmed
  );
}

/**
 * Extracts a custom starting FEN for `/analysis/fromPosition/...`, `/analysis/<FEN>`,
 * or PGN `[FEN "..."]` headers when the game does not begin from the standard setup.
 */
export function extractInitialFenFromUrlAndDom(url?: string): string | undefined {
  if (url) {
    try {
      const parsed = new URL(url, 'https://lichess.org');
      const fromPosMarker = '/analysis/fromPosition/';
      const idx = parsed.pathname.indexOf(fromPosMarker);
      if (idx !== -1) {
        const raw = decodeURIComponent(parsed.pathname.slice(idx + fromPosMarker.length))
          .replace(/_/g, ' ')
          .trim();
        if (isLikelyFen(raw)) {
          return raw;
        }
      } else if (parsed.pathname.startsWith('/analysis/')) {
        const rest = decodeURIComponent(parsed.pathname.slice('/analysis/'.length))
          .replace(/_/g, ' ')
          .trim();
        if (isLikelyFen(rest)) {
          return rest;
        }
      }
    } catch {
      // ignore invalid URL
    }
  }

  if (typeof document !== 'undefined') {
    const pgnNodes = document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLElement>(
      'textarea.pgn, .pgn textarea, .analyse__underboard textarea, .study__share textarea, .pgn .copyable'
    );
    for (const node of pgnNodes) {
      const raw = 'value' in node && typeof node.value === 'string' ? node.value : node.textContent || '';
      const match = raw.match(/\[FEN\s+"([^"]+)"\]/);
      if (match && isLikelyFen(match[1])) {
        return match[1].trim();
      }
    }
  }

  return undefined;
}

/**
 * Extracts current board FEN from Lichess DOM.
 */
export function extractFenFromDom(): string | undefined {
  if (typeof document === 'undefined') {
    return 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  }

  // 1. Try explicit FEN inputs before generic copyable inputs (which also hold share URLs)
  const fenInputs = document.querySelectorAll<HTMLInputElement>(
    '.analyse__fen input, .study__fen input, input.fen, input.copyable'
  );
  for (const input of fenInputs) {
    if (input.value && isLikelyFen(input.value)) {
      return input.value.trim();
    }
  }

  // 2. Try looking in meta tags
  const metaFen = document.querySelector('meta[name="chess-fen"]');
  if (metaFen) {
    const content = metaFen.getAttribute('content');
    if (content && isLikelyFen(content)) return content.trim();
  }

  // 3. Fallback to start position FEN
  return 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
}

interface DomMoveNode {
  el: HTMLElement;
  san: string;
  path?: string;
  inVariation: boolean;
  isMainline: boolean;
  isActive: boolean;
  isCurrent: boolean;
}

const VARIATION_CONTAINER_TAGS = new Set(['INTERRUPT', 'LINES', 'LINE', 'INLINE', 'VARIATION']);

/**
 * Locates the primary Lichess move-list element.
 *
 * Lichess renders the analysis tree as `tview2-inline` or `tview2-column`
 * inside `.analyse__moves`, and renders standard game/round move lists inside
 * `.rm6`, `.game__moves`, or `l4x`.
 */
function findMoveListScope(): ParentNode | null {
  if (typeof document === 'undefined') return null;
  const analysisMoves = document.querySelector<HTMLElement>('.analyse__moves');
  if (analysisMoves) {
    return analysisMoves.querySelector<HTMLElement>('.tview2') || analysisMoves;
  }
  return (
    document.querySelector<HTMLElement>('.tview2') ||
    document.querySelector<HTMLElement>('.rm6') ||
    document.querySelector<HTMLElement>('.game__moves') ||
    document.querySelector<HTMLElement>('.round__app l4x') ||
    document.querySelector<HTMLElement>('l4x')
  );
}

function isInVariation(node: Element, scope: ParentNode): boolean {
  let el: Element | null = node.parentElement;
  while (el) {
    if (VARIATION_CONTAINER_TAGS.has(el.tagName)) return true;
    if (el.parentNode === scope) break;
    el = el.parentElement;
  }
  return false;
}

const sanExtractionCache = new WeakMap<Element, { text: string; san: string }>();

/**
 * Extracts clean SAN from a Lichess move node. Lichess nests `<index>`,
 * `<glyph>` and `<eval>` inside each `<move>`, so reading textContent directly
 * would contaminate the move list.
 */
function extractSanFromNode(node: Element): string {
  const sanEl = node.querySelector('san');
  if (sanEl) {
    const sanText = sanEl.textContent || '';
    const cached = sanExtractionCache.get(node);
    if (cached && cached.text === sanText) {
      return cached.san;
    }
    const clean = sanText
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/(?:[\+\-−]\d+(?:\.\d+)?|#[+\-−]?\d+)\s*$/g, '')
      .replace(/[?!]+$/g, '')
      .trim();
    sanExtractionCache.set(node, { text: sanText, san: clean });
    return clean;
  }

  const rawText = node.textContent || '';
  const cached = sanExtractionCache.get(node);
  if (cached && cached.text === rawText) {
    return cached.san;
  }

  let raw = '';
  if (typeof node.cloneNode === 'function') {
    try {
      const clone = node.cloneNode(true) as Element;
      clone
        .querySelectorAll('eval, .eval, glyph, .glyph, index, .index, san')
        .forEach((el) => el.remove());
      raw = clone.textContent || '';
    } catch {
      raw = rawText;
    }
  } else {
    raw = rawText;
  }

  const clean = raw
    .replace(/\s+/g, ' ')
    .trim()
    // Strip trailing engine evaluation text (+0.2, -1.5, #3)
    .replace(/(?:[\+\-−]\d+(?:\.\d+)?|#[+\-−]?\d+)\s*$/g, '')
    // Strip move quality glyphs (?!, ?, !, etc.)
    .replace(/[?!]+$/g, '')
    .trim();

  sanExtractionCache.set(node, { text: rawText, san: clean });
  return clean;
}

interface ParsedMoveCache {
  scope: ParentNode;
  moveCount: number;
  firstEl: HTMLElement | null;
  lastEl: HTMLElement | null;
  nodes: DomMoveNode[];
  mainlineNodes: DomMoveNode[];
  moves: string[];
}

let lastParsedMoveCache: ParsedMoveCache | null = null;

function collectMoveElements(scope: ParentNode): HTMLElement[] {
  const moveEls = scope.querySelectorAll<HTMLElement>('move');
  if (moveEls.length > 0) {
    return Array.from(moveEls);
  }
  // Older markup / round-view variants still used on some Lichess surfaces.
  return Array.from(
    scope.querySelectorAll<HTMLElement>(
      ':scope > kwdb, :scope > m2, :scope > u5i, l4x > kwdb, rm6 > m2, u8t > u5i'
    )
  );
}

function getMoveNodesWithFastPath(scope: ParentNode): {
  nodes: DomMoveNode[];
  mainlineNodes: DomMoveNode[];
  moves: string[];
} {
  const elements = collectMoveElements(scope);
  const firstEl = elements[0] ?? null;
  const lastEl = elements[elements.length - 1] ?? null;

  if (
    lastParsedMoveCache &&
    lastParsedMoveCache.scope === scope &&
    lastParsedMoveCache.moveCount === elements.length &&
    lastParsedMoveCache.firstEl === firstEl &&
    lastParsedMoveCache.lastEl === lastEl &&
    (!firstEl || firstEl.isConnected)
  ) {
    // Fast path for `.active` / `.current` class-only navigation updates
    for (const node of lastParsedMoveCache.nodes) {
      node.isActive = node.el.classList.contains('active') || node.el.classList.contains('a1t');
      node.isCurrent = node.el.classList.contains('current');
    }
    return {
      nodes: lastParsedMoveCache.nodes,
      mainlineNodes: lastParsedMoveCache.mainlineNodes,
      moves: lastParsedMoveCache.moves,
    };
  }

  let hasMainlineClass = false;
  for (const el of elements) {
    if (el.tagName === 'MOVE' && el.classList.contains('mainline')) {
      hasMainlineClass = true;
      break;
    }
  }

  const nodes: DomMoveNode[] = [];
  const mainlineNodes: DomMoveNode[] = [];
  const moves: string[] = [];

  for (const el of elements) {
    if (el.classList.contains('empty')) continue;

    const san = extractSanFromNode(el);
    if (!san || /^\d+\.?$/.test(san)) continue;

    const inVariation = isInVariation(el, scope);
    // In inline view, `.mainline` is the ground truth. In column/legacy view,
    // anything outside a variation container belongs to the mainline.
    const isMainline = hasMainlineClass ? el.classList.contains('mainline') : !inVariation;

    const node: DomMoveNode = {
      el,
      san,
      path: el.getAttribute('p') || el.dataset.path || undefined,
      inVariation,
      isMainline,
      isActive: el.classList.contains('active') || el.classList.contains('a1t'),
      isCurrent: el.classList.contains('current'),
    };
    nodes.push(node);
    if (isMainline) {
      mainlineNodes.push(node);
      moves.push(san);
    }
  }

  lastParsedMoveCache = {
    scope,
    moveCount: elements.length,
    firstEl,
    lastEl,
    nodes,
    mainlineNodes,
    moves,
  };

  return { nodes, mainlineNodes, moves };
}

/**
 * Rebuilds the full move sequence leading to `activePath` from Lichess tree
 * paths. Each path segment is two characters, one per ply. Collecting every
 * move whose path is a prefix of the active path gives the exact line the
 * board is showing, including user-explored variations.
 */
function buildLineFromPath(nodes: DomMoveNode[], activePath: string): string[] | undefined {
  if (!activePath || activePath.length % 2 !== 0) return undefined;

  const byPath = new Map<string, DomMoveNode>();
  for (const node of nodes) {
    if (!node.path || node.path.length > activePath.length) continue;
    if (!activePath.startsWith(node.path)) continue;
    const existing = byPath.get(node.path);
    // Prefer the mainline occurrence if the same path is rendered twice.
    if (!existing || (node.isMainline && !existing.isMainline)) {
      byPath.set(node.path, node);
    }
  }

  const line: string[] = [];
  for (let length = 2; length <= activePath.length; length += 2) {
    const node = byPath.get(activePath.slice(0, length));
    if (!node) return undefined;
    line.push(node.san);
  }
  return line.length > 0 ? line : undefined;
}

/**
 * Extracts the game's mainline move list and the line currently on the board.
 *
 * The previous implementation selected every `move` node, which also pulled in
 * the user's variation lines. That shifted all later plies and made the
 * extension analyze a completely different move. This version keeps variations
 * separate from the game mainline and uses Lichess's `p` tree paths to resolve
 * the selected line precisely.
 */
export function extractMovesFromDom(): {
  moves: string[];
  selectedMoveSan?: string;
  activePly?: number;
  activeLine?: string[];
  activePath?: string;
  activeIsVariation?: boolean;
} {
  if (typeof document === 'undefined') {
    return { moves: [] };
  }

  const scope = findMoveListScope();
  if (!scope) {
    return { moves: [] };
  }

  const { nodes, mainlineNodes, moves } = getMoveNodesWithFastPath(scope);

  // Prefer the current path (`.active`); fall back to the game's current
  // position (`.current`) only when nothing is actively selected.
  const activeNode =
    nodes.find((node) => node.isActive) || nodes.find((node) => node.isCurrent) || undefined;

  let selectedMoveSan: string | undefined;
  let activePly: number | undefined;
  let activeLine: string[] | undefined;
  let activePath: string | undefined;
  let activeIsVariation: boolean | undefined;

  if (activeNode) {
    selectedMoveSan = activeNode.san;
    activePath = activeNode.path;
    activeIsVariation = !activeNode.isMainline;

    if (activePath) {
      activeLine = buildLineFromPath(nodes, activePath);
      if (activeLine && activeLine.length > 0) {
        activePly = activeLine.length;
      } else {
        // Path exists but the line could not be reconstructed (e.g. collapsed
        // ancestors). Fall back to its depth for mainline positions only.
        const depth = activePath.length / 2;
        if (!activeIsVariation) activePly = depth;
      }
    }

    if (!activePly && activeNode.isMainline) {
      const index = mainlineNodes.indexOf(activeNode);
      if (index >= 0) {
        activePly = index + 1;
        activeLine = mainlineNodes.slice(0, activePly).map((node) => node.san);
      }
    }
  }

  // Fallback: check URL hash for an explicit ply. Lichess writes the current
  // ply into the hash and removes/updates it while navigating.
  if (!activeNode && typeof window !== 'undefined' && window.location?.hash) {
    const hashMatch = window.location.hash.match(/^#(\d+)$/);
    if (hashMatch) {
      const ply = parseInt(hashMatch[1], 10);
      if (ply === 0) {
        activePly = 0;
        activeLine = [];
        selectedMoveSan = undefined;
      } else if (ply > 0 && moves.length >= ply) {
        activePly = ply;
        activeLine = moves.slice(0, ply);
        selectedMoveSan = moves[ply - 1];
      }
    }
  }

  // When a DOM move list exists with no active/current move and no positive hash ply,
  // the board is at the starting position (ply 0). Never jump to the final move.
  if (!activeNode && activePly === undefined && moves.length > 0) {
    activePly = 0;
    activeLine = [];
    selectedMoveSan = undefined;
  }

  return {
    moves,
    selectedMoveSan,
    activePly,
    activeLine,
    activePath,
    activeIsVariation,
  };
}

/**
 * Query full game state. Prefers Lichess API for games, falls back to DOM.
 */
export async function getLichessGameState(fetchFn: typeof fetch = fetch): Promise<LichessGameState> {
  const url = typeof window !== 'undefined' ? window.location.href : 'https://lichess.org';
  const detection = detectPageType(url);

  let isAllowed = false;
  let lockReason: string | undefined;
  let whitePlayer: string | undefined;
  let blackPlayer: string | undefined;
  let opening: string | undefined;
  let apiMoves: string[] | undefined;
  let apiInitialFen: string | undefined;
  let variant: string | undefined;

  if (detection.pageType === 'puzzle') {
    return {
      url,
      pageType: 'puzzle',
      isAllowed: false,
      lockReason: PUZZLE_OFF_MESSAGE,
      timestamp: Date.now(),
    };
  }

  if (detection.pageType === 'other') {
    return {
      url,
      pageType: 'other',
      isAllowed: false,
      lockReason: OTHER_PAGE_MESSAGE,
      timestamp: Date.now(),
    };
  }

  if (detection.pageType === 'analysis' || detection.pageType === 'study') {
    isAllowed = true;
  } else if (detection.pageType === 'game' && detection.gameId) {
    const status = await checkGameStatus(detection.gameId, fetchFn);
    if (status.isFinished) {
      isAllowed = true;
      const apiData =
        getCachedFinishedGameData(detection.gameId) ??
        (await fetchLichessGameApi(detection.gameId, fetchFn));
      if (apiData) {
        if (apiData.moves.length > 0) apiMoves = apiData.moves;
        if (apiData.whitePlayer) whitePlayer = apiData.whitePlayer;
        if (apiData.blackPlayer) blackPlayer = apiData.blackPlayer;
        if (apiData.opening) opening = apiData.opening;
        if (apiData.initialFen) apiInitialFen = apiData.initialFen;
        if (apiData.variant) variant = apiData.variant;
      }
    } else {
      return {
        url,
        pageType: 'game',
        gameId: detection.gameId,
        isAllowed: false,
        lockReason: LOCK_OFF_MESSAGE,
        timestamp: Date.now(),
      };
    }
  }

  // Fallback to DOM player extraction if API did not provide them.
  if (typeof document !== 'undefined') {
    const isFlipped = Boolean(document.querySelector('.cg-wrap.orientation-black'));
    if (!whitePlayer) {
      const explicitWhite = document.querySelector(
        '.game__meta .user-link.white, .game__meta .player.white .user-link, .player.white .username'
      );
      const positionalWhite = isFlipped
        ? document.querySelector('.ruser-top .user-link')
        : document.querySelector('.ruser-bottom .user-link');
      const pWhite = explicitWhite || positionalWhite;
      if (pWhite) whitePlayer = pWhite.textContent?.trim();
    }
    if (!blackPlayer) {
      const explicitBlack = document.querySelector(
        '.game__meta .user-link.black, .game__meta .player.black .user-link, .player.black .username'
      );
      const positionalBlack = isFlipped
        ? document.querySelector('.ruser-bottom .user-link')
        : document.querySelector('.ruser-top .user-link');
      const pBlack = explicitBlack || positionalBlack;
      if (pBlack) blackPlayer = pBlack.textContent?.trim();
    }
    if (!opening) {
      const domOpening = document.querySelector(
        '.game__meta .opening, .analyse__wiki .title, .copyables .opening'
      );
      if (domOpening) opening = domOpening.textContent?.trim();
    }
  }

  const fen = extractFenFromDom();
  const initialFen = extractInitialFenFromUrlAndDom(url) || apiInitialFen;
  const domMovesResult = extractMovesFromDom();
  const moves = apiMoves && apiMoves.length > 0 ? apiMoves : domMovesResult.moves;

  let activePly = domMovesResult.activePly;
  let selectedMoveSan = domMovesResult.selectedMoveSan;
  let activeLine = domMovesResult.activeLine;
  const activePath = domMovesResult.activePath;
  const activeIsVariation = domMovesResult.activeIsVariation;

  // When the selected move is on the game's mainline, rebuild the active line
  // from the authoritative API move list. This avoids DOM truncation issues and
  // guarantees that the ply and SAN always agree.
  if (
    apiMoves &&
    apiMoves.length > 0 &&
    selectedMoveSan &&
    !activeIsVariation &&
    activePly &&
    activePly > 0 &&
    activePly <= apiMoves.length
  ) {
    if (apiMoves[activePly - 1] === selectedMoveSan) {
      activeLine = apiMoves.slice(0, activePly);
    } else {
      const candidates = apiMoves
        .map((san, index) => (san === selectedMoveSan ? index + 1 : -1))
        .filter((ply) => ply > 0);
      if (candidates.length === 1) {
        activePly = candidates[0];
        activeLine = apiMoves.slice(0, activePly);
      }
    }
  }

  // If the selected move is a variation but its line could not be rebuilt,
  // never apply its ply to the game mainline (that would silently analyze a
  // different game move).
  if (activeIsVariation && (!activeLine || activeLine.length === 0)) {
    activePly = undefined;
  }

  // If DOM had no move list at all, resolve hash ply or final move from API moves.
  if (activePly === undefined && !selectedMoveSan && moves.length > 0 && !activeIsVariation) {
    let hashPly: number | undefined;
    if (typeof window !== 'undefined' && window.location?.hash) {
      const hashMatch = window.location.hash.match(/^#(\d+)$/);
      if (hashMatch) {
        hashPly = parseInt(hashMatch[1], 10);
      }
    }
    if (hashPly === 0) {
      activePly = 0;
      activeLine = [];
      selectedMoveSan = undefined;
    } else if (hashPly && hashPly > 0 && hashPly <= moves.length) {
      activePly = hashPly;
      activeLine = moves.slice(0, hashPly);
      selectedMoveSan = moves[hashPly - 1];
    } else {
      selectedMoveSan = moves[moves.length - 1];
      activePly = moves.length;
      activeLine = [...moves];
    }
  } else if (!selectedMoveSan && activePly && activePly > 0 && moves.length >= activePly) {
    selectedMoveSan = moves[activePly - 1];
  }

  return {
    url,
    pageType: detection.pageType,
    gameId: detection.gameId,
    isAllowed,
    lockReason,
    fen,
    initialFen,
    variant,
    moves,
    selectedMoveSan,
    activePly,
    activeLine,
    activePath,
    activeIsVariation,
    whitePlayer,
    blackPlayer,
    opening,
    timestamp: Date.now(),
  };
}

let broadcastSequenceId = 0;
let lastBroadcastSignature: string | null = null;

function computeBroadcastSignature(state: LichessGameState): string {
  return [
    state.url,
    state.pageType,
    state.gameId ?? '',
    state.isAllowed ? '1' : '0',
    state.lockReason ?? '',
    state.fen ?? '',
    state.activePath ?? '',
    state.activePly ?? '',
    state.selectedMoveSan ?? '',
    state.moves?.length ?? 0,
    state.activeLine?.join(',') ?? '',
  ].join('|');
}

/**
 * Broadcast current state to extension runtime, dropping stale async results and duplicate payloads.
 */
export async function broadcastState(): Promise<void> {
  const seq = ++broadcastSequenceId;
  try {
    const state = await getLichessGameState();
    if (seq !== broadcastSequenceId) return;
    const signature = computeBroadcastSignature(state);
    if (signature === lastBroadcastSignature) return;
    lastBroadcastSignature = signature;

    if (typeof chrome !== 'undefined' && chrome.runtime?.sendMessage) {
      const message: ExtensionMessage = {
        type: 'LICHESS_STATE_UPDATE',
        payload: state,
      };
      Promise.resolve(chrome.runtime.sendMessage(message)).catch(() => {
        // Suppress error if sidepanel is not currently open
      });
    }
  } catch (err) {
    console.debug('Failed to broadcast Lichess state:', err);
  }
}

function isIgnoredMutationTarget(target: Node | null): boolean {
  const el = target instanceof Element ? target : target?.parentElement;
  if (!el) return false;
  return Boolean(
    el.closest('.rclock, .clock, .mchat, .ceval-box, .ceval, .eval-gauge, .chat__members, .cg-board, cg-board, coords')
  );
}

function isRelevantMutation(mutation: MutationRecord, moveScope: ParentNode | null): boolean {
  if (isIgnoredMutationTarget(mutation.target)) return false;
  if (!moveScope) return true;

  const target = mutation.target;
  if (moveScope.contains(target)) return true;

  const el = target instanceof Element ? target : target.parentElement;
  if (!el) return true;
  if (el.closest('.analyse__fen, .study__fen, input.fen, input.copyable, .pgn, .cg-wrap, .game__meta')) {
    return true;
  }

  // Detect SPA swap of move list container on childList mutations
  if (mutation.type === 'childList') {
    for (const node of mutation.addedNodes) {
      if (node instanceof Element && node.querySelector?.('.tview2, .analyse__moves, .rm6, .game__moves, l4x, .analyse__fen, .study__fen')) {
        return true;
      }
    }
    if (!document.contains(moveScope as Node)) return true;
  }

  return false;
}

// Initialize listeners (idempotent across repeated executeScript calls).
export function initContentScript(): void {
  lastBroadcastSignature = null;
  if (
    typeof window !== 'undefined' &&
    (window as any).__secondContentScriptInitialized
  ) {
    void broadcastState();
    return;
  }
  if (typeof window !== 'undefined') {
    (window as any).__secondContentScriptInitialized = true;
  }

  // Listen for queries from side panel.
  if (typeof chrome !== 'undefined' && chrome.runtime?.onMessage) {
    chrome.runtime.onMessage.addListener((message: ExtensionMessage | undefined, _sender, sendResponse) => {
      if (message?.type === 'GET_PAGE_STATE') {
        getLichessGameState()
          .then((state) => sendResponse(state))
          .catch(() => sendResponse(undefined));
        return true; // async response
      }
      return false;
    });
  }

  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  const scheduleBroadcast = (delayMs = 150) => {
    if (debounceTimer !== null) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      debounceTimer = null;
      void broadcastState();
    }, delayMs);
  };

  // Observe board & move list changes, ignoring noisy clock/chat/eval/board-drag ticks.
  const targetNode = document.body;
  if (targetNode) {
    let lastCheckedHref = window.location.href;
    const observer = new MutationObserver((mutations) => {
      const currentHref = window.location.href;
      const urlChanged = currentHref !== lastCheckedHref;
      if (urlChanged) {
        lastCheckedHref = currentHref;
      } else {
        const detection = detectPageType(currentHref);
        if (!detection.isAllowedType) {
          return;
        }
      }

      const moveScope = findMoveListScope();
      if (!urlChanged && !mutations.some((m) => isRelevantMutation(m, moveScope))) {
        return;
      }
      scheduleBroadcast(150);
    });

    observer.observe(targetNode, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['class', 'value'],
    });
  }

  // Handle URL navigation inside Lichess SPA (popstate, hashchange, pushState, replaceState).
  window.addEventListener('popstate', () => {
    scheduleBroadcast(200);
  });
  window.addEventListener('hashchange', () => {
    scheduleBroadcast(100);
  });

  if (typeof history !== 'undefined' && !(window as any).__secondHistoryPatched) {
    (window as any).__secondHistoryPatched = true;
    const origPushState = history.pushState.bind(history);
    const origReplaceState = history.replaceState.bind(history);
    history.pushState = (...args: Parameters<History['pushState']>) => {
      origPushState(...args);
      scheduleBroadcast(150);
    };
    history.replaceState = (...args: Parameters<History['replaceState']>) => {
      origReplaceState(...args);
      scheduleBroadcast(150);
    };
  }

  // Initial broadcast
  void broadcastState();
}

// Only run automatically in content script context on Lichess pages, never in sidepanel or Vitest.
if (
  typeof window !== 'undefined' &&
  typeof document !== 'undefined' &&
  !document.getElementById('app-shell') &&
  !(typeof process !== 'undefined' && process?.env?.VITEST) &&
  (window.location?.hostname === 'lichess.org' ||
    window.location?.hostname?.endsWith('.lichess.org'))
) {
  initContentScript();
}
