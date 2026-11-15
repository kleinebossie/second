/**
 * Live-Game Lock (Fail Closed) for Second Chess Assistant.
 *
 * Implements strict fair-play verification per PLAN.md Section 9.3:
 * - Allowed: /analysis, /study/..., and game pages where the Lichess API confirms the game has ended.
 * - Disallowed: active live games, puzzles, storm, racer, streak, TV, or unconfirmed games.
 * - Fail closed: if the API request fails, times out, or returns an ambiguous status -> LOCKED.
 */

import type { ApiGameData, PageType } from './types.js';

export type { PageType, ApiGameData };

export interface PageDetection {
  pageType: PageType;
  gameId?: string;
  isAllowedType: boolean;
}

export interface GameStatusResult {
  isFinished: boolean;
  status?: string;
  error?: string;
}

export interface AccessEvaluation {
  allowed: boolean;
  reason: string;
  pageType: PageType;
  gameId?: string;
  status?: string;
}

export const LOCK_OFF_MESSAGE = 'Second is off during live games. Open the game after it ends.';
export const PUZZLE_OFF_MESSAGE = 'Second is off during puzzles and tactical challenges.';
export const OTHER_PAGE_MESSAGE = 'Navigate to a finished Lichess game, analysis board, or study to use Second.';

/**
 * Standard Lichess game termination statuses indicating a finished game.
 */
export const FINISHED_GAME_STATUSES = new Set([
  'aborted',
  'mate',
  'resign',
  'outoftime',
  'timeout',
  'draw',
  'stalemate',
  'cheat',
  'nostart',
  'unknownfinish',
  'variantend',
]);

export const RESERVED_PATHS = new Set([
  'accounts',
  'activity',
  'analysis',
  'bookmark',
  'broadcast',
  'calendar',
  'download',
  'features',
  'feedback',
  'insights',
  'marathon',
  'messages',
  'password',
  'practice',
  'register',
  'security',
  'settings',
  'streamer',
  'timeline',
  'tournament',
  'training',
  'variants',
]);

/**
 * Detect the Lichess page type from a URL.
 */
export function detectPageType(rawUrl: string | URL): PageDetection {
  let url: URL;
  try {
    url = typeof rawUrl === 'string' ? new URL(rawUrl) : rawUrl;
  } catch {
    return { pageType: 'other', isAllowedType: false };
  }

  const hostname = url.hostname.toLowerCase();
  const isLichess = hostname === 'lichess.org' || hostname.endsWith('.lichess.org');
  const isLocalDev = hostname === 'localhost' || hostname === '127.0.0.1';
  if (!isLichess && !isLocalDev) {
    return { pageType: 'other', isAllowedType: false };
  }

  const pathname = url.pathname.replace(/\/+$/, ''); // trim trailing slashes

  // 1. Analysis board
  if (pathname === '/analysis' || pathname.startsWith('/analysis/')) {
    return { pageType: 'analysis', isAllowedType: true };
  }

  // 2. Studies
  if (pathname.startsWith('/study/') || pathname === '/study') {
    return { pageType: 'study', isAllowedType: true };
  }

  // 3. Puzzles and training modes
  if (
    pathname === '/training' ||
    pathname.startsWith('/training/') ||
    pathname === '/puzzle' ||
    pathname.startsWith('/puzzle/') ||
    pathname === '/storm' ||
    pathname.startsWith('/storm/') ||
    pathname === '/streak' ||
    pathname.startsWith('/streak/') ||
    pathname === '/racer' ||
    pathname.startsWith('/racer/')
  ) {
    return { pageType: 'puzzle', isAllowedType: false };
  }

  // 4. Game pages: /:gameId, /game/:gameId, /round/:gameId (8-char game ID or 12-char with player token, optional /white, /black, or ply number)
  // Lichess game IDs are 8 alphanumeric characters (or 12 for player secret URL)
  const gameMatch = pathname.match(
    /^(?:\/game(?:\/export)?|\/round)?\/([a-zA-Z0-9]{8})(?:[a-zA-Z0-9]{4})?(?:\/(?:white|black|\d+))?(?:\/\d+)?$/
  );
  if (gameMatch) {
    const gameId = gameMatch[1];
    if (!RESERVED_PATHS.has(gameId.toLowerCase())) {
      return { pageType: 'game', gameId, isAllowedType: true };
    }
  }

  // 5. Other pages (TV, home, players, tournaments, etc.)
  return { pageType: 'other', isAllowedType: false };
}

const MAX_STATUS_CACHE_SIZE = 500;
const DEFAULT_RATE_LIMIT_COOLDOWN_MS = 60_000;

const gameStatusCache = new Map<string, { result: GameStatusResult; expiresAt: number }>();
const finishedGameDataCache = new Map<string, { data: ApiGameData; expiresAt: number }>();
const inFlightStatusChecks = new Map<string, Promise<GameStatusResult>>();
let statusCooldownUntil = 0;

function setGameStatusCache(gameId: string, result: GameStatusResult, ttlMs: number): void {
  if (gameStatusCache.has(gameId)) {
    gameStatusCache.delete(gameId);
  } else if (gameStatusCache.size >= MAX_STATUS_CACHE_SIZE) {
    const oldestKey = gameStatusCache.keys().next().value;
    if (oldestKey !== undefined) gameStatusCache.delete(oldestKey);
  }
  gameStatusCache.set(gameId, { result, expiresAt: Date.now() + ttlMs });
}

function setFinishedGameDataCache(gameId: string, data: ApiGameData, ttlMs: number): void {
  if (finishedGameDataCache.has(gameId)) {
    finishedGameDataCache.delete(gameId);
  } else if (finishedGameDataCache.size >= MAX_STATUS_CACHE_SIZE) {
    const oldestKey = finishedGameDataCache.keys().next().value;
    if (oldestKey !== undefined) finishedGameDataCache.delete(oldestKey);
  }
  finishedGameDataCache.set(gameId, { data, expiresAt: Date.now() + ttlMs });
}

export function getCachedFinishedGameData(gameId: string): ApiGameData | undefined {
  const entry = finishedGameDataCache.get(gameId);
  if (!entry) return undefined;
  if (Date.now() >= entry.expiresAt) {
    finishedGameDataCache.delete(gameId);
    return undefined;
  }
  // Refresh LRU order
  finishedGameDataCache.delete(gameId);
  finishedGameDataCache.set(gameId, entry);
  return entry.data;
}

function parseRetryAfterMs(headers: Headers | undefined, fallbackMs: number): number {
  const raw = headers?.get?.('Retry-After');
  if (!raw) return fallbackMs;
  const seconds = Number(raw);
  if (Number.isFinite(seconds) && seconds > 0) {
    return Math.min(seconds * 1000, 300_000);
  }
  return fallbackMs;
}

export function clearGameStatusCache(): void {
  gameStatusCache.clear();
  finishedGameDataCache.clear();
  inFlightStatusChecks.clear();
  statusCooldownUntil = 0;
}

function parseApiGameDataFromStatusPayload(data: any): ApiGameData | undefined {
  if (!data || typeof data !== 'object') return undefined;
  const hasGameDetails =
    typeof data.moves === 'string' ||
    Boolean(data.players) ||
    Boolean(data.opening) ||
    typeof data.initialFen === 'string';
  if (!hasGameDetails) return undefined;

  const movesStr = (typeof data.moves === 'string' ? data.moves : '').trim();
  const moves = movesStr ? movesStr.split(/\s+/) : [];
  const whitePlayer = data?.players?.white?.user?.name || data?.players?.white?.userId;
  const blackPlayer = data?.players?.black?.user?.name || data?.players?.black?.userId;
  const opening = data?.opening?.name;
  const initialFen =
    typeof data?.initialFen === 'string' && data.initialFen.trim() && data.initialFen !== 'startpos'
      ? data.initialFen.trim()
      : undefined;
  const variant =
    typeof data?.variant === 'string'
      ? data.variant
      : typeof data?.variant?.key === 'string'
        ? data.variant.key
        : undefined;

  return { moves, whitePlayer, blackPlayer, opening, initialFen, variant };
}

/**
 * Query Lichess API to check if a specific game has ended.
 * Fail-closed: any network failure, timeout, non-200 response, or unclear status returns isFinished: false.
 * Deduplicates in-flight calls and caches finished games for 24h, live games for 5s, and errors briefly.
 * Also caches finished-game moves and metadata from the single response to avoid a duplicate `/game/export/` fetch.
 */
export async function checkGameStatus(
  gameId: string,
  fetchFn: typeof fetch = fetch
): Promise<GameStatusResult> {
  if (!gameId || !/^[a-zA-Z0-9]{8}$/.test(gameId)) {
    return { isFinished: false, error: 'Invalid game ID' };
  }

  const now = Date.now();
  const cached = gameStatusCache.get(gameId);
  if (cached && now < cached.expiresAt) {
    gameStatusCache.delete(gameId);
    gameStatusCache.set(gameId, cached);
    return cached.result;
  }

  if (now < statusCooldownUntil) {
    return { isFinished: false, error: 'Lichess API rate-limit cooldown active' };
  }

  if (fetchFn === fetch) {
    const pending = inFlightStatusChecks.get(gameId);
    if (pending) return pending;
  }

  const runCheck = async (): Promise<GameStatusResult> => {
    const exportUrl = `https://lichess.org/game/export/${gameId}?moves=true&pgnInJson=false&clocks=false&evals=false&opening=true`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
      const activeFetch = fetchFn || globalThis.fetch;
      const response = await (activeFetch === globalThis.fetch
        ? activeFetch.call(globalThis, exportUrl, {
            method: 'GET',
            headers: { Accept: 'application/json' },
            signal: controller.signal,
          })
        : activeFetch(exportUrl, {
            method: 'GET',
            headers: { Accept: 'application/json' },
            signal: controller.signal,
          }));

      if (response.status === 429) {
        const cooldownMs = parseRetryAfterMs(response.headers, DEFAULT_RATE_LIMIT_COOLDOWN_MS);
        statusCooldownUntil = Date.now() + cooldownMs;
        const rateLimitResult: GameStatusResult = {
          isFinished: false,
          error: 'Lichess API returned status 429',
        };
        setGameStatusCache(gameId, rateLimitResult, Math.min(cooldownMs, 10_000));
        return rateLimitResult;
      }

      if (!response.ok) {
        const httpErrResult: GameStatusResult = {
          isFinished: false,
          error: `Lichess API returned status ${response.status}`,
        };
        setGameStatusCache(gameId, httpErrResult, 5_000);
        return httpErrResult;
      }

      const data = await response.json();
      const status = (data?.status || '').toLowerCase().trim();

      if (!status) {
        const emptyResult: GameStatusResult = { isFinished: false, error: 'Empty status in API response' };
        setGameStatusCache(gameId, emptyResult, 3_000);
        return emptyResult;
      }

      const isFinished = FINISHED_GAME_STATUSES.has(status);
      const result: GameStatusResult = { isFinished, status };

      // Finished games cached for 24h; live games cached for 5s.
      const ttlMs = isFinished ? 86_400_000 : 5_000;
      setGameStatusCache(gameId, result, ttlMs);
      if (isFinished) {
        const gameData = parseApiGameDataFromStatusPayload(data);
        if (gameData) {
          setFinishedGameDataCache(gameId, gameData, ttlMs);
        }
      }
      return result;
    } catch (err: any) {
      const errorResult: GameStatusResult = {
        isFinished: false,
        error: err?.message || 'Network error fetching game status',
      };
      setGameStatusCache(gameId, errorResult, 3_000);
      return errorResult;
    } finally {
      clearTimeout(timeoutId);
    }
  };

  const promise = runCheck();
  if (fetchFn === fetch) {
    inFlightStatusChecks.set(gameId, promise);
    promise.finally(() => {
      if (inFlightStatusChecks.get(gameId) === promise) {
        inFlightStatusChecks.delete(gameId);
      }
    });
  }
  return promise;
}

/**
 * Evaluates whether Second should be enabled or locked for the given URL.
 * Strictly fail-closed.
 */
export async function evaluatePageAccess(
  rawUrl: string | URL,
  fetchFn: typeof fetch = fetch
): Promise<AccessEvaluation> {
  const detection = detectPageType(rawUrl);

  switch (detection.pageType) {
    case 'analysis':
      return {
        allowed: true,
        reason: 'Analysis board is open for study.',
        pageType: 'analysis',
      };

    case 'study':
      return {
        allowed: true,
        reason: 'Lichess Study is open for review.',
        pageType: 'study',
      };

    case 'puzzle':
      return {
        allowed: false,
        reason: PUZZLE_OFF_MESSAGE,
        pageType: 'puzzle',
      };

    case 'game': {
      if (!detection.gameId) {
        return {
          allowed: false,
          reason: LOCK_OFF_MESSAGE,
          pageType: 'game',
        };
      }

      const statusResult = await checkGameStatus(detection.gameId, fetchFn);
      if (statusResult.isFinished) {
        return {
          allowed: true,
          reason: `Game has ended (${statusResult.status}). Review enabled.`,
          pageType: 'game',
          gameId: detection.gameId,
          status: statusResult.status,
        };
      }

      return {
        allowed: false,
        reason: LOCK_OFF_MESSAGE,
        pageType: 'game',
        gameId: detection.gameId,
        status: statusResult.status || 'live_or_unknown',
      };
    }

    case 'other':
    default:
      return {
        allowed: false,
        reason: OTHER_PAGE_MESSAGE,
        pageType: 'other',
      };
  }
}
