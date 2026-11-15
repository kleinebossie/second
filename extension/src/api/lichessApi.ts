/**
 * Standalone Lichess API Client with zero side-effects.
 * Safe to import in both content scripts and extension sidepanels.
 */

import type { ApiGameData } from '../types.js';

export type { ApiGameData };

const MAX_API_CACHE_SIZE = 200;
const API_TIMEOUT_MS = 5000;
const DEFAULT_RATE_LIMIT_COOLDOWN_MS = 60_000;

const apiGameCache = new Map<string, ApiGameData>();
const inFlightApiRequests = new Map<
  string,
  { promise: Promise<ApiGameData | null>; fetchFn: typeof fetch }
>();
let apiCooldownUntil = 0;

export function clearApiGameCache(): void {
  apiGameCache.clear();
  inFlightApiRequests.clear();
  apiCooldownUntil = 0;
}

function setApiGameCache(gameId: string, data: ApiGameData): void {
  if (apiGameCache.has(gameId)) {
    apiGameCache.delete(gameId);
  } else if (apiGameCache.size >= MAX_API_CACHE_SIZE) {
    const oldestKey = apiGameCache.keys().next().value;
    if (oldestKey !== undefined) apiGameCache.delete(oldestKey);
  }
  apiGameCache.set(gameId, data);
}

function extractInitialFenFromApiJson(data: any): string | undefined {
  if (typeof data?.initialFen === 'string' && data.initialFen.trim() && data.initialFen !== 'startpos') {
    return data.initialFen.trim();
  }
  if (typeof data?.pgn === 'string') {
    const match = data.pgn.match(/\[FEN\s+"([^"]+)"\]/);
    if (match?.[1]?.trim()) {
      return match[1].trim();
    }
  }
  return undefined;
}

/**
 * Query Lichess API for official game move list and metadata.
 */
export async function fetchLichessGameApi(
  gameId: string,
  fetchFn: typeof fetch = fetch
): Promise<ApiGameData | null> {
  if (!gameId || !/^[a-zA-Z0-9]{8}$/.test(gameId)) {
    return null;
  }

  const cached = apiGameCache.get(gameId);
  if (cached) {
    apiGameCache.delete(gameId);
    apiGameCache.set(gameId, cached);
    return cached;
  }

  if (Date.now() < apiCooldownUntil) {
    return null;
  }

  const pending = inFlightApiRequests.get(gameId);
  if (pending && pending.fetchFn === fetchFn) {
    return pending.promise;
  }

  const runFetch = async (): Promise<ApiGameData | null> => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    try {
      const activeFetch = fetchFn || globalThis.fetch;
      const url = `https://lichess.org/game/export/${gameId}?moves=true&pgnInJson=false&clocks=false&evals=false&opening=true`;
      const init: RequestInit = {
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      };
      const res = await (activeFetch === globalThis.fetch
        ? activeFetch.call(globalThis, url, init)
        : activeFetch(url, init));

      if (res.status === 429) {
        const retryAfter = Number(res.headers?.get?.('Retry-After'));
        const cooldownMs =
          Number.isFinite(retryAfter) && retryAfter > 0
            ? Math.min(retryAfter * 1000, 300_000)
            : DEFAULT_RATE_LIMIT_COOLDOWN_MS;
        apiCooldownUntil = Date.now() + cooldownMs;
        return null;
      }

      if (!res.ok) return null;

      const data = await res.json();
      const movesStr = (typeof data?.moves === 'string' ? data.moves : '').trim();
      const moves = movesStr ? movesStr.split(/\s+/) : [];
      const whitePlayer = data?.players?.white?.user?.name || data?.players?.white?.userId;
      const blackPlayer = data?.players?.black?.user?.name || data?.players?.black?.userId;
      const opening = data?.opening?.name;
      const initialFen = extractInitialFenFromApiJson(data);
      const variant =
        typeof data?.variant === 'string'
          ? data.variant
          : typeof data?.variant?.key === 'string'
            ? data.variant.key
            : undefined;

      const result: ApiGameData = { moves, whitePlayer, blackPlayer, opening, initialFen, variant };
      setApiGameCache(gameId, result);
      return result;
    } catch {
      return null;
    } finally {
      clearTimeout(timeoutId);
    }
  };

  const promise = runFetch();
  const entry = { promise, fetchFn };
  inFlightApiRequests.set(gameId, entry);
  promise.finally(() => {
    if (inFlightApiRequests.get(gameId) === entry) {
      inFlightApiRequests.delete(gameId);
    }
  });
  return promise;
}
