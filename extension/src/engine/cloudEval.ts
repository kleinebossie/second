/**
 * Polite Lichess Cloud Eval Client.
 *
 * Requirements per PLAN.md Section 9.4:
 * 1. Only after user interaction (user click).
 * 2. Only one request at a time (strict concurrency limit = 1).
 * 3. Cache results by FEN.
 * 4. On HTTP 429: stop all cloud requests for at least 60 seconds.
 * 5. Settings switch turns it off (default is on).
 * 6. If cloud eval has no data -> return null (signals caller to fall back to local Stockfish).
 */

import type { EngineEval, EnginePv, EngineScore } from '../types.js';

export interface CloudEvalOptions {
  enabled?: boolean;
  cooldownMs?: number;
  timeoutMs?: number;
  maxCacheSize?: number;
  maxNegativeCacheSize?: number;
  negativeCacheTtlMs?: number;
  fetchFn?: typeof fetch;
}

export interface GetEvaluationOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

interface InFlightCloudEntry {
  promise: Promise<EngineEval | null>;
  waiters: number;
  cancel: () => void;
}

export class CloudEvalClient {
  private enabled: boolean = true;
  private cooldownUntil: number = 0;
  private isRequestInProgress: boolean = false;
  private requestQueue: Array<(acquired: boolean) => void> = [];
  private cache: Map<string, EngineEval> = new Map();
  private negativeCache: Map<string, number> = new Map();
  private inFlightByFen: Map<string, InFlightCloudEntry> = new Map();
  private readonly cooldownMs: number;
  private readonly defaultTimeoutMs: number;
  private readonly maxCacheSize: number;
  private readonly maxNegativeCacheSize: number;
  private readonly negativeCacheTtlMs: number;
  private readonly fetchFn: typeof fetch;

  constructor(options: CloudEvalOptions = {}) {
    this.enabled = options.enabled ?? true;
    this.cooldownMs = options.cooldownMs ?? 60_000;
    this.defaultTimeoutMs = options.timeoutMs ?? 1_200;
    this.maxCacheSize = options.maxCacheSize ?? 500;
    this.maxNegativeCacheSize = options.maxNegativeCacheSize ?? 500;
    this.negativeCacheTtlMs = options.negativeCacheTtlMs ?? 300_000;
    const rawFetch =
      options.fetchFn ?? (typeof globalThis !== 'undefined' ? globalThis.fetch : undefined);
    this.fetchFn = rawFetch
      ? rawFetch.bind(globalThis)
      : ((async () => {
          throw new Error('fetch is not available');
        }) as typeof fetch);
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  public isEnabled(): boolean {
    return this.enabled;
  }

  public isInCooldown(): boolean {
    return Date.now() < this.cooldownUntil;
  }

  public getCooldownRemainingMs(): number {
    return Math.max(0, this.cooldownUntil - Date.now());
  }

  public clearCache(): void {
    this.cache.clear();
    this.negativeCache.clear();
  }

  private getCacheEntry(normFen: string): EngineEval | undefined {
    const cached = this.cache.get(normFen);
    if (!cached) return undefined;
    this.cache.delete(normFen);
    this.cache.set(normFen, cached);
    return cached;
  }

  public getCached(fen: string): EngineEval | undefined {
    return this.getCacheEntry(this.normalizeFen(fen));
  }

  private isNegativeCached(normFen: string): boolean {
    const expiresAt = this.negativeCache.get(normFen);
    if (expiresAt === undefined) return false;
    if (Date.now() >= expiresAt) {
      this.negativeCache.delete(normFen);
      return false;
    }
    this.negativeCache.delete(normFen);
    this.negativeCache.set(normFen, expiresAt);
    return true;
  }

  private setNegativeCacheEntry(normFen: string): void {
    if (this.maxNegativeCacheSize <= 0 || this.negativeCacheTtlMs <= 0) return;
    if (this.negativeCache.has(normFen)) {
      this.negativeCache.delete(normFen);
    } else if (this.negativeCache.size >= this.maxNegativeCacheSize) {
      const oldestKey = this.negativeCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.negativeCache.delete(oldestKey);
      }
    }
    this.negativeCache.set(normFen, Date.now() + this.negativeCacheTtlMs);
  }

  /**
   * Acquire single-concurrency lock.
   * Returns false if the caller's AbortSignal fires while queued.
   */
  private async acquireLock(signal?: AbortSignal): Promise<boolean> {
    if (signal?.aborted) {
      return false;
    }
    if (!this.isRequestInProgress) {
      this.isRequestInProgress = true;
      return true;
    }
    return new Promise<boolean>((resolve) => {
      let settled = false;
      const onAbort = () => {
        if (settled) return;
        settled = true;
        const idx = this.requestQueue.indexOf(grant);
        if (idx !== -1) {
          this.requestQueue.splice(idx, 1);
        }
        resolve(false);
      };
      const grant = (acquired: boolean) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener('abort', onAbort);
        if (acquired) {
          this.isRequestInProgress = true;
        }
        resolve(acquired);
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      this.requestQueue.push(grant);
    });
  }

  /**
   * Release single-concurrency lock.
   */
  private releaseLock(): void {
    if (this.requestQueue.length > 0) {
      const next = this.requestQueue.shift();
      if (next) next(true);
    } else {
      this.isRequestInProgress = false;
    }
  }

  private setCacheEntry(normFen: string, value: EngineEval): void {
    this.negativeCache.delete(normFen);
    if (this.cache.has(normFen)) {
      this.cache.delete(normFen);
    } else if (this.cache.size >= this.maxCacheSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey !== undefined) {
        this.cache.delete(oldestKey);
      }
    }
    this.cache.set(normFen, value);
  }

  private parseRetryAfterMs(response: Response): number {
    const raw = response.headers?.get?.('Retry-After');
    if (!raw) return 0;
    const secs = Number(raw);
    if (Number.isFinite(secs) && secs > 0) {
      return Math.ceil(secs * 1000);
    }
    const dateMs = Date.parse(raw);
    if (Number.isFinite(dateMs)) {
      return Math.max(0, dateMs - Date.now());
    }
    return 0;
  }

  private normalizeFen(fen: string): string {
    // Normalise FEN to board + turn + castling + ep (first 4 parts)
    const parts = fen.trim().split(/\s+/);
    return parts.slice(0, 4).join(' ');
  }

  private awaitInFlight(
    entry: InFlightCloudEntry,
    fen: string,
    signal?: AbortSignal
  ): Promise<EngineEval | null> {
    if (signal?.aborted) {
      return Promise.resolve(null);
    }
    entry.waiters++;

    return new Promise<EngineEval | null>((resolve) => {
      let settled = false;
      const onAbort = () => {
        if (settled) return;
        settled = true;
        entry.waiters--;
        if (entry.waiters <= 0) {
          entry.cancel();
        }
        resolve(null);
      };
      signal?.addEventListener('abort', onAbort, { once: true });

      entry.promise.then(
        (res) => {
          if (settled) return;
          settled = true;
          signal?.removeEventListener('abort', onAbort);
          entry.waiters--;
          resolve(res ? { ...res, fen } : null);
        },
        () => {
          if (settled) return;
          settled = true;
          signal?.removeEventListener('abort', onAbort);
          entry.waiters--;
          resolve(null);
        }
      );
    });
  }

  /**
   * Fetch evaluation from Lichess Cloud Eval API.
   * Returns EngineEval on success, or null if disabled, in cooldown, aborted, not found, or error.
   */
  public async getEvaluation(
    fen: string,
    options: GetEvaluationOptions | AbortSignal = {}
  ): Promise<EngineEval | null> {
    const resolvedOpts: GetEvaluationOptions =
      typeof AbortSignal !== 'undefined' && options instanceof AbortSignal
        ? { signal: options }
        : (options as GetEvaluationOptions);
    const signal = resolvedOpts.signal;
    const timeoutMs = resolvedOpts.timeoutMs ?? this.defaultTimeoutMs;

    if (!this.enabled || signal?.aborted) {
      return null;
    }

    if (this.isInCooldown()) {
      return null;
    }

    const normFen = this.normalizeFen(fen);
    const cached = this.getCacheEntry(normFen);
    if (cached) {
      return { ...cached, fen };
    }

    if (this.isNegativeCached(normFen)) {
      return null;
    }

    const existingInFlight = this.inFlightByFen.get(normFen);
    if (existingInFlight) {
      return this.awaitInFlight(existingInFlight, fen, signal);
    }

    const entryController = new AbortController();
    const entry: InFlightCloudEntry = {
      promise: Promise.resolve(null),
      waiters: 0,
      cancel: () => entryController.abort(),
    };

    const runFetch = async (): Promise<EngineEval | null> => {
      const acquired = await this.acquireLock(entryController.signal);
      if (!acquired) {
        return null;
      }

      const requestController = new AbortController();
      let timeoutId: ReturnType<typeof setTimeout> | null = null;
      const onEntryAbort = () => requestController.abort();

      try {
        if (entryController.signal.aborted) {
          return null;
        }

        // Recheck caches after acquiring lock
        const doubleCheck = this.getCacheEntry(normFen);
        if (doubleCheck) {
          return doubleCheck;
        }

        if (this.isNegativeCached(normFen) || this.isInCooldown()) {
          return null;
        }

        entryController.signal.addEventListener('abort', onEntryAbort, { once: true });
        if (timeoutMs > 0) {
          timeoutId = setTimeout(() => requestController.abort(), timeoutMs);
        }

        const url = `https://lichess.org/api/cloud-eval?fen=${encodeURIComponent(normFen)}&multiPv=3`;
        const response = await this.fetchFn(url, {
          headers: { Accept: 'application/json' },
          signal: requestController.signal,
        });

        if (response.status === 429) {
          // HTTP 429 Too Many Requests -> backoff for at least 60 seconds
          const retryAfterMs = this.parseRetryAfterMs(response);
          const appliedCooldown = Math.max(this.cooldownMs, retryAfterMs);
          this.cooldownUntil = Date.now() + appliedCooldown;
          console.warn(`Lichess Cloud Eval returned 429. Cooldown active for ${appliedCooldown / 1000}s.`);
          return null;
        }

        if (response.status === 404) {
          this.setNegativeCacheEntry(normFen);
          return null;
        }

        if (!response.ok) {
          return null;
        }

        const data = await response.json();
        if (!data || !Array.isArray(data.pvs) || data.pvs.length === 0) {
          this.setNegativeCacheEntry(normFen);
          return null;
        }

        const pvs: EnginePv[] = [];
        for (const rawPv of data.pvs) {
          if (!rawPv) continue;
          const moves: string[] = typeof rawPv.moves === 'string'
            ? rawPv.moves.trim().split(/\s+/).filter(Boolean)
            : Array.isArray(rawPv.moves)
              ? rawPv.moves.filter((m: unknown): m is string => typeof m === 'string' && m.length > 0)
              : [];

          if (moves.length === 0) continue;

          let score: EngineScore | null = null;
          if (typeof rawPv.mate === 'number' && Number.isFinite(rawPv.mate)) {
            score = { type: 'mate', value: rawPv.mate };
          } else if (typeof rawPv.cp === 'number' && Number.isFinite(rawPv.cp)) {
            score = { type: 'cp', value: rawPv.cp };
          }
          if (!score) continue;

          pvs.push({
            moves,
            score,
          });
        }

        if (pvs.length === 0) {
          this.setNegativeCacheEntry(normFen);
          return null;
        }

        const result: EngineEval = {
          fen: normFen,
          pvs,
        };

        this.setCacheEntry(normFen, result);
        return result;
      } catch (err: any) {
        if (err?.name !== 'AbortError') {
          console.warn('Lichess cloud eval fetch failed:', err);
        }
        return null;
      } finally {
        if (timeoutId !== null) {
          clearTimeout(timeoutId);
        }
        entryController.signal.removeEventListener('abort', onEntryAbort);
        this.releaseLock();
      }
    };

    entry.promise = runFetch().finally(() => {
      if (this.inFlightByFen.get(normFen) === entry) {
        this.inFlightByFen.delete(normFen);
      }
    });
    this.inFlightByFen.set(normFen, entry);

    return this.awaitInFlight(entry, fen, signal);
  }
}
