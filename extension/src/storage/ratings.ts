/**
 * User Ratings Storage and Export Controller.
 *
 * Implements user feedback collection per PLAN.md Section 9.6:
 * - Saves thumbs up/down ratings with full context (FEN, move, prompt, model output, timestamps).
 * - Persists in chrome.storage.local.
 * - Exports formatted JSON dataset for Phase 12 active learning retraining.
 */

export interface UserRating {
  id: string;
  timestamp: number;
  fen: string;
  move: string;
  rating: 'up' | 'down';
  prompt: string;
  output: string;
  reasoning?: string;
  level: string;
  detail: string;
  model: string;
}

const LEGACY_RATINGS_KEY = 'second_user_ratings';
const RATINGS_INDEX_KEY = 'second_user_ratings_index';
const RATING_KEY_PREFIX = 'second_rating_';

function ratingKey(id: string): string {
  return `${RATING_KEY_PREFIX}${id}`;
}

let ratingsWriteQueue: Promise<void> = Promise.resolve();
let cachedIndex: string[] | null = null;
let cachedRatingsMap: Map<string, UserRating> | null = null;
let cachedStorageRef: unknown = undefined;
let attachedOnChangedRef: unknown = undefined;

function getChromeStorageLocal(): chrome.storage.LocalStorageArea | undefined {
  return typeof chrome !== 'undefined' ? chrome.storage?.local : undefined;
}

function invalidateRatingsCache(): void {
  cachedIndex = null;
  cachedRatingsMap = null;
}

function ensureRatingsCacheBinding(): chrome.storage.LocalStorageArea | undefined {
  const storageApi = getChromeStorageLocal();
  if (storageApi !== cachedStorageRef) {
    invalidateRatingsCache();
    cachedStorageRef = storageApi;
  }
  const onChanged = typeof chrome !== 'undefined' ? chrome.storage?.onChanged : undefined;
  if (onChanged && typeof onChanged.addListener === 'function' && onChanged !== attachedOnChangedRef) {
    attachedOnChangedRef = onChanged;
    onChanged.addListener((changes, areaName) => {
      if (areaName !== 'local') return;
      if (
        RATINGS_INDEX_KEY in changes ||
        LEGACY_RATINGS_KEY in changes ||
        Object.keys(changes).some((k) => k.startsWith(RATING_KEY_PREFIX))
      ) {
        invalidateRatingsCache();
      }
    });
  }
  return storageApi;
}

function enqueueRatingsWrite(task: () => Promise<void>): Promise<void> {
  const run = ratingsWriteQueue.then(task, task);
  ratingsWriteQueue = run.then(
    () => undefined,
    () => undefined
  );
  return run;
}

async function chromeGet(
  storageApi: chrome.storage.LocalStorageArea,
  keys: string[]
): Promise<Record<string, any> | undefined> {
  return new Promise((resolve) => {
    storageApi.get(keys, (result) => {
      const lastError = chrome.runtime?.lastError;
      if (lastError) {
        console.warn('Failed to load ratings from chrome.storage.local:', lastError.message);
        resolve(undefined);
        return;
      }
      resolve(result || {});
    });
  });
}

async function chromeSet(
  storageApi: chrome.storage.LocalStorageArea,
  items: Record<string, any>
): Promise<void> {
  return new Promise((resolve, reject) => {
    storageApi.set(items, () => {
      const lastError = chrome.runtime?.lastError;
      if (lastError) {
        reject(new Error(lastError.message || 'Failed to persist ratings'));
        return;
      }
      resolve();
    });
  });
}

async function chromeRemove(
  storageApi: chrome.storage.LocalStorageArea,
  keys: string[]
): Promise<void> {
  return new Promise((resolve, reject) => {
    storageApi.remove(keys, () => {
      const lastError = chrome.runtime?.lastError;
      if (lastError) {
        reject(new Error(lastError.message || 'Failed to clear ratings'));
        return;
      }
      resolve();
    });
  });
}

async function loadChromeIndexAndMigrate(
  storageApi: chrome.storage.LocalStorageArea
): Promise<string[]> {
  if (cachedIndex !== null) {
    return cachedIndex;
  }

  const res = await chromeGet(storageApi, [RATINGS_INDEX_KEY, LEGACY_RATINGS_KEY]);
  if (!res) {
    return [];
  }

  const rawIndex = Array.isArray(res[RATINGS_INDEX_KEY])
    ? (res[RATINGS_INDEX_KEY] as string[])
    : [];
  const legacyList = Array.isArray(res[LEGACY_RATINGS_KEY])
    ? (res[LEGACY_RATINGS_KEY] as UserRating[])
    : null;

  if (legacyList && legacyList.length > 0) {
    // Seamless one-time migration from legacy monolithic array to per-rating keys.
    const indexSet = new Set<string>(rawIndex);
    const itemsToWrite: Record<string, any> = {};
    const migratedMap = new Map<string, UserRating>();
    for (const item of legacyList) {
      if (item && typeof item.id === 'string') {
        indexSet.add(item.id);
        itemsToWrite[ratingKey(item.id)] = item;
        migratedMap.set(item.id, item);
      }
    }
    const mergedIndex = Array.from(indexSet);
    itemsToWrite[RATINGS_INDEX_KEY] = mergedIndex;
    try {
      await chromeSet(storageApi, itemsToWrite);
      await chromeRemove(storageApi, [LEGACY_RATINGS_KEY]);
    } catch {
      // Best effort migration; keep index consistent in memory
    }
    cachedIndex = mergedIndex;
    if (cachedRatingsMap) {
      for (const [id, r] of migratedMap) cachedRatingsMap.set(id, r);
    }
    return mergedIndex;
  }

  cachedIndex = rawIndex;
  return rawIndex;
}

function loadLocalIndexAndMigrate(): string[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const rawIdx = localStorage.getItem(RATINGS_INDEX_KEY);
    const index: string[] = rawIdx ? JSON.parse(rawIdx) : [];
    const safeIndex = Array.isArray(index) ? index.filter((id) => typeof id === 'string') : [];
    const rawLegacy = localStorage.getItem(LEGACY_RATINGS_KEY);
    if (rawLegacy) {
      const legacy = JSON.parse(rawLegacy);
      if (Array.isArray(legacy) && legacy.length > 0) {
        const set = new Set<string>(safeIndex);
        for (const item of legacy) {
          if (item && typeof item.id === 'string') {
            set.add(item.id);
            localStorage.setItem(ratingKey(item.id), JSON.stringify(item));
          }
        }
        const merged = Array.from(set);
        localStorage.setItem(RATINGS_INDEX_KEY, JSON.stringify(merged));
        localStorage.removeItem(LEGACY_RATINGS_KEY);
        return merged;
      }
    }
    return safeIndex;
  } catch {
    return [];
  }
}

export function saveRating(rating: UserRating): Promise<void> {
  return enqueueRatingsWrite(async () => {
    const storageApi = ensureRatingsCacheBinding();
    if (storageApi) {
      const index = await loadChromeIndexAndMigrate(storageApi);
      const alreadyIndexed = index.includes(rating.id);
      const nextIndex = alreadyIndexed ? index : [...index, rating.id];
      const payload: Record<string, any> = {
        [ratingKey(rating.id)]: rating,
      };
      if (!alreadyIndexed) {
        payload[RATINGS_INDEX_KEY] = nextIndex;
      }
      await chromeSet(storageApi, payload);
      cachedIndex = nextIndex;
      if (cachedRatingsMap) {
        cachedRatingsMap.set(rating.id, rating);
      }
      return;
    }

    if (typeof localStorage !== 'undefined') {
      try {
        const index = loadLocalIndexAndMigrate();
        if (!index.includes(rating.id)) {
          index.push(rating.id);
          localStorage.setItem(RATINGS_INDEX_KEY, JSON.stringify(index));
        }
        localStorage.setItem(ratingKey(rating.id), JSON.stringify(rating));
      } catch {
        // ignore
      }
    }
  });
}

export function deleteRating(id: string): Promise<void> {
  return enqueueRatingsWrite(async () => {
    const storageApi = ensureRatingsCacheBinding();
    if (storageApi) {
      const index = await loadChromeIndexAndMigrate(storageApi);
      const nextIndex = index.filter((existingId) => existingId !== id);
      if (nextIndex.length !== index.length) {
        await chromeSet(storageApi, { [RATINGS_INDEX_KEY]: nextIndex });
      }
      await chromeRemove(storageApi, [ratingKey(id)]);
      cachedIndex = nextIndex;
      cachedRatingsMap?.delete(id);
      return;
    }

    if (typeof localStorage !== 'undefined') {
      try {
        const index = loadLocalIndexAndMigrate().filter((existingId) => existingId !== id);
        localStorage.setItem(RATINGS_INDEX_KEY, JSON.stringify(index));
        localStorage.removeItem(ratingKey(id));
      } catch {
        // ignore
      }
    }
  });
}

export async function getRatings(): Promise<UserRating[]> {
  const storageApi = ensureRatingsCacheBinding();
  if (storageApi) {
    const index = await loadChromeIndexAndMigrate(storageApi);
    if (index.length === 0) {
      cachedRatingsMap = new Map();
      return [];
    }
    if (cachedRatingsMap && index.every((id) => cachedRatingsMap!.has(id))) {
      return index.map((id) => cachedRatingsMap!.get(id)!);
    }
    const keys = index.map(ratingKey);
    const items = await chromeGet(storageApi, keys);
    if (!items) return [];
    const map = new Map<string, UserRating>();
    const results: UserRating[] = [];
    for (const id of index) {
      const entry = items[ratingKey(id)];
      if (entry && typeof entry === 'object') {
        map.set(id, entry as UserRating);
        results.push(entry as UserRating);
      }
    }
    cachedRatingsMap = map;
    return results;
  }

  try {
    if (typeof localStorage !== 'undefined') {
      const index = loadLocalIndexAndMigrate();
      const results: UserRating[] = [];
      for (const id of index) {
        const raw = localStorage.getItem(ratingKey(id));
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object') {
            results.push(parsed as UserRating);
          }
        }
      }
      return results;
    }
  } catch {
    // ignore
  }

  return [];
}

export function clearRatings(): Promise<void> {
  return enqueueRatingsWrite(async () => {
    const storageApi = ensureRatingsCacheBinding();
    if (storageApi) {
      const index = await loadChromeIndexAndMigrate(storageApi);
      const keysToRemove = [RATINGS_INDEX_KEY, LEGACY_RATINGS_KEY, ...index.map(ratingKey)];
      await chromeRemove(storageApi, keysToRemove);
      cachedIndex = [];
      cachedRatingsMap = new Map();
      return;
    }

    if (typeof localStorage !== 'undefined') {
      try {
        const index = loadLocalIndexAndMigrate();
        for (const id of index) {
          localStorage.removeItem(ratingKey(id));
        }
        localStorage.removeItem(RATINGS_INDEX_KEY);
        localStorage.removeItem(LEGACY_RATINGS_KEY);
      } catch {
        // ignore
      }
    }
  });
}

export function formatRatingsExportJson(ratings: UserRating[]): string {
  return JSON.stringify(
    {
      version: '1.0',
      exportedAt: new Date().toISOString(),
      count: ratings.length,
      ratings,
    },
    null,
    2
  );
}

export async function exportRatingsAsJsonBlob(): Promise<Blob> {
  const ratings = await getRatings();
  const jsonStr = formatRatingsExportJson(ratings);
  return new Blob([jsonStr], { type: 'application/json' });
}
