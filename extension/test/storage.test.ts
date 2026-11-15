import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { loadSettings, saveSettings, DEFAULT_SETTINGS } from '../src/storage/settings.js';
import {
  saveRating,
  deleteRating,
  getRatings,
  clearRatings,
  formatRatingsExportJson,
  UserRating,
} from '../src/storage/ratings.js';

if (typeof globalThis.localStorage === 'undefined') {
  const store = new Map<string, string>();
  globalThis.localStorage = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => store.set(k, String(v)),
    removeItem: (k: string) => store.delete(k),
    clear: () => store.clear(),
  } as any;
}

describe('Settings Storage Controller', () => {
  const originalChrome = (globalThis as any).chrome;

  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    (globalThis as any).chrome = originalChrome;
  });

  it('loads default settings when empty', async () => {
    const settings = await loadSettings();
    expect(settings).toEqual(DEFAULT_SETTINGS);
  });

  it('saves and updates partial settings', async () => {
    await saveSettings({ level: 'beginner', modelSize: '0.8B' });
    const settings = await loadSettings();

    expect(settings.level).toBe('beginner');
    expect(settings.modelSize).toBe('0.8B');
    expect(settings.cloudEval).toBe(true); // preserved default
  });

  it('sanitizes corrupted stored settings values back to defaults', async () => {
    localStorage.setItem(
      'second_settings',
      JSON.stringify({ level: 'grandmaster', reasoningMode: 'thinking', modelSize: '0.8B', cloudEval: 'yes' })
    );
    const settings = await loadSettings();
    expect(settings).toEqual({
      level: 'club',
      modelSize: '0.8B',
      cloudEval: true,
    });
  });

  it('serializes concurrent saveSettings calls without clobbering updates', async () => {
    const store: Record<string, any> = {};
    (globalThis as any).chrome = {
      runtime: { lastError: null },
      storage: {
        local: {
          get: vi.fn((_keys, cb) => {
            setTimeout(() => cb({ ...store }), 5);
          }),
          set: vi.fn((obj, cb) => {
            setTimeout(() => {
              Object.assign(store, obj);
              cb();
            }, 5);
          }),
        },
      },
    };

    await Promise.all([
      saveSettings({ level: 'beginner' }),
      saveSettings({ modelSize: '0.8B' }),
      saveSettings({ cloudEval: false }),
    ]);

    const final = await loadSettings();
    expect(final).toEqual({
      level: 'beginner',
      modelSize: '0.8B',
      cloudEval: false,
    });
  });

  it('rejects saveSettings and safely falls back in loadSettings on chrome.runtime.lastError', async () => {
    (globalThis as any).chrome = {
      runtime: { lastError: { message: 'QUOTA_BYTES quota exceeded' } },
      storage: {
        local: {
          get: vi.fn((_keys, cb) => cb(undefined)),
          set: vi.fn((_obj, cb) => cb()),
        },
      },
    };

    const loaded = await loadSettings();
    expect(loaded).toEqual(DEFAULT_SETTINGS);
    await expect(saveSettings({ level: 'beginner' })).rejects.toThrow('QUOTA_BYTES');
  });
});

describe('User Ratings Storage & JSON Export', () => {
  beforeEach(async () => {
    localStorage.clear();
  });

  it('saves user rating and updates if ID repeats', async () => {
    const r1: UserRating = {
      id: 'r_1',
      timestamp: 1000,
      fen: 'fen_1',
      move: 'e4',
      rating: 'up',
      prompt: 'prompt_1',
      output: 'good move',
      level: 'club',
      detail: 'short',
      model: '2B',
    };

    await saveRating(r1);
    let all = await getRatings();
    expect(all).toHaveLength(1);
    expect(all[0].rating).toBe('up');

    // User toggles to down
    const r1Updated = { ...r1, rating: 'down' as const };
    await saveRating(r1Updated);

    all = await getRatings();
    expect(all).toHaveLength(1);
    expect(all[0].rating).toBe('down');
  });

  it('formats valid export JSON format', () => {
    const ratings: UserRating[] = [
      {
        id: 'r_1',
        timestamp: 1000,
        fen: 'fen_1',
        move: 'e4',
        rating: 'up',
        prompt: 'prompt_1',
        output: 'good move',
        level: 'club',
        detail: 'short',
        model: '2B',
      },
    ];

    const jsonStr = formatRatingsExportJson(ratings);
    const parsed = JSON.parse(jsonStr);

    expect(parsed.version).toBe('1.0');
    expect(parsed.count).toBe(1);
    expect(parsed.ratings[0].id).toBe('r_1');
  });

  it('clears ratings successfully', async () => {
    await saveRating({
      id: 'r_1',
      timestamp: 1000,
      fen: 'fen_1',
      move: 'e4',
      rating: 'up',
      prompt: 'p',
      output: 'o',
      level: 'club',
      detail: 'short',
      model: '2B',
    });

    await clearRatings();
    const all = await getRatings();
    expect(all).toHaveLength(0);
  });

  it('deletes a specific rating by ID when toggled off', async () => {
    await saveRating({
      id: 'r_keep',
      timestamp: 1000,
      fen: 'fen_1',
      move: 'e4',
      rating: 'up',
      prompt: 'p1',
      output: 'o1',
      level: 'club',
      detail: 'short',
      model: '2B',
    });
    await saveRating({
      id: 'r_drop',
      timestamp: 2000,
      fen: 'fen_2',
      move: 'd4',
      rating: 'down',
      prompt: 'p2',
      output: 'o2',
      level: 'club',
      detail: 'short',
      model: '2B',
    });

    await deleteRating('r_drop');
    const all = await getRatings();
    expect(all).toHaveLength(1);
    expect(all[0].id).toBe('r_keep');
  });

  it('rejects saveRating when chrome.runtime.lastError is set', async () => {
    const originalChrome = (globalThis as any).chrome;
    (globalThis as any).chrome = {
      runtime: { lastError: { message: 'QUOTA_BYTES quota exceeded' } },
      storage: {
        local: {
          get: vi.fn((_keys, cb) => cb(undefined)),
          set: vi.fn((_obj, cb) => cb()),
          remove: vi.fn((_keys, cb) => cb()),
        },
      },
    };

    try {
      const all = await getRatings();
      expect(all).toEqual([]);
      await expect(
        saveRating({
          id: 'r_fail',
          timestamp: 1000,
          fen: 'fen_1',
          move: 'e4',
          rating: 'up',
          prompt: 'p',
          output: 'o',
          level: 'club',
          detail: 'short',
          model: '2B',
        })
      ).rejects.toThrow('QUOTA_BYTES');
    } finally {
      (globalThis as any).chrome = originalChrome;
    }
  });

  it('migrates legacy second_user_ratings array into per-rating keys and index', async () => {
    const originalChrome = (globalThis as any).chrome;
    const store: Record<string, any> = {
      second_user_ratings: [
        {
          id: 'r_legacy',
          timestamp: 500,
          fen: 'fen_legacy',
          move: 'd4',
          rating: 'up',
          prompt: 'p',
          output: 'o',
          level: 'beginner',
          detail: 'short',
          model: '0.8B',
        },
      ],
    };

    (globalThis as any).chrome = {
      runtime: {},
      storage: {
        local: {
          get: vi.fn((keys, cb) => {
            if (typeof keys === 'string') {
              cb({ [keys]: store[keys] });
            } else if (Array.isArray(keys)) {
              const out: Record<string, any> = {};
              for (const k of keys) if (k in store) out[k] = store[k];
              cb(out);
            } else {
              cb({ ...store });
            }
          }),
          set: vi.fn((obj, cb) => {
            Object.assign(store, obj);
            cb();
          }),
          remove: vi.fn((keys, cb) => {
            const list = Array.isArray(keys) ? keys : [keys];
            for (const k of list) delete store[k];
            cb();
          }),
        },
      },
    };

    try {
      const ratings = await getRatings();
      expect(ratings).toHaveLength(1);
      expect(ratings[0].id).toBe('r_legacy');
      expect(store['second_rating_r_legacy']).toMatchObject({ id: 'r_legacy' });
      expect(store['second_user_ratings_index']).toEqual(['r_legacy']);
      expect(store['second_user_ratings']).toBeUndefined();
    } finally {
      (globalThis as any).chrome = originalChrome;
    }
  });
});
