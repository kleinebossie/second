/**
 * Settings Storage Controller for Second Chrome Extension.
 * Persists user configuration in chrome.storage.local with localStorage fallback.
 */

export interface ExtensionSettings {
  level: 'beginner' | 'club';
  modelSize: '2B' | '0.8B';
  cloudEval: boolean;
}

export const DEFAULT_SETTINGS: ExtensionSettings = {
  level: 'club',
  modelSize: '2B',
  cloudEval: true,
};

const SETTINGS_KEY = 'second_settings';

let settingsWriteQueue: Promise<void> = Promise.resolve();
let cachedSettings: ExtensionSettings | null = null;
let cachedStorageRef: unknown = undefined;
let attachedOnChangedRef: unknown = undefined;

function getChromeStorageLocal(): chrome.storage.LocalStorageArea | undefined {
  return typeof chrome !== 'undefined' ? chrome.storage?.local : undefined;
}

function ensureSettingsCacheBinding(): chrome.storage.LocalStorageArea | undefined {
  const storageApi = getChromeStorageLocal();
  if (storageApi !== cachedStorageRef) {
    cachedSettings = null;
    cachedStorageRef = storageApi;
  }
  const onChanged = typeof chrome !== 'undefined' ? chrome.storage?.onChanged : undefined;
  if (onChanged && typeof onChanged.addListener === 'function' && onChanged !== attachedOnChangedRef) {
    attachedOnChangedRef = onChanged;
    onChanged.addListener((changes, areaName) => {
      if (areaName === 'local' && SETTINGS_KEY in changes) {
        const nextVal = changes[SETTINGS_KEY]?.newValue;
        cachedSettings = nextVal !== undefined ? sanitizeSettings(nextVal) : null;
      }
    });
  }
  return storageApi;
}

function enqueueSettingsWrite<T>(task: () => Promise<T>): Promise<T> {
  const run = settingsWriteQueue.then(task, task);
  settingsWriteQueue = run.then(
    () => undefined,
    () => undefined
  );
  return run;
}

function sanitizeSettings(raw: unknown): ExtensionSettings {
  const obj = raw && typeof raw === 'object' && !Array.isArray(raw)
    ? (raw as Record<string, unknown>)
    : {};
  return {
    level: obj.level === 'beginner' || obj.level === 'club' ? obj.level : DEFAULT_SETTINGS.level,
    modelSize:
      obj.modelSize === '2B' || obj.modelSize === '0.8B'
        ? obj.modelSize
        : DEFAULT_SETTINGS.modelSize,
    cloudEval: typeof obj.cloudEval === 'boolean' ? obj.cloudEval : DEFAULT_SETTINGS.cloudEval,
  };
}

export async function loadSettings(): Promise<ExtensionSettings> {
  const storageApi = ensureSettingsCacheBinding();
  if (storageApi) {
    if (cachedSettings) {
      return { ...cachedSettings };
    }
    return new Promise((resolve) => {
      storageApi.get([SETTINGS_KEY], (result) => {
        const lastError = chrome.runtime?.lastError;
        if (lastError) {
          console.warn('Failed to load settings from chrome.storage.local:', lastError.message);
          resolve({ ...DEFAULT_SETTINGS });
          return;
        }
        const stored = result?.[SETTINGS_KEY];
        const sanitized = sanitizeSettings(stored);
        cachedSettings = sanitized;
        resolve({ ...sanitized });
      });
    });
  }

  // Fallback for non-extension environments (tests)
  try {
    if (typeof localStorage !== 'undefined') {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) {
        return sanitizeSettings(JSON.parse(raw));
      }
    }
  } catch {
    // ignore
  }

  return { ...DEFAULT_SETTINGS };
}

export function saveSettings(settings: Partial<ExtensionSettings>): Promise<ExtensionSettings> {
  return enqueueSettingsWrite(async () => {
    const storageApi = ensureSettingsCacheBinding();
    const current = storageApi && cachedSettings ? cachedSettings : await loadSettings();
    const updated: ExtensionSettings = sanitizeSettings({
      ...current,
      ...settings,
    });

    if (storageApi) {
      await new Promise<void>((resolve, reject) => {
        storageApi.set({ [SETTINGS_KEY]: updated }, () => {
          const lastError = chrome.runtime?.lastError;
          if (lastError) {
            reject(new Error(lastError.message || 'Failed to save settings'));
            return;
          }
          resolve();
        });
      });
      cachedSettings = updated;
    } else if (typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(updated));
      } catch {
        // ignore
      }
    }

    return { ...updated };
  });
}
