// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

const {
  mockIsModelLoaded,
  mockGetCurrentModel,
  mockIsModelCached,
  mockLoadModel,
  mockAbortLoad,
  mockGenerate,
  mockStop,
  mockDeleteModelCache,
  mockCheckHardware,
  mockStockfishEvaluate,
  mockCloudGetEvaluation,
} = vi.hoisted(() => ({
  mockIsModelLoaded: vi.fn(() => false),
  mockGetCurrentModel: vi.fn(() => '2B'),
  mockIsModelCached: vi.fn(async () => false),
  mockLoadModel: vi.fn(async (m?: string, _onProgress?: any, _opts?: any) => {
    if (m) mockGetCurrentModel.mockReturnValue(m);
  }),
  mockAbortLoad: vi.fn(),
  mockGenerate: vi.fn(async (_prompt?: string, _onToken?: (t: string) => void, _opts?: any) => ({
    text: 'Moving the knight to 24... Nd4 exposes f7.',
  })),
  mockStop: vi.fn(),
  mockDeleteModelCache: vi.fn(async () => {}),
  mockCheckHardware: vi.fn(async () => ({
    webAssemblySupported: true,
    webGpuSupported: true,
    adapterInfo: 'Mock GPU',
    isWeakDevice: false,
  })),
  mockStockfishEvaluate: vi.fn(async (fen: string) => ({
    fen,
    depth: 14,
    pvs: [{ moves: ['b8c6'], score: { type: 'cp', value: 150 } }],
  })),
  mockCloudGetEvaluation: vi.fn(async (_fen?: string, _opts?: any) => null),
}));

vi.mock('../src/runtime/modelRunner.js', () => {
  return {
    ModelRunner: class {
      isModelLoaded = mockIsModelLoaded;
      getCurrentModel = mockGetCurrentModel;
      isModelCached = mockIsModelCached;
      loadModel = mockLoadModel;
      abortLoad = mockAbortLoad;
      generate = mockGenerate;
      stop = mockStop;
      deleteModelCache = mockDeleteModelCache;
      static checkHardware = mockCheckHardware;
    },
  };
});

vi.mock('../src/lock.js', () => ({
  LOCK_OFF_MESSAGE: 'Second is off during live games. Open the game after it ends.',
  getCachedFinishedGameData: vi.fn(() => undefined),
  evaluatePageAccess: vi.fn(async (url: string) => {
    if (url.includes('analysis') || url.includes('study')) {
      return { allowed: true, reason: 'Analysis board is open for study.', pageType: 'analysis' };
    }
    return {
      allowed: false,
      reason: 'Second is off during live games. Open the game after it ends.',
      pageType: 'game',
      gameId: 'abcdefgh',
    };
  }),
}));

vi.mock('../src/api/lichessApi.js', () => ({
  fetchLichessGameApi: vi.fn(async () => null),
  clearApiGameCache: vi.fn(),
}));

vi.mock('../src/engine/stockfish.js', () => ({
  StockfishEngine: class {
    init = vi.fn(async () => {});
    evaluate = mockStockfishEvaluate;
  },
}));

vi.mock('../src/engine/cloudEval.js', () => ({
  CloudEvalClient: class {
    setEnabled = vi.fn();
    getEvaluation = mockCloudGetEvaluation;
  },
}));

vi.mock('../src/core/promptBuilder.js', () => ({
  buildPromptForInference: vi.fn(() => ({
    prompt: '<|im_start|>user\nMock prompt<|im_end|>',
    userMessage: 'Mock user message',
    systemMessage: 'Mock system message',
  })),
  classifyVerdict: vi.fn(() => 'Blunder'),
}));

vi.mock('../src/core/moveResolution.js', () => ({
  resolveMoveContext: vi.fn((moves: string[], activePly?: number, selectedSan?: string) => ({
    fenBefore: 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
    fenAfter: 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3',
    moveSan: selectedSan || (activePly && moves[activePly - 1]) || 'Nc6',
    moveUci: 'b8c6',
  })),
}));

const mockSaveRating = vi.fn(async (_rating: any) => {});
const mockDeleteRating = vi.fn(async (_id: string) => {});
vi.mock('../src/storage/ratings.js', () => ({
  saveRating: (rating: any) => mockSaveRating(rating),
  deleteRating: (id: string) => mockDeleteRating(id),
  exportRatingsAsJsonBlob: vi.fn(async () => new Blob(['[]'], { type: 'application/json' })),
}));

const mockLoadSettings = vi.fn(async () => ({
  level: 'club',
  modelSize: '2B',
  cloudEval: true,
}));
const mockSaveSettings = vi.fn(async (partial: any) => ({
  level: 'club',
  modelSize: '2B',
  cloudEval: true,
  ...partial,
}));

vi.mock('../src/storage/settings.js', () => ({
  DEFAULT_SETTINGS: {
    level: 'club',
    modelSize: '2B',
    cloudEval: true,
  },
  loadSettings: () => mockLoadSettings(),
  saveSettings: (p: any) => mockSaveSettings(p),
}));

import { initSidepanel } from '../src/sidepanel/main.js';

const sidepanelHtml = fs.readFileSync(
  path.resolve(__dirname, '../sidepanel.html'),
  'utf-8'
);

describe('Sidepanel UI Controller & State Integrity', () => {
  let messageListeners: Array<(msg: any, sender?: any) => void> = [];
  let onActivatedListeners: Array<(info: any) => void> = [];
  let tabResponse: any = null;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockCheckHardware.mockResolvedValue({
      webAssemblySupported: true,
      webGpuSupported: true,
      adapterInfo: 'Mock GPU',
      isWeakDevice: false,
    });
    mockIsModelLoaded.mockReturnValue(false);
    mockGetCurrentModel.mockReturnValue('2B');
    mockIsModelCached.mockResolvedValue(false);
    messageListeners = [];
    onActivatedListeners = [];
    tabResponse = {
      url: 'https://lichess.org/analysis',
      pageType: 'analysis',
      isAllowed: true,
      moves: ['e4', 'e5', 'Nf3', 'Nc6'],
      selectedMoveSan: 'Nc6',
      activePly: 4,
      timestamp: Date.now(),
    };

    document.documentElement.innerHTML = sidepanelHtml;
    Element.prototype.scrollIntoView = vi.fn();

    (globalThis as any).chrome = {
      runtime: {
        onMessage: {
          addListener: (fn: (msg: any, sender?: any) => void) => messageListeners.push(fn),
        },
        lastError: null,
      },
      scripting: {
        executeScript: vi.fn(async () => []),
      },
      tabs: {
        query: vi.fn((_q: any, cb?: (tabs: any[]) => void) => {
          const tabs = [{ id: 1, url: tabResponse?.url || 'https://lichess.org/analysis' }];
          if (cb) cb(tabs);
          return Promise.resolve(tabs);
        }),
        sendMessage: vi.fn((_tabId: number, _msg: any, cb?: (resp: any) => void) => {
          if (cb) cb(tabResponse);
          return Promise.resolve(tabResponse);
        }),
        onActivated: {
          addListener: (fn: (info: any) => void) => onActivatedListeners.push(fn),
        },
        onUpdated: { addListener: vi.fn() },
      },
    };

    await initSidepanel();
    await new Promise((r) => setTimeout(r, 20));
  });

  it('renders Variation 6A header with formatted move number and keeps dock controls gated initially', () => {
    const movePill = document.getElementById('selected-move-pill')!;
    const modelBadge = document.getElementById('model-badge')!;
    const btnLonger = document.getElementById('btn-more-detail') as HTMLButtonElement;
    const btnUp = document.getElementById('btn-rate-up') as HTMLButtonElement;
    const btnDown = document.getElementById('btn-rate-down') as HTMLButtonElement;

    expect(movePill.textContent).toBe('2... Nc6');
    expect(modelBadge.textContent).toBe('Second 2B');
    expect(btnLonger.disabled).toBe(true);
    expect(btnUp.disabled).toBe(true);
    expect(btnDown.disabled).toBe(true);
    expect(btnUp.getAttribute('aria-pressed')).toBe('false');
    expect(btnDown.getAttribute('aria-pressed')).toBe('false');
  });

  it('enforces explicit download consent gate when model is neither loaded nor cached', async () => {
    mockIsModelLoaded.mockReturnValue(false);
    mockIsModelCached.mockResolvedValue(false);

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 30));

    // Must NOT call loadModel without user clicking Download Model
    expect(mockLoadModel).not.toHaveBeenCalled();
    expect(document.getElementById('status-indicator')!.textContent).toBe('Model required');
    expect(document.getElementById('output-text')!.textContent).toContain(
      'Click Download Model in the banner above'
    );
  });

  it('fails closed immediately if active tab switched to a live game before Explain Move', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    // Simulate user switching to a live game right before clicking Explain Move
    tabResponse = {
      url: 'https://lichess.org/abcdefgh',
      pageType: 'game',
      gameId: 'abcdefgh',
      isAllowed: false,
      lockReason: 'Second is off during live games. Open the game after it ends.',
      timestamp: Date.now(),
    };

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 30));

    expect(mockGenerate).not.toHaveBeenCalled();
    expect(document.getElementById('lock-banner')!.style.display).toBe('flex');
    expect(btnExplain.disabled).toBe(true);
    expect(document.getElementById('status-indicator')!.textContent).toBe('Locked');
  });

  it('streams highlighted commentary, sets verdict, enables ratings with generating model, and invalidates on move change', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 40));

    const outputText = document.getElementById('output-text')!;
    expect(outputText.innerHTML).toContain('tok-piece');
    expect(outputText.innerHTML).toContain('tok-move');
    expect(outputText.innerHTML).toContain('tok-square');

    const verdictEl = document.getElementById('selected-move-verdict')!;
    expect(verdictEl.style.display).toBe('inline-block');
    expect(verdictEl.textContent).toBeTruthy();

    const btnUp = document.getElementById('btn-rate-up') as HTMLButtonElement;
    const btnLonger = document.getElementById('btn-more-detail') as HTMLButtonElement;
    expect(btnUp.disabled).toBe(false);
    expect(btnLonger.disabled).toBe(false);

    // Navigating to a new move resets rating state and keeps the 2B model badge
    for (const listener of messageListeners) {
      listener({
        type: 'LICHESS_STATE_UPDATE',
        payload: {
          ...tabResponse,
          moves: ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5'],
          selectedMoveSan: 'Bb5',
          activePly: 5,
        },
      });
    }

    expect(document.getElementById('selected-move-pill')!.textContent).toBe('3. Bb5');
    expect(verdictEl.style.display).toBe('none');
    expect(btnUp.disabled).toBe(true);
    expect(btnUp.getAttribute('aria-pressed')).toBe('false');
    expect(document.getElementById('model-badge')!.textContent).toBe('Second 2B');
  });

  it('manages settings modal keyboard focus, Escape key, and inert background', () => {
    const btnOpen = document.getElementById('btn-settings-open') as HTMLButtonElement;
    const modal = document.getElementById('settings-modal')!;
    const appShell = document.getElementById('app-shell')!;
    const settingLevel = document.getElementById('setting-level')!;

    btnOpen.focus();
    btnOpen.click();

    expect(modal.classList.contains('open')).toBe(true);
    expect(modal.getAttribute('aria-hidden')).toBe('false');
    expect(appShell.hasAttribute('inert')).toBe(true);
    expect(document.activeElement).toBe(settingLevel);

    // Pressing Escape closes modal and restores focus to trigger
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(modal.classList.contains('open')).toBe(false);
    expect(modal.getAttribute('aria-hidden')).toBe('true');
    expect(appShell.hasAttribute('inert')).toBe(false);
    expect(document.activeElement).toBe(btnOpen);
  });

  it('aborts in-flight generation when user navigates to a new move and discards stale commentary', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    let resolveSlowGen: (v: any) => void = () => {};
    mockGenerate.mockImplementationOnce(() => new Promise((resolve) => {
      resolveSlowGen = resolve;
    }));

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 20));

    expect(document.getElementById('status-indicator')!.textContent).toBe('Streaming...');

    // User navigates away while generation is in flight
    for (const listener of messageListeners) {
      listener({
        type: 'LICHESS_STATE_UPDATE',
        payload: {
          ...tabResponse,
          moves: ['e4', 'e5', 'Nf3', 'Nc6', 'd4'],
          selectedMoveSan: 'd4',
          activePly: 5,
        },
      });
    }

    expect(document.getElementById('selected-move-pill')!.textContent).toBe('3. d4');

    // Late resolving in-flight promise must be discarded due to generationEpoch mismatch
    resolveSlowGen({ text: 'Stale commentary for Nc6' });
    await new Promise((r) => setTimeout(r, 20));

    expect(document.getElementById('output-text')!.textContent).not.toContain('Stale commentary for Nc6');
    expect(document.getElementById('btn-rate-up')!.getAttribute('aria-pressed')).toBe('false');

    // Critically: Explain Move must be re-enabled and ready to explain the new move immediately
    expect(btnExplain.disabled).toBe(false);
    expect(document.getElementById('status-indicator')!.textContent).toBe('Ready');

    tabResponse = {
      ...tabResponse,
      moves: ['e4', 'e5', 'Nf3', 'Nc6', 'd4'],
      selectedMoveSan: 'd4',
      activePly: 5,
    };
    mockGenerate.mockResolvedValueOnce({ text: 'Playing 3. d4 opens the center.' });
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 30));

    expect(document.getElementById('output-text')!.textContent).toContain('Playing 3. d4 opens the center.');
    expect(btnExplain.disabled).toBe(false);
  });

  it('does not self-abort when revalidateFreshTabState returns a newer move clicked just before Explain Move', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    // Content script has a newer move than what sidepanel last received
    tabResponse = {
      ...tabResponse,
      moves: ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5'],
      selectedMoveSan: 'Bb5',
      activePly: 5,
    };
    mockGenerate.mockResolvedValueOnce({ text: '3. Bb5 pins the knight.' });

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 40));

    expect(document.getElementById('selected-move-pill')!.textContent).toBe('3. Bb5');
    expect(mockGenerate).toHaveBeenCalledTimes(1);
    expect(document.getElementById('output-text')!.textContent).toContain('3. Bb5 pins the knight.');
    expect(btnExplain.disabled).toBe(false);
  });

  it('supports Longer commentary flow and rating toggle-off', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    mockGenerate.mockImplementationOnce(async (_prompt?: string, onToken?: (t: string) => void) => {
      onToken?.('2... Nc6 develops normally.');
      return { text: '2... Nc6 develops normally.' };
    });

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 40));

    // Click Longer button to generate detailed commentary
    const btnLonger = document.getElementById('btn-more-detail') as HTMLButtonElement;
    expect(btnLonger.disabled).toBe(false);
    mockGenerate.mockResolvedValueOnce({ text: 'Extended analysis of 2... Nc6.' });
    btnLonger.click();
    await new Promise((r) => setTimeout(r, 40));

    expect(btnLonger.disabled).toBe(true);
    expect(btnLonger.textContent).toBe('Longer commentary');
    expect(btnLonger.title).toContain('already been generated');
    expect(btnLonger.classList.contains('active-mode')).toBe(true);

    // Vote up, then click again to toggle off
    const btnUp = document.getElementById('btn-rate-up') as HTMLButtonElement;
    btnUp.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(btnUp.getAttribute('aria-pressed')).toBe('true');
    expect(mockSaveRating).toHaveBeenCalledTimes(1);

    btnUp.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(btnUp.getAttribute('aria-pressed')).toBe('false');
    expect(mockDeleteRating).toHaveBeenCalledTimes(1);
  });

  it('handles explicit model download progress, mid-download Explain Move protection, and download failure', async () => {
    mockIsModelLoaded.mockReturnValue(false);
    mockIsModelCached.mockResolvedValue(false);

    let rejectDownload: (err: Error) => void = () => {};
    mockLoadModel.mockImplementationOnce(
      (_m: any, onProgress: any) =>
        new Promise((_resolve, reject) => {
          onProgress({ phase: 'downloading', percent: 45, loadedBytes: 45, totalBytes: 100, message: '45% downloaded' });
          rejectDownload = reject;
        })
    );

    const btnDownload = document.getElementById('btn-download-model') as HTMLButtonElement;
    const progressTrack = document.getElementById('download-progress-track')!;
    const progressText = document.getElementById('download-progress-text')!;
    btnDownload.click();
    await new Promise((r) => setTimeout(r, 20));

    expect(btnDownload.disabled).toBe(true);
    expect(progressTrack.style.display).toBe('block');
    expect(progressTrack.getAttribute('aria-valuenow')).toBe('45');
    expect(progressText.textContent).toBe('45% downloaded');

    // Clicking Explain Move mid-download must NOT wipe the progress bar
    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(progressTrack.style.display).toBe('block');
    expect(btnDownload.disabled).toBe(true);

    // Simulate download failure -> error shown and download button re-enabled
    rejectDownload(new Error('Network offline'));
    await new Promise((r) => setTimeout(r, 20));
    expect(progressText.textContent).toContain('Download failed: Network offline');
    expect(btnDownload.disabled).toBe(false);
    expect(document.getElementById('status-indicator')!.textContent).toBe('Download error');

    // Retry succeeds -> banner hides
    mockLoadModel.mockResolvedValueOnce(undefined);
    btnDownload.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(document.getElementById('model-banner')!.style.display).toBe('none');
    expect(document.getElementById('status-indicator')!.textContent).toBe('Ready');
  });

  it('supports inline delete-weights confirmation and modal Tab/Shift+Tab focus trapping', async () => {
    const btnOpen = document.getElementById('btn-settings-open') as HTMLButtonElement;
    const modal = document.getElementById('settings-modal')!;
    const btnDelete = document.getElementById('btn-delete-model') as HTMLButtonElement;
    const confirmRow = document.getElementById('delete-confirm-row')!;
    const btnNo = document.getElementById('btn-confirm-delete-no') as HTMLButtonElement;
    const btnYes = document.getElementById('btn-confirm-delete-yes') as HTMLButtonElement;
    const settingLevel = document.getElementById('setting-level') as HTMLSelectElement;

    btnOpen.click();
    expect(modal.classList.contains('open')).toBe(true);

    // Force layout-visible offsetParent in jsdom for focusables check
    const modalButtons = Array.from(
      modal.querySelectorAll<HTMLElement>('button, select, input')
    );
    for (const el of modalButtons) {
      Object.defineProperty(el, 'offsetParent', { configurable: true, get: () => modal });
    }

    const btnSettingsClose = document.getElementById('btn-settings-close') as HTMLButtonElement;

    // Shift+Tab from first focusable (close button) wraps to last visible focusable (delete button)
    btnSettingsClose.focus();
    const shiftTabEvt = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, cancelable: true });
    document.dispatchEvent(shiftTabEvt);
    expect(shiftTabEvt.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(btnDelete);

    // Tab from last visible focusable wraps back to first
    const tabEvt = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: false, cancelable: true });
    document.dispatchEvent(tabEvt);
    expect(tabEvt.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(btnSettingsClose);

    // Inline delete confirmation: Cancel preserves cache
    btnDelete.click();
    expect(confirmRow.style.display).toBe('flex');
    expect(document.activeElement).toBe(btnNo);

    btnNo.click();
    expect(confirmRow.style.display).toBe('none');
    expect(mockDeleteModelCache).not.toHaveBeenCalled();

    // Confirm deletes cache and closes modal
    btnDelete.click();
    btnYes.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(mockDeleteModelCache).toHaveBeenCalledTimes(1);
    expect(modal.classList.contains('open')).toBe(false);
    expect(document.getElementById('model-banner')!.style.display).toBe('flex');
  });

  it('falls back across currentWindow when lastFocusedWindow returns no tabs during revalidation', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    (globalThis as any).chrome.tabs.query = vi.fn((q: any, cb?: (tabs: any[]) => void) => {
      const tabs = q.lastFocusedWindow
        ? []
        : [{ id: 1, url: 'https://lichess.org/analysis' }];
      if (cb) cb(tabs);
      return Promise.resolve(tabs);
    });

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 40));

    expect(mockGenerate).toHaveBeenCalledTimes(1);
    expect(document.getElementById('lock-banner')!.style.display).toBe('none');
  });

  it('fails closed when tab query returns no active tab', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    // Simulate query returning empty tabs array
    (globalThis as any).chrome.tabs.query = vi.fn(() => Promise.resolve([]));

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 20));

    expect(mockGenerate).not.toHaveBeenCalled();
    expect(document.getElementById('lock-banner')!.style.display).toBe('flex');
    expect(document.getElementById('lock-banner-text')!.textContent).toContain('No active Lichess tab found');
  });

  it('ignores LICHESS_STATE_UPDATE messages from background or non-active tabs', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 40));
    expect(document.getElementById('output-text')!.textContent).toContain('Nd4');

    // Inactive background tab or different window active tab emits update
    for (const listener of messageListeners) {
      listener(
        {
          type: 'LICHESS_STATE_UPDATE',
          payload: {
            ...tabResponse,
            isAllowed: false,
            lockReason: 'Second is off during live games.',
            pageType: 'game',
          },
        },
        { tab: { id: 99, active: false } }
      );
      listener(
        {
          type: 'LICHESS_STATE_UPDATE',
          payload: {
            ...tabResponse,
            moves: ['d4'],
            selectedMoveSan: 'd4',
            activePly: 1,
          },
        },
        { tab: { id: 99, active: true } }
      );
    }

    expect(document.getElementById('selected-move-pill')!.textContent).toBe('2... Nc6');
    expect(document.getElementById('output-text')!.textContent).toContain('Nd4');
    expect(document.getElementById('lock-banner')!.style.display).toBe('none');
  });

  it('fails closed in queryActiveTab() when chrome.tabs.query returns no tabs on activation', async () => {
    (globalThis as any).chrome.tabs.query = vi.fn((_q: any, cb?: (tabs: any[]) => void) => {
      if (cb) cb([]);
      return Promise.resolve([]);
    });

    for (const fn of onActivatedListeners) {
      fn({ tabId: 999 });
    }
    await new Promise((r) => setTimeout(r, 20));

    expect(document.getElementById('lock-banner')!.style.display).toBe('flex');
    expect(document.getElementById('lock-banner-text')!.textContent).toContain('No active Lichess tab found');
    expect((document.getElementById('btn-explain-move') as HTMLButtonElement).disabled).toBe(true);
  });

  it('recovers freshly navigated allowed page during revalidation by injecting content.js without locking', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    let injected = false;
    (globalThis as any).chrome.scripting.executeScript = vi.fn(async () => {
      injected = true;
      return [];
    });
    (globalThis as any).chrome.tabs.sendMessage = vi.fn(async () => {
      if (!injected) throw new Error('Receiving end does not exist');
      return {
        ...tabResponse,
        url: 'https://lichess.org/analysis#4',
      };
    });

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 40));

    expect(chrome.scripting.executeScript).toHaveBeenCalledTimes(1);
    expect(mockGenerate).toHaveBeenCalledTimes(1);
    expect(document.getElementById('lock-banner')!.style.display).toBe('none');
  });

  it('restores modal close focus to btnSettingsOpen if trigger was disabled while modal was open', () => {
    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    const btnSettingsOpen = document.getElementById('btn-settings-open') as HTMLButtonElement;

    btnExplain.focus();
    btnSettingsOpen.click();

    // Simulate page locking and disabling btnExplain while modal is open
    btnExplain.disabled = true;
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(document.activeElement).toBe(btnSettingsOpen);
  });

  it('does not invalidate commentary when FEN hydrates from undefined to defined for the same move', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    const baseWithoutFen = { ...tabResponse };
    delete baseWithoutFen.fen;

    for (const listener of messageListeners) {
      listener({ type: 'LICHESS_STATE_UPDATE', payload: baseWithoutFen });
    }

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 40));
    expect(document.getElementById('output-text')!.textContent).toContain('Nd4');

    // DOM content script subsequently hydrates FEN for the identical move/ply
    for (const listener of messageListeners) {
      listener({
        type: 'LICHESS_STATE_UPDATE',
        payload: {
          ...baseWithoutFen,
          fen: 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
        },
      });
    }

    expect(document.getElementById('output-text')!.textContent).toContain('Nd4');
  });

  it('passes maxTokens budgets for short and Longer commentary', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    // 1. Short -> 256
    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(mockGenerate).toHaveBeenLastCalledWith(
      expect.any(String),
      expect.any(Function),
      expect.objectContaining({ maxTokens: 256 })
    );

    // 2. Long -> 450
    const btnLonger = document.getElementById('btn-more-detail') as HTMLButtonElement;
    btnLonger.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(mockGenerate).toHaveBeenLastCalledWith(
      expect.any(String),
      expect.any(Function),
      expect.objectContaining({ maxTokens: 450 })
    );
  });

  it('shows Stop button during generation, aborts evaluation/model on click, and discards late tokens', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    let resolveSlowGen: (v: any) => void = () => {};
    mockGenerate.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveSlowGen = resolve;
        })
    );

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    const btnStop = document.getElementById('btn-stop-generation') as HTMLButtonElement;

    btnExplain.click();
    await new Promise((r) => setTimeout(r, 20));

    expect(btnExplain.style.display).toBe('none');
    expect(btnStop.style.display).toBe('inline-flex');
    expect(mockCloudGetEvaluation).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );

    btnStop.click();
    expect(mockStop).toHaveBeenCalled();
    expect(mockAbortLoad).toHaveBeenCalled();
    expect(btnStop.style.display).toBe('none');
    expect(btnExplain.disabled).toBe(false);
    expect(document.getElementById('output-text')!.textContent).toContain('Generation stopped');

    resolveSlowGen({ text: 'Late stopped output' });
    await new Promise((r) => setTimeout(r, 20));
    expect(document.getElementById('output-text')!.textContent).not.toContain('Late stopped output');
  });

  it('handles activePly === 0 cleanly with -- pill, disabled Explain Move, and clear placeholder', () => {
    for (const listener of messageListeners) {
      listener({
        type: 'LICHESS_STATE_UPDATE',
        payload: {
          ...tabResponse,
          moves: ['e4', 'e5', 'Nf3', 'Nc6'],
          selectedMoveSan: undefined,
          activePly: 0,
        },
      });
    }

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    expect(document.getElementById('selected-move-pill')!.textContent).toBe('--');
    expect(btnExplain.disabled).toBe(true);
    expect(document.getElementById('status-indicator')!.textContent).toBe('Select a move');
    expect(document.getElementById('output-text')!.textContent).toContain('Select a move to explain.');

    // Selecting a move re-enables Explain Move
    for (const listener of messageListeners) {
      listener({
        type: 'LICHESS_STATE_UPDATE',
        payload: {
          ...tabResponse,
          moves: ['e4', 'e5', 'Nf3', 'Nc6'],
          selectedMoveSan: 'e4',
          activePly: 1,
        },
      });
    }

    expect(document.getElementById('selected-move-pill')!.textContent).toBe('1. e4');
    expect(btnExplain.disabled).toBe(false);
  });

  it('invalidates stale commentary when switching between variations with identical ply and SAN', async () => {
    mockIsModelLoaded.mockReturnValue(true);
    mockIsModelCached.mockResolvedValue(true);

    for (const listener of messageListeners) {
      listener({
        type: 'LICHESS_STATE_UPDATE',
        payload: {
          ...tabResponse,
          activeLine: ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6'],
          activePath: 'pathA',
          selectedMoveSan: 'Nf6',
          activePly: 6,
          fen: undefined,
        },
      });
    }

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 30));
    expect(document.getElementById('output-text')!.textContent).toContain('Nd4');

    // Switch to another variation of equal length ending on Nf6
    for (const listener of messageListeners) {
      listener({
        type: 'LICHESS_STATE_UPDATE',
        payload: {
          ...tabResponse,
          activeLine: ['e4', 'e5', 'Bc4', 'Nc6', 'Nf3', 'Nf6'],
          activePath: 'pathB',
          selectedMoveSan: 'Nf6',
          activePly: 6,
          fen: undefined,
        },
      });
    }

    expect(document.getElementById('output-text')!.textContent).not.toContain('Nd4');
    expect((document.getElementById('btn-rate-up') as HTMLButtonElement).disabled).toBe(true);
  });

  it('surfaces weak-device warning and unsupported WebAssembly warning in model banner while 0.8B is hidden', async () => {
    mockCheckHardware.mockResolvedValueOnce({
      webAssemblySupported: true,
      webGpuSupported: false,
      adapterInfo: '',
      isWeakDevice: true,
    });

    document.documentElement.innerHTML = sidepanelHtml;
    await initSidepanel();
    await new Promise((r) => setTimeout(r, 20));

    const deviceNote = document.getElementById('model-banner-device-note')!;
    const btnSwitch = document.getElementById('btn-switch-08b') as HTMLButtonElement;
    const settingModel = document.getElementById('setting-model') as HTMLSelectElement;
    expect(deviceNote.style.display).toBe('flex');
    expect(deviceNote.textContent).toContain('2B');
    expect(btnSwitch.style.display).toBe('none');
    expect(Array.from(settingModel.options).map((o) => o.value)).toEqual(['2B']);

    // Unsupported WebAssembly disables download and shows error note
    mockCheckHardware.mockResolvedValueOnce({
      webAssemblySupported: false,
      webGpuSupported: false,
      adapterInfo: '',
      isWeakDevice: true,
    });

    document.documentElement.innerHTML = sidepanelHtml;
    await initSidepanel();
    await new Promise((r) => setTimeout(r, 20));

    const wasmNote = document.getElementById('model-banner-device-note')!;
    const btnDownload = document.getElementById('btn-download-model') as HTMLButtonElement;
    expect(wasmNote.classList.contains('is-error')).toBe(true);
    expect(wasmNote.textContent).toContain('WebAssembly is disabled or unsupported');
    expect(btnDownload.disabled).toBe(true);
  });

  it('sets Retry Download state on failure, handles empty ratings export, and locks when tab URL is undefined', async () => {
    mockIsModelLoaded.mockReturnValue(false);
    mockIsModelCached.mockResolvedValue(false);
    mockLoadModel.mockRejectedValueOnce(new Error('CDN unreachable'));

    const btnDownload = document.getElementById('btn-download-model') as HTMLButtonElement;
    const progressTrack = document.getElementById('download-progress-track')!;
    btnDownload.click();
    await new Promise((r) => setTimeout(r, 20));

    expect(progressTrack.classList.contains('is-error')).toBe(true);
    expect(btnDownload.textContent).toBe('Retry Download');

    // Empty ratings export reports status without crashing
    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    const btnExport = document.getElementById('btn-export-ratings') as HTMLButtonElement;
    btnExport.click();
    await new Promise((r) => setTimeout(r, 20));
    expect(document.getElementById('settings-feedback')!.textContent).toBe('No ratings saved yet.');

    // Non-Lichess tab with undefined URL (after removing "tabs" permission) fails closed without script injection
    (globalThis as any).chrome.tabs.query = vi.fn((_q: any, cb?: (tabs: any[]) => void) => {
      const tabs = [{ id: 42, active: true }];
      if (cb) cb(tabs);
      return Promise.resolve(tabs);
    });
    for (const fn of onActivatedListeners) {
      fn({ tabId: 42 });
    }
    await new Promise((r) => setTimeout(r, 20));

    expect(document.getElementById('lock-banner')!.style.display).toBe('flex');
    expect(btnExplain.disabled).toBe(true);
    expect(chrome.scripting.executeScript).not.toHaveBeenCalled();
  });

  it('renders Monograph inline brand mark, welcome card, semantic status tone, and keeps button icons on label updates', async () => {
    const brandSvg = document.querySelector('.app-header .brand svg.brand-icon');
    expect(brandSvg).not.toBeNull();
    expect(brandSvg!.getAttribute('viewBox')).toBe('0 0 24 24');
    expect(document.querySelector('.brand-subtitle')?.textContent).toBe('Local second');

    tabResponse = {
      url: 'https://lichess.org/analysis',
      pageType: 'analysis',
      isAllowed: true,
      moves: ['e4'],
      activePly: 0,
      selectedMoveSan: '',
      timestamp: Date.now(),
    };
    await initSidepanel();
    await new Promise((r) => setTimeout(r, 20));

    const status = document.getElementById('status-indicator')!;
    const movePill = document.getElementById('selected-move-pill')!;
    const outputText = document.getElementById('output-text')!;

    expect(status.textContent).toBe('Select a move');
    expect(status.dataset.tone).toBe('idle');
    expect(movePill.classList.contains('is-empty')).toBe(true);
    expect(outputText.querySelector('.welcome-card')).not.toBeNull();
    expect(outputText.querySelector('.welcome-tagline')?.textContent).toBe(
      'Your local chess second.'
    );
    expect(outputText.textContent).toContain('Select a move to explain.');

    mockLoadModel.mockRejectedValueOnce(new Error('Offline'));
    const btnDownload = document.getElementById('btn-download-model') as HTMLButtonElement;
    btnDownload.click();
    await new Promise((r) => setTimeout(r, 20));

    expect(status.dataset.tone).toBe('error');
    expect(btnDownload.querySelector('svg')).not.toBeNull();
    expect(btnDownload.textContent).toBe('Retry Download');
  });

  it('loads cached model in parallel with Stockfish (multiPv: 1) and preserves finalized <p> nodes while streaming', async () => {
    mockIsModelLoaded.mockReturnValue(false);
    mockIsModelCached.mockResolvedValue(true);

    let resolveLoad!: () => void;
    let resolveSf!: (val: any) => void;
    const events: string[] = [];

    mockLoadModel.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          events.push('load_started');
          resolveLoad = resolve;
        })
    );
    mockStockfishEvaluate.mockImplementation(
      () =>
        new Promise((resolve) => {
          events.push('sf_started');
          resolveSf = resolve;
        })
    );
    mockCloudGetEvaluation.mockImplementation(
      (_fen?: string, opts?: { signal?: AbortSignal }) =>
        new Promise((resolve) => {
          opts?.signal?.addEventListener('abort', () => {
            events.push('cloud_aborted');
            resolve(null);
          });
        })
    );

    let capturedFirstPara: Element | null = null;
    mockGenerate.mockImplementationOnce(
      async (_prompt?: string, onToken?: (t: string, r: boolean) => void) => {
        onToken?.('First paragraph.', false);
        onToken?.('\n\nSecond ', false);
        await new Promise((r) => setTimeout(r, 25));
        capturedFirstPara = document.querySelector('#output-text p.commentary-para');
        onToken?.('paragraph.', false);
        await new Promise((r) => setTimeout(r, 25));
        const sameFirstPara = document.querySelector('#output-text p.commentary-para');
        expect(sameFirstPara).toBe(capturedFirstPara);
        return { text: 'First paragraph.\n\nSecond paragraph.' };
      }
    );

    const btnExplain = document.getElementById('btn-explain-move') as HTMLButtonElement;
    btnExplain.click();
    await new Promise((r) => setTimeout(r, 15));

    // Model load and Stockfish evaluation must both start concurrently before either finishes
    expect(events).toContain('load_started');
    expect(events).toContain('sf_started');
    expect(mockStockfishEvaluate).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ multiPv: 3 })
    );

    // Stockfish winning the race aborts pending Cloud Eval
    resolveSf({ depth: 18, multipv: 3, pvs: [{ pv: ['e2e4'], scoreCp: 35 }] });
    await new Promise((r) => setTimeout(r, 10));
    expect(events).toContain('cloud_aborted');

    // Finish model load and let second eval resolve
    resolveLoad();
    await new Promise((r) => setTimeout(r, 10));
    expect(mockStockfishEvaluate).toHaveBeenLastCalledWith(
      expect.any(String),
      expect.objectContaining({ multiPv: 1 })
    );
    resolveSf({ depth: 18, multipv: 1, pvs: [{ pv: ['e7e5'], scoreCp: 30 }] });
    await new Promise((r) => setTimeout(r, 80));

    expect(capturedFirstPara).not.toBeNull();
    expect(document.querySelectorAll('#output-text p.commentary-para')).toHaveLength(2);
    expect(document.getElementById('status-indicator')!.textContent).toBe('Ready');
  });

  it('prevents slower apiTask fallback from overwriting DOM-selected move in evaluateCurrentUrl', async () => {
    const { evaluatePageAccess } = await import('../src/lock.js');
    const { fetchLichessGameApi } = await import('../src/api/lichessApi.js');

    let resolveSlowApi!: (val: any) => void;
    vi.mocked(evaluatePageAccess).mockResolvedValueOnce({
      allowed: true,
      reason: 'Finished game.',
      pageType: 'game',
      gameId: 'race1234',
    });
    vi.mocked(fetchLichessGameApi).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveSlowApi = resolve;
        })
    );

    tabResponse = {
      url: 'https://lichess.org/race1234',
      pageType: 'game',
      gameId: 'race1234',
      isAllowed: true,
      moves: ['e4', 'e5', 'Nf3', 'Nc6'],
      selectedMoveSan: 'e4',
      activePly: 1,
      timestamp: Date.now(),
    };

    // Trigger evaluateCurrentUrl by failing initial GET_PAGE_STATE, then succeeding after executeScript
    let sendCount = 0;
    (globalThis as any).chrome.scripting = {
      executeScript: vi.fn(async () => []),
    };
    (globalThis as any).chrome.tabs.sendMessage = vi.fn(
      (_tabId: number, _msg: any, cb?: (resp: any) => void) => {
        sendCount++;
        const resp = sendCount === 1 ? undefined : tabResponse;
        if (cb) cb(resp);
        return Promise.resolve(resp);
      }
    );

    await initSidepanel();
    await new Promise((r) => setTimeout(r, 20));
    expect(document.getElementById('selected-move-pill')!.textContent).toBe('1. e4');

    // Slow API finishes after scriptTask already delivered activePly=1; must not overwrite to 2... Nc6
    resolveSlowApi({
      moves: ['e4', 'e5', 'Nf3', 'Nc6'],
      whitePlayer: 'Alice',
      blackPlayer: 'Bob',
    });
    await new Promise((r) => setTimeout(r, 20));
    expect(document.getElementById('selected-move-pill')!.textContent).toBe('1. e4');
  });
});
