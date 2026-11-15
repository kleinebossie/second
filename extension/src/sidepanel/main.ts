/**
 * Side Panel Main UI Controller for Second Chrome Extension.
 *
 * Implements full interactive lifecycle:
 * - Fail-closed fair-play live game locking with monotonic request sequencing.
 * - Explicit user consent gate before downloading local model weights (~1.2 GB / ~508 MB).
 * - Local Stockfish WASM + Polite Lichess Cloud Eval integration.
 * - Streaming move commentary.
 * - Semantic chess token highlighting (Moves, Squares, Pieces) in streamed commentary.
 * - Variation 6A center-aligned header (`24... Nd4 · Blunder` + `Second 2B` / `Second 0.8B`).
 * - Floating island dock with ghost-to-pill "Longer" button and naked vector thumbs.
 * - Accessible modal dialog with focus trapping, Escape handling, and focus restoration.
 */

import type { ModelRunner, HardwareSupport } from '../runtime/modelRunner.js';
import { StockfishEngine } from '../engine/stockfish.js';
import { CloudEvalClient } from '../engine/cloudEval.js';
import {
  evaluatePageAccess,
  getCachedFinishedGameData,
  LOCK_OFF_MESSAGE,
} from '../lock.js';
import {
  loadSettings,
  saveSettings,
  DEFAULT_SETTINGS,
  ExtensionSettings,
} from '../storage/settings.js';

type CoachLevel = ExtensionSettings['level'];
type ModelSize = ExtensionSettings['modelSize'];
import {
  saveRating,
  deleteRating,
  exportRatingsAsJsonBlob,
  UserRating,
} from '../storage/ratings.js';
import type { Verdict } from '../core/promptBuilder.js';
import { resolveMoveContext, MoveContext } from '../core/moveResolution.js';
import { fetchLichessGameApi } from '../api/lichessApi.js';
import type { LichessGameState, EngineEval } from '../types.js';
import {
  formatCommentaryHtml,
  formatParagraphInnerHtml,
  formatMoveWithNumber,
  escapeHtml,
} from './highlighter.js';

export { formatMoveWithNumber };

// Controller Singletons (ModelRunner and promptBuilder are lazy-loaded to keep initial UI parse fast)
const stockfish = new StockfishEngine();
const cloudEval = new CloudEvalClient();

let modelRunnerModulePromise: Promise<typeof import('../runtime/modelRunner.js')> | null = null;
let modelRunnerInstance: ModelRunner | null = null;
let promptBuilderModulePromise: Promise<typeof import('../core/promptBuilder.js')> | null = null;

function getModelRunnerModule(): Promise<typeof import('../runtime/modelRunner.js')> {
  if (!modelRunnerModulePromise) {
    modelRunnerModulePromise = import('../runtime/modelRunner.js');
  }
  return modelRunnerModulePromise;
}

async function getModelRunner(): Promise<ModelRunner> {
  if (modelRunnerInstance) return modelRunnerInstance;
  const mod = await getModelRunnerModule();
  if (!modelRunnerInstance) {
    modelRunnerInstance = new mod.ModelRunner();
  }
  return modelRunnerInstance;
}

function getPromptBuilderModule(): Promise<typeof import('../core/promptBuilder.js')> {
  if (!promptBuilderModulePromise) {
    promptBuilderModulePromise = import('../core/promptBuilder.js');
  }
  return promptBuilderModulePromise;
}

let inFlightCacheCheck: Promise<boolean> | null = null;
let inFlightCacheModel: ModelSize | null = null;

async function checkModelCachedDedup(targetModel: ModelSize): Promise<boolean> {
  const runner = await getModelRunner();
  if (runner.isModelLoaded() && runner.getCurrentModel() === targetModel) {
    return true;
  }
  if (inFlightCacheCheck && inFlightCacheModel === targetModel) {
    return inFlightCacheCheck;
  }
  inFlightCacheModel = targetModel;
  const p = runner.isModelCached(targetModel).finally(() => {
    if (inFlightCacheCheck === p) {
      inFlightCacheCheck = null;
      inFlightCacheModel = null;
    }
  });
  inFlightCacheCheck = p;
  return p;
}

let currentSettings: ExtensionSettings = { ...DEFAULT_SETTINGS };
let currentLichessState: LichessGameState | null = null;
let settingsReady = false;
let isGenerating = false;
let isDownloadingModel = false;
let isSavingRating = false;
let stateEvalSequenceId = 0;
let generationEpoch = 0;
let generationAbortController: AbortController | null = null;
let activeTabId: number | null = null;
let activeStreamCancel: (() => void) | null = null;
let lastHardwareSupport: HardwareSupport | null = null;
let hasMoveSelected = false;
let lastResolvedSan: string | undefined = undefined;
let lastResolvedPly: number | undefined = undefined;
let lastResolvedPath: string | undefined = undefined;
let lastResolvedLineKey: string | undefined = undefined;
let lastWarmupError: unknown = null;

export interface ExplanationRecord {
  id: string;
  fen: string;
  move: string;
  prompt: string;
  text: string;
  level: CoachLevel;
  detail: 'short' | 'long';
  model: ModelSize;
}

let lastExplanation: ExplanationRecord | null = null;
let currentVote: 'up' | 'down' | null = null;

// DOM Elements
let appShell = document.getElementById('app-shell')!;
let lockBanner = document.getElementById('lock-banner')!;
let lockBannerTitle = document.getElementById('lock-banner-title')!;
let lockBannerText = document.getElementById('lock-banner-text')!;
let modelBanner = document.getElementById('model-banner')!;
let modelBannerDeviceNote = document.getElementById('model-banner-device-note')!;
let modelBannerDeviceText = document.getElementById('model-banner-device-text')!;
let btnSwitch08b = document.getElementById('btn-switch-08b') as HTMLButtonElement;
let btnDownloadModel = document.getElementById('btn-download-model') as HTMLButtonElement;
let downloadProgressTrack = document.getElementById('download-progress-track')!;
let downloadProgressFill = document.getElementById('download-progress-fill')!;
let downloadProgressText = document.getElementById('download-progress-text')!;

let selectedMovePill = document.getElementById('selected-move-pill')!;
let verdictSep = document.getElementById('verdict-sep')!;
let selectedMoveVerdict = document.getElementById('selected-move-verdict')!;
let modelBadge = document.getElementById('model-badge')!;

let btnExplainMove = document.getElementById('btn-explain-move') as HTMLButtonElement;
let btnStopGeneration = document.getElementById('btn-stop-generation') as HTMLButtonElement;
let btnMoreDetail = document.getElementById('btn-more-detail') as HTMLButtonElement;

let statusIndicator = document.getElementById('status-indicator')!;
let proseScrollContainer = document.getElementById('prose-scroll-container')!;
let outputText = document.getElementById('output-text')!;

let btnRateUp = document.getElementById('btn-rate-up') as HTMLButtonElement;
let btnRateDown = document.getElementById('btn-rate-down') as HTMLButtonElement;

// Settings Modal Elements
let settingsModal = document.getElementById('settings-modal')!;
let btnSettingsOpen = document.getElementById('btn-settings-open') as HTMLButtonElement;
let btnSettingsClose = document.getElementById('btn-settings-close') as HTMLButtonElement;
let settingLevel = document.getElementById('setting-level') as HTMLSelectElement;
let settingModel = document.getElementById('setting-model') as HTMLSelectElement;
let settingCloudEval = document.getElementById('setting-cloud-eval') as HTMLInputElement;
let btnExportRatings = document.getElementById('btn-export-ratings') as HTMLButtonElement;
let btnDeleteModel = document.getElementById('btn-delete-model') as HTMLButtonElement;
let deleteConfirmRow = document.getElementById('delete-confirm-row')!;
let btnConfirmDeleteYes = document.getElementById('btn-confirm-delete-yes') as HTMLButtonElement;
let btnConfirmDeleteNo = document.getElementById('btn-confirm-delete-no') as HTMLButtonElement;
let settingsFeedback = document.getElementById('settings-feedback')!;
let hwStatusInfo = document.getElementById('hw-status-info')!;

let previouslyFocusedElement: HTMLElement | null = null;

/**
 * Updates the center-aligned verdict text (`· Blunder`) in the 6A header.
 */
function setHeaderVerdict(verdict?: Verdict | null): void {
  if (!verdict) {
    verdictSep.style.display = 'none';
    selectedMoveVerdict.style.display = 'none';
    selectedMoveVerdict.textContent = '';
    selectedMoveVerdict.className = 'verdict-label';
    return;
  }

  verdictSep.style.display = 'inline';
  selectedMoveVerdict.style.display = 'inline-block';
  selectedMoveVerdict.textContent = verdict;
  const slug = verdict.toLowerCase().replace(/\s+/g, '-');
  selectedMoveVerdict.className = `verdict-label verdict-${slug}`;
}

/**
 * Updates the thumbs up/down button states and `aria-pressed` attributes.
 */
function setRatingVoteState(vote: 'up' | 'down' | null, enabled: boolean): void {
  btnRateUp.disabled = !enabled;
  btnRateDown.disabled = !enabled;

  if (vote === 'up') {
    btnRateUp.className = 'btn-thumb up active-up';
    btnRateUp.setAttribute('aria-pressed', 'true');
    btnRateDown.className = 'btn-thumb down';
    btnRateDown.setAttribute('aria-pressed', 'false');
  } else if (vote === 'down') {
    btnRateDown.className = 'btn-thumb down active-down';
    btnRateDown.setAttribute('aria-pressed', 'true');
    btnRateUp.className = 'btn-thumb up';
    btnRateUp.setAttribute('aria-pressed', 'false');
  } else {
    btnRateUp.className = 'btn-thumb up';
    btnRateUp.setAttribute('aria-pressed', 'false');
    btnRateDown.className = 'btn-thumb down';
    btnRateDown.setAttribute('aria-pressed', 'false');
  }
}

/**
 * Updates the "Longer" button state in the floating dock.
 */
function setLongerButtonState(mode: 'idle' | 'ready' | 'longer-complete'): void {
  if (mode === 'idle') {
    btnMoreDetail.disabled = true;
    btnMoreDetail.textContent = 'Longer';
    btnMoreDetail.title = 'Generate a longer, deeper explanation for this move';
    btnMoreDetail.classList.remove('active-mode');
  } else if (mode === 'ready') {
    btnMoreDetail.disabled = isGenerating || !currentLichessState?.isAllowed;
    btnMoreDetail.textContent = 'Longer';
    btnMoreDetail.title = 'Generate a longer, deeper explanation for this move';
    btnMoreDetail.classList.remove('active-mode');
  } else {
    btnMoreDetail.disabled = true;
    btnMoreDetail.textContent = 'Longer commentary';
    btnMoreDetail.title = 'Longer commentary has already been generated for this move';
    btnMoreDetail.classList.add('active-mode');
  }
}

const BRAND_EMBLEM_SVG = `<svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
  <path fill="#e5c07b" d="M 11.2 2.2 L 12.8 3.9 C 16.8 4.4, 19.2 7.2, 18.8 11.0 C 18.4 13.9, 16.0 15.9, 11.8 17.5 L 6.1 17.5 C 7.0 15.0, 10.4 12.9, 12.9 10.4 C 14.0 9.3, 13.5 7.6, 11.8 7.3 C 10.5 7.1, 9.4 7.9, 8.7 9.2 L 6.3 10.7 C 5.1 11.1, 4.1 10.1, 4.7 8.8 L 7.0 5.8 C 7.8 4.7, 9.0 4.0, 10.4 3.8 Z"/>
  <circle cx="8.5" cy="6.8" r="0.65" fill="#16181e"/>
  <rect x="5.7" y="18.8" width="10.1" height="2.2" rx="0.7" fill="#f1ebd9"/>
  <rect x="16.7" y="18.8" width="2.2" height="2.2" rx="0.7" fill="#34d399"/>
</svg>`;

type StatusTone = 'ready' | 'busy' | 'locked' | 'error' | 'idle';

function setStatusIndicator(text: string, tone: StatusTone = 'idle'): void {
  statusIndicator.textContent = text;
  statusIndicator.dataset.tone = tone;
}

function setDownloadButtonText(label: string): void {
  const span = btnDownloadModel.querySelector('span');
  if (span) {
    span.textContent = label;
  } else {
    btnDownloadModel.textContent = label;
  }
}

/**
 * Renders an accessible welcome or notice placeholder into `#output-text`.
 */
function renderPlaceholder(
  message: string,
  variant: 'welcome' | 'notice' = 'welcome'
): void {
  outputText.setAttribute('aria-live', 'polite');
  outputText.setAttribute('aria-busy', 'false');
  if (variant === 'notice') {
    outputText.innerHTML = `<div class="commentary-notice"><p class="commentary-placeholder">${escapeHtml(message)}</p></div>`;
    return;
  }
  outputText.innerHTML = `<div class="welcome-card">
    <div class="welcome-emblem" aria-hidden="true">${BRAND_EMBLEM_SVG}</div>
    <div class="welcome-tagline">Your local chess second.</div>
    <p class="commentary-placeholder">${escapeHtml(message)}</p>
    <div class="welcome-pills" aria-hidden="true">
      <span class="welcome-pill"><span class="welcome-dot is-emerald"></span>On-device AI</span>
      <span class="welcome-pill"><span class="welcome-dot is-gold"></span>Post-game only</span>
    </div>
  </div>`;
}

/**
 * Renders an accessible loading indicator into `#output-text`.
 */
function renderLoadingState(stageText: string): void {
  outputText.setAttribute('aria-live', 'polite');
  outputText.setAttribute('aria-busy', 'true');
  outputText.innerHTML = `<div class="commentary-loading" role="status"><span class="loading-dot" aria-hidden="true"></span><span>${escapeHtml(stageText)}</span></div>`;
}

/**
 * Renders an accessible error alert into `#output-text`.
 */
function renderErrorState(message: string): void {
  outputText.setAttribute('aria-live', 'polite');
  outputText.setAttribute('aria-busy', 'false');
  outputText.innerHTML = `<div class="commentary-error" role="alert">${escapeHtml(message)}</div>`;
}

/**
 * Renders raw commentary text into `#output-text` using the semantic chess highlighter.
 */
function renderCommentary(rawText: string): void {
  if (!rawText || !rawText.trim()) {
    const variant =
      currentLichessState && !currentLichessState.isAllowed ? 'notice' : 'welcome';
    renderPlaceholder(getDefaultPlaceholderText(), variant);
    return;
  }
  outputText.innerHTML = formatCommentaryHtml(rawText);
}

function isLichessUrl(rawUrl?: string): boolean {
  if (!rawUrl) return false;
  try {
    const hostname = new URL(rawUrl).hostname.toLowerCase();
    return hostname === 'lichess.org' || hostname.endsWith('.lichess.org');
  } catch {
    return false;
  }
}

function syncActionControls(): void {
  if (isGenerating) {
    btnExplainMove.disabled = true;
    btnExplainMove.style.display = 'none';
    btnStopGeneration.style.display = 'inline-flex';
    btnStopGeneration.disabled = false;
  } else {
    btnStopGeneration.style.display = 'none';
    btnExplainMove.style.display = '';
    btnExplainMove.disabled = !currentLichessState?.isAllowed || !hasMoveSelected;
  }

  const lockModelSettings = isGenerating || isDownloadingModel;
  settingModel.disabled = lockModelSettings;
  btnDeleteModel.disabled = lockModelSettings;
}

function getDefaultPlaceholderText(): string {
  if (currentLichessState && !currentLichessState.isAllowed) {
    if (currentLichessState.pageType === 'game') {
      return 'Open this game after it ends to review moves.';
    }
    return 'Open a finished game, analysis board, or study on Lichess to review moves.';
  }
  if (currentLichessState?.isAllowed && !hasMoveSelected) {
    return 'Select a move to explain.';
  }
  return 'Select any move on a finished Lichess game or analysis board, then click Explain Move.';
}

/**
 * Aborts any in-flight position evaluation or model generation and restores interactive controls.
 */
function abortActiveGeneration(options: { userStopped?: boolean } = {}): void {
  generationEpoch++;
  if (activeStreamCancel) {
    activeStreamCancel();
    activeStreamCancel = null;
  }
  if (generationAbortController) {
    generationAbortController.abort();
    generationAbortController = null;
  }
  modelRunnerInstance?.stop();
  isGenerating = false;
  outputText.setAttribute('aria-live', 'polite');
  outputText.setAttribute('aria-busy', 'false');
  syncActionControls();
  if (!isDownloadingModel) {
    if (!currentLichessState?.isAllowed) {
      setStatusIndicator('Locked', 'locked');
    } else {
      setStatusIndicator(
        hasMoveSelected ? 'Ready' : 'Select a move',
        hasMoveSelected ? 'ready' : 'idle'
      );
    }
  }
  if (options.userStopped) {
    lastExplanation = null;
    currentVote = null;
    setHeaderVerdict(null);
    setRatingVoteState(null, false);
    setLongerButtonState('idle');
    renderPlaceholder('Generation stopped. Click Explain Move to try again.', 'notice');
    if (!btnExplainMove.disabled) {
      btnExplainMove.focus();
    }
  }
}

/**
 * Clears any existing commentary, verdict, and rating state
 * when the selected move, model, or page access changes.
 */
function invalidateCurrentCommentary(): void {
  lastExplanation = null;
  currentVote = null;
  setHeaderVerdict(null);
  setRatingVoteState(null, false);
  setLongerButtonState('idle');
  modelBadge.textContent = `Second ${currentSettings.modelSize}`;
  const variant =
    currentLichessState && !currentLichessState.isAllowed ? 'notice' : 'welcome';
  renderPlaceholder(getDefaultPlaceholderText(), variant);
}

/**
 * Checks whether a scroll container is currently near the bottom so streaming
 * only auto-scrolls when the user has not scrolled up to read earlier lines.
 */
function isNearBottom(el: HTMLElement, thresholdPx = 42): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= thresholdPx;
}

/**
 * Updates the download progress bar and throttles screen-reader live status updates.
 */
let lastAnnouncedPercent = -1;
function updateDownloadProgressUi(percent: number, message: string, phase: string): void {
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));
  downloadProgressTrack.style.display = 'block';
  downloadProgressTrack.setAttribute('aria-valuenow', String(clamped));
  downloadProgressTrack.setAttribute('aria-valuetext', `${clamped}% - ${message}`);
  downloadProgressFill.style.width = `${Math.max(clamped, 3)}%`;
  downloadProgressText.style.display = 'block';
  downloadProgressText.textContent = message;

  // Throttle aria-live polite announcements to 10% increments to avoid flooding screen readers
  if (clamped === 0 || clamped === 100 || Math.abs(clamped - lastAnnouncedPercent) >= 10) {
    lastAnnouncedPercent = clamped;
    setStatusIndicator(`${phase === 'downloading' ? 'Downloading' : 'Loading'} ${clamped}%`, 'busy');
  }
}

/**
 * Initialize application with parallel startup pipelining.
 */
async function init(): Promise<void> {
  setupEventListeners();

  // Attach Chrome listeners immediately so tab/message events are never missed during async startup
  if (typeof chrome !== 'undefined') {
    if (chrome.runtime?.onMessage) {
      chrome.runtime.onMessage.addListener((message, sender) => {
        if (message.type === 'LICHESS_STATE_UPDATE') {
          if (sender?.tab) {
            if (sender.tab.active === false) return;
            if (
              activeTabId !== null &&
              sender.tab.id !== undefined &&
              sender.tab.id !== activeTabId
            ) {
              return;
            }
          }
          stateEvalSequenceId++;
          updateLichessState(message.payload);
        }
      });
    }

    if (chrome.tabs?.onActivated) {
      chrome.tabs.onActivated.addListener((activeInfo) => {
        if (typeof activeInfo?.tabId === 'number') {
          activeTabId = activeInfo.tabId;
        }
        queryActiveTab();
      });
    }

    if (chrome.tabs?.onUpdated) {
      chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
        if (tab && tab.active === false && (activeTabId === null || tabId !== activeTabId)) {
          return;
        }
        if (changeInfo.status === 'complete' || changeInfo.url) {
          queryActiveTab();
        }
      });
    }
  }

  // Launch independent startup tasks concurrently
  stockfish.init().catch((err) => console.debug('Stockfish warm-up failed:', err));
  const queryTabPromise = queryActiveTab();
  const modelModPromise = getModelRunnerModule();
  // Preload promptBuilder chunk asynchronously in background without blocking initial UI
  void getPromptBuilderModule().catch(() => {});

  const settingsPromise = loadSettings()
    .then(async (loaded) => {
      // Keep v1.0 locked to 2B until Second-1.1-0.8B GGUF is published on Hugging Face.
      currentSettings = loaded.modelSize === '0.8B' ? await saveSettings({ modelSize: '2B' }) : loaded;
      applySettingsToUi();
    })
    .catch((err) => {
      console.debug('Failed to load settings:', err);
    })
    .finally(() => {
      settingsReady = true;
    });

  const hardwarePromise = modelModPromise
    .then((mod) => mod.ModelRunner.checkHardware())
    .then((hw) => {
      lastHardwareSupport = hw;
      let hwSummary = `WASM: ${hw.webAssemblySupported ? 'Supported' : 'Unavailable'} | WebGPU: ${hw.webGpuSupported ? 'Active' : 'CPU Wasm fallback'}`;
      if (hw.adapterInfo) hwSummary += ` (${hw.adapterInfo})`;
      if (hw.isWeakDevice) hwSummary += ' [Weak Device detected]';
      hwStatusInfo.textContent = `Hardware: ${hwSummary}`;
      updateModelBannerDeviceNote();
      if (!hw.webAssemblySupported) {
        modelBanner.style.display = 'flex';
      }
    })
    .catch((err) => {
      console.debug('Hardware check error:', err);
    });

  // Once settings + ModelRunner module resolve, run one deduplicated cache check for banner & warm-up
  const bannerAndWarmupPromise = Promise.all([settingsPromise, modelModPromise]).then(async () => {
    const cached = await checkModelBanner();
    if (cached) {
      void warmUpModel({ knownCached: true });
    }
  });

  await Promise.all([settingsPromise, hardwarePromise, bannerAndWarmupPromise, queryTabPromise]);
}

function applySettingsToUi(): void {
  settingLevel.value = currentSettings.level;
  settingModel.value = currentSettings.modelSize;
  settingCloudEval.checked = currentSettings.cloudEval;
  if (!lastExplanation) {
    modelBadge.textContent = `Second ${currentSettings.modelSize}`;
  }
  cloudEval.setEnabled(currentSettings.cloudEval);
}

function updateModelBannerDeviceNote(): void {
  if (lastHardwareSupport && !lastHardwareSupport.webAssemblySupported) {
    modelBannerDeviceNote.style.display = 'flex';
    modelBannerDeviceNote.classList.add('is-error');
    modelBannerDeviceText.textContent =
      'WebAssembly is disabled or unsupported in this browser. Local AI inference cannot run.';
    btnSwitch08b.style.display = 'none';
    btnDownloadModel.disabled = true;
    return;
  }

  modelBannerDeviceNote.classList.remove('is-error');
  btnSwitch08b.style.display = 'none';
  if (lastHardwareSupport?.isWeakDevice) {
    // Re-enable btnSwitch08b and 0.8B prompt once Second-1.1-0.8B GGUF weights are published.
    modelBannerDeviceNote.style.display = 'flex';
    modelBannerDeviceText.textContent =
      'Lightweight device detected — local 2B (~1.2 GB) inference may run slowly on this device.';
  } else {
    modelBannerDeviceNote.style.display = 'none';
  }
}

function checkModelBanner(): Promise<boolean> {
  const targetModel = currentSettings.modelSize;
  const modelDesc = document.getElementById('model-banner-desc');
  if (modelDesc) {
    const approxSize = targetModel === '0.8B' ? '~508 MB' : '~1.2 GB';
    modelDesc.textContent = `Download the local Second ${targetModel} coaching weights (${approxSize}) to generate move explanations offline.`;
  }

  if (isDownloadingModel) {
    modelBanner.style.display = 'flex';
    updateModelBannerDeviceNote();
    return Promise.resolve(false);
  }

  downloadProgressTrack.classList.remove('is-error');
  downloadProgressTrack.style.display = 'none';
  downloadProgressText.style.display = 'none';
  setDownloadButtonText('Download Model');
  btnDownloadModel.disabled = lastHardwareSupport ? !lastHardwareSupport.webAssemblySupported : false;
  updateModelBannerDeviceNote();

  if (lastHardwareSupport && !lastHardwareSupport.webAssemblySupported) {
    modelBanner.style.display = 'flex';
    return Promise.resolve(false);
  }

  if (
    modelRunnerInstance?.isModelLoaded() &&
    modelRunnerInstance.getCurrentModel() === targetModel
  ) {
    modelBanner.style.display = 'none';
    return Promise.resolve(true);
  }
  modelBanner.style.display = 'flex';
  return checkModelCachedDedup(targetModel)
    .then((cached) => {
      if (targetModel !== currentSettings.modelSize || isDownloadingModel) {
        return false;
      }
      if (lastHardwareSupport && !lastHardwareSupport.webAssemblySupported) {
        return false;
      }
      if (
        cached ||
        (modelRunnerInstance?.isModelLoaded() &&
          modelRunnerInstance.getCurrentModel() === targetModel)
      ) {
        modelBanner.style.display = 'none';
        return true;
      }
      return false;
    })
    .catch(() => false);
}

let warmupPromise: Promise<void> | null = null;
let warmupTargetModel: ModelSize | null = null;
let warmupUnavailable = false;
let isWarmingModel = false;

/**
 * Loads an already-downloaded model in the background. Never initiates an
 * unconsented network download if the model is not cached.
 */
function warmUpModel(options: { knownCached?: boolean } = {}): Promise<void> {
  if (!settingsReady || isDownloadingModel) return Promise.resolve();
  if (lastHardwareSupport && !lastHardwareSupport.webAssemblySupported) {
    return Promise.resolve();
  }
  const targetModel = currentSettings.modelSize;
  if (
    modelRunnerInstance?.isModelLoaded() &&
    modelRunnerInstance.getCurrentModel() === targetModel
  ) {
    return Promise.resolve();
  }
  if (warmupPromise && warmupTargetModel === targetModel) return warmupPromise;
  if (warmupUnavailable && warmupTargetModel === targetModel) return Promise.resolve();

  warmupTargetModel = targetModel;
  warmupUnavailable = false;
  lastWarmupError = null;

  const currentWarmup = (async () => {
    const runner = await getModelRunner();
    const cached =
      options.knownCached !== undefined
        ? options.knownCached
        : await checkModelCachedDedup(targetModel);
    if (targetModel !== currentSettings.modelSize || isDownloadingModel) {
      return;
    }
    if (!cached) {
      warmupUnavailable = true;
      if (!isGenerating && !isDownloadingModel) {
        if (!currentLichessState?.isAllowed) {
          setStatusIndicator('Locked', 'locked');
        } else {
          setStatusIndicator(
            hasMoveSelected ? 'Ready' : 'Select a move',
            hasMoveSelected ? 'ready' : 'idle'
          );
        }
      }
      return;
    }
    if (
      runner.isModelLoaded() &&
      runner.getCurrentModel() === targetModel
    ) {
      return;
    }

    isWarmingModel = true;
    setStatusIndicator('Warming model...', 'busy');
    lastAnnouncedPercent = -1;
    try {
      await runner.loadModel(targetModel, (p) => {
        if (targetModel !== currentSettings.modelSize || isDownloadingModel) return;
        if (p.phase === 'downloading' || p.phase === 'loading') {
          const clamped = Math.round(p.percent);
          if (Math.abs(clamped - lastAnnouncedPercent) >= 20) {
            lastAnnouncedPercent = clamped;
            setStatusIndicator(`Loading ${clamped}%`, 'busy');
          }
        }
      });
    } finally {
      isWarmingModel = false;
    }
    if (targetModel !== currentSettings.modelSize || isDownloadingModel) {
      return;
    }
    modelBanner.style.display = 'none';
    if (!isGenerating) {
      if (!currentLichessState?.isAllowed) {
        setStatusIndicator('Locked', 'locked');
      } else {
        setStatusIndicator(
          hasMoveSelected ? 'Ready' : 'Select a move',
          hasMoveSelected ? 'ready' : 'idle'
        );
      }
    }
  })()
    .catch((err) => {
      lastWarmupError = err;
      console.debug('Background model warm-up skipped:', err);
    })
    .finally(() => {
      if (warmupPromise === currentWarmup) {
        warmupPromise = null;
      }
    });

  warmupPromise = currentWarmup;
  return currentWarmup;
}

/**
 * Accessible Settings Modal Open / Close / Focus Trap / Escape Key
 */
function openSettingsModal(): void {
  previouslyFocusedElement = document.activeElement as HTMLElement | null;
  deleteConfirmRow.style.display = 'none';
  btnDeleteModel.setAttribute('aria-expanded', 'false');
  settingsFeedback.textContent = '';
  syncActionControls();
  settingsModal.classList.add('open');
  settingsModal.setAttribute('aria-hidden', 'false');
  btnSettingsOpen.setAttribute('aria-expanded', 'true');
  appShell.setAttribute('inert', '');
  appShell.setAttribute('aria-hidden', 'true');
  settingLevel.focus();
}

function closeSettingsModal(): void {
  if (!settingsModal.classList.contains('open')) return;
  settingsModal.classList.remove('open');
  settingsModal.setAttribute('aria-hidden', 'true');
  btnSettingsOpen.setAttribute('aria-expanded', 'false');
  appShell.removeAttribute('inert');
  appShell.removeAttribute('aria-hidden');
  deleteConfirmRow.style.display = 'none';
  btnDeleteModel.setAttribute('aria-expanded', 'false');
  if (
    previouslyFocusedElement &&
    document.contains(previouslyFocusedElement) &&
    !(previouslyFocusedElement as HTMLButtonElement).disabled &&
    typeof previouslyFocusedElement.focus === 'function'
  ) {
    previouslyFocusedElement.focus();
  } else {
    btnSettingsOpen.focus();
  }
}

function handleModalKeydown(e: KeyboardEvent): void {
  if (!settingsModal.classList.contains('open')) return;

  if (e.key === 'Escape') {
    e.preventDefault();
    closeSettingsModal();
    return;
  }

  if (e.key === 'Tab') {
    const focusables = Array.from(
      settingsModal.querySelectorAll<HTMLElement>(
        'button:not(:disabled), select:not(:disabled), input:not(:disabled), [tabindex]:not([tabindex="-1"])'
      )
    )
      .filter((el) => {
        if (el.closest('[style*="display: none"]')) return false;
        if (typeof window !== 'undefined' && typeof window.getComputedStyle === 'function') {
          if (window.getComputedStyle(el).display === 'none') return false;
        }
        return el.offsetParent !== null || el === document.activeElement;
      })
      .sort((a, b) =>
        a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1
      );

    if (focusables.length === 0) return;

    const first = focusables[0];
    const last = focusables[focusables.length - 1];

    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
}

async function switchModelSizeSetting(modelSize: ModelSize): Promise<void> {
  currentSettings = await saveSettings({ modelSize });
  settingModel.value = modelSize;
  if (!lastExplanation) {
    modelBadge.textContent = `Second ${currentSettings.modelSize}`;
  }
  warmupUnavailable = false;
  lastWarmupError = null;
  const cached = await checkModelBanner();
  if (cached) {
    void warmUpModel({ knownCached: true });
  }
}

function setupEventListeners(): void {
  // Settings modal accessibility
  btnSettingsOpen.addEventListener('click', openSettingsModal);
  btnSettingsClose.addEventListener('click', closeSettingsModal);
  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) closeSettingsModal();
  });
  document.removeEventListener('keydown', handleModalKeydown);
  document.addEventListener('keydown', handleModalKeydown);

  settingLevel.addEventListener('change', async () => {
    try {
      const level: CoachLevel = settingLevel.value === 'beginner' ? 'beginner' : 'club';
      currentSettings = await saveSettings({ level });
    } catch {
      settingsFeedback.textContent = 'Could not save coach style.';
    }
  });
  settingModel.addEventListener('change', async () => {
    if (isDownloadingModel || isGenerating) {
      settingModel.value = currentSettings.modelSize;
      return;
    }
    try {
      // Clamp to 2B while 0.8B is disabled; restore '0.8B' handling when v1.1-0.8B ships.
      const modelSize: ModelSize = '2B';
      await switchModelSizeSetting(modelSize);
    } catch {
      settingsFeedback.textContent = 'Could not save model size.';
    }
  });
  settingCloudEval.addEventListener('change', async () => {
    try {
      currentSettings = await saveSettings({ cloudEval: settingCloudEval.checked });
      cloudEval.setEnabled(currentSettings.cloudEval);
    } catch {
      settingsFeedback.textContent = 'Could not save cloud eval setting.';
    }
  });

  btnSwitch08b.addEventListener('click', async () => {
    if (isDownloadingModel || isGenerating) return;
    try {
      await switchModelSizeSetting('0.8B');
    } catch {
      // ignore
    }
  });

  // Explicit Consent Model Download
  btnDownloadModel.addEventListener('click', async () => {
    if (isDownloadingModel) return;
    if (lastHardwareSupport && !lastHardwareSupport.webAssemblySupported) return;
    isDownloadingModel = true;
    const targetModel = currentSettings.modelSize;
    btnDownloadModel.disabled = true;
    setDownloadButtonText('Downloading...');
    btnSwitch08b.disabled = true;
    downloadProgressTrack.classList.remove('is-error');
    syncActionControls();
    lastAnnouncedPercent = -1;
    updateDownloadProgressUi(3, 'Connecting to model download...', 'downloading');

    try {
      const runner = await getModelRunner();
      await runner.loadModel(
        targetModel,
        (p) => {
          updateDownloadProgressUi(p.percent, p.message, p.phase);
        },
        { forceReload: true }
      );
      if (targetModel === currentSettings.modelSize) {
        modelBanner.style.display = 'none';
      }
      btnDownloadModel.disabled = false;
      setDownloadButtonText('Download Model');
      downloadProgressTrack.style.display = 'none';
      downloadProgressText.style.display = 'none';
      if (!isGenerating) {
        if (!currentLichessState?.isAllowed) {
          setStatusIndicator('Locked', 'locked');
        } else {
          setStatusIndicator(
            hasMoveSelected ? 'Ready' : 'Select a move',
            hasMoveSelected ? 'ready' : 'idle'
          );
        }
      }
    } catch (err: any) {
      downloadProgressTrack.classList.add('is-error');
      downloadProgressText.style.display = 'block';
      downloadProgressText.textContent = `Download failed: ${err?.message || err}`;
      setStatusIndicator('Download error', 'error');
      setDownloadButtonText('Retry Download');
      btnDownloadModel.disabled = false;
    } finally {
      isDownloadingModel = false;
      btnSwitch08b.disabled = isGenerating;
      syncActionControls();
      if (targetModel !== currentSettings.modelSize) {
        void checkModelBanner();
      }
    }
  });

  // Inline Delete Model Confirmation (replaces jarring native confirm/alert)
  btnDeleteModel.addEventListener('click', () => {
    if (isGenerating || isDownloadingModel) return;
    deleteConfirmRow.style.display = 'flex';
    btnDeleteModel.setAttribute('aria-expanded', 'true');
    btnConfirmDeleteNo.focus();
  });

  btnConfirmDeleteNo.addEventListener('click', () => {
    deleteConfirmRow.style.display = 'none';
    btnDeleteModel.setAttribute('aria-expanded', 'false');
    btnDeleteModel.focus();
  });

  btnConfirmDeleteYes.addEventListener('click', async () => {
    btnConfirmDeleteYes.disabled = true;
    btnConfirmDeleteNo.disabled = true;
    try {
      const runner = await getModelRunner();
      runner.abortLoad?.();
      await runner.deleteModelCache();
      deleteConfirmRow.style.display = 'none';
      btnDeleteModel.setAttribute('aria-expanded', 'false');
      warmupUnavailable = true;
      lastWarmupError = null;
      invalidateCurrentCommentary();
      void checkModelBanner();
      setStatusIndicator('Weights deleted', 'idle');
      closeSettingsModal();
    } catch (err: any) {
      settingsFeedback.textContent = `Failed to delete cached weights: ${err?.message || err}`;
    } finally {
      btnConfirmDeleteYes.disabled = false;
      btnConfirmDeleteNo.disabled = false;
    }
  });

  // Export ratings
  btnExportRatings.addEventListener('click', async () => {
    settingsFeedback.textContent = '';
    try {
      const blob = await exportRatingsAsJsonBlob();
      let countLabel = '';
      if (typeof blob.text === 'function') {
        try {
          const parsed = JSON.parse(await blob.text());
          const count = Array.isArray(parsed)
            ? parsed.length
            : Array.isArray(parsed?.ratings)
              ? parsed.ratings.length
              : 0;
          if (count === 0) {
            settingsFeedback.textContent = 'No ratings saved yet.';
            return;
          }
          countLabel = ` (${count})`;
        } catch {
          // proceed with download if blob is non-JSON mock
        }
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `second_ratings_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      settingsFeedback.textContent = `Ratings exported${countLabel}.`;
    } catch (err: any) {
      settingsFeedback.textContent = `Export failed: ${err?.message || err}`;
    }
  });

  // Actions
  btnExplainMove.addEventListener('click', () => runExplanation('short'));
  btnStopGeneration.addEventListener('click', () => {
    modelRunnerInstance?.abortLoad?.();
    abortActiveGeneration({ userStopped: true });
  });
  btnMoreDetail.addEventListener('click', () => runExplanation('long'));

  // Ratings
  btnRateUp.addEventListener('click', () => recordRating('up'));
  btnRateDown.addEventListener('click', () => recordRating('down'));
}

/**
 * Query active tab in Chrome with sequence protection against out-of-order responses.
 */
function queryActiveTab(): void {
  if (typeof chrome === 'undefined' || !chrome.tabs?.query) return;
  const seq = ++stateEvalSequenceId;

  const tryQuery = (queryInfo: chrome.tabs.QueryInfo, fallback?: () => void) => {
    chrome.tabs.query(queryInfo, (tabs) => {
      if (seq !== stateEvalSequenceId) return;
      const tab = tabs?.find((t) => isLichessUrl(t.url)) || tabs?.[0];
      if (tab?.id) {
        activeTabId = tab.id;
        if (!tab.url || !isLichessUrl(tab.url)) {
          void evaluateCurrentUrl(tab.url || '', undefined, seq);
          return;
        }
        chrome.tabs.sendMessage(tab.id, { type: 'GET_PAGE_STATE' }, (response) => {
          if (seq !== stateEvalSequenceId) return;
          if (chrome.runtime.lastError || !response) {
            void evaluateCurrentUrl(tab.url || '', tab.id, seq);
            return;
          }
          updateLichessState(response);
        });
      } else if (fallback) {
        fallback();
      } else {
        activeTabId = null;
        updateLichessState({
          url: '',
          pageType: 'other',
          isAllowed: false,
          lockReason: 'No active Lichess tab found.',
          timestamp: Date.now(),
        });
      }
    });
  };

  tryQuery({ active: true, lastFocusedWindow: true }, () => {
    tryQuery({ active: true, currentWindow: true }, () => {
      tryQuery({ active: true });
    });
  });
}

async function evaluateCurrentUrl(url: string, tabId?: number, seq?: number): Promise<void> {
  const activeSeq = seq ?? ++stateEvalSequenceId;
  const evalResult = await evaluatePageAccess(url);
  if (activeSeq !== stateEvalSequenceId) return;

  if (!evalResult.allowed) {
    updateLichessState({
      url,
      pageType: evalResult.pageType,
      gameId: evalResult.gameId,
      isAllowed: false,
      lockReason: evalResult.reason,
      timestamp: Date.now(),
    });
    return;
  }

  let appliedInitialState = false;
  const applyApiDataState = (apiData: Awaited<ReturnType<typeof fetchLichessGameApi>>) => {
    if (!apiData || activeSeq !== stateEvalSequenceId) return;
    const moves = apiData.moves || [];
    let activePly = moves.length;
    try {
      const parsedUrl = new URL(url);
      const hashMatch = parsedUrl.hash.match(/^#(\d+)$/);
      if (hashMatch) {
        const hashPly = parseInt(hashMatch[1], 10);
        if (hashPly >= 0 && hashPly <= moves.length) {
          activePly = hashPly;
        }
      }
    } catch {
      // ignore invalid URL
    }
    const state: LichessGameState = {
      url,
      pageType: evalResult.pageType,
      gameId: evalResult.gameId,
      isAllowed: true,
      moves,
      selectedMoveSan: activePly > 0 ? moves[activePly - 1] : undefined,
      activePly,
      whitePlayer: apiData.whitePlayer,
      blackPlayer: apiData.blackPlayer,
      opening: apiData.opening,
      initialFen: apiData.initialFen,
      variant: apiData.variant,
      timestamp: Date.now(),
    };
    updateLichessState(state);
    appliedInitialState = true;
  };

  // Reuse game data already fetched by checkGameStatus if available, otherwise run API fetch in parallel with script injection
  const cachedApiData = evalResult.gameId ? getCachedFinishedGameData(evalResult.gameId) : undefined;
  if (cachedApiData) {
    applyApiDataState(cachedApiData);
  }

  let gotScriptState = false;

  const apiTask =
    evalResult.gameId && !cachedApiData
      ? fetchLichessGameApi(evalResult.gameId)
          .then((apiData) => {
            if (gotScriptState || activeSeq !== stateEvalSequenceId) return;
            applyApiDataState(apiData);
          })
          .catch((err) => {
            console.debug('Failed to fetch game details from API:', err);
          })
      : Promise.resolve();

  const scriptTask =
    tabId && typeof chrome !== 'undefined' && chrome.scripting?.executeScript
      ? chrome.scripting
          .executeScript({
            target: { tabId },
            files: ['content.js'],
          })
          .then(() => {
            if (activeSeq !== stateEvalSequenceId) return false;
            return new Promise<boolean>((resolve) => {
              chrome.tabs.sendMessage(tabId, { type: 'GET_PAGE_STATE' }, (resp) => {
                if (activeSeq !== stateEvalSequenceId) {
                  resolve(true);
                  return;
                }
                if (!chrome.runtime.lastError && resp) {
                  gotScriptState = true;
                  updateLichessState(resp);
                  resolve(true);
                } else {
                  resolve(false);
                }
              });
            });
          })
          .catch(() => false)
      : Promise.resolve(false);

  await Promise.all([apiTask, scriptTask]);
  if (activeSeq !== stateEvalSequenceId) return;

  if (!gotScriptState && !appliedInitialState) {
    updateLichessState({
      url,
      pageType: evalResult.pageType,
      gameId: evalResult.gameId,
      isAllowed: true,
      moves: [],
      timestamp: Date.now(),
    });
  }
}

/**
 * Locates the active Lichess tab using the same multi-window fallback order as queryActiveTab().
 */
async function findActiveLichessTab(): Promise<chrome.tabs.Tab | undefined> {
  const queries: chrome.tabs.QueryInfo[] = [
    { active: true, lastFocusedWindow: true },
    { active: true, currentWindow: true },
    { active: true },
  ];

  for (const queryInfo of queries) {
    const tabs = await chrome.tabs.query(queryInfo);
    if (tabs && tabs.length > 0) {
      return tabs.find((t) => isLichessUrl(t.url)) || tabs[0];
    }
  }
  return undefined;
}

/**
 * Revalidates the active Lichess tab right before generation.
 * Strictly FAIL-CLOSED for disallowed or live pages, while recovering
 * freshly navigated allowed pages via content script injection or API fallback.
 */
async function revalidateFreshTabState(): Promise<LichessGameState | null> {
  if (typeof chrome === 'undefined' || !chrome.tabs?.query) {
    return currentLichessState;
  }

  try {
    const tab = await findActiveLichessTab();
    if (!tab || !tab.url) {
      activeTabId = null;
      return {
        url: '',
        pageType: 'other',
        isAllowed: false,
        lockReason: tab
          ? 'Navigate to a finished Lichess game, analysis board, or study to use Second.'
          : 'No active Lichess tab found.',
        timestamp: Date.now(),
      };
    }

    if (typeof tab.id === 'number') {
      activeTabId = tab.id;
    }

    if (tab.id && chrome.tabs.sendMessage) {
      try {
        const response: LichessGameState | undefined = await chrome.tabs.sendMessage(tab.id, {
          type: 'GET_PAGE_STATE',
        });
        if (response) {
          return response;
        }
      } catch {
        // Content script unreachable; evaluate access directly below
      }
    }

    // Fail-closed URL/API check when content script did not respond
    const evalResult = await evaluatePageAccess(tab.url || '');
    if (!evalResult.allowed) {
      return {
        url: tab.url || '',
        pageType: evalResult.pageType,
        gameId: evalResult.gameId,
        isAllowed: false,
        lockReason: evalResult.reason,
        timestamp: Date.now(),
      };
    }

    // Attempt to inject content script once on confirmed-allowed pages if missing
    if (tab.id && chrome.scripting?.executeScript && chrome.tabs.sendMessage) {
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['content.js'],
        });
        const retryResp: LichessGameState | undefined = await chrome.tabs.sendMessage(tab.id, {
          type: 'GET_PAGE_STATE',
        });
        if (retryResp) {
          return retryResp;
        }
      } catch {
        // Fall through to API / empty allowed state below
      }
    }

    if (evalResult.gameId) {
      try {
        const apiData =
          getCachedFinishedGameData(evalResult.gameId) ??
          (await fetchLichessGameApi(evalResult.gameId));
        if (apiData) {
          const moves = apiData.moves || [];
          let activePly = moves.length;
          try {
            const parsedUrl = new URL(tab.url);
            const hashMatch = parsedUrl.hash.match(/^#(\d+)$/);
            if (hashMatch) {
              const hashPly = parseInt(hashMatch[1], 10);
              if (hashPly >= 0 && hashPly <= moves.length) {
                activePly = hashPly;
              }
            }
          } catch {
            // ignore
          }
          return {
            url: tab.url,
            pageType: evalResult.pageType,
            gameId: evalResult.gameId,
            isAllowed: true,
            moves,
            selectedMoveSan: activePly > 0 ? moves[activePly - 1] : undefined,
            activePly,
            whitePlayer: apiData.whitePlayer,
            blackPlayer: apiData.blackPlayer,
            opening: apiData.opening,
            initialFen: apiData.initialFen,
            variant: apiData.variant,
            timestamp: Date.now(),
          };
        }
      } catch {
        // ignore API fallback error
      }
    }

    return {
      url: tab.url,
      pageType: evalResult.pageType,
      gameId: evalResult.gameId,
      isAllowed: true,
      moves: [],
      timestamp: Date.now(),
    };
  } catch {
    return {
      url: '',
      pageType: 'other',
      isAllowed: false,
      lockReason: 'Could not inspect active tab.',
      timestamp: Date.now(),
    };
  }
}

/**
 * Builds the move context for the position currently shown on the Lichess
 * board. When the user is inside a variation, `activeLine` contains the exact
 * variation moves.
 */
function resolveStateMoveContext(state: LichessGameState): MoveContext {
  const line =
    state.activeLine && state.activeLine.length > 0 ? state.activeLine : undefined;
  const moves = line ?? state.moves ?? [];
  const activePly = line ? line.length : state.activePly;
  return resolveMoveContext(
    moves,
    activePly,
    state.selectedMoveSan,
    state.fen,
    state.initialFen
  );
}

function updateLichessState(
  state: LichessGameState,
  options: { abortInFlight?: boolean } = {}
): void {
  const abortInFlight = options.abortInFlight ?? true;
  const prevState = currentLichessState;
  const prevAllowed = prevState?.isAllowed;
  const prevGameId = prevState?.gameId;
  const prevFen = prevState?.fen;

  currentLichessState = state;

  if (!state.isAllowed) {
    hasMoveSelected = false;
    lastResolvedSan = undefined;
    lastResolvedPly = undefined;
    lastResolvedPath = undefined;
    lastResolvedLineKey = undefined;
    selectedMovePill.textContent = '--';
    selectedMovePill.classList.add('is-empty');
    applyAccessResult(false, state.lockReason, state.pageType);
    if (abortInFlight && isGenerating) {
      abortActiveGeneration();
    }
    invalidateCurrentCommentary();
    syncActionControls();
    return;
  }

  void warmUpModel();

  const moveCtx = resolveStateMoveContext(state);
  const activePly =
    state.activeLine && state.activeLine.length > 0
      ? state.activeLine.length
      : state.activePly;
  const isStartPlyZero = activePly === 0 && !state.selectedMoveSan;
  const rawSan = isStartPlyZero ? undefined : moveCtx.moveSan || state.selectedMoveSan;
  hasMoveSelected = Boolean(rawSan);

  if (rawSan) {
    selectedMovePill.textContent = formatMoveWithNumber(rawSan, activePly);
    selectedMovePill.classList.remove('is-empty');
  } else {
    selectedMovePill.textContent = '--';
    selectedMovePill.classList.add('is-empty');
  }

  applyAccessResult(true, state.lockReason, state.pageType);

  const nextPath = state.activePath || undefined;
  const nextLineKey =
    state.activeLine && state.activeLine.length > 0 ? state.activeLine.join(',') : undefined;

  // Only treat FEN/path/line differences as a change once a previous value was known,
  // so initial API-then-DOM hydration for the same move does not abort/invalidate commentary.
  const fenChanged =
    prevFen !== undefined && state.fen !== undefined && state.fen !== prevFen;
  const pathChanged =
    lastResolvedPath !== undefined && nextPath !== undefined && nextPath !== lastResolvedPath;
  const lineChanged =
    lastResolvedLineKey !== undefined &&
    nextLineKey !== undefined &&
    nextLineKey !== lastResolvedLineKey;
  const stateChanged =
    rawSan !== lastResolvedSan ||
    activePly !== lastResolvedPly ||
    state.isAllowed !== prevAllowed ||
    state.gameId !== prevGameId ||
    fenChanged ||
    pathChanged ||
    lineChanged;

  lastResolvedSan = rawSan;
  lastResolvedPly = activePly;
  lastResolvedPath = nextPath ?? (stateChanged ? undefined : lastResolvedPath);
  lastResolvedLineKey = nextLineKey ?? (stateChanged ? undefined : lastResolvedLineKey);

  if (stateChanged) {
    if (abortInFlight && isGenerating) {
      abortActiveGeneration();
    }
    invalidateCurrentCommentary();
  }

  syncActionControls();
  if (!isGenerating && !isDownloadingModel && !isWarmingModel) {
    setStatusIndicator(
      hasMoveSelected ? 'Ready' : 'Select a move',
      hasMoveSelected ? 'ready' : 'idle'
    );
  }
}

function applyAccessResult(allowed: boolean, reason?: string, pageType?: string): void {
  if (allowed) {
    lockBanner.style.display = 'none';
    syncActionControls();
    if (lastExplanation && lastExplanation.detail === 'short') {
      setLongerButtonState('ready');
    }
    if (!isGenerating && !isDownloadingModel && statusIndicator.textContent === 'Locked') {
      setStatusIndicator(
        hasMoveSelected ? 'Ready' : 'Select a move',
        hasMoveSelected ? 'ready' : 'idle'
      );
    }
  } else {
    lockBanner.style.display = 'flex';
    const isLiveOrPuzzle = pageType === 'game' || pageType === 'puzzle';
    lockBannerTitle.textContent = isLiveOrPuzzle
      ? 'Fair-Play Lock Active'
      : 'Lichess Analysis Required';
    lockBannerText.textContent = reason || LOCK_OFF_MESSAGE;
    setStatusIndicator('Locked', 'locked');
    syncActionControls();
    setLongerButtonState('idle');
    setRatingVoteState(null, false);
  }
}

/**
 * Evaluate position by racing Cloud Eval and local Stockfish (`multiPv: 3` default).
 */
async function getPositionEvaluation(
  fen: string,
  signal?: AbortSignal,
  options: { multiPv?: number } = {}
): Promise<EngineEval> {
  if (signal?.aborted) {
    throw new Error('Evaluation aborted');
  }
  const multiPv = options.multiPv ?? 3;

  if (!currentSettings.cloudEval) {
    setStatusIndicator('Stockfish...', 'busy');
    return await stockfish.evaluate(fen, { signal, multiPv });
  }

  setStatusIndicator('Evaluating...', 'busy');
  const cloudController = new AbortController();
  const sfController = new AbortController();
  const onParentAbort = () => {
    cloudController.abort();
    sfController.abort();
  };
  signal?.addEventListener('abort', onParentAbort, { once: true });

  const cloudPromise = cloudEval
    .getEvaluation(fen, { signal: cloudController.signal })
    .catch(() => null);
  const sfPromise = stockfish.evaluate(fen, { signal: sfController.signal, multiPv });

  try {
    const winner = await Promise.race([
      cloudPromise.then((res) => (res ? { source: 'cloud' as const, res } : null)),
      sfPromise.then((res) => ({ source: 'sf' as const, res })),
    ]);

    if (winner?.source === 'cloud') {
      sfController.abort();
      void sfPromise.catch(() => {});
      return winner.res;
    }
    if (winner?.source === 'sf') {
      cloudController.abort();
      return winner.res;
    }

    return await sfPromise;
  } catch (err) {
    cloudController.abort();
    sfController.abort();
    throw err;
  } finally {
    signal?.removeEventListener('abort', onParentAbort);
  }
}

/**
 * Run Move Explanation.
 * Strictly enforces:
 * 1. Synchronous re-entrancy guard (`isGenerating`) so double clicks cannot overlap.
 * 2. Fail-closed fresh tab revalidation (locks immediately if tab switched to live game).
 * 3. Explicit user consent before any network model download.
 */
async function runExplanation(detail: 'short' | 'long'): Promise<void> {
  if (isGenerating) return;
  isGenerating = true;
  syncActionControls();
  btnMoreDetail.disabled = true;

  const thisEpoch = ++generationEpoch;
  const abortController = new AbortController();
  generationAbortController = abortController;
  const capturedLevel = currentSettings.level;

  try {
    // 1. Revalidate active tab state immediately (Fail-Closed)
    const fresh = await revalidateFreshTabState();
    if (thisEpoch !== generationEpoch) return;
    if (fresh) {
      // Update to the latest board move without aborting this freshly initiated run
      updateLichessState(fresh, { abortInFlight: false });
      syncActionControls();
      btnMoreDetail.disabled = true;
    }

    if (!currentLichessState || !currentLichessState.isAllowed) {
      applyAccessResult(
        false,
        currentLichessState?.lockReason || LOCK_OFF_MESSAGE,
        currentLichessState?.pageType
      );
      renderPlaceholder(getDefaultPlaceholderText(), 'notice');
      return;
    }

    const moveCtx = resolveStateMoveContext(currentLichessState);

    if (!moveCtx.moveSan || !moveCtx.fenAfter) {
      hasMoveSelected = false;
      setStatusIndicator('Select a move', 'idle');
      renderPlaceholder('Select a move to explain.', 'welcome');
      return;
    }

    if (lastHardwareSupport && !lastHardwareSupport.webAssemblySupported) {
      checkModelBanner();
      setStatusIndicator('WASM unavailable', 'error');
      renderErrorState('WebAssembly is disabled or unsupported in this browser. Local AI inference cannot run.');
      return;
    }

    // 2. Enforce Explicit Model Download Consent Gate
    const targetModel: ModelSize = currentSettings.modelSize;

    if (isDownloadingModel) {
      if (detail === 'short' || !lastExplanation) {
        renderPlaceholder(
          `Second ${targetModel} weights are currently downloading. Wait for the download in the banner above to complete.`,
          'notice'
        );
      }
      const prefersReducedMotion =
        typeof window !== 'undefined' &&
        window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      modelBanner.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
      return;
    }

    const runner = await getModelRunner();
    if (thisEpoch !== generationEpoch) return;

    const isTargetLoaded =
      runner.isModelLoaded() && runner.getCurrentModel() === targetModel;

    if (!isTargetLoaded) {
      const isCached = await checkModelCachedDedup(targetModel);
      if (thisEpoch !== generationEpoch) return;

      if (!isCached) {
        // Never start a 1.2 GB / 508 MB download without explicit click on "Download Model"
        if (detail === 'short' || !lastExplanation) {
          invalidateCurrentCommentary();
        }
        checkModelBanner();
        setStatusIndicator('Model required', 'idle');
        const approxSize = targetModel === '0.8B' ? '~508 MB' : '~1.2 GB';
        if (detail === 'short' || !lastExplanation) {
          renderPlaceholder(
            `Second ${targetModel} weights (${approxSize}) are not downloaded yet. Click Download Model in the banner above to download them first.`,
            'notice'
          );
        }
        const prefersReducedMotion =
          typeof window !== 'undefined' &&
          window.matchMedia &&
          window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        modelBanner.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
        return;
      }
    }

    // 3. Clear previous explanation & rating state immediately before evaluation/loading starts
    lastExplanation = null;
    currentVote = null;
    setRatingVoteState(null, false);
    setHeaderVerdict(null);
    modelBadge.textContent = `Second ${targetModel}`;
    renderLoadingState(
      isTargetLoaded
        ? 'Analyzing position with Stockfish...'
        : `Loading local Second ${targetModel} model...`
    );
    setStatusIndicator(isTargetLoaded ? 'Evaluating...' : 'Loading model...', 'busy');

    const loadModelTask = (async () => {
      if (isTargetLoaded) return;
      try {
        const hadInFlightWarmup = Boolean(warmupPromise && warmupTargetModel === targetModel);
        if (hadInFlightWarmup) {
          await warmupPromise;
          if (thisEpoch !== generationEpoch) return;
        }

        if (!runner.isModelLoaded() || runner.getCurrentModel() !== targetModel) {
          if (hadInFlightWarmup && lastWarmupError) {
            throw lastWarmupError;
          }
          await runner.loadModel(targetModel, (p) => {
            if (thisEpoch !== generationEpoch) return;
            if (p.phase === 'downloading' || p.phase === 'loading') {
              const pct = Math.max(0, Math.min(100, Math.round(p.percent)));
              renderLoadingState(`Loading local Second ${targetModel} model (${pct}%)...`);
              setStatusIndicator(`Loading ${pct}%`, 'busy');
            }
          });
        }
        if (thisEpoch !== generationEpoch) return;
        modelBanner.style.display = 'none';
      } catch (err) {
        abortController.abort();
        throw err;
      }
    })();

    const evalTask = (async () => {
      const evalBefore = await getPositionEvaluation(moveCtx.fenBefore, abortController.signal, {
        multiPv: 3,
      });
      if (thisEpoch !== generationEpoch) {
        return { evalBefore, evalAfter: null as EngineEval | null };
      }

      let evalAfter: EngineEval | null = null;
      if (moveCtx.fenAfter) {
        evalAfter = await getPositionEvaluation(moveCtx.fenAfter, abortController.signal, {
          multiPv: 1,
        });
      }
      return { evalBefore, evalAfter };
    })();

    const [, evalResult, promptBuilderMod] = await Promise.all([
      loadModelTask,
      evalTask,
      getPromptBuilderModule(),
    ]);
    if (thisEpoch !== generationEpoch) return;

    const { evalBefore, evalAfter } = evalResult;

    // Update the center-aligned verdict in the 6A header (`24... Nd4 · Blunder`)
    if (evalAfter && moveCtx.moveUci) {
      const verdict = promptBuilderMod.classifyVerdict(evalBefore, evalAfter, moveCtx.moveUci);
      setHeaderVerdict(verdict);
    } else {
      setHeaderVerdict(null);
    }

    const inferenceReq = promptBuilderMod.buildPromptForInference({
      fen: moveCtx.fenBefore,
      moveUci: moveCtx.moveUci,
      evalBefore,
      evalAfter,
      level: capturedLevel,
      detail,
    });

    setStatusIndicator('Streaming...', 'busy');
    renderLoadingState('Generating explanation...');
    outputText.setAttribute('aria-live', 'off');
    outputText.setAttribute('aria-busy', 'true');
    let streamedCommentary = '';
    let commentaryDirty = false;
    let rafPending = false;
    let rafId: number | null = null;
    let streamFinished = false;
    let streamInitialized = false;
    let finalizedParaCount = 0;
    const streamParaEls: HTMLParagraphElement[] = [];

    const cancelPendingRaf = () => {
      streamFinished = true;
      rafPending = false;
      if (rafId !== null && typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    };
    activeStreamCancel = cancelPendingRaf;

    const updateIncrementalCommentary = (raw: string): void => {
      const trimmedLeading = raw.replace(/\r\n/g, '\n').trimStart();
      if (!trimmedLeading) return;

      if (!streamInitialized) {
        outputText.innerHTML = '';
        streamInitialized = true;
      }

      const rawParts = trimmedLeading.split(/\n{2,}/);
      const completedParts = rawParts
        .slice(0, -1)
        .map((p) => p.trim())
        .filter(Boolean);
      const activePart = rawParts[rawParts.length - 1].trim();

      for (let i = finalizedParaCount; i < completedParts.length; i++) {
        let pEl = streamParaEls[i];
        if (!pEl) {
          pEl = document.createElement('p');
          pEl.className = 'commentary-para';
          streamParaEls[i] = pEl;
          outputText.appendChild(pEl);
        }
        pEl.innerHTML = formatParagraphInnerHtml(completedParts[i]);
        finalizedParaCount = i + 1;
      }

      if (activePart.length > 0) {
        let activeEl = streamParaEls[finalizedParaCount];
        if (!activeEl) {
          activeEl = document.createElement('p');
          activeEl.className = 'commentary-para';
          streamParaEls[finalizedParaCount] = activeEl;
          outputText.appendChild(activeEl);
        }
        activeEl.innerHTML = formatParagraphInnerHtml(activePart);
      } else {
        while (streamParaEls.length > finalizedParaCount) {
          streamParaEls.pop()?.remove();
        }
      }
    };

    const flushStreamToDom = () => {
      rafPending = false;
      rafId = null;
      if (streamFinished || thisEpoch !== generationEpoch) return;

      const hasCommentaryUpdate = commentaryDirty && streamedCommentary.trim().length > 0;
      if (!hasCommentaryUpdate) return;

      // Batch all layout reads before DOM writes to prevent forced synchronous reflow
      const shouldStickCommentary = isNearBottom(proseScrollContainer);

      commentaryDirty = false;
      updateIncrementalCommentary(streamedCommentary);

      if (shouldStickCommentary) {
        proseScrollContainer.scrollTop = proseScrollContainer.scrollHeight;
      }
    };

    const scheduleFlush = () => {
      if (!rafPending && !streamFinished) {
        rafPending = true;
        if (typeof requestAnimationFrame === 'function') {
          rafId = requestAnimationFrame(flushStreamToDom);
        } else {
          flushStreamToDom();
        }
      }
    };

    const maxTokens = detail === 'long' ? 450 : 256;

    const result = await runner.generate(
      inferenceReq.prompt,
      (token) => {
        if (thisEpoch !== generationEpoch) return;
        streamedCommentary += token;
        commentaryDirty = true;
        scheduleFlush();
      },
      {
        maxTokens,
        userMessage: inferenceReq.userMessage,
        systemMessage: inferenceReq.systemMessage,
      }
    );

    cancelPendingRaf();
    if (activeStreamCancel === cancelPendingRaf) {
      activeStreamCancel = null;
    }

    if (thisEpoch !== generationEpoch) return;

    const finalCommentary = (result.text || streamedCommentary).trim();
    outputText.setAttribute('aria-live', 'polite');
    outputText.setAttribute('aria-busy', 'false');

    if (!finalCommentary) {
      renderPlaceholder(getDefaultPlaceholderText(), 'welcome');
      lastExplanation = null;
      currentVote = null;
      setRatingVoteState(null, false);
      setLongerButtonState('idle');
      setStatusIndicator(
        hasMoveSelected ? 'Ready' : 'Select a move',
        hasMoveSelected ? 'ready' : 'idle'
      );
      return;
    }

    renderCommentary(finalCommentary);

    setStatusIndicator('Ready', 'ready');

    lastExplanation = {
      id: `exp_${Date.now()}`,
      fen: moveCtx.fenBefore,
      move: moveCtx.moveSan || 'position',
      prompt: inferenceReq.prompt,
      text: finalCommentary,
      level: capturedLevel,
      detail,
      model: targetModel,
    };
    currentVote = null;

    setRatingVoteState(null, true);
    setLongerButtonState(detail === 'long' ? 'longer-complete' : 'ready');
  } catch (err: any) {
    if (thisEpoch !== generationEpoch) return;
    setStatusIndicator('Error', 'error');
    lastExplanation = null;
    currentVote = null;
    setRatingVoteState(null, false);
    setLongerButtonState('idle');
    renderErrorState(`Explanation could not be completed: ${err?.message || err}`);
  } finally {
    if (thisEpoch === generationEpoch) {
      activeStreamCancel?.();
      activeStreamCancel = null;
      if (generationAbortController === abortController) {
        generationAbortController = null;
      }
      isGenerating = false;
      outputText.setAttribute('aria-live', 'polite');
      outputText.setAttribute('aria-busy', 'false');
      syncActionControls();
      if (currentLichessState?.isAllowed && lastExplanation) {
        setLongerButtonState(
          lastExplanation.detail === 'long' ? 'longer-complete' : 'ready'
        );
      }
    }
  }
}

/**
 * Record or toggle off user feedback rating.
 * Associates the rating with the exact model (`lastExplanation.model`) that generated the text.
 */
async function recordRating(type: 'up' | 'down'): Promise<void> {
  if (!lastExplanation || isGenerating || isSavingRating) return;
  const targetExplanation = lastExplanation;
  isSavingRating = true;

  try {
    if (currentVote === type) {
      await deleteRating(targetExplanation.id);
      if (lastExplanation?.id !== targetExplanation.id) return;
      currentVote = null;
      setRatingVoteState(null, true);
      setStatusIndicator('Ready', 'ready');
      return;
    }

    const ratingObj: UserRating = {
      id: targetExplanation.id,
      timestamp: Date.now(),
      fen: targetExplanation.fen,
      move: targetExplanation.move,
      rating: type,
      prompt: targetExplanation.prompt,
      output: targetExplanation.text,
      level: targetExplanation.level,
      detail: targetExplanation.detail,
      model: targetExplanation.model,
    };

    await saveRating(ratingObj);
    if (lastExplanation?.id !== targetExplanation.id) return;
    currentVote = type;
    setRatingVoteState(type, true);
    setStatusIndicator('Rating saved', 'ready');
  } catch {
    if (lastExplanation?.id === targetExplanation.id) {
      setStatusIndicator('Save error', 'error');
    }
  } finally {
    isSavingRating = false;
  }
}

export async function initSidepanel(): Promise<void> {
  appShell = document.getElementById('app-shell')!;
  lockBanner = document.getElementById('lock-banner')!;
  lockBannerTitle = document.getElementById('lock-banner-title')!;
  lockBannerText = document.getElementById('lock-banner-text')!;
  modelBanner = document.getElementById('model-banner')!;
  modelBannerDeviceNote = document.getElementById('model-banner-device-note')!;
  modelBannerDeviceText = document.getElementById('model-banner-device-text')!;
  btnSwitch08b = document.getElementById('btn-switch-08b') as HTMLButtonElement;
  btnDownloadModel = document.getElementById('btn-download-model') as HTMLButtonElement;
  downloadProgressTrack = document.getElementById('download-progress-track')!;
  downloadProgressFill = document.getElementById('download-progress-fill')!;
  downloadProgressText = document.getElementById('download-progress-text')!;

  selectedMovePill = document.getElementById('selected-move-pill')!;
  verdictSep = document.getElementById('verdict-sep')!;
  selectedMoveVerdict = document.getElementById('selected-move-verdict')!;
  modelBadge = document.getElementById('model-badge')!;

  btnExplainMove = document.getElementById('btn-explain-move') as HTMLButtonElement;
  btnStopGeneration = document.getElementById('btn-stop-generation') as HTMLButtonElement;
  btnMoreDetail = document.getElementById('btn-more-detail') as HTMLButtonElement;

  statusIndicator = document.getElementById('status-indicator')!;
  proseScrollContainer = document.getElementById('prose-scroll-container')!;
  outputText = document.getElementById('output-text')!;

  btnRateUp = document.getElementById('btn-rate-up') as HTMLButtonElement;
  btnRateDown = document.getElementById('btn-rate-down') as HTMLButtonElement;

  settingsModal = document.getElementById('settings-modal')!;
  btnSettingsOpen = document.getElementById('btn-settings-open') as HTMLButtonElement;
  btnSettingsClose = document.getElementById('btn-settings-close') as HTMLButtonElement;
  settingLevel = document.getElementById('setting-level') as HTMLSelectElement;
  settingModel = document.getElementById('setting-model') as HTMLSelectElement;
  settingCloudEval = document.getElementById('setting-cloud-eval') as HTMLInputElement;
  btnExportRatings = document.getElementById('btn-export-ratings') as HTMLButtonElement;
  btnDeleteModel = document.getElementById('btn-delete-model') as HTMLButtonElement;
  deleteConfirmRow = document.getElementById('delete-confirm-row')!;
  btnConfirmDeleteYes = document.getElementById('btn-confirm-delete-yes') as HTMLButtonElement;
  btnConfirmDeleteNo = document.getElementById('btn-confirm-delete-no') as HTMLButtonElement;
  settingsFeedback = document.getElementById('settings-feedback')!;
  hwStatusInfo = document.getElementById('hw-status-info')!;

  currentSettings = { ...DEFAULT_SETTINGS };
  currentLichessState = null;
  settingsReady = false;
  isGenerating = false;
  isDownloadingModel = false;
  isSavingRating = false;
  generationAbortController = null;
  activeTabId = null;
  activeStreamCancel = null;
  lastHardwareSupport = null;
  hasMoveSelected = false;
  lastResolvedSan = undefined;
  lastResolvedPly = undefined;
  lastResolvedPath = undefined;
  lastResolvedLineKey = undefined;
  lastWarmupError = null;
  lastExplanation = null;
  currentVote = null;
  warmupPromise = null;
  warmupTargetModel = null;
  warmupUnavailable = false;
  isWarmingModel = false;
  modelRunnerModulePromise = null;
  modelRunnerInstance = null;
  promptBuilderModulePromise = null;
  inFlightCacheCheck = null;
  inFlightCacheModel = null;

  await init();
}

// Auto-start in browser extension environment
if (typeof process === 'undefined' || !process.env?.VITEST) {
  void initSidepanel();
}

