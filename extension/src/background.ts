/**
 * Background Service Worker for Second Chrome Extension (Manifest V3).
 */

import type { ExtensionMessage, PingResponse } from './types.js';

function configureSidePanelBehavior(): void {
  if (chrome.sidePanel && typeof chrome.sidePanel.setPanelBehavior === 'function') {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch((err) => {
      console.error('Failed to set side panel behavior:', err);
    });
  }
}

chrome.runtime.onInstalled.addListener(() => {
  configureSidePanelBehavior();
});

chrome.runtime.onStartup.addListener(() => {
  configureSidePanelBehavior();
});

// Fallback if setPanelBehavior is unavailable or failed.
if (chrome.action?.onClicked) {
  chrome.action.onClicked.addListener((tab) => {
    if (chrome.sidePanel && typeof chrome.sidePanel.open === 'function' && tab?.windowId !== undefined) {
      chrome.sidePanel.open({ windowId: tab.windowId }).catch((err) => {
        console.error('Failed to open side panel:', err);
      });
    }
  });
}

// Forward messages between tabs and side panel if needed.
chrome.runtime.onMessage.addListener((message: ExtensionMessage | undefined, _sender, sendResponse) => {
  if (message?.type === 'PING') {
    const resp: PingResponse = { status: 'ok' };
    sendResponse(resp);
    return false;
  }
  return false;
});
