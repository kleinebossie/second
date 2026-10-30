// Background service worker for Second Manifest V3 test extension.
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

// Open the side panel when the user clicks the extension toolbar icon
chrome.runtime.onInstalled.addListener(() => {
  if (chrome.sidePanel && typeof chrome.sidePanel.setPanelBehavior === "function") {
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch((err) => {
      console.error("Failed to set side panel behavior:", err);
    });
  }
});
