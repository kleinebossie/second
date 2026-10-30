// Entry point for web page and Chrome extension side panel.
// ASD-STE100 guidelines: short sentences, active voice, simple terms.

import { initApp } from "./app.js";

document.addEventListener("DOMContentLoaded", () => {
  initApp().catch((err) => {
    console.error("Failed to initialize benchmark harness:", err);
  });
});
