# Extension Testing Guide

This document describes how to test the Second Chrome extension.

## 1. Load Unpacked Extension

1. Open Google Chrome.
2. Navigate to `chrome://extensions`.
3. Enable **Developer mode** in the top right corner.
4. Click **Load unpacked**.
5. Select the `extension/` directory (or built distribution directory).

## 2. Open Side Panel

1. Navigate to a game on `https://lichess.org`.
2. Click the extension icon in the toolbar.
3. Verify that the side panel opens on the right side of the window.

## 3. Verify Local Inference

1. Select a move or position in the game.
2. Click **Explain Move**.
3. Verify that the model generates text in the side panel.
4. Verify that the extension does not make external network requests for model inference.

## 4. Fair-Play Verification

1. When a game is active and not finished, verify that analysis stays disabled.
2. When the game ends, verify that analysis becomes enabled.
