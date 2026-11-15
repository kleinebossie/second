# Second Brand Guide

## 1. Brand Identity

Second is an open-source, privacy-first local AI chess coach extension for Lichess.
It runs inside the Chrome side panel.

- **Product Name:** Second
- **Extension Store Title:** Second - Local AI Chess Coach
- **Primary Tagline:** Your local chess second.
- **Short Description:** Clear chess move explanations on your device. Active only after games end.

In chess, a "second" is a trusted analyst who helps a player prepare and review games.
Second sits beside the player after a game finishes and explains move ideas in plain English.

---

## 2. Logo System

### 2.1 The Mark

The Second mark combines a sculpted Staunton chess Knight with a subtle numeral `2`.
The Knight faces left inside a dark rounded-square plaque.
The curved neck and diagonal sweep trace the numeral `2` above a horizontal plinth.

Three semantic elements form the emblem:

1. **Warm Gold Knight (`#e5c07b`):** Represents tactical insight and move notation.
2. **Warm Parchment Plinth (`#f1ebd9`):** Anchors the piece like a printed chess monograph.
3. **Emerald Anchor (`#34d399`):** Signals on-device execution and verified post-game safety.

### 2.2 Logo Variants

| Asset | File | Use Case |
| :--- | :--- | :--- |
| Master Mark | [`docs/brand/logo-mark.svg`](brand/logo-mark.svg) | Docs, app badges, and vector scaling. |
| Extension Master SVG | [`extension/public/icons/logo.svg`](../extension/public/icons/logo.svg) | Extension vector source (`48px` and `128px`). |
| Horizontal Lockup | [`docs/brand/logo-lockup.svg`](brand/logo-lockup.svg) | Repository headers and documentation headers. |
| Repository Banner | [`docs/brand/banner.svg`](brand/banner.svg) | Top hero banner in `README.md`. |
| Social Preview SVG | [`docs/brand/social-preview.svg`](brand/social-preview.svg) | Vector master for OpenGraph and GitHub social cards. |
| Extension Icons | `extension/public/icons/icon{16,32,48,128}.png` | Chrome toolbar, extensions page, and manifest icons. |
| Store Promo Small | `extension/store/assets/promo-small-440x280.png` | Chrome Web Store small promo tile (`440x280`). |
| Store Promo Marquee | `extension/store/assets/promo-marquee-1400x560.png` | Chrome Web Store marquee tile (`1400x560`). |
| Social Preview PNG | `extension/store/assets/social-preview-1280x640.png` | GitHub repository social preview image (`1280x640`). |

### 2.3 Inline Sidepanel Mark (`20x20`)

Use the bare vector mark inside the sidepanel top bar (`.app-header` on `#16181e`).
Omit the outer plaque in the top bar so the Knight stays large and sharp at `20px`.

```svg
<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
  <path fill="#e5c07b" d="
    M 11.2 2.2
    L 12.8 3.9
    C 16.8 4.4, 19.2 7.2, 18.8 11.0
    C 18.4 13.9, 16.0 15.9, 11.8 17.5
    L 6.1 17.5
    C 7.0 15.0, 10.4 12.9, 12.9 10.4
    C 14.0 9.3, 13.5 7.6, 11.8 7.3
    C 10.5 7.1, 9.4 7.9, 8.7 9.2
    L 6.3 10.7
    C 5.1 11.1, 4.1 10.1, 4.7 8.8
    L 7.0 5.8
    C 7.8 4.7, 9.0 4.0, 10.4 3.8
    Z
  "/>
  <circle cx="8.5" cy="6.8" r="0.65" fill="#16181e"/>
  <rect x="5.7" y="18.8" width="10.1" height="2.2" rx="0.7" fill="#f1ebd9"/>
  <rect x="16.7" y="18.8" width="2.2" height="2.2" rx="0.7" fill="#34d399"/>
</svg>
```

### 2.4 Optical Scaling Rules

Small icon sizes blur thin hairlines and inner cutouts.
`extension/generate_icons.py` uses optical variants for small icons:

- **16x16 (`icon16.png`):** Uses an edge-to-edge plaque, a bold flat gold silhouette, and no eye cutout.
- **32x32 (`icon32.png`):** Keeps a crisp circular eye cutout and removes the translucent flank shadow.
- **48x48 and 128x128 (`icon48.png`, `icon128.png`):** Render directly from `extension/public/icons/logo.svg`.

### 2.5 Clear Space and Restrictions

- Keep clear space around the badge equal to at least 20% of the badge width.
- Never stretch, skew, or rotate the Knight mark.
- Never replace the Monograph palette colors with bright neon greens or flat yellows.
- Never place dark text over the dark `#121316` or `#16181e` surfaces without contrast verification.
- If a background is light, always use the full badged `logo-mark.svg` or `logo-lockup.svg`.

---

## 3. Color System (Monograph Palette)

All brand assets share the CSS tokens defined in `extension/src/sidepanel/styles.css`.

### 3.1 Surface Tokens

| Token | Hex / Value | Role |
| :--- | :--- | :--- |
| `--bg-canvas` | `#121316` | Primary dark canvas and reading room background. |
| `--bg-chrome` | `#16181e` | Top header bar, banners, and card surfaces. |
| `--bg-elevated` | `#1c1f28` | Elevated badges, plaques, and interactive cards. |
| `--bg-input` | `#0e1015` | Recessed progress tracks and deep wells. |
| `--bg-pill-hover` | `#252834` | Secondary button fill and pill hover state. |
| `--border-subtle` | `rgba(255, 255, 255, 0.07)` | Hairline dividers between panel sections. |
| `--border-dock` | `rgba(255, 255, 255, 0.09)` | Borders around banners and the floating dock. |
| `--border-hover` | `#363b4c` | Interactive control borders. |

### 3.2 Brand and Semantic Accents

| Token | Hex | Role |
| :--- | :--- | :--- |
| `--tok-move-color` | `#e5c07b` | Warm Gold. Used for the Knight mark, move notation, and focus rings. |
| `--text-warm-ghost` | `#f1ebd9` | Warm Parchment. Used for the Knight plinth, lead sentences, and editorial highlights. |
| `--verdict-good` | `#34d399` | Emerald. Used for the plinth anchor, "Ready" status, and "Best move" / "Good" badges. |
| `--verdict-best` | `#60a5fa` | Blue. Used for "Best" and "Only move" badges. |
| `--verdict-inaccuracy` | `#fcd34d` | Soft Gold. Used for inaccuracy labels. |
| `--verdict-mistake` | `#fbbf24` | Amber. Used for mistake labels. |
| `--verdict-blunder` | `#f87171` | Coral Red. Used for blunders and the live-game Fair-Play Lock banner. |

### 3.3 Text Hierarchy

| Token | Hex | Role |
| :--- | :--- | :--- |
| `--text-primary` | `#ffffff` | Wordmark, headings, move titles, and piece names. |
| `--text-body` | `#e2e8f0` | Primary commentary prose. |
| `--text-secondary` | `#94a3b8` | Subtitles, banner descriptions, and helper copy. |
| `--text-muted` | `#8e8a83` | Model badges, timestamps, and quiet metadata. |
| `--tok-square-color` | `#cbd5e1` | Chess square and file references. |

---

## 4. Typography

Second uses two font stacks to separate human commentary from chess notation.

- **Sans Stack (`--font-sans`):** `'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`
  Use for the `Second` wordmark, UI buttons, headings, and commentary paragraphs.
- **Monospace Stack (`--font-mono`):** `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace`
  Use for SAN moves (`14. Nd5`), board squares (`d5`, `e-file`), status labels, and model tags.

### Type Rules

- Set the `Second` wordmark in semi-bold or bold (`600`–`700`) with tight tracking (`-0.02em` to `-0.03em`).
- Use sentence case for buttons, banners, and feature pills.
- Avoid all-caps labels except short technical spec tags like `ON-DEVICE AI · POST-GAME ONLY`.

---

## 5. Brand Voice (ASD-STE100)

Write all product copy, documentation, and error messages in ASD-STE100 Simplified Technical English:

- Keep sentences short (`25` words or fewer).
- Use active voice.
- Put one idea in each sentence.
- State the condition before the action.
- Use plain, direct words instead of marketing filler.
- State facts accurately: the extension runs local language models in WebAssembly (`wllama`) and Stockfish in WebAssembly.

---

## 6. Regenerating Assets

Run `generate_icons.py` whenever you update `extension/public/icons/logo.svg` or `docs/brand/social-preview.svg`:

```bash
python3 extension/generate_icons.py
```

The script requires Google Chrome or Chromium and Python `Pillow`.
It renders all four transparent extension icons (`16`, `32`, `48`, `128`) and all three store promo PNGs (`440x280`, `1400x560`, `1280x640`).
