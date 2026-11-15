#!/usr/bin/env python3
"""Render Second extension icons and store assets from SVG/HTML masters with headless Chrome."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image

EXT_DIR = Path(__file__).resolve().parent
REPO_DIR = EXT_DIR.parent
ICONS_DIR = EXT_DIR / "public" / "icons"
STORE_ASSETS_DIR = EXT_DIR / "store" / "assets"
DOCS_BRAND_DIR = REPO_DIR / "docs" / "brand"

# Simplified 16x16 optical mark. Removes fine hairlines and tiny details that blur at 16 px.
OPTICAL_16_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <rect x="0.5" y="0.5" width="15" height="15" rx="3.5" fill="#16181e" stroke="rgba(229,192,123,0.40)" stroke-width="0.8"/>
  <path fill="#e5c07b" d="
    M 7.5 1.9
    L 8.6 3.0
    C 11.2 3.4, 12.7 5.2, 12.3 7.6
    C 12.0 9.4, 10.5 10.6, 7.9 11.6
    L 4.0 11.6
    C 4.6 10.0, 6.8 8.6, 8.4 7.1
    C 9.1 6.4, 8.8 5.2, 7.6 5.0
    C 6.8 4.9, 6.1 5.5, 5.6 6.3
    L 4.1 7.3
    C 3.3 7.5, 2.7 6.8, 3.1 6.0
    L 4.6 4.0
    C 5.2 3.3, 6.0 2.9, 6.9 2.7
    Z
  "/>
  <rect x="3.8" y="12.4" width="6.4" height="1.7" rx="0.5" fill="#f1ebd9"/>
  <rect x="10.8" y="12.4" width="1.7" height="1.7" rx="0.5" fill="#34d399"/>
</svg>"""

# Simplified 32x32 optical mark. Keeps bold silhouette and high-contrast edges.
OPTICAL_32_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <defs>
    <linearGradient id="bg32" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1e212b"/>
      <stop offset="100%" stop-color="#121316"/>
    </linearGradient>
    <linearGradient id="gold32" x1="0.15" y1="0" x2="0.85" y2="1">
      <stop offset="0%" stop-color="#f3ead5"/>
      <stop offset="35%" stop-color="#e5c07b"/>
      <stop offset="100%" stop-color="#c69845"/>
    </linearGradient>
  </defs>
  <rect x="0.75" y="0.75" width="30.5" height="30.5" rx="7" fill="url(#bg32)" stroke="rgba(229,192,123,0.34)" stroke-width="1"/>
  <path fill="url(#gold32)" d="
    M 15.0 4.0
    L 17.0 6.2
    C 21.8 6.8, 24.6 10.2, 24.0 14.8
    C 23.5 18.4, 20.6 20.8, 15.6 22.8
    L 8.6 22.8
    C 9.7 19.7, 13.8 17.1, 16.9 14.0
    C 18.3 12.7, 17.7 10.6, 15.6 10.2
    C 14.0 9.9, 12.7 11.0, 11.9 12.6
    L 8.9 14.5
    C 7.4 15.0, 6.3 13.7, 7.0 12.1
    L 9.8 8.4
    C 10.8 7.1, 12.3 6.2, 13.9 5.9
    Z
  "/>
  <circle cx="11.8" cy="9.6" r="0.85" fill="#16181e"/>
  <rect x="8.2" y="24.2" width="12.2" height="2.7" rx="0.9" fill="#f1ebd9"/>
  <rect x="21.4" y="24.2" width="2.7" height="2.7" rx="0.9" fill="#34d399"/>
</svg>"""


def find_chrome() -> str:
    """Locate a Chrome or Chromium binary on PATH."""
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("Google Chrome or Chromium is required to rasterize SVG assets.")


def render_html_to_png(chrome_bin: str, html: str, out_path: Path, width: int, height: int, tmp_dir: Path, name: str, transparent: bool = False) -> None:
    """Render an HTML document with headless Chrome at an exact pixel size."""
    html_file = tmp_dir / f"{name}.html"
    html_file.write_text(html, encoding="utf-8")

    cmd = [
        chrome_bin,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={width},{height}",
        f"--screenshot={out_path}",
    ]
    if transparent:
        cmd.append("--default-background-color=00000000")
    cmd.append(html_file.as_uri())

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with Image.open(out_path) as img:
        if img.size != (width, height):
            raise RuntimeError(f"Expected {width}x{height} for {out_path}, got {img.size}")


def wrap_svg_html(svg_markup: str, width: int, height: int, transparent: bool) -> str:
    """Wrap an inline SVG in a zero-margin HTML page for exact rasterization."""
    bg = "transparent" if transparent else "#121316"
    sized_svg = svg_markup.replace("<svg ", f'<svg width="{width}" height="{height}" style="display:block;" ', 1)
    return (
        "<!doctype html><html><body "
        f'style="margin:0;padding:0;overflow:hidden;width:{width}px;height:{height}px;background:{bg};">'
        f"{sized_svg}</body></html>"
    )


def build_promo_small_html(master_svg: str) -> str:
    """Build the 440x280 Chrome Web Store small promo tile HTML."""
    icon_svg = master_svg.replace("<svg ", '<svg width="78" height="78" style="display:block;" ', 1)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 440px;
    height: 280px;
    overflow: hidden;
    background:
      radial-gradient(circle at 26% 36%, rgba(229, 192, 123, 0.12), transparent 54%),
      radial-gradient(circle at 84% 78%, rgba(52, 211, 153, 0.08), transparent 48%),
      linear-gradient(145deg, #16181e 0%, #121316 65%, #0e1015 100%);
    color: #e2e8f0;
    font-family: 'Inter', 'Lato', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 32px 34px 28px 34px;
    position: relative;
  }}
  .grid-bg {{
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(45deg, rgba(255,255,255,0.016) 25%, transparent 25%),
      linear-gradient(-45deg, rgba(255,255,255,0.016) 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.016) 75%),
      linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.016) 75%);
    background-size: 44px 44px;
    background-position: 0 0, 0 22px, 22px -22px, -22px 0;
    pointer-events: none;
  }}
  .top-row {{
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 18px;
  }}
  .wordmark {{
    display: flex;
    flex-direction: column;
    gap: 3px;
  }}
  .title {{
    font-size: 40px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #ffffff;
    line-height: 1.05;
  }}
  .tagline {{
    font-size: 16px;
    font-weight: 600;
    color: #e5c07b;
    line-height: 1.25;
  }}
  .subcopy {{
    position: relative;
    z-index: 1;
    font-size: 14px;
    line-height: 1.45;
    color: #94a3b8;
    max-width: 365px;
  }}
  .pills {{
    position: relative;
    z-index: 1;
    display: flex;
    gap: 10px;
  }}
  .pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 8px;
    background: #1c1f28;
    border: 1px solid rgba(255, 255, 255, 0.09);
    font-family: ui-monospace, 'Ubuntu Mono', 'Liberation Mono', Menlo, monospace;
    font-size: 11.5px;
    font-weight: 600;
    color: #f1ebd9;
  }}
  .dot {{
    width: 7px;
    height: 7px;
    border-radius: 2px;
    flex-shrink: 0;
  }}
  .dot-emerald {{ background: #34d399; }}
  .dot-gold {{ background: #e5c07b; }}
</style>
</head>
<body>
  <div class="grid-bg"></div>
  <div class="top-row">
    {icon_svg}
    <div class="wordmark">
      <div class="title">Second</div>
      <div class="tagline">Your local chess second.</div>
    </div>
  </div>
  <p class="subcopy">Plain-English move explanations on your device. Built for post-game Lichess review.</p>
  <div class="pills">
    <span class="pill"><span class="dot dot-emerald"></span>100% On-Device</span>
    <span class="pill"><span class="dot dot-gold"></span>Post-Game Only</span>
  </div>
</body>
</html>"""


def build_promo_marquee_html(master_svg: str) -> str:
    """Build the 1400x560 Chrome Web Store marquee promo tile HTML."""
    icon_svg = master_svg.replace("<svg ", '<svg width="108" height="108" style="display:block;" ', 1)
    bare_mini = """<svg width="20" height="20" viewBox="0 0 24 24" style="display:block;flex-shrink:0;">
      <path fill="#e5c07b" d="M 11.2 2.2 L 12.8 3.9 C 16.8 4.4, 19.2 7.2, 18.8 11.0 C 18.4 13.9, 16.0 15.9, 11.8 17.5 L 6.1 17.5 C 7.0 15.0, 10.4 12.9, 12.9 10.4 C 14.0 9.3, 13.5 7.6, 11.8 7.3 C 10.5 7.1, 9.4 7.9, 8.7 9.2 L 6.3 10.7 C 5.1 11.1, 4.1 10.1, 4.7 8.8 L 7.0 5.8 C 7.8 4.7, 9.0 4.0, 10.4 3.8 Z"/>
      <circle cx="8.5" cy="6.8" r="0.65" fill="#16181e"/>
      <rect x="5.7" y="18.8" width="10.1" height="2.2" rx="0.7" fill="#f1ebd9"/>
      <rect x="16.7" y="18.8" width="2.2" height="2.2" rx="0.7" fill="#34d399"/>
    </svg>"""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1400px;
    height: 560px;
    overflow: hidden;
    background:
      radial-gradient(circle at 20% 42%, rgba(229, 192, 123, 0.12), transparent 46%),
      radial-gradient(circle at 82% 76%, rgba(52, 211, 153, 0.08), transparent 42%),
      linear-gradient(135deg, #16181e 0%, #121316 58%, #0e1015 100%);
    color: #e2e8f0;
    font-family: 'Inter', 'Lato', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 56px 76px;
    gap: 56px;
    position: relative;
  }}
  .grid-bg {{
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(45deg, rgba(255,255,255,0.015) 25%, transparent 25%),
      linear-gradient(-45deg, rgba(255,255,255,0.015) 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.015) 75%),
      linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.015) 75%);
    background-size: 64px 64px;
    background-position: 0 0, 0 32px, 32px -32px, -32px 0;
    pointer-events: none;
  }}
  .left {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    gap: 24px;
    max-width: 640px;
  }}
  .brand-lockup {{
    display: flex;
    align-items: center;
    gap: 24px;
  }}
  .brand-text {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .brand-name {{
    font-size: 72px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #ffffff;
    line-height: 1;
  }}
  .brand-tagline {{
    font-size: 25px;
    font-weight: 600;
    color: #e5c07b;
    line-height: 1.25;
  }}
  .desc {{
    font-size: 20px;
    line-height: 1.5;
    color: #94a3b8;
  }}
  .badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 16px;
    border-radius: 10px;
    background: #1c1f28;
    border: 1px solid rgba(255, 255, 255, 0.10);
    font-family: ui-monospace, 'Ubuntu Mono', 'Liberation Mono', Menlo, monospace;
    font-size: 13.5px;
    font-weight: 600;
    color: #f1ebd9;
  }}
  .dot {{
    width: 9px;
    height: 9px;
    border-radius: 2.5px;
  }}
  .card {{
    position: relative;
    z-index: 1;
    width: 530px;
    height: 444px;
    border-radius: 16px;
    background: #121316;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.55);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
  }}
  .card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 18px;
    background: #16181e;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }}
  .card-brand {{
    display: flex;
    align-items: center;
    gap: 9px;
  }}
  .card-brand-title {{
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
  }}
  .card-status {{
    font-family: ui-monospace, 'Ubuntu Mono', monospace;
    font-size: 12px;
    color: #34d399;
    padding-left: 9px;
    border-left: 1px solid rgba(255, 255, 255, 0.09);
  }}
  .btn-explain {{
    background: #252834;
    border: 1px solid #363b4c;
    color: #ffffff;
    font-size: 12.5px;
    font-weight: 500;
    padding: 6px 13px;
    border-radius: 7px;
  }}
  .move-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px 12px 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    font-family: ui-monospace, 'Ubuntu Mono', monospace;
  }}
  .move-left {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
  }}
  .move-sep {{ color: rgba(255, 255, 255, 0.25); }}
  .move-verdict {{ color: #34d399; font-weight: 600; font-size: 14.5px; }}
  .model-tag {{ color: #8e8a83; font-size: 12px; }}
  .prose-area {{
    flex: 1;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }}
  .think-pill {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(229, 192, 123, 0.05);
    border: 1px solid rgba(229, 192, 123, 0.18);
    font-size: 12.5px;
    color: #94a3b8;
  }}
  .think-tag {{
    font-family: ui-monospace, 'Ubuntu Mono', monospace;
    color: #e5c07b;
    font-weight: 600;
    margin-right: 8px;
  }}
  .lead {{
    font-size: 17px;
    font-weight: 600;
    color: #f1ebd9;
    line-height: 1.45;
  }}
  .body-copy {{
    font-size: 15px;
    line-height: 1.55;
    color: #e2e8f0;
  }}
  .tok-piece {{ color: #ffffff; font-weight: 700; }}
  .tok-sq {{
    font-family: ui-monospace, 'Ubuntu Mono', monospace;
    color: #cbd5e1;
    border-bottom: 1.5px solid rgba(203, 213, 225, 0.7);
  }}
  .tok-move {{
    font-family: ui-monospace, 'Ubuntu Mono', monospace;
    color: #e5c07b;
    font-weight: 600;
    border-bottom: 1.5px solid rgba(229, 192, 123, 0.6);
  }}
  .dock {{
    margin: 0 20px 18px 20px;
    padding: 9px 12px;
    border-radius: 11px;
    background: #16181e;
    border: 1px solid rgba(255, 255, 255, 0.09);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .dock-left {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .btn-longer {{
    background: #1c1f28;
    border: 1px solid #363b4c;
    color: #e2e8f0;
    font-size: 12px;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 7px;
  }}
  .dock-meta {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: ui-monospace, 'Ubuntu Mono', monospace;
    font-size: 11.5px;
    color: #94a3b8;
  }}
  .thumbs {{
    display: flex;
    gap: 6px;
  }}
  .thumb-btn {{
    width: 30px;
    height: 28px;
    border-radius: 7px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    align-items: center;
    justify-content: center;
  }}
</style>
</head>
<body>
  <div class="grid-bg"></div>
  <div class="left">
    <div class="brand-lockup">
      {icon_svg}
      <div class="brand-text">
        <div class="brand-name">Second</div>
        <div class="brand-tagline">Your local chess second.</div>
      </div>
    </div>
    <p class="desc">Clear, plain-English move commentary inside your Chrome side panel. Runs locally in WebAssembly on Lichess post-game review.</p>
    <div class="badges">
      <span class="badge"><span class="dot" style="background:#34d399;"></span>Local WASM AI</span>
      <span class="badge"><span class="dot" style="background:#e5c07b;"></span>Fair-Play Locked</span>
      <span class="badge"><span class="dot" style="background:#f1ebd9;"></span>Open Source</span>
    </div>
  </div>

  <div class="card">
    <div class="card-header">
      <div class="card-brand">
        {bare_mini}
        <span class="card-brand-title">Second</span>
        <span class="card-status">Ready</span>
      </div>
      <span class="btn-explain">Explain Move</span>
    </div>
    <div class="move-bar">
      <div class="move-left">
        <span>14. Nd5</span>
        <span class="move-sep">·</span>
        <span class="move-verdict">Best move</span>
      </div>
      <span class="model-tag">Second 2B</span>
    </div>
    <div class="prose-area">
      <div class="think-pill">
        <div><span class="think-tag">&lt;think&gt;</span>Reasoning trace</div>
        <span>▾</span>
      </div>
      <p class="lead">Places the <span class="tok-piece">knight</span> on a strong outpost at <span class="tok-sq">d5</span>.</p>
      <p class="body-copy">This jump attacks the defender on <span class="tok-sq">f6</span> and opens the <span class="tok-sq">e-file</span> for your rook. After <span class="tok-move">14...Nxd5 15. exd5</span>, White gains clear space and targets the <span class="tok-sq">e7</span> pawn.</p>
    </div>
    <div class="dock">
      <div class="dock-left">
        <span class="btn-longer">Longer</span>
        <span class="dock-meta"><span class="dot" style="background:#34d399;width:7px;height:7px;"></span>Local WASM · Stockfish</span>
      </div>
      <div class="thumbs">
        <span class="thumb-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2h0a3.13 3.13 0 0 1 3 3.88Z"/></svg>
        </span>
        <span class="thumb-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22h0a3.13 3.13 0 0 1-3-3.88Z"/></svg>
        </span>
      </div>
    </div>
  </div>
</body>
</html>"""


def main() -> None:
    chrome_bin = find_chrome()
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    STORE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    master_svg_path = ICONS_DIR / "logo.svg"
    master_svg = master_svg_path.read_text(encoding="utf-8")

    social_svg_path = DOCS_BRAND_DIR / "social-preview.svg"
    social_svg = social_svg_path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Render extension icons with transparent PNG backgrounds.
        icon_specs = [
            (16, OPTICAL_16_SVG),
            (32, OPTICAL_32_SVG),
            (48, master_svg),
            (128, master_svg),
        ]
        for size, svg_src in icon_specs:
            out_file = ICONS_DIR / f"icon{size}.png"
            html = wrap_svg_html(svg_src, size, size, transparent=True)
            render_html_to_png(chrome_bin, html, out_file, size, size, tmp_dir, f"icon{size}", transparent=True)
            print(f"Generated {out_file} ({size}x{size})")

        # Render Chrome Web Store and social preview PNGs with full-bleed backgrounds.
        promo_small = STORE_ASSETS_DIR / "promo-small-440x280.png"
        render_html_to_png(chrome_bin, build_promo_small_html(master_svg), promo_small, 440, 280, tmp_dir, "promo_small", transparent=False)
        print(f"Generated {promo_small} (440x280)")

        promo_marquee = STORE_ASSETS_DIR / "promo-marquee-1400x560.png"
        render_html_to_png(chrome_bin, build_promo_marquee_html(master_svg), promo_marquee, 1400, 560, tmp_dir, "promo_marquee", transparent=False)
        print(f"Generated {promo_marquee} (1400x560)")

        social_png = STORE_ASSETS_DIR / "social-preview-1280x640.png"
        render_html_to_png(chrome_bin, wrap_svg_html(social_svg, 1280, 640, transparent=False), social_png, 1280, 640, tmp_dir, "social_preview", transparent=False)
        print(f"Generated {social_png} (1280x640)")


if __name__ == "__main__":
    main()
