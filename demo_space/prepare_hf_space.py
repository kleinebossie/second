"""Stage a standalone Hugging Face Space bundle for Second demo.

Copies demo_space files and core/python into demo_space/dist/ so the Space works standalone.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = DEMO_DIR / "dist"


def stage_demo_space(output_dir: Path | str = DEFAULT_OUT_DIR) -> Path:
    """Copy Space files and shared Python core into output directory."""
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for fname in ("README.md", "app.py", "requirements.txt", "packages.txt"):
        shutil.copy2(DEMO_DIR / fname, out / fname)

    core_dst = out / "core"
    core_dst.mkdir(parents=True, exist_ok=True)
    init_src = PROJECT_ROOT / "core" / "__init__.py"
    if init_src.is_file():
        shutil.copy2(init_src, core_dst / "__init__.py")

    shutil.copytree(
        PROJECT_ROOT / "core" / "python",
        core_dst / "python",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py"),
    )
    return out


def main() -> None:
    """Parse CLI arguments and build standalone Space folder."""
    parser = argparse.ArgumentParser(description="Stage standalone HF Space directory.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Target staging folder.")
    args = parser.parse_args()
    staged = stage_demo_space(args.out)
    print(f"Staged Hugging Face Demo Space in {staged}")


if __name__ == "__main__":
    main()
