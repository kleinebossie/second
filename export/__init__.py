"""Export package for browser deployment of Second chess models.

This package contains tools to merge LoRA adapters, strip non-text layers,
convert models to GGUF format, quantize weights, shard files for browser caching,
verify post-quantization quality gates, and publish models to Hugging Face Hub.
All docstrings follow ASD-STE100 Simplified Technical English.
"""

from __future__ import annotations

__version__ = "1.0.0"
