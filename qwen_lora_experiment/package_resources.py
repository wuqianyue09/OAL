"""Import-safe locations of immutable files shipped with this package."""

from __future__ import annotations
from pathlib import Path


def qwen2_attention_compatibility_fixture_path() -> Path:
    """Return the package-owned Qwen2 attention compatibility evidence path.

    This helper deliberately does not import Torch or Transformers, and does
    not stat the path at import time.  Callers retain responsibility for
    reporting an absent package-data file in their own error vocabulary.
    """
    return Path(__file__).with_name("fixtures") / "qwen2_attention_compatibility.json"
