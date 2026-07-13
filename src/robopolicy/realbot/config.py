"""Load configs/smolvla_so101.yaml.

Kept intentionally dependency-light (yaml only, no torch/lerobot) so the voice
layer and the camera-detection helper import cheaply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# repo root = .../src/robopolicy/realbot/config.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "smolvla_so101.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the SO-101 SmolVLA config. Defaults to configs/smolvla_so101.yaml."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def has_placeholder(value: Any) -> bool:
    """True if a config value is still an unfilled '<FILL_ME ...>' placeholder."""
    return isinstance(value, str) and value.strip().startswith("<")
