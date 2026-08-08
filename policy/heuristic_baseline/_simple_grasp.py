"""Locate and configure the adjacent simple-grasp checkout."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def default_simple_grasp_root() -> Path:
    """Return the root requested by the environment or local-development default."""
    return Path(
        os.environ.get("SIMPLE_GRASP_ROOT", Path.home() / "projects" / "simple-grasp")
    ).expanduser()


def ensure_simple_grasp_importable(root: str | Path | None = None) -> Path:
    """Add ``root/src`` ahead of other imports and return its resolved root."""
    simple_grasp_root = (
        default_simple_grasp_root() if root is None else Path(root).expanduser()
    ).resolve()
    src = simple_grasp_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(
            f"simple-grasp source not found at {src}; set SIMPLE_GRASP_ROOT"
        )
    while str(src) in sys.path:
        sys.path.remove(str(src))
    sys.path.insert(0, str(src))
    return simple_grasp_root
