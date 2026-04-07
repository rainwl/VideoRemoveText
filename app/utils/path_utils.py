"""Cross-platform path helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_frames(folder: str | os.PathLike, suffix: str = ".png") -> list[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([f for f in p.iterdir() if f.suffix.lower() == suffix.lower()])


def which_or_raise(binary: str) -> str:
    found = shutil.which(binary)
    if not found:
        raise FileNotFoundError(
            f"Required binary {binary!r} was not found on PATH. "
            "Please install it (e.g. `brew install ffmpeg` on macOS, "
            "`apt-get install ffmpeg` on Debian/Ubuntu)."
        )
    return found
