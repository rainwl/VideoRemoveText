"""Tiny logging helpers so the rest of the code can use a single import."""

from __future__ import annotations

import logging
import sys


_INITIALIZED = False


def get_logger(name: str = "subrm") -> logging.Logger:
    global _INITIALIZED
    if not _INITIALIZED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("subrm")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _INITIALIZED = True
    return logging.getLogger(name if name.startswith("subrm") else f"subrm.{name}")
