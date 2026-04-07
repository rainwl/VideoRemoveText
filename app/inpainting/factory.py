"""Backend factory."""

from __future__ import annotations

from ..config import AppConfig
from .base import InpaintingBackend
from .lama_backend import LaMaBackend
from .opencv_backend import OpenCVInpaintBackend
from .propainter_backend import ProPainterBackend
from .e2fgvi_backend import E2FGVIBackend


def build_backend(cfg: AppConfig) -> InpaintingBackend:
    name = cfg.backend.lower()
    if name == "lama":
        return LaMaBackend(cfg)
    if name == "opencv":
        return OpenCVInpaintBackend(cfg)
    if name == "propainter":
        return ProPainterBackend(cfg)
    if name == "e2fgvi":
        return E2FGVIBackend(cfg)
    raise ValueError(f"Unknown inpainting backend: {cfg.backend!r}")
