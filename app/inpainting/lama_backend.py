"""LaMa image inpainting backend.

Uses the ``simple-lama-inpainting`` package, which downloads the LaMa
weights on first use and exposes a one-line API. We add a small layer
that:

* picks a sensible torch device (cuda → mps → cpu);
* converts BGR uint8 ↔ PIL RGB cleanly;
* dilates the mask a bit further before feeding it to LaMa, so the
  network has some breathing room around character edges and avoids
  haloing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import AppConfig
from ..utils.logging_utils import get_logger
from .base import InpaintingBackend

log = get_logger("inpaint.lama")


def _pick_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class LaMaBackend(InpaintingBackend):
    name = "lama"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.device = _pick_device(cfg.lama_device)
        self._lama = None  # lazy

    def warmup(self) -> None:
        if self._lama is not None:
            return
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as e:
            raise ImportError(
                "simple-lama-inpainting is not installed. Run "
                "`pip install simple-lama-inpainting`."
            ) from e
        log.info("Loading LaMa on device=%s (first run downloads ~200MB weights)...", self.device)
        # SimpleLama uses LAMA_MODEL env to optionally pin a custom checkpoint;
        # otherwise it auto-downloads.
        try:
            self._lama = SimpleLama(device=self.device)
        except TypeError:
            # Older versions don't accept device kwarg.
            self._lama = SimpleLama()
        log.info("LaMa ready.")

    def inpaint(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._lama is None:
            self.warmup()

        # Optionally widen the mask a few pixels — LaMa is more forgiving
        # if the masked area covers anti-aliased edges fully.
        if self.cfg.lama_dilate_extra > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask, k, iterations=self.cfg.lama_dilate_extra)

        if not mask.any():
            return frame_bgr  # nothing to do

        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_mask = Image.fromarray(mask)

        result = self._lama(pil_img, pil_mask)  # PIL RGB
        out_rgb = np.array(result)
        # SimpleLama may return a slightly different size on rare inputs;
        # force-resize to match.
        if out_rgb.shape[:2] != frame_bgr.shape[:2]:
            out_rgb = cv2.resize(
                out_rgb,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
