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

from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

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
        self._pil_image = None

    def warmup(self) -> None:
        if self._lama is not None:
            return
        try:
            from simple_lama_inpainting import SimpleLama
            from PIL import Image
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
        self._pil_image = Image
        log.info("LaMa ready.")

    def _crop_to_mask(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
        pts = cv2.findNonZero(mask)
        if pts is None:
            h, w = frame_bgr.shape[:2]
            return (0, 0, w, h), frame_bgr, mask

        x, y, w, h = cv2.boundingRect(pts)
        pad = max(0, self.cfg.lama_crop_padding) + max(0, self.cfg.lama_dilate_extra) * 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        return (
            (x1, y1, x2, y2),
            frame_bgr[y1:y2, x1:x2],
            mask[y1:y2, x1:x2].copy(),
        )

    def inpaint(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._lama is None:
            self.warmup()

        if not mask.any():
            return frame_bgr

        (x1, y1, x2, y2), frame_crop, mask_crop = self._crop_to_mask(frame_bgr, mask)

        # Optionally widen the mask a few pixels — LaMa is more forgiving
        # if the masked area covers anti-aliased edges fully.
        if self.cfg.lama_dilate_extra > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_crop = cv2.dilate(mask_crop, k, iterations=self.cfg.lama_dilate_extra)

        rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
        pil_img = self._pil_image.fromarray(rgb)
        pil_mask = self._pil_image.fromarray(mask_crop)

        result = self._lama(pil_img, pil_mask)  # PIL RGB
        out_rgb = np.array(result)
        # SimpleLama may return a slightly different size on rare inputs;
        # force-resize to match.
        if out_rgb.shape[:2] != frame_crop.shape[:2]:
            out_rgb = cv2.resize(
                out_rgb,
                (frame_crop.shape[1], frame_crop.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        out = frame_bgr.copy()
        out[y1:y2, x1:x2] = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out

    def inpaint_video(
        self,
        frames_bgr: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        if len(frames_bgr) != len(masks):
            raise ValueError("frames and masks must have the same length")
        active = sum(int(mask.any()) for mask in masks)
        log.info("LaMa will process %d/%d frames with non-empty masks", active, len(masks))
        return [self.inpaint(f, m) for f, m in tqdm(zip(frames_bgr, masks), total=len(frames_bgr), desc="lama")]
