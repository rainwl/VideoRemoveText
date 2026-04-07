"""Fast OpenCV-based inpainting backend.

This backend is much faster than LaMa because it only runs a classical
algorithm (`cv2.inpaint`) over the subtitle crop. It is a good default
for quick previews and many subtitle-on-video cases where the occluded
region is narrow and localized.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from ..config import AppConfig
from .base import InpaintingBackend


class OpenCVInpaintBackend(InpaintingBackend):
    name = "opencv"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

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
        pad = max(8, self.cfg.lama_crop_padding // 2)
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
        if not mask.any():
            return frame_bgr

        (x1, y1, x2, y2), frame_crop, mask_crop = self._crop_to_mask(frame_bgr, mask)
        if self.cfg.lama_dilate_extra > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_crop = cv2.dilate(mask_crop, kernel, iterations=max(1, self.cfg.lama_dilate_extra // 2))

        repaired_crop = cv2.inpaint(
            frame_crop,
            mask_crop,
            inpaintRadius=float(self.cfg.opencv_inpaint_radius),
            flags=cv2.INPAINT_TELEA,
        )
        out = frame_bgr.copy()
        out[y1:y2, x1:x2] = repaired_crop
        return out

    def inpaint_video(
        self,
        frames_bgr: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        if len(frames_bgr) != len(masks):
            raise ValueError("frames and masks must have the same length")
        return [self.inpaint(frame, mask) for frame, mask in tqdm(zip(frames_bgr, masks), total=len(frames_bgr), desc="opencv")]
