"""Color-threshold subtitle mask generator.

This is the heart of the project. We do **not** use OCR or any text
detection model — instead we exploit two facts about hard subtitles:

1. They live inside a known, fixed ROI (e.g. the bottom band of the
   frame).
2. Within a single video they have a single dominant color (white or
   black) with possibly an outline of the opposite color.

Pipeline per frame
------------------
1. Crop the ROI.
2. Build an initial binary mask combining RGB channel-wise thresholds
   **and** HSV value/saturation thresholds. The intersection is much
   more robust than either alone (RGB catches "almost white", HSV
   helps reject saturated colors that happen to be bright).
3. Post-process: open → close → dilate.
4. Drop tiny connected components (noise) and merge nearby components
   so an entire line of text becomes one blob.
5. Paste the cleaned mask back onto a full-frame canvas; everything
   outside the ROI stays black.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import AppConfig
from ..models import ROI


@dataclass
class MaskParams:
    subtitle_color: str
    white_rgb_threshold: int
    hsv_white_v_min: int
    hsv_white_s_max: int
    black_rgb_threshold: int
    hsv_black_v_max: int
    hsv_black_s_max: int
    morph_kernel_size: int
    open_iterations: int
    close_iterations: int
    dilate_iterations: int
    min_component_area: int
    merge_distance: int

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "MaskParams":
        return cls(
            subtitle_color=cfg.subtitle_color,
            white_rgb_threshold=cfg.white_rgb_threshold,
            hsv_white_v_min=cfg.hsv_white_v_min,
            hsv_white_s_max=cfg.hsv_white_s_max,
            black_rgb_threshold=cfg.black_rgb_threshold,
            hsv_black_v_max=cfg.hsv_black_v_max,
            hsv_black_s_max=cfg.hsv_black_s_max,
            morph_kernel_size=cfg.morph_kernel_size,
            open_iterations=cfg.open_iterations,
            close_iterations=cfg.close_iterations,
            dilate_iterations=cfg.dilate_iterations,
            min_component_area=cfg.min_component_area,
            merge_distance=cfg.merge_distance,
        )


def _initial_white_mask(bgr: np.ndarray, p: MaskParams) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    rgb_mask = (
        (r >= p.white_rgb_threshold)
        & (g >= p.white_rgb_threshold)
        & (b >= p.white_rgb_threshold)
    )
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hsv_mask = (v >= p.hsv_white_v_min) & (s <= p.hsv_white_s_max)
    return (rgb_mask & hsv_mask).astype(np.uint8) * 255


def _initial_black_mask(bgr: np.ndarray, p: MaskParams) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    rgb_mask = (
        (r <= p.black_rgb_threshold)
        & (g <= p.black_rgb_threshold)
        & (b <= p.black_rgb_threshold)
    )
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hsv_mask = (v <= p.hsv_black_v_max) & (s <= p.hsv_black_s_max)
    return (rgb_mask & hsv_mask).astype(np.uint8) * 255


def _morphology(mask: np.ndarray, p: MaskParams) -> np.ndarray:
    k = max(1, p.morph_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    out = mask
    if p.open_iterations > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=p.open_iterations)
    if p.close_iterations > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=p.close_iterations)
    if p.dilate_iterations > 0:
        out = cv2.dilate(out, kernel, iterations=p.dilate_iterations)
    return out


def _filter_components(mask: np.ndarray, p: MaskParams) -> np.ndarray:
    """Drop tiny components, then merge nearby ones via a single closing pass."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= p.min_component_area:
            out[labels == i] = 255

    if p.merge_distance > 0 and out.any():
        d = max(1, p.merge_distance)
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d * 2 + 1, d * 2 + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, merge_kernel)
    return out


def build_subtitle_mask(
    frame_bgr: np.ndarray,
    roi: ROI,
    params: MaskParams,
) -> np.ndarray:
    """Return a full-frame uint8 mask (255 = subtitle, 0 = background).

    Outside ``roi`` the mask is always zero.
    """
    h, w = frame_bgr.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    crop = frame_bgr[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    if crop.size == 0:
        return full_mask

    if params.subtitle_color == "white":
        m = _initial_white_mask(crop, params)
    else:
        m = _initial_black_mask(crop, params)

    m = _morphology(m, params)
    m = _filter_components(m, params)

    full_mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = m
    return full_mask
