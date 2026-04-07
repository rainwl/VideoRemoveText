"""Simple temporal smoothing for per-frame subtitle masks.

Two strategies are supported:

* ``union`` — for each frame take the bitwise OR of itself and the
  ``temporal_window`` neighbours on each side. This is aggressive but
  reliable: it covers any frame where a single character was missed.

* ``vote`` — for each pixel keep it only if it appeared in at least
  ``temporal_vote_min`` frames inside the window. This is good when
  there are sporadic false positives in the background.

We do not warp masks with optical flow because subtitles are static
and the ROI is fixed; a plain window is enough to kill flicker.
"""

from __future__ import annotations

from typing import List

import numpy as np


def smooth_masks(
    masks: List[np.ndarray],
    window: int,
    mode: str = "union",
    vote_min: int = 2,
) -> List[np.ndarray]:
    if window <= 0 or len(masks) == 0:
        return masks

    n = len(masks)
    out: List[np.ndarray] = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        stack = masks[lo:hi]
        if mode == "union":
            acc = np.zeros_like(stack[0])
            for m in stack:
                acc = np.maximum(acc, m)
            out.append(acc)
        elif mode == "vote":
            counts = np.zeros(stack[0].shape, dtype=np.int16)
            for m in stack:
                counts += (m > 0).astype(np.int16)
            keep = (counts >= vote_min).astype(np.uint8) * 255
            out.append(keep)
        else:
            raise ValueError(f"unknown temporal mode: {mode}")
    return out
