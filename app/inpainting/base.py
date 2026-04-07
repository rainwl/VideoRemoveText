"""Abstract base class for inpainting backends.

Concrete backends only need to implement :meth:`inpaint` (per-frame)
and optionally :meth:`inpaint_video` (full clip). The pipeline calls
:meth:`inpaint_video` with the entire frame list and the corresponding
mask list — image-level backends like LaMa fall back to a simple
per-frame loop, while video backends like ProPainter / E2FGVI can
override it to do real spatio-temporal inpainting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class InpaintingBackend(ABC):
    name: str = "base"

    @abstractmethod
    def inpaint(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint a single frame. ``mask`` is uint8 with 255 = remove."""

    def inpaint_video(
        self,
        frames_bgr: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Default implementation: per-frame loop."""
        if len(frames_bgr) != len(masks):
            raise ValueError("frames and masks must have the same length")
        return [self.inpaint(f, m) for f, m in zip(frames_bgr, masks)]

    def warmup(self) -> None:
        """Optional: load weights, allocate buffers, etc."""
