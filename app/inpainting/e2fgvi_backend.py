"""Placeholder backend for E2FGVI (https://github.com/MCG-NKU/E2FGVI).

Same idea as the ProPainter stub: API in place, real integration
deferred. To wire it up:

1. Clone E2FGVI and install its dependencies.
2. Place the ``E2FGVI-HQ-CVPR22.pth`` checkpoint under
   ``weights/E2FGVI``.
3. Import their ``test.py`` style inference loop and call it from
   :meth:`inpaint_video`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import AppConfig
from .base import InpaintingBackend


class E2FGVIBackend(InpaintingBackend):
    name = "e2fgvi"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def warmup(self) -> None:  # pragma: no cover
        raise NotImplementedError(
            "E2FGVI backend is a stub. See app/inpainting/e2fgvi_backend.py."
        )

    def inpaint(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("E2FGVI is video-level; use inpaint_video.")

    def inpaint_video(
        self,
        frames_bgr: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:  # pragma: no cover
        raise NotImplementedError(
            "E2FGVI backend is not implemented yet — see file docstring."
        )
