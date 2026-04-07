"""Placeholder backend for ProPainter (https://github.com/sczhou/ProPainter).

ProPainter is a video inpainting model: it considers a window of
frames jointly so the result is much more temporally stable than a
per-frame image model. Integrating it cleanly involves cloning the
upstream repo, downloading several checkpoints and importing their
``inference_propainter`` module. We deliberately do **not** ship that
heavy dependency by default; instead this file establishes the API
contract so that swapping LaMa for ProPainter is a one-line change in
:mod:`app.inpainting.factory`.

To finish the integration:

1. ``pip install -r requirements_propainter.txt`` (their list).
2. Drop the ProPainter ``model`` and ``RAFT`` checkpoints under
   ``weights/ProPainter``.
3. Replace ``NotImplementedError`` below with calls into their
   ``inference_propainter.main_worker`` (or import it as a function).
4. Override :meth:`inpaint_video` so that the *whole* clip is
   processed at once — that is where ProPainter shines.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import AppConfig
from .base import InpaintingBackend


class ProPainterBackend(InpaintingBackend):
    name = "propainter"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def warmup(self) -> None:  # pragma: no cover
        raise NotImplementedError(
            "ProPainter backend is a stub. See app/inpainting/propainter_backend.py "
            "for integration instructions."
        )

    def inpaint(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("ProPainter is video-level; use inpaint_video.")

    def inpaint_video(
        self,
        frames_bgr: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:  # pragma: no cover
        # TODO: call ProPainter's inference here.
        raise NotImplementedError(
            "ProPainter backend is not implemented yet — see file docstring."
        )
