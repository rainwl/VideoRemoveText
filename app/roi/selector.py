"""ROI selection: command-line tuple parsing and interactive box drawing.

The interactive selector reads the first frame of the video, lets the
user drag a rectangle over the subtitle area and confirms with Enter.
If the user passes ``--roi`` on the command line, the interactive UI
is skipped (CLI takes precedence).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2

from ..models import ROI
from ..utils.logging_utils import get_logger

log = get_logger("roi")


def parse_roi_string(s: str) -> Tuple[int, int, int, int]:
    """Parse ``"x,y,w,h"`` to a tuple of ints.

    Raises ``ValueError`` on malformed input.
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--roi expects 'x,y,w,h', got {s!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"--roi must contain integers, got {s!r}") from e
    if w <= 0 or h <= 0:
        raise ValueError(f"--roi width/height must be positive, got {s!r}")
    return x, y, w, h


def select_roi_interactive(video_path: str, window_name: str = "Select subtitle ROI") -> Tuple[int, int, int, int]:
    """Pop up an OpenCV window so the user can drag a rectangle on the first frame.

    Returns ``(x, y, w, h)``. Press Enter / Space to confirm, ``c`` to cancel.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for ROI selection: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame of video: {video_path}")

    log.info("Drag a rectangle over the subtitle area, then press ENTER/SPACE.")
    log.info("Press 'c' to cancel.")

    # showCrosshair=False is more comfortable for subtitle bands
    rect = cv2.selectROI(window_name, frame, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection cancelled or empty.")
    log.info(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
    return x, y, w, h


def resolve_roi(
    video_path: str,
    cli_roi: Optional[Tuple[int, int, int, int]],
    interactive: bool,
    frame_w: int,
    frame_h: int,
) -> ROI:
    """Decide between CLI ROI and interactive ROI, then clip to frame size."""
    if cli_roi is not None:
        roi = ROI.from_tuple(cli_roi)
    elif interactive:
        roi = ROI.from_tuple(select_roi_interactive(video_path))
    else:
        # Default fallback: bottom 18% of the frame as the subtitle band.
        h_band = max(1, int(frame_h * 0.18))
        roi = ROI(x=0, y=frame_h - h_band, w=frame_w, h=h_band)
        log.warning(
            "No ROI provided; falling back to bottom 18%% band: %s",
            roi.as_tuple(),
        )
    return roi.clip_to(frame_w, frame_h)
