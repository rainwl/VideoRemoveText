"""Central configuration for the subtitle removal pipeline.

All knobs that the rest of the codebase reads live in :class:`AppConfig`.
The CLI in ``app.main`` builds an instance of this and passes it to the
pipeline. Defaults are tuned to be reasonable for typical 1080p videos
with bottom-aligned hard subtitles, but every value can be overridden
from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


SubtitleColor = str  # "white" or "black"
Backend = str        # "lama" | "opencv" | "propainter" | "e2fgvi"


@dataclass
class AppConfig:
    # ---- I/O --------------------------------------------------------
    input_path: str = ""
    output_path: str = "output.mp4"
    work_dir: str = "work"
    save_intermediate: bool = True

    # ---- Run mode ---------------------------------------------------
    preview: bool = False
    preview_seconds: float = 3.0

    # ---- ROI --------------------------------------------------------
    # (x, y, w, h) in absolute pixels of the source video.
    # When None and ``roi_interactive`` is True the user is asked to
    # draw the box over the first frame.
    roi: Optional[Tuple[int, int, int, int]] = None
    roi_interactive: bool = False

    # ---- Subtitle color ---------------------------------------------
    subtitle_color: SubtitleColor = "white"

    # White color thresholds
    white_rgb_threshold: int = 200       # min value on each RGB channel
    hsv_white_v_min: int = 200           # HSV V min for white
    hsv_white_s_max: int = 60            # HSV S max for white

    # Black color thresholds
    black_rgb_threshold: int = 60        # max value on each RGB channel
    hsv_black_v_max: int = 70            # HSV V max for black
    hsv_black_s_max: int = 90            # HSV S max for black (relax a bit)

    # ---- Mask post-processing ---------------------------------------
    morph_kernel_size: int = 3
    dilate_iterations: int = 2
    close_iterations: int = 1
    open_iterations: int = 1
    min_component_area: int = 25
    merge_distance: int = 6              # used to merge nearby components

    # ---- Temporal smoothing -----------------------------------------
    temporal_window: int = 2             # +/- frames; 0 disables smoothing
    temporal_mode: str = "union"         # union | vote
    temporal_vote_min: int = 2           # min hits across window for vote

    # ---- Inpainting backend -----------------------------------------
    backend: Backend = "lama"
    lama_device: str = "auto"            # "auto" | "cpu" | "cuda" | "mps"
    lama_dilate_extra: int = 4           # extra dilation feeding LaMa
    lama_crop_padding: int = 48          # extra context around subtitle bbox
    opencv_inpaint_radius: float = 3.0

    # ---- FFmpeg -----------------------------------------------------
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    output_video_codec: str = "libx264"
    output_pix_fmt: str = "yuv420p"
    output_crf: int = 18

    # ---- Misc -------------------------------------------------------
    # Optional list of (frame_index_start, frame_index_end_exclusive) ranges
    # to actually process. If empty, all frames are processed. Useful for
    # debugging.
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)

    def validate(self) -> None:
        if self.subtitle_color not in ("white", "black"):
            raise ValueError(
                f"subtitle_color must be 'white' or 'black', got {self.subtitle_color!r}"
            )
        if self.backend not in ("lama", "opencv", "propainter", "e2fgvi"):
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.temporal_window < 0:
            raise ValueError("temporal_window must be >= 0")
        if self.lama_crop_padding < 0:
            raise ValueError("lama_crop_padding must be >= 0")
        if self.opencv_inpaint_radius <= 0:
            raise ValueError("opencv_inpaint_radius must be > 0")
        if self.roi is not None:
            x, y, w, h = self.roi
            if w <= 0 or h <= 0:
                raise ValueError(f"invalid roi size: {self.roi}")
