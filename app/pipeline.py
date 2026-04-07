"""End-to-end pipeline that ties everything together.

Stages
------
1. Probe input video.
2. Resolve ROI (CLI / interactive / fallback).
3. ffmpeg → extract frames (and optionally limit to preview duration).
4. ffmpeg → extract original audio.
5. For each frame:
     - build a per-frame subtitle mask via color thresholds.
6. Temporal smoothing across frame masks.
7. Inpainting backend → repaired frames.
8. ffmpeg → re-encode repaired frames into a silent video.
9. ffmpeg → mux original audio back into the final output.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .config import AppConfig
from .ffmpeg_tools.ffmpeg_wrapper import (
    encode_video_from_frames,
    extract_audio,
    extract_frames,
    mux_audio,
    probe_video,
)
from .inpainting.factory import build_backend
from .models import ROI, VideoInfo
from .roi.selector import resolve_roi
from .subtitle_mask.color_mask import MaskParams, build_subtitle_mask
from .temporal.smoother import smooth_masks
from .utils.logging_utils import get_logger
from .utils.path_utils import clean_dir, ensure_dir, list_frames

log = get_logger("pipeline")


def _save_preview(
    frame: np.ndarray, mask: np.ndarray, repaired: np.ndarray, dst: Path
) -> None:
    """Save a side-by-side preview: [original | mask | repaired]."""
    h, w = frame.shape[:2]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel = np.concatenate([frame, mask_bgr, repaired], axis=1)
    cv2.imwrite(str(dst), panel)


def run_pipeline(cfg: AppConfig) -> str:
    cfg.validate()

    if not cfg.input_path:
        raise ValueError("cfg.input_path is empty")
    if not Path(cfg.input_path).exists():
        raise FileNotFoundError(f"Input video not found: {cfg.input_path}")

    work = Path(cfg.work_dir)
    frames_dir = work / "frames"
    masks_dir = work / "masks"
    repaired_dir = work / "repaired_frames"
    preview_dir = work / "preview"
    audio_path = work / "audio.m4a"
    silent_video = work / "silent.mp4"

    if cfg.save_intermediate:
        for d in (frames_dir, masks_dir, repaired_dir, preview_dir):
            clean_dir(d)
    else:
        clean_dir(frames_dir)
        clean_dir(repaired_dir)

    # 1. Probe
    info: VideoInfo = probe_video(cfg.input_path, cfg.ffprobe_bin)
    log.info("Video: %dx%d @ %.3f fps, %.2fs, ~%d frames",
             info.width, info.height, info.fps, info.duration, info.n_frames)

    # 2. ROI
    roi: ROI = resolve_roi(
        video_path=cfg.input_path,
        cli_roi=cfg.roi,
        interactive=cfg.roi_interactive,
        frame_w=info.width,
        frame_h=info.height,
    )
    log.info("Using ROI: %s", roi.as_tuple())

    # 3. Extract frames
    duration_limit = cfg.preview_seconds if cfg.preview else None
    log.info("Extracting frames%s ...",
             f" (preview {cfg.preview_seconds}s)" if cfg.preview else "")
    extract_frames(cfg.input_path, str(frames_dir), cfg.ffmpeg_bin, duration_limit)
    frame_files = list_frames(frames_dir, ".png")
    if not frame_files:
        raise RuntimeError("ffmpeg did not produce any frames")
    log.info("Extracted %d frames", len(frame_files))

    # 4. Audio
    log.info("Extracting audio...")
    has_audio = extract_audio(cfg.input_path, str(audio_path), cfg.ffmpeg_bin, duration_limit)

    # 5. Per-frame masks
    log.info("Building subtitle masks (%s color)...", cfg.subtitle_color)
    params = MaskParams.from_config(cfg)
    masks: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    for fp in tqdm(frame_files, desc="mask"):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {fp}")
        frames.append(img)
        masks.append(build_subtitle_mask(img, roi, params))

    # 6. Temporal smoothing
    if cfg.temporal_window > 0:
        log.info("Temporal smoothing (window=%d, mode=%s)...",
                 cfg.temporal_window, cfg.temporal_mode)
        masks = smooth_masks(
            masks,
            window=cfg.temporal_window,
            mode=cfg.temporal_mode,
            vote_min=cfg.temporal_vote_min,
        )

    if cfg.save_intermediate:
        for fp, m in zip(frame_files, masks):
            cv2.imwrite(str(masks_dir / fp.name), m)

    # 7. Inpainting
    log.info("Loading inpainting backend: %s", cfg.backend)
    backend = build_backend(cfg)
    backend.warmup()

    log.info("Inpainting %d frames...", len(frames))
    repaired = backend.inpaint_video(frames, masks)

    # Persist repaired frames (we always need this for re-encoding).
    for fp, img in zip(frame_files, repaired):
        cv2.imwrite(str(repaired_dir / fp.name), img)

    if cfg.save_intermediate:
        # write a few side-by-side previews
        sample_idx = list(range(0, len(frames), max(1, len(frames) // 10)))[:10]
        for i in sample_idx:
            _save_preview(
                frames[i], masks[i], repaired[i],
                preview_dir / f"preview_{i:08d}.png",
            )

    # 8. Re-encode
    log.info("Encoding repaired frames into video...")
    encode_video_from_frames(
        frames_dir=str(repaired_dir),
        out_video_path=str(silent_video),
        fps=info.fps,
        ffmpeg_bin=cfg.ffmpeg_bin,
        codec=cfg.output_video_codec,
        pix_fmt=cfg.output_pix_fmt,
        crf=cfg.output_crf,
    )

    # 9. Mux audio (if any)
    out_path = Path(cfg.output_path)
    ensure_dir(out_path.parent if str(out_path.parent) else ".")
    if has_audio:
        log.info("Muxing original audio back...")
        mux_audio(str(silent_video), str(audio_path), str(out_path), cfg.ffmpeg_bin)
    else:
        # Just rename / copy.
        import shutil
        shutil.copyfile(silent_video, out_path)

    log.info("Done. Output written to: %s", out_path)
    return str(out_path)
