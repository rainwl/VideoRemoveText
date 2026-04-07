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

import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .config import AppConfig
from .ffmpeg_tools.ffmpeg_wrapper import (
    encode_video_from_frames,
    encode_video_from_ndarrays,
    extract_audio,
    mux_audio,
    probe_video,
)
from .inpainting.factory import build_backend
from .models import ROI, VideoInfo
from .roi.selector import resolve_roi
from .subtitle_mask.color_mask import MaskParams, build_subtitle_mask
from .temporal.smoother import smooth_masks
from .utils.logging_utils import get_logger
from .utils.path_utils import clean_dir, ensure_dir

log = get_logger("pipeline")


def _save_preview(
    frame: np.ndarray, mask: np.ndarray, repaired: np.ndarray, dst: Path
) -> None:
    """Save a side-by-side preview: [original | mask | repaired]."""
    h, w = frame.shape[:2]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel = np.concatenate([frame, mask_bgr, repaired], axis=1)
    cv2.imwrite(str(dst), panel)


def _decode_frames(
    video_path: str,
    max_frames: int | None = None,
    save_dir: Path | None = None,
) -> tuple[List[str], List[np.ndarray]]:
    """Decode frames directly from the source video.

    This avoids the previous decode -> PNG -> re-read round trip, which
    was adding noticeable latency for short clips.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_names: List[str] = []
    frames: List[np.ndarray] = []
    idx = 1

    try:
        while True:
            if max_frames is not None and idx > max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            name = f"frame_{idx:08d}.png"
            frame_names.append(name)
            frames.append(frame)
            if save_dir is not None:
                cv2.imwrite(str(save_dir / name), frame)
            idx += 1
    finally:
        cap.release()

    return frame_names, frames


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
        clean_dir(work)

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

    # 3. Decode frames
    duration_limit = cfg.preview_seconds if cfg.preview else None
    max_frames = None
    if duration_limit is not None and duration_limit > 0:
        max_frames = max(1, int(math.ceil(duration_limit * info.fps)))

    log.info("Decoding frames%s ...",
             f" (preview {cfg.preview_seconds}s)" if cfg.preview else "")
    frame_names, frames = _decode_frames(
        cfg.input_path,
        max_frames=max_frames,
        save_dir=frames_dir if cfg.save_intermediate else None,
    )
    if not frames:
        raise RuntimeError("Video decoding did not produce any frames")
    log.info("Decoded %d frames", len(frames))

    # 4. Audio
    log.info("Extracting audio...")
    has_audio = extract_audio(cfg.input_path, str(audio_path), cfg.ffmpeg_bin, duration_limit)

    # 5. Per-frame masks
    log.info("Building subtitle masks (%s color)...", cfg.subtitle_color)
    params = MaskParams.from_config(cfg)
    masks: List[np.ndarray] = []
    for img in tqdm(frames, desc="mask"):
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
        for name, m in zip(frame_names, masks):
            cv2.imwrite(str(masks_dir / name), m)

    # 7. Inpainting
    log.info("Loading inpainting backend: %s", cfg.backend)
    backend = build_backend(cfg)
    backend.warmup()

    log.info("Inpainting %d frames...", len(frames))
    repaired = backend.inpaint_video(frames, masks)

    if cfg.save_intermediate:
        for name, img in zip(frame_names, repaired):
            cv2.imwrite(str(repaired_dir / name), img)
        # write a few side-by-side previews
        sample_idx = list(range(0, len(frames), max(1, len(frames) // 10)))[:10]
        for i in sample_idx:
            _save_preview(
                frames[i], masks[i], repaired[i],
                preview_dir / f"preview_{i:08d}.png",
            )

    # 8. Re-encode
    log.info("Encoding repaired frames into video...")
    if cfg.save_intermediate:
        encode_video_from_frames(
            frames_dir=str(repaired_dir),
            out_video_path=str(silent_video),
            fps=info.fps,
            ffmpeg_bin=cfg.ffmpeg_bin,
            codec=cfg.output_video_codec,
            pix_fmt=cfg.output_pix_fmt,
            crf=cfg.output_crf,
        )
    else:
        encode_video_from_ndarrays(
            frames=repaired,
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
