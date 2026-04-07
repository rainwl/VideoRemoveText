"""Thin wrappers around ffmpeg / ffprobe.

We deliberately call the binaries via ``subprocess`` instead of using
``imageio-ffmpeg`` or ``ffmpeg-python``. The reason is reproducibility:
the user almost certainly already has FFmpeg installed and our
options stay visible in any error message.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import VideoInfo
from ..utils.logging_utils import get_logger
from ..utils.path_utils import ensure_dir, which_or_raise

log = get_logger("ffmpeg")


def _run(cmd: list[str]) -> None:
    log.debug("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{proc.stderr.decode(errors='ignore')}"
        )


def probe_video(path: str, ffprobe_bin: str = "ffprobe") -> VideoInfo:
    which_or_raise(ffprobe_bin)
    if not Path(path).exists():
        raise FileNotFoundError(f"Input video does not exist: {path}")

    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.decode(errors='ignore')}")

    data = json.loads(proc.stdout.decode())
    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    rfr = stream.get("r_frame_rate", "0/1")
    num, den = (rfr.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    nb = stream.get("nb_frames")
    n_frames = int(nb) if nb and nb != "N/A" else int(round(fps * duration))

    if width == 0 or height == 0 or fps == 0.0:
        raise RuntimeError(f"Could not determine video properties for {path}")

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        n_frames=n_frames,
    )


def extract_frames(
    video_path: str,
    out_dir: str,
    ffmpeg_bin: str = "ffmpeg",
    duration: Optional[float] = None,
) -> None:
    """Extract frames as zero-padded PNGs into ``out_dir``.

    If ``duration`` is given, only that many seconds from the start are
    extracted (used by --preview).
    """
    which_or_raise(ffmpeg_bin)
    ensure_dir(out_dir)
    out_pattern = str(Path(out_dir) / "frame_%08d.png")

    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", video_path, "-vsync", "0", out_pattern]
    _run(cmd)


def extract_audio(
    video_path: str,
    out_audio_path: str,
    ffmpeg_bin: str = "ffmpeg",
    duration: Optional[float] = None,
) -> bool:
    """Extract audio losslessly to AAC. Returns False if the source has no audio."""
    which_or_raise(ffmpeg_bin)
    ensure_dir(Path(out_audio_path).parent)

    cmd = [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", video_path, "-vn", "-acodec", "copy", out_audio_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # Most common: source has no audio stream. Don't blow up.
        log.warning("No audio stream copied (%s)", proc.stderr.decode(errors='ignore').strip().splitlines()[-1] if proc.stderr else "?")
        return False
    return True


def encode_video_from_frames(
    frames_dir: str,
    out_video_path: str,
    fps: float,
    ffmpeg_bin: str = "ffmpeg",
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    crf: int = 18,
) -> None:
    which_or_raise(ffmpeg_bin)
    ensure_dir(Path(out_video_path).parent)

    in_pattern = str(Path(frames_dir) / "frame_%08d.png")
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.6f}",
        "-i", in_pattern,
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        "-crf", str(crf),
        out_video_path,
    ]
    _run(cmd)


def encode_video_from_ndarrays(
    frames: list[np.ndarray],
    out_video_path: str,
    fps: float,
    ffmpeg_bin: str = "ffmpeg",
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    crf: int = 18,
) -> None:
    if not frames:
        raise ValueError("frames must not be empty")

    which_or_raise(ffmpeg_bin)
    ensure_dir(Path(out_video_path).parent)

    first = frames[0]
    height, width = first.shape[:2]
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s:v", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-an",
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        "-crf", str(crf),
        out_video_path,
    ]
    log.debug("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("all frames must have the same size")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        rc = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if rc != 0:
        raise RuntimeError(
            f"Command failed (rc={rc}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{stderr.decode(errors='ignore')}"
        )


def mux_audio(
    video_no_audio: str,
    audio_path: str,
    out_path: str,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    which_or_raise(ffmpeg_bin)
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_no_audio,
        "-i", audio_path,
        "-c", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        out_path,
    ]
    _run(cmd)
