"""Command line entry point.

Example
-------

    python -m app.main \
        --input input.mp4 \
        --output output.mp4 \
        --roi 0,820,1920,180 \
        --subtitle-color white

Use ``--roi-interactive`` to draw the ROI on the first frame.
Use ``--preview`` to process only the first 3 seconds.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from .config import AppConfig
from .pipeline import run_pipeline
from .roi.selector import parse_roi_string
from .utils.logging_utils import get_logger

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subrm",
        description="Remove burned-in subtitles from a video using a fixed ROI + color mask + AI inpainting.",
    )

    # I/O
    p.add_argument("--input", "-i", required=True, help="Input video path (mp4).")
    p.add_argument("--output", "-o", default="output.mp4", help="Output video path.")
    p.add_argument("--work-dir", default="work", help="Working directory for intermediate files.")
    p.add_argument("--no-save-intermediate", action="store_true",
                   help="Disable writing masks/preview frames to disk.")

    # ROI
    p.add_argument("--roi", default=None,
                   help="Subtitle ROI as 'x,y,w,h' in pixels. Takes precedence over --roi-interactive.")
    p.add_argument("--roi-interactive", action="store_true",
                   help="Pick ROI by drawing a rectangle on the first frame.")

    # Subtitle color
    p.add_argument("--subtitle-color", choices=["white", "black"], default="white")

    # Mode
    p.add_argument("--preview", action="store_true",
                   help="Process only the first --preview-seconds seconds (quick check).")
    p.add_argument("--preview-seconds", type=float, default=3.0)

    # Backend
    p.add_argument("--backend", choices=["lama", "propainter", "e2fgvi"], default="lama")
    p.add_argument("--lama-device", default="auto",
                   help="auto / cpu / cuda / mps")
    p.add_argument("--lama-dilate-extra", type=int, default=4)

    # Color thresholds
    p.add_argument("--white-rgb-threshold", type=int, default=200)
    p.add_argument("--hsv-white-v-min", type=int, default=200)
    p.add_argument("--hsv-white-s-max", type=int, default=60)
    p.add_argument("--black-rgb-threshold", type=int, default=60)
    p.add_argument("--hsv-black-v-max", type=int, default=70)
    p.add_argument("--hsv-black-s-max", type=int, default=90)

    # Mask post-processing
    p.add_argument("--morph-kernel-size", type=int, default=3)
    p.add_argument("--open-iterations", type=int, default=1)
    p.add_argument("--close-iterations", type=int, default=1)
    p.add_argument("--dilate-iterations", type=int, default=2)
    p.add_argument("--min-component-area", type=int, default=25)
    p.add_argument("--merge-distance", type=int, default=6)

    # Temporal
    p.add_argument("--temporal-window", type=int, default=2)
    p.add_argument("--temporal-mode", choices=["union", "vote"], default="union")
    p.add_argument("--temporal-vote-min", type=int, default=2)

    # ffmpeg
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    p.add_argument("--codec", default="libx264")
    p.add_argument("--pix-fmt", default="yuv420p")
    p.add_argument("--crf", type=int, default=18)

    return p


def args_to_config(args: argparse.Namespace) -> AppConfig:
    roi = parse_roi_string(args.roi) if args.roi else None
    return AppConfig(
        input_path=args.input,
        output_path=args.output,
        work_dir=args.work_dir,
        save_intermediate=not args.no_save_intermediate,
        preview=args.preview,
        preview_seconds=args.preview_seconds,
        roi=roi,
        roi_interactive=args.roi_interactive,
        subtitle_color=args.subtitle_color,
        white_rgb_threshold=args.white_rgb_threshold,
        hsv_white_v_min=args.hsv_white_v_min,
        hsv_white_s_max=args.hsv_white_s_max,
        black_rgb_threshold=args.black_rgb_threshold,
        hsv_black_v_max=args.hsv_black_v_max,
        hsv_black_s_max=args.hsv_black_s_max,
        morph_kernel_size=args.morph_kernel_size,
        open_iterations=args.open_iterations,
        close_iterations=args.close_iterations,
        dilate_iterations=args.dilate_iterations,
        min_component_area=args.min_component_area,
        merge_distance=args.merge_distance,
        temporal_window=args.temporal_window,
        temporal_mode=args.temporal_mode,
        temporal_vote_min=args.temporal_vote_min,
        backend=args.backend,
        lama_device=args.lama_device,
        lama_dilate_extra=args.lama_dilate_extra,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        output_video_codec=args.codec,
        output_pix_fmt=args.pix_fmt,
        output_crf=args.crf,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = args_to_config(args)
        run_pipeline(cfg)
        return 0
    except FileNotFoundError as e:
        log.error("File not found: %s", e)
        return 2
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        return 3
    except Exception as e:  # pragma: no cover
        log.error("Pipeline failed: %s", e)
        log.debug("%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
