"""Gradio web UI for the subtitle remover.

Wraps :func:`app.pipeline.run_pipeline` so a non-technical user can drag a
video into the browser, draw a subtitle box on the first frame, and get a
cleaned video back. All ffmpeg/ffprobe binaries come from the
``static-ffmpeg`` pip package, so the user does not need to install ffmpeg
themselves.

Run with::

    python -m app.web
"""

from __future__ import annotations

import shutil
import socket
import tempfile
import threading
import time
import uuid
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from . import __version__
from .config import AppConfig
from .pipeline import run_pipeline


# ---------------------------------------------------------------------------
# ffmpeg/ffprobe resolution (static-ffmpeg)
# ---------------------------------------------------------------------------

def _resolve_ffmpeg_bins() -> Tuple[str, str]:
    """Return (ffmpeg, ffprobe) paths.

    Tries ``static_ffmpeg`` first (the bundled binaries), then falls back
    to whatever is on PATH so a developer with brew-installed ffmpeg can
    still run this directly.
    """
    try:
        from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

        ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
        return ffmpeg, ffprobe
    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    return ffmpeg, ffprobe


FFMPEG_BIN, FFPROBE_BIN = _resolve_ffmpeg_bins()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_first_frame(video_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    # Gradio expects RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _draw_roi_overlay(frame_rgb: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Return a copy of the frame with a translucent red rectangle drawn."""
    img = frame_rgb.copy()
    H, W = img.shape[:2]
    x1 = max(0, min(int(x), W - 1))
    y1 = max(0, min(int(y), H - 1))
    x2 = max(0, min(int(x) + int(w), W))
    y2 = max(0, min(int(y) + int(h), H))
    if x2 > x1 and y2 > y1:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), thickness=-1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
    return img


def _corners_to_roi(
    left_x: int,
    bottom_y: int,
    right_x: int,
    top_y: int,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(left_x), int(right_x)))
    x2 = min(frame_w, max(int(left_x), int(right_x)))
    y1 = max(0, min(int(top_y), int(bottom_y)))
    y2 = min(frame_h, max(int(top_y), int(bottom_y)))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _wait_and_open_browser(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(0.5)


def _find_available_port(preferred: int = 7860, search_span: int = 50) -> int:
    for port in range(preferred, preferred + search_span):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise OSError(f"Cannot find empty port in range: {preferred}-{preferred + search_span - 1}")


def _video_dims(video_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return w, h


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def on_video_uploaded(video_path: Optional[str]):
    """When a video is uploaded, extract the first frame and set up sliders."""
    if not video_path:
        return (
            None,
            gr.update(maximum=1920, value=0),
            gr.update(maximum=1080, value=0),
            gr.update(maximum=1920, value=1920),
            gr.update(maximum=1080, value=900),
            "",
        )

    frame = _read_first_frame(video_path)
    if frame is None:
        return (
            None,
            gr.update(maximum=1920, value=0),
            gr.update(maximum=1080, value=0),
            gr.update(maximum=1920, value=1920),
            gr.update(maximum=1080, value=900),
            "无法读取视频第一帧",
        )

    H, W = frame.shape[:2]
    # Default ROI: bottom 18% band, full width
    h_band = max(1, int(H * 0.18))
    left_x, bottom_y, right_x, top_y = 0, H, W, H - h_band
    x0, y0, w0, h0 = _corners_to_roi(left_x, bottom_y, right_x, top_y, W, H)
    preview = _draw_roi_overlay(frame, x0, y0, w0, h0)

    return (
        preview,
        gr.update(maximum=W, value=left_x),
        gr.update(maximum=H, value=bottom_y),
        gr.update(maximum=W, value=right_x),
        gr.update(maximum=H, value=top_y),
        f"视频尺寸: {W} × {H}",
    )


def on_roi_changed(video_path: Optional[str], left_x: int, bottom_y: int, right_x: int, top_y: int):
    if not video_path:
        return None
    frame = _read_first_frame(video_path)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x, y, roi_w, roi_h = _corners_to_roi(left_x, bottom_y, right_x, top_y, w, h)
    return _draw_roi_overlay(frame, x, y, roi_w, roi_h)


def on_preset_bottom(video_path: Optional[str]):
    """Set ROI to bottom 18% band."""
    if not video_path:
        return gr.update(), gr.update(), gr.update(), gr.update(), None
    W, H = _video_dims(video_path)
    h_band = max(1, int(H * 0.18))
    left_x, bottom_y, right_x, top_y = 0, H, W, H - h_band
    frame = _read_first_frame(video_path)
    if frame is not None:
        x0, y0, w0, h0 = _corners_to_roi(left_x, bottom_y, right_x, top_y, W, H)
        preview = _draw_roi_overlay(frame, x0, y0, w0, h0)
    else:
        preview = None
    return left_x, bottom_y, right_x, top_y, preview


def process_video(
    video_path: Optional[str],
    left_x: int,
    bottom_y: int,
    right_x: int,
    top_y: int,
    subtitle_color: str,
    process_mode: str,
    preview_only: bool,
    dilate_iterations: int,
    lama_dilate_extra: int,
    temporal_window: int,
    progress: gr.Progress = gr.Progress(),
):
    if not video_path:
        raise gr.Error("请先上传一个视频")
    frame_w, frame_h = _video_dims(video_path)
    x, y, w, h = _corners_to_roi(left_x, bottom_y, right_x, top_y, frame_w, frame_h)
    if w <= 1 or h <= 1:
        raise gr.Error("请把左下角和右上角框开，形成有效的字幕区域")

    progress(0.05, desc="准备工作目录...")
    request_id = uuid.uuid4().hex[:8]
    work_dir = Path(tempfile.gettempdir()) / f"VideoRemoveText_{request_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "output.mp4"

    cfg = AppConfig(
        input_path=video_path,
        output_path=str(output_path),
        work_dir=str(work_dir),
        save_intermediate=False,
        preview=bool(preview_only),
        preview_seconds=3.0,
        roi=(int(x), int(y), int(w), int(h)),
        roi_interactive=False,
        subtitle_color=str(subtitle_color),
        dilate_iterations=int(dilate_iterations),
        temporal_window=int(temporal_window),
        backend="opencv" if process_mode == "fast" else "lama",
        lama_device="auto",
        lama_dilate_extra=int(lama_dilate_extra),
        ffmpeg_bin=FFMPEG_BIN,
        ffprobe_bin=FFPROBE_BIN,
    )

    if process_mode == "fast":
        progress(0.05, desc="开始处理（速度优先）...")
    else:
        progress(0.05, desc="开始处理（质量优先，首次会下载约 200MB 模型）...")

    # Run the pipeline in a worker thread so we can tick the progress bar.
    # The pipeline itself is synchronous and emits its own tqdm bars to the
    # terminal — we just need to keep the browser-side bar visibly alive.
    result_holder: dict = {}

    def _worker():
        try:
            result_holder["path"] = run_pipeline(cfg)
        except Exception as e:  # noqa: BLE001
            result_holder["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    start_ts = time.time()
    # Animate the bar between 5% and 95% based on a soft estimate so the user
    # gets feedback. We don't know the true total cost, so we use an asymptote.
    while t.is_alive():
        elapsed = time.time() - start_ts
        # Soft progress: approaches 0.95 but never reaches it until the worker is done.
        # 60s elapsed → ~0.55, 120s → ~0.78, 240s → ~0.91
        frac = 0.05 + 0.90 * (1 - 1 / (1 + elapsed / 45.0))
        progress(frac, desc=f"处理中... 已用时 {int(elapsed)}s")
        time.sleep(1.0)

    t.join()

    if "error" in result_holder:
        raise gr.Error(f"处理失败: {result_holder['error']}")

    progress(1.0, desc=f"完成 (用时 {int(time.time() - start_ts)}s)")
    return result_holder["path"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="视频字幕去除器") as demo:
        gr.Markdown(
            """
            # 视频字幕去除器
            上传一段视频，框选字幕所在的区域，点击运行即可去除硬字幕。
            首次运行需要下载约 200 MB 的 AI 模型权重，请耐心等待。
            """
        )
        gr.Markdown(f"当前界面版本：`v{__version__}`")

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="1. 上传视频", sources=["upload"])
                info_md = gr.Markdown("")

                gr.Markdown("### 2. 框选字幕区域")
                roi_preview = gr.Image(
                    label="字幕区域预览（红框就是会被去除的区域）",
                    interactive=False,
                    height=300,
                )
                gr.Markdown("坐标改成两个角点：左下角 `(x, y)` 和右上角 `(x, y)`。")
                with gr.Row():
                    preset_btn = gr.Button("⬇ 底部 18%（默认）", size="sm")

                left_x_slider = gr.Slider(0, 1920, value=0, step=1, label="左下角 X")
                bottom_y_slider = gr.Slider(0, 1080, value=1080, step=1, label="左下角 Y")
                right_x_slider = gr.Slider(0, 1920, value=1920, step=1, label="右上角 X")
                top_y_slider = gr.Slider(0, 1080, value=900, step=1, label="右上角 Y")

            with gr.Column(scale=1):
                gr.Markdown("### 3. 字幕颜色")
                color_radio = gr.Radio(
                    choices=["white", "black"],
                    value="white",
                    label="字幕颜色",
                    info="白字（最常见）选 white；黑字描边选 black",
                )

                gr.Markdown("### 4. 运行模式")
                mode_radio = gr.Radio(
                    choices=[
                        ("速度优先（推荐）", "fast"),
                        ("质量优先（LaMa）", "quality"),
                    ],
                    value="fast",
                    label="处理模式",
                    info="速度优先会快很多；复杂背景可以切到质量优先",
                )
                preview_chk = gr.Checkbox(
                    value=True,
                    label="仅预览前 3 秒（建议先勾选验证效果）",
                )

                with gr.Accordion("高级设置（一般不用动）", open=False):
                    dilate_slider = gr.Slider(
                        0, 6, value=2, step=1,
                        label="膨胀迭代次数",
                        info="字幕没去干净就调大",
                    )
                    lama_extra_slider = gr.Slider(
                        0, 12, value=4, step=1,
                        label="LaMa 额外膨胀",
                        info="残留边缘 / 光晕调大",
                    )
                    temporal_slider = gr.Slider(
                        0, 5, value=2, step=1,
                        label="时域平滑窗口",
                        info="闪烁严重调大到 3-4",
                    )

                run_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

                gr.Markdown("### 5. 处理结果")
                video_out = gr.Video(label="输出视频", interactive=False)

        # ---- Wire up events ----
        video_in.change(
            on_video_uploaded,
            inputs=[video_in],
            outputs=[roi_preview, left_x_slider, bottom_y_slider, right_x_slider, top_y_slider, info_md],
        )

        for s in (left_x_slider, bottom_y_slider, right_x_slider, top_y_slider):
            s.release(
                on_roi_changed,
                inputs=[video_in, left_x_slider, bottom_y_slider, right_x_slider, top_y_slider],
                outputs=[roi_preview],
            )

        preset_btn.click(
            on_preset_bottom,
            inputs=[video_in],
            outputs=[left_x_slider, bottom_y_slider, right_x_slider, top_y_slider, roi_preview],
        )

        run_btn.click(
            process_video,
            inputs=[
                video_in,
                left_x_slider, bottom_y_slider, right_x_slider, top_y_slider,
                color_radio,
                mode_radio,
                preview_chk,
                dilate_slider,
                lama_extra_slider,
                temporal_slider,
            ],
            outputs=[video_out],
        )

    return demo


def main() -> None:
    # Make sure static-ffmpeg has fetched its binaries before launch
    print(f"[web] ffmpeg = {FFMPEG_BIN}")
    print(f"[web] ffprobe = {FFPROBE_BIN}")
    demo = build_ui()
    port = _find_available_port(7860, 50)
    url = f"http://127.0.0.1:{port}"
    print(f"[web] url = {url}")
    threading.Thread(target=_wait_and_open_browser, args=(url,), daemon=True).start()
    demo.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        share=False,
    )


if __name__ == "__main__":
    main()
