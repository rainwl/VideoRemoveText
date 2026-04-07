# Hard Subtitle Remover (No-OCR, Color-Mask + AI Inpainting)

A small, self-contained Python project that removes burned-in (hard)
subtitles from a local video. Detection is **purely based on a fixed
ROI plus a color threshold** — there is no OCR, no text detection
model, and no commercial API. The actual filling-in is done by an AI
inpainting backend (LaMa by default; ProPainter / E2FGVI ports are
stubbed and ready to plug in).

## Features

- Fixed-ROI subtitle detection (CLI tuple **or** interactive box drawing).
- Two color modes: `white` and `black`, each combining RGB + HSV
  thresholds for robust anti-aliased edge handling.
- Morphological cleanup, connected-component filtering and small-gap
  merging so a full line of text becomes one mask blob.
- Simple temporal smoothing (`union` or `vote`) over a sliding window
  to kill flicker and recover the occasional missed frame.
- Pluggable inpainting backend (`lama` and `opencv` work out of the box;
  `propainter` / `e2fgvi` stubs included).
- Original audio is preserved automatically.
- Preview mode that processes only the first 3 seconds.
- Faster LaMa path: decode frames directly into memory and only
  inpaint a tight crop around the subtitle mask instead of the whole
  frame.
- Saves intermediate `frames/`, `masks/`, `repaired_frames/` and
  side-by-side `preview/` images for debugging.

## Project layout

```
app/
  main.py              # CLI entry
  pipeline.py          # end-to-end orchestration
  config.py            # AppConfig dataclass (all knobs)
  models.py            # ROI / VideoInfo dataclasses
  roi/selector.py      # CLI parsing + interactive cv2.selectROI
  subtitle_mask/color_mask.py
  temporal/smoother.py
  inpainting/
    base.py
    lama_backend.py
    propainter_backend.py    # stub
    e2fgvi_backend.py        # stub
    factory.py
  ffmpeg_tools/ffmpeg_wrapper.py
scripts/download_lama.sh
requirements.txt
README.md
```

## 1. Install

### 1.1 System: FFmpeg

You need both `ffmpeg` and `ffprobe` available on `PATH`.

| OS | Command |
|----|---------|
| macOS (Homebrew) | `brew install ffmpeg` |
| Ubuntu / Debian  | `sudo apt-get install -y ffmpeg` |
| Windows          | Download from https://www.gyan.dev/ffmpeg/builds/ and add `bin/` to `PATH` |

Verify:

```bash
ffmpeg -version
ffprobe -version
```

### 1.2 Python deps

Python 3.10+ is recommended. Inside a virtualenv:

```bash
pip install -r requirements.txt
```

The `simple-lama-inpainting` package brings in `torch` and downloads
the LaMa weights on first use (~200MB) into its own cache. To force
the download up front:

```bash
bash scripts/download_lama.sh
```

GPU is optional. The pipeline auto-picks `cuda` → `mps` (Apple
Silicon) → `cpu`. Override with `--lama-device cpu`.

## 2. Quick start

Process a 1080p video where subtitles sit in the bottom 180 px band:

```bash
python -m app.main \
    --input  input.mp4 \
    --output output.mp4 \
    --roi    0,820,1920,180 \
    --subtitle-color white
```

For the fastest non-debug run, add `--no-save-intermediate` so the
pipeline skips writing mask / preview artifacts.

Pick ROI visually instead:

```bash
python -m app.main -i input.mp4 -o output.mp4 --roi-interactive
```

A window opens on the first frame; drag a rectangle, press
**Enter/Space** to confirm. In the web UI, the ROI is controlled by
two corner points: left-bottom `(x, y)` and right-top `(x, y)`.

Quickly preview the first 3 seconds:

```bash
python -m app.main -i input.mp4 -o preview.mp4 --roi 0,820,1920,180 \
    --subtitle-color white --preview
```

Check `work/preview/preview_*.png` for `[original | mask | repaired]`
side-by-side debug images.

## 3. Tuning the color mask

If your subtitles are not being fully removed, the most common fixes:

- **White text on bright backgrounds** → loosen `--white-rgb-threshold`
  to e.g. `185`, raise `--dilate-iterations` to `3`.
- **Black text with thick outlines** → use `--subtitle-color black` and
  raise `--hsv-black-v-max` to `90`, `--hsv-black-s-max` to `110`.
- **Halo / leftover edges** → increase `--lama-dilate-extra` to `6–8`.
- **LaMa crops too tightly around subtitles** → increase
  `--lama-crop-padding` to `64–96`.
- **Text not connected into single blob** → raise `--merge-distance` to
  `10–14`.
- **False positives in the background** → narrow the ROI, or switch
  `--temporal-mode vote` with `--temporal-vote-min 3`.

All knobs are listed in `python -m app.main --help`.

## 4. Switching the inpainting backend

The default CLI backend is LaMa (image-level, higher quality):

```bash
--backend lama
```

For a much faster run, use the OpenCV backend:

```bash
--backend opencv --no-save-intermediate
```

The `propainter` and `e2fgvi` choices are wired into the factory but
intentionally left as stubs — the upstream repos require their own
checkpoints and a fairly intrusive install. To plug them in:

1. Clone the upstream repo and install its requirements.
2. Drop checkpoints under `weights/<Name>/`.
3. Implement `inpaint_video()` in
   `app/inpainting/propainter_backend.py` /
   `app/inpainting/e2fgvi_backend.py`. The contract is documented in
   each file's docstring.

The pipeline already calls `backend.inpaint_video(frames, masks)`, so
swapping is a one-line change once the backend is implemented.

## 5. Output

- `output.mp4` — final video with subtitles removed and original audio.
- `work/frames/` — extracted source frames.
- `work/masks/` — final per-frame subtitle masks (after smoothing).
- `work/repaired_frames/` — inpainted frames before re-encoding.
- `work/preview/` — sample side-by-side debug images.

Use `--no-save-intermediate` to skip masks/preview if you only want
the final video.

## 6. Troubleshooting

| Symptom | Most likely cause |
|---------|-------------------|
| `ffmpeg not found on PATH` | Install FFmpeg, restart shell |
| `simple-lama-inpainting is not installed` | `pip install simple-lama-inpainting` |
| `Failed to read first frame` | Video is corrupt / unsupported codec; try `ffmpeg -i input.mp4 -c copy fixed.mp4` |
| Subtitles partially removed | Lower the RGB threshold (white) or raise it (black); add more `--dilate-iterations` |
| Background ghosting | Increase `--lama-dilate-extra`; consider switching to a video backend |
| Flickering between frames | Increase `--temporal-window` to `3–4` |
| Processing is still too slow | Add `--backend opencv --no-save-intermediate`; if you stay on LaMa, lower `--lama-crop-padding` if the ROI is already tight |

## 7. Limitations (current implementation)

- Works best when the subtitle area is **fixed and known**. Moving
  captions need a wider ROI.
- LaMa is an *image* inpainter, so consecutive frames may differ
  slightly. The temporal smoother fights this on the *mask* side, but
  for cinematic results swap in ProPainter / E2FGVI.
- Very complex backgrounds (fast motion, fine textures) directly under
  the subtitle may show mild artifacts.
- Same-color text on same-color background (e.g. white text on a
  white wall) is fine to detect but offers little for the inpainter to
  reconstruct.

## 8. License

MIT for the project skeleton itself. Each integrated model
(LaMa / ProPainter / E2FGVI) keeps its own license — please check the
upstream repos before commercial use.
