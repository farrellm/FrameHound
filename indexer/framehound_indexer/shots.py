"""Shot boundaries, and the ffmpeg frame reader the visual stages share.

The detector is a numpy port of PySceneDetect's ``AdaptiveDetector``: each
frame's content score is the mean absolute HSV change from the previous frame,
and a frame starts a new shot when its score is ``ratio`` times the average of
its ``window`` neighbours on each side (and at least ``min_content``). Comparing
with the neighbourhood instead of a fixed threshold keeps fast camera motion
from reading as a cut.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Probe:
    width: int
    height: int
    duration_s: float
    aspect: float  # display aspect ratio (sample aspect applied)
    fps: float


def probe(media: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,sample_aspect_ratio,avg_frame_rate,duration:format=duration", "-of", "json",
         str(media)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    st = data["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    sar = 1.0
    if st.get("sample_aspect_ratio", "0:1") not in ("0:1", "1:1", "N/A", None):
        num, den = st["sample_aspect_ratio"].split(":")
        sar = int(num) / int(den) if int(den) else 1.0
    duration = float(data.get("format", {}).get("duration") or st.get("duration") or 0)
    num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return Probe(w, h, duration, w * sar / h, fps if 1 <= fps <= 240 else 24.0)


def even(x: float) -> int:
    return max(2, round(x / 2) * 2)


def video_frames(media: Path, vf: str, width: int, height: int, hwaccel: bool = False,
                 batch: int = 256) -> Iterator[np.ndarray]:
    """Decode ``media`` through filter ``vf`` (which must end at width×height rgb24)
    and yield uint8 batches of shape (n, height, width, 3).

    With ``hwaccel``, decoding is tried on the GPU (NVDEC) first; if ffmpeg fails
    before producing a frame, it is retried on the CPU.
    """
    attempts = [True, False] if hwaccel else [False]
    for hw in attempts:
        produced = False
        try:
            for frames in _decode(media, vf, width, height, hw, batch):
                produced = True
                yield frames
            return
        except RuntimeError as e:
            if produced or not hw:
                raise
            print(f"GPU decode failed ({e}); decoding on the CPU", file=sys.stderr)


def _decode(media: Path, vf: str, width: int, height: int, hw: bool,
            batch: int) -> Iterator[np.ndarray]:
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file"]
    if hw:
        argv += ["-hwaccel", "cuda"]
    argv += ["-i", str(media), "-map", "0:v:0", "-an", "-sn", "-vf", vf,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL)
    err: list[bytes] = []
    t = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
    t.start()
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size * batch)
            if not buf:
                break
            n = len(buf) // size
            if n:
                yield np.frombuffer(buf[: n * size], dtype=np.uint8).reshape(n, height, width, 3)
    finally:
        proc.stdout.close()
        rc = proc.wait()
        t.join(5)
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc}: {b''.join(err).decode(errors='replace')[-500:]}")


def hsv(frames: np.ndarray) -> np.ndarray:
    """RGB uint8 (n, h, w, 3) → HSV float32 with H in [0, 180) and S, V in [0, 255],
    OpenCV's ranges, which PySceneDetect's thresholds assume."""
    rgb = frames.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = v - mn
    s = np.where(v > 0, d / np.maximum(v, 1e-6) * 255.0, 0.0)
    safe = np.maximum(d, 1e-6)
    h = np.where(v == r, (g - b) / safe, np.where(v == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe))
    h = np.where(d > 0, (h * 30.0) % 180.0, 0.0)
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def content_scores(frames: np.ndarray, prev: np.ndarray | None = None) -> tuple[np.ndarray,
                                                                                  np.ndarray]:
    """Mean |ΔH|, |ΔS|, |ΔV| between consecutive frames, averaged over channels.

    ``prev`` is the last HSV frame of the previous batch (None for the first,
    whose first score is 0). Returns (scores, last HSV frame).
    """
    cur = hsv(frames)
    first = cur[:1] if prev is None else prev[None]
    stacked = np.concatenate([first, cur])
    diff = np.abs(np.diff(stacked, axis=0))
    # Hue is circular.
    diff[..., 0] = np.minimum(diff[..., 0], 180.0 - diff[..., 0])
    return diff.mean(axis=(1, 2, 3)).astype(np.float32), cur[-1]


def adaptive_cuts(scores: np.ndarray, fps: float, ratio: float = 3.0, min_content: float = 15.0,
                  window: int = 2, min_shot_s: float = 0.6) -> list[int]:
    """Frame indices where a new shot starts (excluding frame 0)."""
    n = len(scores)
    min_len = max(round(min_shot_s * fps), 1)
    cuts: list[int] = []
    last = 0
    for i in range(1, n):
        s = scores[i]
        if s < min_content or i - last < min_len:
            continue
        nb = np.concatenate([scores[max(1, i - window):i], scores[i + 1:i + 1 + window]])
        avg = float(nb.mean()) if nb.size else 0.0
        r = s / avg if avg > 1e-6 else 255.0
        if r >= ratio:
            cuts.append(i)
            last = i
    return cuts


def detect_shots(media: Path, cfg: dict, duration_s: float, hwaccel: bool = False,
                 on_progress: Callable[[float], None] | None = None) -> list[tuple[float, float]]:
    """Decode every frame small and return [(start, end)] seconds per shot."""
    p = probe(media)
    duration = duration_s or p.duration_s
    w = even(int(cfg.get("width", 96)))
    h = even(w / p.aspect)
    scores: list[np.ndarray] = []
    prev = None
    n = 0
    for frames in video_frames(media, f"scale={w}:{h}:flags=area,format=rgb24", w, h, hwaccel):
        sc, prev = content_scores(frames, prev)
        scores.append(sc)
        n += len(frames)
        if on_progress and duration:
            on_progress(min(n / (duration * p.fps), 0.99))
    all_scores = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    if n == 0 or duration <= 0:
        return [(0.0, max(duration, 0.0))]
    fps = n / duration
    cuts = adaptive_cuts(all_scores, fps, ratio=cfg.get("ratio", 3.0),
                         min_content=cfg.get("min_content", 15.0), window=cfg.get("window", 2),
                         min_shot_s=cfg.get("min_shot_s", 0.6))
    dt = duration / n
    bounds = [0, *cuts, n]
    return [(round(a * dt, 3), round(b * dt if b < n else duration, 3))
            for a, b in itertools.pairwise(bounds)]
