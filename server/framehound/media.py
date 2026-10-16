"""ffprobe / ffmpeg under resource limits, with a file-only protocol whitelist."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .errors import Cancelled, FHError

FORMAT_ALLOW = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2", "matroska", "webm", "avi", "mpeg",
                "mpegts", "ogg", "flv", "asf"}
VIDEO_CODEC_ALLOW = {"h264", "hevc", "mpeg4", "mpeg2video", "mpeg1video", "vp8", "vp9", "av1",
                     "theora", "msmpeg4v2", "msmpeg4v3", "mjpeg", "wmv2", "wmv3", "vc1", "cinepak",
                     "flv1", "h263", "prores", "dvvideo"}


def limited(cmd: list[str]) -> list[str]:
    """Wrap ``cmd`` in prlimit (CPU, memory, file size).

    prlimit rather than ``preexec_fn``, which can deadlock when jobs run in
    threads (the api's inline mode).
    """
    s = settings()
    if not shutil.which("prlimit"):
        return cmd
    mem = s.ffmpeg_mem_mb * 1024 * 1024
    return ["prlimit", f"--cpu={s.ffmpeg_cpu_s}", f"--as={mem}",
            f"--fsize={s.max_bytes * 2}", "--", *cmd]


@dataclass
class Probe:
    duration_s: float
    width: int
    height: int
    fps: float
    format_name: str
    video_codec: str
    has_audio: bool


def ffprobe(path: Path) -> Probe:
    try:
        out = subprocess.run(
            limited(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-print_format",
                     "json", "-show_format", "-show_streams", str(path)]),
            capture_output=True, check=True, timeout=120,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise FHError("bad_media", "That file isn't a movie FrameHound can read.",
                      detail=e.stderr.decode(errors="replace")[-500:]) from e
    info = json.loads(out)
    video = next((st for st in info.get("streams", []) if st.get("codec_type") == "video"
                  and not st.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise FHError("bad_media", "That file has no video in it.", detail="no video stream")
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or video.get("duration") or 0)
    num, _, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return Probe(
        duration_s=duration, width=int(video.get("width") or 0),
        height=int(video.get("height") or 0), fps=round(fps, 3) if fps else 24.0,
        format_name=fmt.get("format_name", ""), video_codec=video.get("codec_name", ""),
        has_audio=any(st.get("codec_type") == "audio" for st in info.get("streams", [])),
    )


def validate(p: Probe, size: int) -> None:
    s = settings()
    if not set(p.format_name.split(",")) & FORMAT_ALLOW:
        raise FHError("bad_media", "FrameHound doesn't support this kind of video file.",
                      detail=f"container {p.format_name}")
    if p.video_codec not in VIDEO_CODEC_ALLOW:
        raise FHError("bad_media", "FrameHound doesn't support this video's format.",
                      detail=f"codec {p.video_codec}")
    if p.duration_s <= 0:
        raise FHError("bad_media", "FrameHound couldn't tell how long this movie is.",
                      detail="duration 0")
    if p.duration_s > s.max_duration_s:
        hours = s.max_duration_s / 3600
        raise FHError("too_long", f"This movie is longer than the {hours:g}-hour limit.",
                      detail=f"duration {p.duration_s:.0f}s")
    if size > s.max_bytes:
        raise FHError("too_big", "This file is larger than the size limit.",
                      detail=f"{size} bytes > {s.max_bytes}")


def run_ffmpeg(args: list[str], duration_s: float, on_progress: Callable[[float], None],
               check_cancel: Callable[[], None] = lambda: None) -> None:
    """Run ffmpeg with ``-progress``; report fraction done; kill on cancel."""
    s = settings()
    threads = ["-threads", str(s.ffmpeg_threads)] if s.ffmpeg_threads else []
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y",
           "-progress", "pipe:1", "-nostats", *threads, *args]
    proc = subprocess.Popen(limited(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            key, _, val = line.strip().partition("=")
            if key == "out_time_us" and val.isdigit() and duration_s > 0:
                on_progress(min(int(val) / 1e6 / duration_s, 1.0))
            try:
                check_cancel()
            except Cancelled:
                proc.kill()
                raise
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read() if proc.stderr else ""
        raise FHError("ffmpeg_failed", "FrameHound couldn't process this video.",
                      detail=err[-800:])


def transcode_proxy(src: Path, dst: Path, duration_s: float, on_progress, check_cancel) -> None:
    tmp = dst.with_name(dst.stem + ".part.mp4")
    run_ffmpeg(
        ["-protocol_whitelist", "file", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", "scale=-2:'min(720,ih)':flags=bicubic,format=yuv420p",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-movflags", "+faststart", str(tmp)],
        duration_s, on_progress, check_cancel,
    )
    tmp.replace(dst)


CROP_EXPAND = 2.5


def face_crop(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """The crop (w, h, x, y in pixels) for a face at ``box`` (x, y, w, h as fractions
    of a ``width``×``height`` frame): the face expanded 2.5× around its centre, grown
    to portrait 9:16, or to 1:1 when 9:16 won't fit the frame, then kept inside it.
    A fixed box at the peak; following a moving face is out of scope (design.md §7)."""
    fx, fy, fw, fh = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    cx, cy = fx + fw / 2, fy + fh / 2
    side = max(fw, fh) * CROP_EXPAND
    for aspect in (9 / 16, 1.0):  # width / height: the narrowest box holding the square
        w, h = side, side / aspect
        if h <= height and w <= width:
            break
    # Too big for the frame either way: the largest box of that aspect that fits.
    scale = min(1.0, width / w, height / h)
    w, h = w * scale, h * scale
    w, h = max(2, int(w) // 2 * 2), max(2, int(h) // 2 * 2)
    x = round(min(max(cx - w / 2, 0), width - w))
    y = round(min(max(cy - h / 2, 0), height - h))
    return w, h, x, y


def cut_clip(src: Path, dst: Path, start: float, end: float, mode: str,
             on_progress, check_cancel, crop: tuple[int, int, int, int] | None = None) -> None:
    dur = end - start
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    head = ["-ss", f"{start:.3f}", "-protocol_whitelist", "file", "-i", str(src),
            "-t", f"{dur:.3f}", "-map", "0:v:0", "-map", "0:a:0?"]
    if mode == "fast" and crop is None:
        body = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        vf = ["-vf", "crop={}:{}:{}:{}".format(*crop)] if crop else []
        body = [*vf, "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k"]
    run_ffmpeg([*head, *body, "-movflags", "+faststart", str(tmp)], dur, on_progress,
               check_cancel)
    tmp.replace(dst)
