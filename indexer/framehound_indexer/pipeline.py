"""The indexing pipeline: ``python -m framehound_indexer job.json``.

Each stage writes its output and a ``{stage}.done`` marker under
``{work_dir}/work/``; a relaunch in the same work dir resumes at the first
stage without a marker. Progress is appended to ``{work_dir}/progress.jsonl``
as ``{"stage", "state", "pct", "eta_s", "t"}`` lines, ending with
``{"stage": "done"}`` or ``{"stage": "error", "message": ...}``.

job.json::

    {
      "movie_id": "...",
      "work_dir": "/content/fh/{movie_id}",
      "media": {"path": "/data/movies/.../source.mp4"}   # or {"url": ..., "sha256": ...}
      "duration_s": 5543.2,
      "models_lock": {...},          # the host's lock, so both sides agree
      "stages": ["fetch", "shots", "visual", "faces", "audio", "sentences", "bundle"],
      "device": "auto",              # auto | cpu | cuda
      "zstd_level": 19
    }

The audio lane (audio, sentences) runs in a thread alongside the visual lane
(shots, visual, faces). On a GPU the GPU-bound stages (audio, visual, faces) take
turns, since sharing a saturated GPU only slows both and spoils their ETAs;
shot detection (CPU and NVDEC) still overlaps Whisper. On a CPU the lanes
overlap freely, so ffmpeg decoding overlaps inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, bundle
from .lock import ModelsLock, load_lock

DEFAULT_STAGES = ["fetch", "shots", "visual", "faces", "audio", "sentences", "bundle"]
AUDIO_LANE = {"audio", "sentences"}
GPU_STAGES = {"audio", "visual", "faces"}  # each saturates the GPU on its own


class Progress:
    """Appends progress lines, rate-limited per stage, with a naive ETA."""

    def __init__(self, path: Path):
        self.fh = open(path, "a", buffering=1)  # noqa: SIM115 (lives for the run)
        self.started: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.lock = threading.Lock()

    def emit(self, stage: str, pct: float, state: str = "running", force: bool = False, **kw):
        with self.lock:
            self._emit(stage, pct, state, force, **kw)

    def _emit(self, stage: str, pct: float, state: str, force: bool, **kw):
        now = time.time()
        start = self.started.setdefault(stage, now)
        if not force and state == "running" and now - self.last.get(stage, 0) < 1.0:
            return
        self.last[stage] = now
        eta = None
        if state == "done":
            eta = 0
        elif pct > 0.02:
            eta = round((now - start) * (1 - pct) / pct)
        line = {"stage": stage, "state": state, "pct": round(min(max(pct, 0.0), 1.0), 4),
                "eta_s": eta, "t": round(now, 3), **kw}
        self.fh.write(json.dumps(line) + "\n")


@dataclass
class Ctx:
    job: dict[str, Any]
    work: Path  # {work_dir}/work
    out: Path  # {work_dir}
    lock: ModelsLock
    progress: Progress
    durations: dict[str, float] = field(default_factory=dict)
    gpu_turn: threading.Lock | None = None  # held by a GPU stage while it runs

    def media(self) -> Path:
        found = sorted(self.work.glob("media.*"))
        if not found:
            raise RuntimeError("fetch stage produced no media file")
        return found[0]

    def device(self) -> str:
        want = self.job.get("device", "auto")
        if want != "auto":
            return want
        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001 (no usable CUDA runtime)
            return "cpu"


    def ort_device(self) -> str:
        """Where SigLIP runs: CUDA if requested/available to onnxruntime."""
        want = self.job.get("device", "auto")
        if want == "cpu":
            return "cpu"
        from .encoders import preload_cuda, providers

        if want == "auto" and self.device() != "cuda":
            return "cpu"
        preload_cuda()
        return "cuda" if "CUDAExecutionProvider" in providers("cuda") else "cpu"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def stage_fetch(ctx: Ctx) -> None:
    media = ctx.job["media"]
    if "path" in media:
        src = Path(media["path"])
        dst = ctx.work / f"media{src.suffix or '.bin'}"
        dst.unlink(missing_ok=True)
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copyfile(src, dst)
        return
    import httpx

    url, want_sha = media["url"], media.get("sha256")
    suffix = Path(url.split("?")[0]).suffix[:6] or ".bin"
    part = ctx.work / f"download{suffix}.part"
    h = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(part, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if total:
                    ctx.progress.emit("fetch", done / total)
    if want_sha and h.hexdigest() != want_sha:
        part.unlink()
        raise RuntimeError(f"sha256 mismatch: got {h.hexdigest()}, want {want_sha}")
    part.rename(ctx.work / f"media{suffix}")


def compute_type(device: str) -> str:
    """Best compute type the device supports (old GPUs lack efficient fp16)."""
    import ctranslate2

    supported = ctranslate2.get_supported_compute_types(device)
    prefs = ["float16", "int8_float16", "int8", "float32"] if device == "cuda" else ["int8", "float32"]
    return next((t for t in prefs if t in supported), "default")


def _transcribe(ctx: Ctx, audio: np.ndarray, device: str) -> dict[str, list]:
    from faster_whisper import WhisperModel

    from .encoders import snapshot

    cfg = ctx.lock.whisper
    model = WhisperModel(
        str(snapshot(cfg["repo"], cfg["revision"])),
        device=device,
        compute_type=compute_type(device),
        cpu_threads=int(ctx.job.get("cpu_threads") or os.cpu_count() or 4),
    )
    segments, info = model.transcribe(
        audio,
        beam_size=cfg.get("beam_size", 5),
        vad_filter=cfg.get("vad_filter", True),
        word_timestamps=True,
        language=cfg.get("language"),
        condition_on_previous_text=False,
    )
    duration = info.duration or float(ctx.job.get("duration_s") or 0) or 1.0
    rows: dict[str, list] = {"seg": [], "start": [], "end": [], "text": [], "words_json": []}
    max_nsp = cfg.get("max_no_speech_prob", 0.6)
    for seg in segments:
        ctx.progress.emit("audio", seg.end / duration)
        text = seg.text.strip()
        # Whisper's own heuristic for hallucinated text over music or silence.
        if not text or (seg.no_speech_prob > max_nsp and seg.avg_logprob < -1.0):
            continue
        words = [
            {"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
             "p": round(w.probability, 3)}
            for w in (seg.words or [])
            if w.word.strip()
        ]
        rows["seg"].append(len(rows["seg"]))
        rows["start"].append(float(words[0]["start"] if words else seg.start))
        rows["end"].append(float(words[-1]["end"] if words else seg.end))
        rows["text"].append(text)
        rows["words_json"].append(json.dumps(words))
    return rows


def stage_audio(ctx: Ctx) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Decode with ffmpeg ourselves and hand Whisper samples, so we don't depend on
    # faster-whisper's PyAV decoding (which breaks across PyAV releases).
    raw = ctx.work / "audio.f32"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file",
         "-i", str(ctx.media()), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", str(raw)],
        check=True,
    )
    audio = np.fromfile(raw, dtype=np.float32)
    device = ctx.device()
    print(f"audio: whisper on {device}", file=sys.stderr, flush=True)
    try:
        rows = _transcribe(ctx, audio, device)
    except RuntimeError as e:
        # "auto" can pick a GPU whose CUDA libraries (cuBLAS/cuDNN) aren't usable.
        if device != "cuda" or ctx.job.get("device", "auto") != "auto":
            raise
        print(f"CUDA transcription failed ({e}); falling back to CPU", file=sys.stderr)
        rows = _transcribe(ctx, audio, "cpu")
    table = pa.table(
        rows,
        schema=pa.schema([("seg", pa.int32()), ("start", pa.float64()), ("end", pa.float64()),
                          ("text", pa.string()), ("words_json", pa.string())]),
    )
    pq.write_table(table, ctx.work / bundle.TRANSCRIPT)
    raw.unlink(missing_ok=True)


def sentence_units(texts: list[str]) -> list[tuple[int, int, str]]:
    """Every segment, plus 3-segment sliding windows for lines that span segments."""
    units = [(i, i, t) for i, t in enumerate(texts)]
    units += [(i, i + 2, " ".join(texts[i : i + 3])) for i in range(len(texts) - 2)]
    return units


def stage_sentences(ctx: Ctx) -> None:
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .encoders import SentenceEncoder

    texts = pq.read_table(ctx.work / bundle.TRANSCRIPT, columns=["text"]).column("text")
    units = sentence_units(texts.to_pylist())
    enc = SentenceEncoder(ctx.lock)
    index = faiss.IndexFlatIP(enc.dim)
    batch = 256
    for i in range(0, len(units), batch):
        vecs = enc.encode([u[2] for u in units[i : i + batch]])
        index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        ctx.progress.emit("sentences", (i + batch) / max(len(units), 1))
    faiss.write_index(index, str(ctx.work / bundle.SENTENCES_INDEX))
    pq.write_table(
        pa.table({"row": pa.array(range(len(units)), pa.int32()),
                  "seg_lo": pa.array([u[0] for u in units], pa.int32()),
                  "seg_hi": pa.array([u[1] for u in units], pa.int32())}),
        ctx.work / bundle.SENTENCES_TABLE,
    )


def stage_shots(ctx: Ctx) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .shots import detect_shots

    shots = detect_shots(ctx.media(), ctx.lock.shots, float(ctx.job.get("duration_s") or 0),
                         hwaccel=ctx.device() == "cuda",
                         on_progress=lambda f: ctx.progress.emit("shots", f))
    pq.write_table(
        pa.table({"idx": pa.array(range(len(shots)), pa.int32()),
                  "start": pa.array([a for a, _ in shots], pa.float64()),
                  "end": pa.array([b for _, b in shots], pa.float64())}),
        ctx.work / bundle.SHOTS,
    )


THUMB_W = 320


def stage_visual(ctx: Ctx) -> None:
    """Decode at ``fps`` once: visual embeddings for every frame, plus each shot's
    thumbnail (the sampled frame nearest its middle)."""
    import queue

    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from .encoders import make_visual_encoder
    from .shots import even, probe, video_frames

    cfg = ctx.lock.visual
    fps = float(cfg.get("fps", 2))
    media = ctx.media()
    p = probe(media)
    duration = float(ctx.job.get("duration_s") or 0) or p.duration_s
    shots = pq.read_table(ctx.work / bundle.SHOTS).to_pydict()
    starts = np.asarray(shots["start"], dtype=np.float64)
    size = int(cfg["image_size"])
    h = max(size, THUMB_W * 9 // 16)
    w = even(h * p.aspect)
    h = even(h)
    thumbs_dir = ctx.work / bundle.THUMBS
    thumbs_dir.mkdir(exist_ok=True)
    # Frame k is at k / fps; each shot's thumbnail is the frame nearest its middle.
    want: dict[int, list[int]] = {}
    for idx, (a, b) in enumerate(zip(shots["start"], shots["end"], strict=True)):
        k = round((a + b) / 2 * fps)
        k = min(max(k, int(np.ceil(a * fps))), max(int(np.floor(b * fps - 1e-9)), 0))
        want.setdefault(max(k, 0), []).append(idx)

    device = ctx.ort_device()
    enc = make_visual_encoder(ctx.lock, device=device)
    print(f"visual: {enc.vision_backend()}", file=sys.stderr, flush=True)
    batch = int(cfg.get("batch_size", 32)) if device == "cuda" else 16
    index = faiss.IndexFlatIP(enc.dim)
    q: queue.Queue = queue.Queue(maxsize=4)
    # Fit thumbnails in a THUMB_W box (a portrait source stays portrait).
    tw, th = (THUMB_W, even(THUMB_W / p.aspect)) if p.aspect >= 1 else \
        (even(THUMB_W * p.aspect), THUMB_W)

    def save_thumb(frame: np.ndarray, idxs: list[int]) -> None:
        im = Image.fromarray(frame).resize((tw, th), Image.BILINEAR)
        for i in idxs:
            im.save(thumbs_dir / f"{i}.jpg", quality=80)

    def produce() -> None:
        # Decoding, thumbnails and resizing overlap with inference.
        try:
            k = 0
            for frames in video_frames(media, f"fps={fps:g},scale={w}:{h}:flags=bicubic,"
                                       "format=rgb24", w, h, hwaccel=device == "cuda",
                                       batch=batch):
                for f in frames:
                    if k in want:
                        save_thumb(f, want.pop(k))
                    k += 1
                q.put((frames[-1], enc.preprocess(list(frames)), len(frames)))
            q.put(None)
        except BaseException as e:  # noqa: BLE001 (handed to the consumer)
            q.put(e)

    threading.Thread(target=produce, daemon=True).start()
    n = 0
    last = None
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        last, pixels, count = item
        index.add(np.ascontiguousarray(enc.encode_pixels(pixels), dtype=np.float32))
        n += count
        if duration:
            ctx.progress.emit("visual", n / fps / duration)
    if last is not None:
        for idxs in want.values():  # shots past the last sampled frame
            save_thumb(last, idxs)
    if n == 0:
        raise RuntimeError("no frames decoded")
    t = np.arange(n, dtype=np.float64) / fps
    shot_idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, max(len(starts) - 1, 0))
    faiss.write_index(index, str(ctx.work / bundle.CLIP_INDEX))
    pq.write_table(
        pa.table({"row": pa.array(range(n), pa.int32()), "t": pa.array(t, pa.float64()),
                  "shot_idx": pa.array(shot_idx, pa.int32())}),
        ctx.work / bundle.FRAMES,
    )


def stage_faces(ctx: Ctx) -> None:
    """Decode at the visual ``fps`` again (so frame k is ``frames.parquet`` row k),
    find faces with SCRFD, and embed each with ArcFace. Boxes are stored as
    fractions of the frame, so a crop applies to the source at any resolution."""
    import queue

    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .faces import FaceEncoder, align, normalized_box
    from .shots import even, probe, video_frames

    cfg = ctx.lock.faces
    fps = float(ctx.lock.visual.get("fps", 2))
    n_frames = pq.read_metadata(ctx.work / bundle.FRAMES).num_rows
    media = ctx.media()
    p = probe(media)
    duration = float(ctx.job.get("duration_s") or 0) or p.duration_s
    h = even(min(int(cfg["decode_height"]), p.height))
    w = even(h * p.aspect)
    min_px = float(cfg["min_face_px"]) * h / int(cfg["decode_height"])
    device = ctx.ort_device()
    enc = FaceEncoder(ctx.lock, device=device)
    det, _ = enc.sessions()
    print(f"faces: onnxruntime providers {det.get_providers()}", file=sys.stderr, flush=True)
    index = faiss.IndexFlatIP(enc.dim)
    rows: dict[str, list] = {"frame_row": [], "x": [], "y": [], "w": [], "h": [],
                             "det_score": []}
    q: queue.Queue = queue.Queue(maxsize=4)

    def produce() -> None:
        try:
            for frames in video_frames(media, f"fps={fps:g},scale={w}:{h}:flags=bicubic,"
                                       "format=rgb24", w, h, hwaccel=device == "cuda",
                                       batch=32):
                q.put(frames)
            q.put(None)
        except BaseException as e:  # noqa: BLE001 (handed to the consumer)
            q.put(e)

    threading.Thread(target=produce, daemon=True).start()
    k = 0
    crops: list[np.ndarray] = []

    def flush() -> None:
        if crops:
            index.add(np.ascontiguousarray(enc.embed(crops), dtype=np.float32))
            crops.clear()

    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        for frame in item:
            # Frames past the visual index (a rounding difference at the end) are dropped.
            if k < n_frames:
                for face in enc.detect(frame):
                    if face.size < min_px:
                        continue
                    x, y, bw, bh = normalized_box(face, w, h)
                    rows["frame_row"].append(k)
                    rows["x"].append(x)
                    rows["y"].append(y)
                    rows["w"].append(bw)
                    rows["h"].append(bh)
                    rows["det_score"].append(face.score)
                    crops.append(align(frame, face))
            k += 1
        if len(crops) >= int(cfg.get("batch_size", 64)):
            flush()
        if duration:
            ctx.progress.emit("faces", k / fps / duration)
    flush()
    n = len(rows["frame_row"])
    if index.ntotal != n:
        raise RuntimeError(f"faces: {index.ntotal} embeddings for {n} boxes")
    faiss.write_index(index, str(ctx.work / bundle.FACES_INDEX))
    pq.write_table(
        pa.table({"row": pa.array(range(n), pa.int32()),
                  "frame_row": pa.array(rows["frame_row"], pa.int32()),
                  **{c: pa.array(rows[c], pa.float32()) for c in ("x", "y", "w", "h",
                                                                 "det_score")}}),
        ctx.work / bundle.FACES_TABLE,
    )


def stage_bundle(ctx: Ctx) -> None:
    import faiss
    import pyarrow.parquet as pq

    def rows(name: str) -> int:
        return pq.read_metadata(ctx.work / name).num_rows

    def ntotal(name: str) -> int:
        return faiss.read_index(str(ctx.work / name), faiss.IO_FLAG_MMAP).ntotal

    names: list[str] = []
    contents: list[str] = []
    counts: dict[str, int] = {}
    if (ctx.work / bundle.SENTENCES_INDEX).exists():
        names += [bundle.TRANSCRIPT, bundle.SENTENCES_TABLE, bundle.SENTENCES_INDEX]
        contents.append("dialogue")
        counts.update(transcript_segments=rows(bundle.TRANSCRIPT),
                      sentences=rows(bundle.SENTENCES_TABLE),
                      sentences_index=ntotal(bundle.SENTENCES_INDEX))
    if (ctx.work / bundle.CLIP_INDEX).exists():
        names += [bundle.SHOTS, bundle.FRAMES, bundle.CLIP_INDEX, bundle.THUMBS]
        contents.append("visual")
        counts.update(shots=rows(bundle.SHOTS), frames=rows(bundle.FRAMES),
                      clip_index=ntotal(bundle.CLIP_INDEX),
                      thumbs=len(list((ctx.work / bundle.THUMBS).glob("*.jpg"))))
    if (ctx.work / bundle.FACES_INDEX).exists():
        names += [bundle.FACES_TABLE, bundle.FACES_INDEX]
        contents.append("faces")
        counts.update(faces=rows(bundle.FACES_TABLE), faces_index=ntotal(bundle.FACES_INDEX))
    durations = {}
    for marker in ctx.work.glob("*.done"):
        try:
            durations[marker.stem] = float(marker.read_text() or 0)
        except ValueError:
            pass
    manifest = {
        "format": bundle.FORMAT,
        "movie_id": ctx.job.get("movie_id"),
        "indexer_version": __version__,
        "models_lock_hash": ctx.lock.hash,
        "models_lock": ctx.lock.data,
        "created_at": time.time(),
        "contents": contents,
        "counts": counts,
        "durations_s": durations,
        "files": names,
    }
    (ctx.work / bundle.MANIFEST).write_text(json.dumps(manifest, indent=1))
    bundle.pack(ctx.work, [bundle.MANIFEST, *names], ctx.out / bundle.BUNDLE_NAME,
                level=int(ctx.job.get("zstd_level", 19)))


STAGES: dict[str, Callable[[Ctx], None]] = {
    "fetch": stage_fetch,
    "shots": stage_shots,
    "visual": stage_visual,
    "faces": stage_faces,
    "audio": stage_audio,
    "sentences": stage_sentences,
    "bundle": stage_bundle,
}


def _run_stage(ctx: Ctx, stage: str) -> None:
    marker = ctx.work / f"{stage}.done"
    if marker.exists():
        ctx.progress.emit(stage, 1.0, state="done", force=True, resumed=True)
        return
    turn = ctx.gpu_turn if stage in GPU_STAGES else None
    if turn is not None:
        turn.acquire()  # before the first emit, so the wait stays "pending"
    try:
        ctx.progress.emit(stage, 0.0, force=True)
        t0 = time.time()
        STAGES[stage](ctx)
        marker.write_text(f"{time.time() - t0:.3f}")
    finally:
        if turn is not None:
            turn.release()
    ctx.progress.emit(stage, 1.0, state="done", force=True)


def run(job_path: str | os.PathLike[str]) -> None:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["work_dir"])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    lock = ModelsLock(job["models_lock"]) if "models_lock" in job else load_lock()
    progress = Progress(out / "progress.jsonl")
    ctx = Ctx(job=job, work=work, out=out, lock=lock, progress=progress)
    stages = job.get("stages", DEFAULT_STAGES)
    try:
        for stage in stages:
            if stage not in STAGES:
                raise ValueError(f"unknown stage {stage!r}")
        # fetch first, bundle last; between them the audio lane runs in a thread.
        head = [st for st in stages if st == "fetch"]
        tail = [st for st in stages if st == "bundle"]
        middle = [st for st in stages if st not in ("fetch", "bundle")]
        audio = [st for st in middle if st in AUDIO_LANE]
        visual = [st for st in middle if st not in AUDIO_LANE]
        for stage in head:
            _run_stage(ctx, stage)
        failure: list[BaseException] = []

        def audio_lane() -> None:
            try:
                for stage in audio:
                    _run_stage(ctx, stage)
            except BaseException as e:  # noqa: BLE001 (re-raised on the main thread)
                failure.append(e)

        if ctx.device() == "cuda":
            ctx.gpu_turn = threading.Lock()
            if "visual" in visual and not (work / "visual.done").exists():
                from .encoders import prefetch_visual

                threading.Thread(target=prefetch_visual, args=(lock,), daemon=True).start()
        lane = None
        if audio and visual and job.get("parallel", True):
            lane = threading.Thread(target=audio_lane, name="audio-lane", daemon=True)
            lane.start()
        else:
            visual = middle
        for stage in visual:
            if failure:
                break
            _run_stage(ctx, stage)
        if lane is not None:
            lane.join()
        if failure:
            raise failure[0]
        for stage in tail:
            _run_stage(ctx, stage)
        progress.emit("done", 1.0, state="done", force=True)
    except BaseException as e:
        progress.emit("error", 0.0, state="failed", force=True, message=f"{type(e).__name__}: {e}")
        raise


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit("usage: python -m framehound_indexer job.json")
    try:
        run(argv[0])
    except BaseException:  # noqa: BLE001 (the error line is already written)
        # A still-running audio thread (Whisper) must not keep a failed run alive.
        sys.stderr.flush()
        os._exit(1)
