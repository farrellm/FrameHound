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
      "stages": ["fetch", "audio", "sentences", "bundle"],
      "device": "auto",              # auto | cpu | cuda
      "zstd_level": 19
    }

M1 implements the transcript-only stages; shots, frames, visual, faces and
thumbs arrive in M2 and M3.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, bundle
from .lock import ModelsLock, load_lock

DEFAULT_STAGES = ["fetch", "audio", "sentences", "bundle"]


class Progress:
    """Appends progress lines, rate-limited per stage, with a naive ETA."""

    def __init__(self, path: Path):
        self.fh = open(path, "a", buffering=1)  # noqa: SIM115 (lives for the run)
        self.started: dict[str, float] = {}
        self.last: dict[str, float] = {}

    def emit(self, stage: str, pct: float, state: str = "running", force: bool = False, **kw):
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


def stage_bundle(ctx: Ctx) -> None:
    import faiss
    import pyarrow.parquet as pq

    names = [bundle.TRANSCRIPT, bundle.SENTENCES_TABLE, bundle.SENTENCES_INDEX]
    counts = {
        "transcript_segments": pq.read_metadata(ctx.work / bundle.TRANSCRIPT).num_rows,
        "sentences": pq.read_metadata(ctx.work / bundle.SENTENCES_TABLE).num_rows,
        "sentences_index": faiss.read_index(str(ctx.work / bundle.SENTENCES_INDEX)).ntotal,
    }
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
        "contents": ["dialogue"],
        "counts": counts,
        "durations_s": durations,
        "files": names,
    }
    (ctx.work / bundle.MANIFEST).write_text(json.dumps(manifest, indent=1))
    bundle.pack(ctx.work, [bundle.MANIFEST, *names], ctx.out / bundle.BUNDLE_NAME,
                level=int(ctx.job.get("zstd_level", 19)))


STAGES: dict[str, Callable[[Ctx], None]] = {
    "fetch": stage_fetch,
    "audio": stage_audio,
    "sentences": stage_sentences,
    "bundle": stage_bundle,
}


def run(job_path: str | os.PathLike[str]) -> None:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["work_dir"])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    lock = ModelsLock(job["models_lock"]) if "models_lock" in job else load_lock()
    progress = Progress(out / "progress.jsonl")
    ctx = Ctx(job=job, work=work, out=out, lock=lock, progress=progress)
    try:
        for stage in job.get("stages", DEFAULT_STAGES):
            if stage not in STAGES:
                raise ValueError(f"unknown stage {stage!r}")
            marker = work / f"{stage}.done"
            if marker.exists():
                progress.emit(stage, 1.0, state="done", force=True, resumed=True)
                continue
            progress.emit(stage, 0.0, force=True)
            t0 = time.time()
            STAGES[stage](ctx)
            marker.write_text(f"{time.time() - t0:.3f}")
            progress.emit(stage, 1.0, state="done", force=True)
        progress.emit("done", 1.0, state="done", force=True)
    except BaseException as e:
        progress.emit("error", 0.0, state="failed", force=True, message=f"{type(e).__name__}: {e}")
        raise


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit("usage: python -m framehound_indexer job.json")
    run(argv[0])
