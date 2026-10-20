"""Visual (text → frame) search: SigLIP similarities over the 2 Hz frame index,
turned into candidate segments (design.md §6, "From frame scores to candidate
segments")."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from framehound_indexer import bundle

from .. import db
from .faces import FaceIndex
from .heat import max_pool, nms, robust_z, spans_to_series

SMOOTH_S = 1.0
MERGE_GAP_S = 1.5
SNAP_S = 1.0
PAD_S = 0.5
TOP_PCT = 99.0
THUMBS_PER_CANDIDATE = 3
HEAT_FLOOR = 0.75  # of the threshold; weaker frames are noise in the strip


@dataclass
class Segment:
    start: float
    end: float
    peak_t: float
    peak_z: float
    core: tuple[float, float]  # before padding: what the thumbnails show

    @property
    def score(self) -> float:
        return self.peak_z * (1 + 0.1 * math.log1p(max(self.end - self.start, 0.0)))


def moving_max(z: np.ndarray, times: np.ndarray, window_s: float = SMOOTH_S) -> np.ndarray:
    """Max over a centred ``window_s`` window, filling single-frame dropouts."""
    if z.size < 2:
        return z.copy()
    dt = float(np.median(np.diff(times))) or 0.5
    r = max(1, round(window_s / dt / 2))
    out = z.copy()
    for k in range(1, r + 1):
        np.maximum(out[k:], z[:-k], out=out[k:])
        np.maximum(out[:-k], z[k:], out=out[:-k])
    return out


def _snap(t: float, bounds: np.ndarray, within: float) -> float:
    if bounds.size == 0:
        return t
    i = int(np.argmin(np.abs(bounds - t)))
    return float(bounds[i]) if abs(bounds[i] - t) <= within else t


def segments(z: np.ndarray, times: np.ndarray, shots: list[tuple[float, float]],
             duration_s: float, z_min: float) -> tuple[list[Segment], float, np.ndarray]:
    """Frame z-scores → ranked segments.

    Returns (segments, threshold used, smoothed z). ``z`` is already robust-z
    normalized per movie; ``times`` are the frame times in seconds.
    """
    if z.size == 0:
        return [], z_min, z
    zs = moving_max(z, times)
    thr = max(z_min, float(np.percentile(zs, TOP_PCT)))
    keep = np.flatnonzero(zs >= thr)
    if keep.size == 0:
        return [], thr, zs
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.5
    runs: list[list[int]] = [[int(keep[0])]]
    for i in keep[1:]:
        if times[i] - times[runs[-1][-1]] - dt < MERGE_GAP_S:
            runs[-1].append(int(i))
        else:
            runs.append([int(i)])
    starts = np.array([a for a, _ in shots], dtype=np.float64)
    ends = np.array([b for _, b in shots], dtype=np.float64)
    bounds = np.unique(np.concatenate([starts, ends])) if shots else np.zeros(0)
    out = []
    for run in runs:
        lo, hi = run[0], run[-1]
        a, b = float(times[lo]), float(times[hi]) + dt
        sa, sb = _snap(a, bounds, SNAP_S), _snap(b, bounds, SNAP_S)
        # Around a short run both edges can snap to the same boundary: keep it unsnapped.
        if sb - sa >= dt:
            a, b = sa, sb
        core = (a, b)
        a, b = max(a - PAD_S, 0.0), min(b + PAD_S, duration_s or b + PAD_S)
        # The peak is the strongest raw frame: smoothing only widens the run.
        idx = np.arange(lo, hi + 1)
        p = int(idx[np.argmax(z[idx])])
        out.append(Segment(a, b, float(times[p]), float(z[p]), core))
    out.sort(key=lambda s: -s.score)
    return out, thr, zs


class VisualIndex:
    """One movie's frame times, shots, and memory-mapped SigLIP index."""

    def __init__(self, movie_id: str, index_dir: Path):
        self.movie_id = movie_id
        with db.db() as conn:
            fr = conn.execute("SELECT t FROM frames WHERE movie_id=? ORDER BY row",
                              (movie_id,)).fetchall()
            sh = conn.execute('SELECT idx, start, "end" FROM shots WHERE movie_id=? '
                              "ORDER BY idx", (movie_id,)).fetchall()
        self.times = np.array([r[0] for r in fr], dtype=np.float64)
        self.shots = [(float(r[1]), float(r[2])) for r in sh]
        self.shot_ids = [int(r[0]) for r in sh]
        self.index = faiss.read_index(str(index_dir / bundle.CLIP_INDEX), faiss.IO_FLAG_MMAP)

    def scores(self, qvec: np.ndarray) -> np.ndarray:
        n = self.index.ntotal
        sims, ids = self.index.search(qvec.reshape(1, -1).astype(np.float32), n)
        out = np.zeros(n, dtype=np.float32)
        valid = ids[0] >= 0
        out[ids[0][valid]] = sims[0][valid]
        return out

    def thumbs(self, start: float, end: float) -> list[str]:
        """Up to three thumbnails of the shots the span covers, spread across it."""
        hit = [i for i, (a, b) in zip(self.shot_ids, self.shots, strict=True)
               if b > start and a < end]
        if len(hit) > THUMBS_PER_CANDIDATE:
            pick = np.linspace(0, len(hit) - 1, THUMBS_PER_CANDIDATE).round().astype(int)
            hit = [hit[k] for k in pick]
        return [thumb_url(self.movie_id, i) for i in hit]

    def search(self, qvec: np.ndarray, duration_s: float, top_k: int, heat_bins: int,
               z_min: float, faces: FaceIndex | None = None) -> dict:
        """Describe it: the query vector against every frame."""
        return self.rank(robust_z(self.scores(qvec)), duration_s, top_k, heat_bins, z_min,
                         faces=faces)

    def search_image(self, svec: np.ndarray, face_vec: np.ndarray | None, duration_s: float,
                     top_k: int, heat_bins: int, z_min: float, faces: FaceIndex | None,
                     match: float, bonus: float, scale: float) -> dict:
        """Show a picture: SigLIP looks, fused with ArcFace identity when the picture
        has a face and the movie has a face index."""
        z = robust_z(self.scores(svec))
        face_sims = face_rows = None
        if face_vec is not None and faces is not None:
            face_sims, face_rows = faces.match(face_vec)
            z, _ = fuse(z, face_sims, match, z_min, bonus, scale)
        return self.rank(z, duration_s, top_k, heat_bins, z_min, faces=faces,
                         face_sims=face_sims, face_rows=face_rows, match=match)

    def rank(self, z: np.ndarray, duration_s: float, top_k: int, heat_bins: int, z_min: float,
             faces: FaceIndex | None = None, face_sims: np.ndarray | None = None,
             face_rows: np.ndarray | None = None, match: float = 1.0) -> dict:
        """Frame z-scores → candidates and the heat series. A candidate whose peak frame
        holds a face matching the query (``face_sims`` ≥ ``match``) is a face hit."""
        segs, thr, zs = segments(z, self.times, self.shots, duration_s, z_min)
        cands = []
        for s in segs:
            p = int(np.clip(np.searchsorted(self.times, s.peak_t - 1e-6), 0,
                            max(self.times.size - 1, 0)))
            c = {"start": round(s.start, 3), "end": round(s.end, 3), "score": round(s.score, 4),
                 "peak_t": round(s.peak_t, 3), "reason": "visual",
                 "thumbs": self.thumbs(*s.core),
                 "has_face": bool(faces is not None and p < faces.has_face.size
                                  and faces.has_face[p])}
            if face_sims is not None and face_rows is not None and face_sims[p] >= match:
                c["reason"] = "face"
                c["face_bbox"] = faces.box(int(face_rows[p])) if faces else None
                c["has_face"] = True
            cands.append(c)
        cands = nms(cands)[:top_k]
        dt = float(np.median(np.diff(self.times))) if self.times.size > 1 else 0.5
        heat = np.where(zs >= HEAT_FLOOR * thr, zs, 0)
        series = spans_to_series([(float(t), float(t) + dt, float(v))
                                  for t, v in zip(self.times, heat, strict=True)],
                                 duration_s or float(self.times[-1] + dt))
        return {"candidates": cands, "heat": max_pool(series, heat_bins),
                "heat_threshold": round(thr, 3)}


def fuse(z_visual: np.ndarray, face_sims: np.ndarray, match: float, z_min: float,
         bonus: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``max(z_visual, z_face)``, where a face at or above ``match`` scores
    ``z_min + bonus + (sim − match)·scale``: identity is far more specific than looks,
    so every true face match clears the threshold and outranks looks alone.

    Returns the fused z and a mask of the frames whose score came from the face."""
    z_face = np.where(face_sims >= match, z_min + bonus + (face_sims - match) * scale, -np.inf)
    from_face = z_face > z_visual
    return np.where(from_face, z_face, z_visual).astype(np.float32), from_face


def thumb_url(movie_id: str, shot_idx: int) -> str:
    return f"/media/{movie_id}/thumbs/{shot_idx}.jpg"


_cache: dict[str, tuple[float, VisualIndex]] = {}
_cache_lock = threading.Lock()


def visual_index(movie_id: str, index_dir: Path) -> VisualIndex:
    """Cache per movie, invalidated when the index is re-imported."""
    stamp = (index_dir / bundle.MANIFEST).stat().st_mtime
    with _cache_lock:
        hit = _cache.get(movie_id)
        if hit and hit[0] == stamp:
            return hit[1]
    idx = VisualIndex(movie_id, index_dir)
    with _cache_lock:
        _cache[movie_id] = (stamp, idx)
        if len(_cache) > 16:
            _cache.pop(next(iter(_cache)))
    return idx


def evict(movie_id: str) -> None:
    with _cache_lock:
        _cache.pop(movie_id, None)
