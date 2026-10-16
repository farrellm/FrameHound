"""Face (picture → frame) matching: ArcFace similarities over every face found in
the movie, reduced to the best face per 2 Hz frame (design.md §6, Image)."""

from __future__ import annotations

import threading
from pathlib import Path

import faiss
import numpy as np
from framehound_indexer import bundle

from .. import db


class FaceIndex:
    """One movie's face boxes (fractions of the frame) and memory-mapped ArcFace index."""

    def __init__(self, movie_id: str, index_dir: Path, n_frames: int):
        with db.db() as conn:
            rs = conn.execute("SELECT frame_row, x, y, w, h FROM faces WHERE movie_id=? "
                              "ORDER BY row", (movie_id,)).fetchall()
        self.n_frames = n_frames
        self.frame_row = np.array([r[0] for r in rs], dtype=np.int64)
        self.boxes = np.array([r[1:] for r in rs], dtype=np.float32).reshape(-1, 4)
        self.has_face = np.zeros(n_frames, dtype=bool)
        self.has_face[self.frame_row[self.frame_row < n_frames]] = True
        self.index = faiss.read_index(str(index_dir / bundle.FACES_INDEX), faiss.IO_FLAG_MMAP)
        if self.index.ntotal != len(rs):
            raise RuntimeError(f"faces index {self.index.ntotal} != table {len(rs)}")

    def sims(self, qvec: np.ndarray) -> np.ndarray:
        n = self.index.ntotal
        out = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out
        sims, ids = self.index.search(qvec.reshape(1, -1).astype(np.float32), n)
        valid = ids[0] >= 0
        out[ids[0][valid]] = sims[0][valid]
        return out

    def match(self, qvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per frame: the best face similarity (-1 where the frame has no face) and
        that face's row (-1 where none)."""
        return best_per_frame(self.sims(qvec), self.frame_row, self.n_frames)

    def box(self, face_row: int) -> list[float]:
        return [round(float(v), 4) for v in self.boxes[face_row]]


def best_per_frame(sims: np.ndarray, frame_row: np.ndarray,
                   n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(n_frames, -1.0, dtype=np.float32)
    which = np.full(n_frames, -1, dtype=np.int64)
    ok = frame_row < n_frames
    rows = np.flatnonzero(ok)
    if rows.size == 0:
        return best, which
    fr = frame_row[rows]
    order = np.lexsort((sims[rows], fr))  # by frame, then similarity ascending
    fr_sorted = fr[order]
    last = np.r_[fr_sorted[1:] != fr_sorted[:-1], True]  # the best face of each frame
    pick = rows[order[last]]
    best[frame_row[pick]] = sims[pick]
    which[frame_row[pick]] = pick
    return best, which


_cache: dict[str, tuple[float, FaceIndex]] = {}
_cache_lock = threading.Lock()


def face_index(movie_id: str, index_dir: Path, n_frames: int) -> FaceIndex:
    """Cache per movie, invalidated when the index is re-imported."""
    stamp = (index_dir / bundle.MANIFEST).stat().st_mtime
    with _cache_lock:
        hit = _cache.get(movie_id)
        if hit and hit[0] == stamp:
            return hit[1]
    idx = FaceIndex(movie_id, index_dir, n_frames)
    with _cache_lock:
        _cache[movie_id] = (stamp, idx)
        if len(_cache) > 16:
            _cache.pop(next(iter(_cache)))
    return idx


def evict(movie_id: str) -> None:
    with _cache_lock:
        _cache.pop(movie_id, None)
