"""Dialogue search: fuzzy lexical match + semantic match over the transcript.

1. Lexical: rapidfuzz partial_ratio of the normalized query against each
   segment and each 3-segment window, keeping matches >= 85.
2. Semantic: bge-small embedding of the query against sentences.faiss,
   z-normalized per movie, which catches paraphrases.
3. Precise bounds: a lexical hit is aligned to word timestamps.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
from framehound_indexer import bundle
from rapidfuzz import fuzz

from .. import db
from .heat import max_pool, nms, robust_z, spans_to_series

LEXICAL_MIN = 85
SEMANTIC_MIN_COS = 0.70
LEXICAL_BONUS = 0.10
PAD_S = 0.15

_PUNCT = re.compile(r"[^\w\s']+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip(" '")


def looks_like_quote(q: str) -> bool:
    return bool(re.search(r"[\"“”]", q)) or q.strip().startswith("'")


@dataclass
class Unit:
    seg_lo: int
    seg_hi: int
    text: str  # normalized


class DialogueIndex:
    """One movie's transcript, units, and memory-mapped sentence index."""

    def __init__(self, movie_id: str, index_dir: Path):
        with db.db() as conn:
            segs = db.rows(conn, 'SELECT seg, start, "end", text, words_json FROM '
                           "transcript_segments WHERE movie_id=? ORDER BY seg", movie_id)
        self.segs = segs
        self.words = [json.loads(s["words_json"] or "[]") for s in segs]
        table = pq.read_table(index_dir / bundle.SENTENCES_TABLE).to_pydict()
        texts = [normalize(s["text"]) for s in segs]
        self.units = [
            Unit(lo, hi, " ".join(texts[lo : hi + 1]))
            for lo, hi in zip(table["seg_lo"], table["seg_hi"], strict=True)
        ]
        self.index = faiss.read_index(str(index_dir / bundle.SENTENCES_INDEX), faiss.IO_FLAG_MMAP)

    # scoring ---------------------------------------------------------------

    def lexical(self, qn: str) -> np.ndarray:
        out = np.zeros(len(self.units), dtype=np.float32)
        for i, u in enumerate(self.units):
            if not u.text:
                continue
            # partial_ratio aligns the shorter string inside the longer one, so a
            # short segment wholly inside a long query would score 100. Only let
            # partial matching run when the unit is at least as long as the query.
            if len(u.text) >= len(qn):
                out[i] = fuzz.partial_ratio(qn, u.text)
            else:
                out[i] = fuzz.ratio(qn, u.text)
        return out

    def semantic(self, qvec: np.ndarray | None) -> np.ndarray:
        n = self.index.ntotal
        if qvec is None or n == 0:
            return np.zeros(n, dtype=np.float32)
        sims, ids = self.index.search(qvec.reshape(1, -1).astype(np.float32), n)
        out = np.zeros(n, dtype=np.float32)
        valid = ids[0] >= 0
        out[ids[0][valid]] = sims[0][valid]
        return out

    def word_bounds(self, u: Unit, qn: str) -> tuple[float, float, list[dict]]:
        """Align the query to the unit's words; return (start, end, matched words)."""
        words = [w for seg in range(u.seg_lo, u.seg_hi + 1) for w in self.words[seg]]
        if not words:
            return self.segs[u.seg_lo]["start"], self.segs[u.seg_hi]["end"], []
        norm = [normalize(w["word"]) for w in words]
        nq = max(len(qn.split()), 1)
        best, best_span = -1.0, (0, len(words) - 1)
        for size in range(max(nq - 2, 1), nq + 3):
            for i in range(max(len(words) - size, 0) + 1):
                cand = " ".join(t for t in norm[i : i + size] if t)
                score = fuzz.ratio(qn, cand)
                if score > best:
                    best, best_span = score, (i, min(i + size, len(words)) - 1)
        lo, hi = best_span
        matched = [{"word": w["word"], "start": w["start"], "end": w["end"]}
                   for w in words[lo : hi + 1]]
        return words[lo]["start"], words[hi]["end"], matched

    # search ----------------------------------------------------------------

    def search(self, q: str, qvec: np.ndarray | None, duration_s: float, top_k: int,
               heat_bins: int, z_min: float) -> dict:
        qn = normalize(q)
        if not qn or not self.units:
            return {"candidates": [], "heat": [0.0] * heat_bins, "heat_threshold": z_min}
        lex = self.lexical(qn)
        sem = self.semantic(qvec)
        sem_z = robust_z(sem)

        cands = []
        for i, u in enumerate(self.units):
            lex_hit = lex[i] >= LEXICAL_MIN
            sem_hit = sem[i] >= SEMANTIC_MIN_COS and sem_z[i] >= z_min
            if not (lex_hit or sem_hit):
                continue
            lex_score = float(lex[i]) / 100 + LEXICAL_BONUS if lex_hit else 0.0
            score = max(lex_score, float(sem[i]) if sem_hit else 0.0)
            if lex_hit:
                start, end, matched = self.word_bounds(u, qn)
            else:
                start, end = self.segs[u.seg_lo]["start"], self.segs[u.seg_hi]["end"]
                matched = [{"word": w["word"], "start": w["start"], "end": w["end"]}
                           for s in range(u.seg_lo, u.seg_hi + 1) for w in self.words[s]]
            text = " ".join(self.segs[s]["text"] for s in range(u.seg_lo, u.seg_hi + 1))
            cands.append({
                "start": round(max(start - PAD_S, 0.0), 3),
                "end": round(min(end + PAD_S, duration_s), 3),
                "score": round(score, 4),
                "peak_t": round(start, 3),
                "reason": "dialogue",
                "match": "lexical" if lex_hit and lex_score >= score else "semantic",
                "thumbs": [],
                "has_face": False,
                "matched_words": matched,
                "text": text,
            })
        cands = nms(cands)[:top_k]

        # Heat: each segment's best score (its own or any window containing it),
        # z-normalized over segments and painted onto the 2 Hz series.
        per_unit = np.maximum(lex / 100, sem)
        per_seg = np.zeros(len(self.segs), dtype=np.float32)
        for i, u in enumerate(self.units):
            for s in range(u.seg_lo, u.seg_hi + 1):
                per_seg[s] = max(per_seg[s], per_unit[i])
        z = np.clip(robust_z(per_seg), 0, None)
        series = spans_to_series(
            [(s["start"], s["end"], float(z[i])) for i, s in enumerate(self.segs)], duration_s)
        return {"candidates": cands, "heat": max_pool(series, heat_bins), "heat_threshold": z_min}


_cache: dict[str, tuple[float, DialogueIndex]] = {}
_cache_lock = threading.Lock()


def dialogue_index(movie_id: str, index_dir: Path) -> DialogueIndex:
    """Cache per movie, invalidated when the index is re-imported."""
    stamp = (index_dir / bundle.MANIFEST).stat().st_mtime
    with _cache_lock:
        hit = _cache.get(movie_id)
        if hit and hit[0] == stamp:
            return hit[1]
    idx = DialogueIndex(movie_id, index_dir)
    with _cache_lock:
        _cache[movie_id] = (stamp, idx)
        if len(_cache) > 16:
            _cache.pop(next(iter(_cache)))
    return idx


def evict(movie_id: str) -> None:
    with _cache_lock:
        _cache.pop(movie_id, None)
