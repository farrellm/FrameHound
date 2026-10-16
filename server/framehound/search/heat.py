"""Score series helpers shared by every query type."""

from __future__ import annotations

import numpy as np

SAMPLE_HZ = 2.0


def robust_z(scores: np.ndarray) -> np.ndarray:
    """(s - median) / MAD; raw similarities drift with the query, so thresholds use z."""
    if scores.size == 0:
        return scores.astype(np.float32)
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    if mad < 1e-6:
        # More than half the values are identical; fall back to the mean absolute
        # deviation, which still lets the rare outliers stand out.
        mad = float(np.mean(np.abs(scores - med))) or 1.0
    return ((scores - med) / mad).astype(np.float32)


def spans_to_series(spans: list[tuple[float, float, float]], duration_s: float,
                    hz: float = SAMPLE_HZ) -> np.ndarray:
    """Paint (start, end, value) spans onto a 2 Hz series, taking the max where they overlap."""
    n = max(int(np.ceil(duration_s * hz)), 1)
    out = np.zeros(n, dtype=np.float32)
    for start, end, value in spans:
        lo = max(int(np.floor(start * hz)), 0)
        hi = min(max(int(np.ceil(end * hz)), lo + 1), n)
        if lo < n:
            np.maximum(out[lo:hi], value, out=out[lo:hi])
    return out


def max_pool(series: np.ndarray, bins: int) -> list[float]:
    """Max-pool into ``bins`` buckets, so short spikes stay visible at coarse resolution."""
    if series.size == 0:
        return [0.0] * bins
    edges = np.linspace(0, series.size, bins + 1)
    out = []
    for i in range(bins):
        lo = int(edges[i])
        hi = max(int(np.ceil(edges[i + 1])), lo + 1)
        out.append(round(float(series[lo:min(hi, series.size)].max(initial=0.0)), 3))
    return out


def nms(cands: list[dict], max_overlap: float = 0.3) -> list[dict]:
    """Keep the best candidate of any group overlapping by more than ``max_overlap``
    of the shorter span. Ties prefer the tighter span."""
    keep: list[dict] = []
    for c in sorted(cands, key=lambda c: (-c["score"], c["end"] - c["start"])):
        ok = True
        for k in keep:
            inter = min(c["end"], k["end"]) - max(c["start"], k["start"])
            shorter = min(c["end"] - c["start"], k["end"] - k["start"]) or 1e-6
            if inter > 0 and inter / shorter > max_overlap:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep
