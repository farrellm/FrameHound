"""Burned-in subtitles: the transcript words inside a clip as an .srt, starting at 0."""

from __future__ import annotations

import json

MAX_LINE = 42  # characters per line, two lines per cue
MAX_CUE_S = 4.0
GAP_S = 0.8  # a pause this long starts a new cue
MIN_CUE_S = 0.5


def srt_time(t: float) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap(words: list[str]) -> str:
    """Two balanced lines when one won't fit."""
    text = " ".join(words)
    if len(text) <= MAX_LINE or len(words) < 2:
        return text
    best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i]))
                                                     - len(" ".join(words[i:]))))
    return " ".join(words[:best]) + "\n" + " ".join(words[best:])


def cues(segments: list[list[dict]], start: float, end: float) -> list[tuple[float, float, str]]:
    """Cues ``(start, end, text)`` relative to ``start`` from per-segment word lists.

    Words are kept when their middle falls inside ``[start, end]``. A cue ends at
    a segment boundary, a pause, two full lines, or ``MAX_CUE_S``.
    """
    groups: list[list[dict]] = []
    for words in segments:
        cur: list[dict] = []
        for w in words:
            text = (w.get("word") or "").strip()
            if not text or not start <= (w["start"] + w["end"]) / 2 <= end:
                continue
            if cur:
                length = len(" ".join(x["word"].strip() for x in cur)) + 1 + len(text)
                if (w["start"] - cur[-1]["end"] > GAP_S or length > 2 * MAX_LINE
                        or w["end"] - cur[0]["start"] > MAX_CUE_S):
                    groups.append(cur)
                    cur = []
            cur.append(w)
        if cur:
            groups.append(cur)
    out: list[tuple[float, float, str]] = []
    for g in groups:
        a = max(g[0]["start"], start) - start
        b = min(g[-1]["end"], end) - start
        b = min(max(b, a + MIN_CUE_S), end - start)
        out.append((round(a, 3), round(b, 3), _wrap([w["word"].strip() for w in g])))
    # Overlaps (from MIN_CUE_S, or overlapping segments) end at the next cue's start.
    fixed = []
    for i, (a, b, text) in enumerate(out):
        if i + 1 < len(out):
            b = min(b, out[i + 1][0])
        if b > a:
            fixed.append((a, b, text))
    return fixed


def build_srt(segments: list[list[dict]], start: float, end: float) -> str:
    return "".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n"
                   for i, (a, b, text) in enumerate(cues(segments, start, end), 1))


def clip_srt(conn, movie_id: str, start: float, end: float) -> str:
    """The .srt for a clip of ``movie_id`` from the imported transcript."""
    rows = conn.execute('SELECT words_json FROM transcript_segments WHERE movie_id=? AND '
                        '"end">=? AND start<=? ORDER BY start', (movie_id, start, end)).fetchall()
    return build_srt([json.loads(r[0] or "[]") for r in rows], start, end)
