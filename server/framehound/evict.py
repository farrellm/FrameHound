"""Disk quota: LRU eviction by ``last_accessed_at`` once movies/ (under DATA_DIR and
MEDIA_DIR together) passes MAX_DATA_GB.

Cheapest loss first (design.md §10):

1. clip files,
2. a ready movie's ``source.*``: it stays searchable and plays from the proxy,
   and shows "re-download to clip",
3. everything else of the movie (proxy, index): it becomes ``evicted``, and
   Re-download brings it back through ingest and indexing.

Movies with a queued or running job, and those in ``protect``, are never touched.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
from pathlib import Path

from . import db
from .config import settings
from .search import dialogue, faces, visual
from .torrent import TORRENT_FILE

log = logging.getLogger("framehound.evict")

INDEX_TABLES = ("transcript_segments", "transcript_fts", "shots", "frames", "faces")


def tree_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def usage() -> int:
    s = settings()
    roots = dict.fromkeys([s.data_dir / "movies", s.media_dir / "movies"])
    return sum(tree_bytes(r) for r in roots)


def enforce_quota(protect: set[str] | frozenset[str] = frozenset(), need: int = 0) -> list[str]:
    """Evict until movies/ plus ``need`` more bytes fits the quota.

    Returns what was done, as ``"{kind}:{movie_id}"`` strings (for logs and tests).
    """
    s = settings()
    if s.max_data_gb <= 0:
        return []
    # The api's hourly pass and an ingest on worker-io must not both evict for the
    # same overage.
    s.data_dir.mkdir(parents=True, exist_ok=True)
    with open(s.data_dir / "evict.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _enforce(s, protect, need)


def _enforce(s, protect, need: int) -> list[str]:
    limit = int(s.max_data_gb * 1024**3)
    over = usage() + need - limit
    if over <= 0:
        return []
    with db.db() as conn:
        busy = {r[0] for r in conn.execute(
            "SELECT DISTINCT movie_id FROM jobs WHERE state IN ('queued','running')")}
        movies = [m for m in db.rows(
            conn, "SELECT id, status, source_file FROM movies WHERE status NOT IN "
            "('merged','evicted','ingesting','indexing') "
            "ORDER BY COALESCE(last_accessed_at, created_at)")
            if m["id"] not in busy and m["id"] not in protect]
        done: list[str] = []
        for step in (_clips, _source, _all):
            for m in movies:
                if over <= 0:
                    break
                freed = step(conn, m)
                if freed:
                    over -= freed
                    done.append(f"{step.__name__.strip('_')}:{m['id']}")
            if over <= 0:
                break
    if done:
        log.warning("evicted to stay under %g GB: %s", s.max_data_gb, ", ".join(done))
    if over > 0:
        log.warning("still %d bytes over the %g GB quota after eviction", over, s.max_data_gb)
    return done


def _clips(conn, m: dict) -> int:
    freed = 0
    for c in db.rows(conn, "SELECT id, path FROM clips WHERE movie_id=?", m["id"]):
        if c["path"]:
            p = Path(c["path"])
            freed += p.stat().st_size if p.exists() else 0
            p.unlink(missing_ok=True)
            p.with_suffix(".srt").unlink(missing_ok=True)
        conn.execute("DELETE FROM clips WHERE id=?", (c["id"],))
    return freed


def _source(conn, m: dict) -> int:
    if m["status"] != "ready" or not m["source_file"]:
        return 0
    src = settings().media_movie_dir(m["id"]) / m["source_file"]
    if not src.exists():
        return 0
    size = src.stat().st_size
    src.unlink()
    return size


def _all(conn, m: dict) -> int:
    dirs = [d for d in settings().movie_dirs(m["id"]) if d.exists()]
    if not dirs:
        return 0
    size = 0
    for child in (c for d in dirs for c in d.iterdir()):
        if child.name == TORRENT_FILE:  # small, and needed to download it again
            continue
        if child.is_dir():
            size += tree_bytes(child)
            shutil.rmtree(child, ignore_errors=True)
        else:
            size += child.stat().st_size
            child.unlink(missing_ok=True)
    with db.tx(conn):
        for table in INDEX_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE movie_id=?", (m["id"],))
        conn.execute("UPDATE movies SET status='evicted', error=NULL WHERE id=?", (m["id"],))
    for mod in (dialogue, visual, faces):
        mod.evict(m["id"])
    return size
