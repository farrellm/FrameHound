"""SQLite (WAL) shared by the api and the workers through the /data volume."""

from __future__ import annotations

import secrets
import sqlite3
import string
import time
from collections.abc import Iterator
from contextlib import contextmanager

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies(
  id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, title TEXT, source_kind TEXT, source_ref TEXT,
  source_url TEXT, source_md5 TEXT, source_file TEXT,
  duration_s REAL, width INT, height INT, fps REAL, bytes INT, models_lock_hash TEXT,
  status TEXT NOT NULL, error TEXT, merged_into TEXT,
  created_at REAL, last_accessed_at REAL);
CREATE INDEX IF NOT EXISTS movies_source_ref ON movies(source_ref);

CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, movie_id TEXT, kind TEXT NOT NULL, state TEXT NOT NULL,
  stage TEXT, pct REAL, eta_s REAL, message TEXT, error TEXT, error_code TEXT,
  attempt INT DEFAULT 0, colab_session TEXT, gpu TEXT, compute_units_used REAL,
  backend_handle TEXT, cancel_requested INT DEFAULT 0, seq INT DEFAULT 0,
  started_at REAL, finished_at REAL, heartbeat_at REAL, created_at REAL);
CREATE INDEX IF NOT EXISTS jobs_movie ON jobs(movie_id, created_at);

CREATE TABLE IF NOT EXISTS job_stages(
  job_id TEXT, stage TEXT, ord INT, state TEXT NOT NULL DEFAULT 'pending',
  pct REAL DEFAULT 0, eta_s REAL, PRIMARY KEY(job_id, stage));

CREATE TABLE IF NOT EXISTS shots(movie_id TEXT, idx INT, start REAL, "end" REAL,
  PRIMARY KEY(movie_id, idx));
CREATE TABLE IF NOT EXISTS frames(movie_id TEXT, row INT, t REAL, shot_idx INT,
  PRIMARY KEY(movie_id, row));
CREATE TABLE IF NOT EXISTS faces(movie_id TEXT, row INT, frame_row INT, x REAL, y REAL,
  w REAL, h REAL, det_score REAL, PRIMARY KEY(movie_id, row));
CREATE TABLE IF NOT EXISTS transcript_segments(movie_id TEXT, seg INT, start REAL, "end" REAL,
  text TEXT, words_json TEXT, PRIMARY KEY(movie_id, seg));
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
  text, movie_id UNINDEXED, seg UNINDEXED);

CREATE TABLE IF NOT EXISTS queries(id TEXT PRIMARY KEY, movie_id TEXT, type TEXT, q TEXT,
  image_sha TEXT, created_at REAL, latency_ms REAL);
CREATE TABLE IF NOT EXISTS clips(id TEXT PRIMARY KEY, movie_id TEXT, start REAL, "end" REAL,
  mode TEXT, format TEXT, path TEXT, job_id TEXT, created_at REAL,
  crop_box TEXT /* JSON [x, y, w, h], fractions of the frame */);
"""

_ALPHABET = string.ascii_lowercase + string.digits


def new_id(n: int = 10) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(n))


def now() -> float:
    return time.time()


def connect() -> sqlite3.Connection:
    s = settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(s.db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def init() -> None:
    with db() as conn:
        conn.executescript(SCHEMA)
        # Columns added after a release: CREATE TABLE IF NOT EXISTS leaves old tables be.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)")}
        if "crop_box" not in cols:
            conn.execute("ALTER TABLE clips ADD COLUMN crop_box TEXT")


def row(conn: sqlite3.Connection, sql: str, *args) -> dict | None:
    r = conn.execute(sql, args).fetchone()
    return dict(r) if r else None


def rows(conn: sqlite3.Connection, sql: str, *args) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]
