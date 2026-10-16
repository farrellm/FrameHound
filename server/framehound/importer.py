"""Verify an index bundle and load it into SQLite + movies/{id}/index/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import faiss
import pyarrow.parquet as pq
from framehound_indexer import bundle, load_lock

from . import db
from .config import settings
from .errors import FHError


def _bad(detail: str) -> FHError:
    return FHError("bad_index", "The index came back damaged. Retry indexing.", detail=detail)


def import_bundle(movie_id: str, bundle_path: Path) -> dict:
    mdir = settings().movie_dir(movie_id)
    staging = mdir / "index.new"
    shutil.rmtree(staging, ignore_errors=True)
    bundle.unpack(bundle_path, staging)
    try:
        manifest = json.loads((staging / bundle.MANIFEST).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise _bad(f"manifest unreadable: {e}") from e
    if manifest.get("format") != bundle.FORMAT:
        raise _bad(f"bundle format {manifest.get('format')} != {bundle.FORMAT}")
    if manifest.get("models_lock_hash") != load_lock().hash:
        raise FHError("models_mismatch", "This movie was indexed with different models than "
                      "the server uses. Re-index it to search.",
                      detail=f"bundle {manifest.get('models_lock_hash')} != {load_lock().hash}")
    counts = manifest.get("counts", {})
    transcript = pq.read_table(staging / bundle.TRANSCRIPT).to_pylist()
    n_sent = pq.read_metadata(staging / bundle.SENTENCES_TABLE).num_rows
    n_index = faiss.read_index(str(staging / bundle.SENTENCES_INDEX), faiss.IO_FLAG_MMAP).ntotal
    if (len(transcript), n_sent, n_index) != (counts.get("transcript_segments"),
                                              counts.get("sentences"), counts.get("sentences_index")):
        raise _bad(f"row counts {len(transcript)}/{n_sent}/{n_index} don't match manifest {counts}")
    if n_sent != n_index:
        raise _bad(f"sentences table {n_sent} != index {n_index}")

    with db.db() as conn, db.tx(conn):
        conn.execute("DELETE FROM transcript_segments WHERE movie_id=?", (movie_id,))
        conn.execute("DELETE FROM transcript_fts WHERE movie_id=?", (movie_id,))
        conn.executemany(
            'INSERT INTO transcript_segments(movie_id, seg, start, "end", text, words_json) '
            "VALUES (?,?,?,?,?,?)",
            [(movie_id, r["seg"], r["start"], r["end"], r["text"], r["words_json"])
             for r in transcript],
        )
        conn.executemany("INSERT INTO transcript_fts(text, movie_id, seg) VALUES (?,?,?)",
                         [(r["text"], movie_id, r["seg"]) for r in transcript])
        conn.execute("UPDATE movies SET models_lock_hash=? WHERE id=?",
                     (manifest["models_lock_hash"], movie_id))
        live, old = mdir / "index", mdir / "index.old"
        shutil.rmtree(old, ignore_errors=True)
        if live.exists():
            live.rename(old)
        staging.rename(live)
    shutil.rmtree(old, ignore_errors=True)
    bundle_path.unlink(missing_ok=True)
    return manifest
