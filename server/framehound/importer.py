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


def _ntotal(path: Path) -> int:
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP).ntotal


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
    contents = set(manifest.get("contents", []))
    transcript: list[dict] = []
    if "dialogue" in contents:
        transcript = pq.read_table(staging / bundle.TRANSCRIPT).to_pylist()
        n_sent = pq.read_metadata(staging / bundle.SENTENCES_TABLE).num_rows
        n_index = _ntotal(staging / bundle.SENTENCES_INDEX)
        if (len(transcript), n_sent, n_index) != (counts.get("transcript_segments"),
                                                  counts.get("sentences"),
                                                  counts.get("sentences_index")):
            raise _bad(f"row counts {len(transcript)}/{n_sent}/{n_index} don't match "
                       f"manifest {counts}")
        if n_sent != n_index:
            raise _bad(f"sentences table {n_sent} != index {n_index}")
    shots: list[dict] = []
    frames: list[dict] = []
    if "visual" in contents:
        shots = pq.read_table(staging / bundle.SHOTS).to_pylist()
        frames = pq.read_table(staging / bundle.FRAMES).to_pylist()
        n_clip = _ntotal(staging / bundle.CLIP_INDEX)
        if (len(shots), len(frames), n_clip) != (counts.get("shots"), counts.get("frames"),
                                                 counts.get("clip_index")):
            raise _bad(f"row counts {len(shots)}/{len(frames)}/{n_clip} don't match "
                       f"manifest {counts}")
        if len(frames) != n_clip:
            raise _bad(f"frames table {len(frames)} != index {n_clip}")
    faces: list[dict] = []
    if "faces" in contents:
        if "visual" not in contents:
            raise _bad("faces without a visual index")
        faces = pq.read_table(staging / bundle.FACES_TABLE).to_pylist()
        n_faces = _ntotal(staging / bundle.FACES_INDEX)
        if (len(faces), n_faces) != (counts.get("faces"), counts.get("faces_index")):
            raise _bad(f"face counts {len(faces)}/{n_faces} don't match manifest {counts}")
        if any(not 0 <= r["frame_row"] < len(frames) for r in faces):
            raise _bad("a face points past the frames table")

    with db.db() as conn, db.tx(conn):
        for table in ("transcript_segments", "transcript_fts", "shots", "frames", "faces"):
            conn.execute(f"DELETE FROM {table} WHERE movie_id=?", (movie_id,))
        conn.executemany(
            'INSERT INTO transcript_segments(movie_id, seg, start, "end", text, words_json) '
            "VALUES (?,?,?,?,?,?)",
            [(movie_id, r["seg"], r["start"], r["end"], r["text"], r["words_json"])
             for r in transcript],
        )
        conn.executemany("INSERT INTO transcript_fts(text, movie_id, seg) VALUES (?,?,?)",
                         [(r["text"], movie_id, r["seg"]) for r in transcript])
        conn.executemany('INSERT INTO shots(movie_id, idx, start, "end") VALUES (?,?,?,?)',
                         [(movie_id, r["idx"], r["start"], r["end"]) for r in shots])
        conn.executemany("INSERT INTO frames(movie_id, row, t, shot_idx) VALUES (?,?,?,?)",
                         [(movie_id, r["row"], r["t"], r["shot_idx"]) for r in frames])
        conn.executemany("INSERT INTO faces(movie_id, row, frame_row, x, y, w, h, det_score) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         [(movie_id, r["row"], r["frame_row"], r["x"], r["y"], r["w"], r["h"],
                           r["det_score"]) for r in faces])
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
