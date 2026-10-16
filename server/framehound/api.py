"""FastAPI app: REST + SSE. Search runs in process against memory-mapped indexes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from framehound_indexer import bundle, load_lock
from pydantic import BaseModel, Field

from . import db, jobs, queue, tasks
from .config import settings
from .errors import FHError
from .resolver import Collection, SourceResolver
from .search import dialogue

log = logging.getLogger("framehound.api")

_encoder_state = {"state": "loading", "error": None}


def _warm_encoders() -> None:
    try:
        from framehound_indexer.encoders import sentence_encoder

        sentence_encoder()
        _encoder_state["state"] = "ready"
    except Exception as e:  # search degrades to lexical-only dialogue matching
        log.exception("sentence encoder failed to load")
        _encoder_state.update(state="error", error=f"{type(e).__name__}: {e}")


def _gc_clips_forever() -> None:
    while True:
        try:
            gc_clips()
        except Exception:
            log.exception("clip gc failed")
        time.sleep(3600)


def gc_clips() -> None:
    cutoff = db.now() - settings().clip_ttl_s
    with db.db() as conn:
        for c in db.rows(conn, "SELECT id, path FROM clips WHERE created_at < ?", cutoff):
            if c["path"]:
                Path(c["path"]).unlink(missing_ok=True)
            conn.execute("DELETE FROM clips WHERE id=?", (c["id"],))


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    if not settings().redis_url:
        # Inline mode: jobs ran in this process's threads, so none survived a restart.
        with db.db() as conn:
            conn.execute("UPDATE jobs SET heartbeat_at=0 WHERE state='running'")
            jobs.reap_stale(conn)
            for j in db.rows(conn, "SELECT id, kind FROM jobs WHERE state='queued'"):
                queue.enqueue(j["kind"], j["id"])
    if os.environ.get("FH_SKIP_ENCODERS") != "1":
        threading.Thread(target=_warm_encoders, daemon=True).start()
    threading.Thread(target=_gc_clips_forever, daemon=True).start()
    yield


app = FastAPI(title="FrameHound", version="0.1.0", lifespan=lifespan)


@app.exception_handler(FHError)
async def fh_error(_: Request, exc: FHError):
    return JSONResponse(exc.to_dict(), status_code=exc.status)


def _not_found(what: str = "movie") -> FHError:
    return FHError("not_found", f"That {what} doesn't exist.", status=404)


# serialization ----------------------------------------------------------------

def _movie_out(conn, m: dict) -> dict:
    mdir = settings().movie_dir(m["id"])
    active = db.row(conn, "SELECT * FROM jobs WHERE movie_id=? AND kind IN ('ingest','index') "
                    "ORDER BY created_at DESC LIMIT 1", m["id"])
    manifest = _manifest(m["id"])
    contents = set(manifest.get("contents", [])) if manifest else set()
    return {
        "id": m["id"], "title": m["title"], "status": m["status"], "error": m["error"],
        "merged_into": m["merged_into"], "duration_s": m["duration_s"], "width": m["width"],
        "height": m["height"], "fps": m["fps"], "source_kind": m["source_kind"],
        "created_at": m["created_at"],
        "has_source": bool(m["source_file"]) and (mdir / m["source_file"]).exists(),
        "has_proxy": (mdir / "proxy.mp4").exists(),
        "capabilities": {"dialogue": "dialogue" in contents, "visual": "visual" in contents,
                         "faces": "faces" in contents},
        "models_ok": (m["models_lock_hash"] == load_lock().hash) if manifest else None,
        "job": _job_out(active) if active else None,
        "thumbs": [],  # shot thumbnails arrive with the shots stage (M2)
    }


def _job_out(j: dict, stages: list[dict] | None = None) -> dict:
    out = {k: j[k] for k in ("id", "movie_id", "kind", "state", "stage", "pct", "eta_s",
                             "error", "error_code", "gpu", "compute_units_used", "seq")}
    out["detail"] = j["message"] if j["state"] == "failed" else None
    if stages is not None:
        out["stages"] = stages
    return out


def _stages(conn, job_id: str) -> list[dict]:
    return db.rows(conn, "SELECT stage, state, pct, eta_s FROM job_stages WHERE job_id=? "
                   "ORDER BY ord", job_id)


def _manifest(movie_id: str) -> dict | None:
    p = settings().movie_dir(movie_id) / "index" / bundle.MANIFEST
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _get_movie(conn, movie_id: str) -> dict:
    m = db.row(conn, "SELECT * FROM movies WHERE id=?", movie_id)
    if m is None:
        raise _not_found()
    return m


# health ---------------------------------------------------------------------------

@app.get("/health")
def health():
    s = settings()
    du = shutil.disk_usage(s.data_dir) if s.data_dir.exists() else None
    return {
        "ok": True,
        "redis": queue.redis_ok(),
        "disk_free_bytes": du.free if du else None,
        "index_backend": s.index_backend,
        "colab": {"state": "not_configured"},
        "encoders": _encoder_state,
        "models_lock_hash": load_lock().hash,
    }


# movies -------------------------------------------------------------------------

class AddMovie(BaseModel):
    source: str = Field(min_length=1, max_length=4096)


@app.get("/movies")
def list_movies():
    with db.db() as conn:
        jobs.reap_stale(conn)
        ms = db.rows(conn, "SELECT * FROM movies WHERE status<>'merged' "
                     "ORDER BY created_at DESC")
        return {"movies": [_movie_out(conn, m) for m in ms]}


@app.post("/movies", status_code=201)
async def add_movie(request: Request):
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/"):
        raise FHError("unsupported_source", ".torrent files aren't supported yet. "
                      "Paste a web or archive.org link instead.")
    try:
        body = AddMovie.model_validate(await request.json())
    except Exception as e:
        raise FHError("unrecognized", "Paste a link to a movie.", detail=str(e)) from e
    return await run_in_threadpool(_add_movie, body.source.strip())


def _add_movie(source: str):
    resolver = SourceResolver()
    try:
        res = resolver.resolve(source)
    finally:
        resolver.client.close()
    if isinstance(res, Collection):
        return JSONResponse(_collection_out(res), status_code=200)
    with db.db() as conn, db.tx(conn):
        dup = db.row(conn, "SELECT id FROM movies WHERE source_ref=? AND status<>'merged'",
                     res.source_ref)
        if dup:
            raise FHError("duplicate", "You've already added this movie.", status=409,
                          movie_id=dup["id"])
        movie_id = db.new_id()
        conn.execute(
            "INSERT INTO movies(id, title, source_kind, source_ref, source_url, source_md5, "
            "source_file, bytes, status, created_at, last_accessed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (movie_id, res.title, res.kind, res.source_ref, res.url, res.md5, res.filename,
             res.expected_size, "ingesting", db.now(), db.now()),
        )
        job_id = tasks.start_ingest(conn, movie_id)
    queue.enqueue("ingest", job_id)
    return {"movie_id": movie_id, "job_id": job_id}


def _collection_out(c: Collection) -> dict:
    return {"collection": {"identifier": c.identifier, "title": c.title, "total": c.total},
            "choices": c.choices}


@app.get("/collections/{identifier}")
def collection_items(identifier: str, q: str | None = None):
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", identifier):
        raise _not_found("collection")
    resolver = SourceResolver()
    try:
        return _collection_out(resolver.collection(identifier, q=q))
    finally:
        resolver.client.close()


@app.get("/movies/{movie_id}")
def get_movie(movie_id: str):
    with db.db() as conn:
        jobs.reap_stale(conn)
        m = _get_movie(conn, movie_id)
        return _movie_out(conn, m)


@app.delete("/movies/{movie_id}", status_code=204)
def delete_movie(movie_id: str):
    with db.db() as conn:
        _get_movie(conn, movie_id)
        conn.execute("UPDATE jobs SET cancel_requested=1 WHERE movie_id=? AND state IN "
                     "('queued','running')", (movie_id,))
        with db.tx(conn):
            for table in ("transcript_segments", "transcript_fts", "shots", "frames", "faces",
                          "clips", "queries"):
                conn.execute(f"DELETE FROM {table} WHERE movie_id=?", (movie_id,))
            conn.execute("DELETE FROM movies WHERE id=? OR merged_into=?", (movie_id, movie_id))
    dialogue.evict(movie_id)
    shutil.rmtree(settings().movie_dir(movie_id), ignore_errors=True)


@app.post("/movies/{movie_id}/retry", status_code=202)
def retry_movie(movie_id: str):
    """Restart from the first missing step: download/proxy, else indexing (also re-index)."""
    with db.db() as conn, db.tx(conn):
        m = _get_movie(conn, movie_id)
        if m["status"] in tasks.ACTIVE:
            raise FHError("busy", "This movie is already being worked on.", status=409)
        mdir = settings().movie_dir(movie_id)
        ready_media = m["sha256"] and (mdir / (m["source_file"] or "")).exists() \
            and (mdir / "proxy.mp4").exists()
        if ready_media:
            kind, job_id = "index", tasks.start_index(conn, movie_id)
        else:
            kind, job_id = "ingest", tasks.start_ingest(conn, movie_id)
    queue.enqueue(kind, job_id)
    return {"job_id": job_id, "kind": kind}


@app.get("/movies/{movie_id}/index-progress")
def index_progress(movie_id: str):
    from .index_backend import get_backend

    with db.db() as conn:
        jobs.reap_stale(conn)
        m = _get_movie(conn, movie_id)
        latest = {}
        for kind in ("ingest", "index"):
            latest[kind] = db.row(conn, "SELECT * FROM jobs WHERE movie_id=? AND kind=? "
                                  "ORDER BY created_at DESC LIMIT 1", movie_id, kind)
        # An index job older than the newest ingest belongs to a previous attempt.
        if latest["ingest"] and latest["index"] and \
                latest["index"]["created_at"] < latest["ingest"]["created_at"]:
            latest["index"] = None
        stages = []
        for kind in ("ingest", "index"):
            j = latest[kind]
            if j:
                stages += [{**st, "job": kind} for st in _stages(conn, j["id"])]
            elif kind == "index":
                stages += [{"stage": st, "state": "pending", "pct": 0, "eta_s": None,
                            "job": "index"} for st in get_backend().planned_stages()]
            else:
                stages += [{"stage": st, "state": "done", "pct": 1, "eta_s": 0, "job": "ingest"}
                           for st in jobs.INGEST_STAGES]
        current = latest["index"] or latest["ingest"]
        running = [st["eta_s"] for st in stages if st["state"] == "running" and st["eta_s"]]
        return {
            "movie_id": movie_id, "status": m["status"], "error": m["error"],
            "job": _job_out(current) if current else None,
            "stages": stages,
            "eta_s": max(running) if running else None,
            "gpu": latest["index"]["gpu"] if latest["index"] else None,
            "compute_units_used": latest["index"]["compute_units_used"] if latest["index"] else None,
        }


@app.get("/movies/{movie_id}/shots")
def shots(movie_id: str, from_: float = Query(default=0.0, alias="from"),
          to: float = Query(default=1e12)):
    lo, hi = from_, to
    with db.db() as conn:
        _get_movie(conn, movie_id)
        rs = db.rows(conn, 'SELECT idx, start, "end" FROM shots WHERE movie_id=? AND "end">=? '
                     "AND start<=? ORDER BY idx", movie_id, lo, hi)
    return [{**r, "thumb_url": f"/media/{movie_id}/thumbs/{r['idx']}.jpg"} for r in rs]


# search ----------------------------------------------------------------------------

class SearchBody(BaseModel):
    type: str = Field(pattern="^(text|image|dialogue)$")
    q: str | None = Field(default=None, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    also_dialogue: bool = False
    heat_bins: int | None = Field(default=None, ge=64, le=4096)


@app.post("/movies/{movie_id}/search")
async def search(movie_id: str, request: Request):
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/"):
        form = await request.form()
        raw = {k: v for k, v in form.items() if isinstance(v, str)}
        raw.setdefault("type", "image")
    else:
        raw = await request.json()
    try:
        body = SearchBody.model_validate(raw)
    except Exception as e:
        raise FHError("bad_query", "That search couldn't be read.", detail=str(e)) from e
    return await run_in_threadpool(_search, movie_id, body)


def _search(movie_id: str, body: SearchBody) -> dict:
    s = settings()
    t0 = time.perf_counter()
    with db.db() as conn:
        m = _get_movie(conn, movie_id)
        conn.execute("UPDATE movies SET last_accessed_at=? WHERE id=?", (db.now(), movie_id))
    if m["status"] != "ready":
        raise FHError("not_ready", "This movie isn't ready to search yet.", status=409)
    manifest = _manifest(movie_id) or {}
    if manifest.get("models_lock_hash") != load_lock().hash:
        raise FHError("models_mismatch", "This movie was indexed with older models. "
                      "Re-index it to search.", status=409)
    contents = set(manifest.get("contents", []))
    q = (body.q or "").strip()
    use_dialogue = body.type == "dialogue" or (
        body.type == "text" and (body.also_dialogue or dialogue.looks_like_quote(q)))
    if body.type in ("text", "image") and "visual" not in contents and not use_dialogue:
        raise FHError("no_visual_index", "This movie was indexed for dialogue only, so "
                      "Describe it and Show a picture aren't available yet. Quote a line instead.",
                      status=409)
    if not q:
        raise FHError("bad_query", "Type something to find.")
    if "dialogue" not in contents:
        raise FHError("no_dialogue_index", "This movie has no dialogue index.", status=409)

    qvec = None
    if _encoder_state["state"] != "error":
        try:
            from framehound_indexer.encoders import encode_sentence

            qvec = encode_sentence([q])[0]
        except Exception:
            log.exception("query encoding failed; lexical only")
    idx = dialogue.dialogue_index(movie_id, s.movie_dir(movie_id) / "index")
    result = idx.search(q, qvec, m["duration_s"] or 0.0, body.top_k or s.top_k,
                        body.heat_bins or s.heat_bins, s.z_min)
    latency = (time.perf_counter() - t0) * 1000
    with db.db() as conn:
        conn.execute("INSERT INTO queries(id, movie_id, type, q, created_at, latency_ms) "
                     "VALUES (?,?,?,?,?,?)",
                     (db.new_id(), movie_id, body.type, q, db.now(), latency))
    result["latency_ms"] = round(latency, 1)
    result["semantic"] = qvec is not None
    return result


# clips -----------------------------------------------------------------------------

class ClipBody(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    mode: str = Field(default="accurate", pattern="^(accurate|fast)$")
    format: str = Field(default="mp4", pattern="^(mp4|gif)$")
    subs: bool = False
    crop: bool = False


@app.post("/movies/{movie_id}/clips", status_code=201)
def make_clip(movie_id: str, body: ClipBody):
    s = settings()
    if body.format != "mp4" or body.subs or body.crop:
        raise FHError("unsupported_format", "GIF, subtitles and face crop arrive in a later "
                      "version. Cut a plain video clip for now.")
    dur = body.end - body.start
    if dur < 0.1:
        raise FHError("bad_span", "Set an out point after the in point.")
    if dur > s.max_clip_s + 1e-6:
        raise FHError("too_long", f"Clips are limited to {s.max_clip_s:g} seconds. "
                      "Move the in or out point.")
    with db.db() as conn, db.tx(conn):
        m = _get_movie(conn, movie_id)
        if m["duration_s"] and body.end > m["duration_s"] + 0.5:
            raise FHError("bad_span", "The out point is past the end of the movie.")
        if not (s.movie_dir(movie_id) / (m["source_file"] or "")).exists():
            raise FHError("source_missing", "The movie file is gone. Re-download it to clip.",
                          status=409)
        job_id = jobs.create(conn, movie_id, "clip", ["cutting"])
        clip_id = db.new_id()
        conn.execute('INSERT INTO clips(id, movie_id, start, "end", mode, format, job_id, '
                     "created_at) VALUES (?,?,?,?,?,?,?,?)",
                     (clip_id, movie_id, body.start, min(body.end, m["duration_s"] or body.end),
                      body.mode, body.format, job_id, db.now()))
    queue.enqueue("clip", job_id)
    return {"clip_id": clip_id, "job_id": job_id}


def _get_clip(conn, clip_id: str) -> dict:
    c = db.row(conn, "SELECT * FROM clips WHERE id=?", clip_id)
    if c is None:
        raise _not_found("clip")
    return c


@app.get("/clips/{clip_id}")
def get_clip(clip_id: str):
    with db.db() as conn:
        c = _get_clip(conn, clip_id)
        j = db.row(conn, "SELECT * FROM jobs WHERE id=?", c["job_id"])
    return {"id": c["id"], "movie_id": c["movie_id"], "start": c["start"], "end": c["end"],
            "mode": c["mode"], "format": c["format"], "ready": bool(c["path"]),
            "file_url": f"/clips/{c['id']}/file", "job": _job_out(j) if j else None}


def _slug(text: str | None) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "clip").strip("-")
    return s[:60] or "clip"


def _hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600}-{t % 3600 // 60:02d}-{t % 60:02d}"


@app.get("/clips/{clip_id}/file")
def clip_file(clip_id: str):
    with db.db() as conn:
        c = _get_clip(conn, clip_id)
        m = db.row(conn, "SELECT title FROM movies WHERE id=?", c["movie_id"]) or {}
    if not c["path"] or not Path(c["path"]).exists():
        raise FHError("not_ready", "This clip isn't ready yet.", status=409)
    name = f"{_slug(m.get('title'))}-{_hms(c['start'])}.{c['format']}"
    return FileResponse(c["path"], media_type="video/mp4" if c["format"] == "mp4"
                        else "image/gif", filename=name, content_disposition_type="inline")


@app.get("/media/{movie_id}/proxy.mp4")
def proxy(movie_id: str):
    if not re.fullmatch(r"[a-z0-9]{1,32}", movie_id):
        raise _not_found()
    path = settings().movie_dir(movie_id) / "proxy.mp4"
    if not path.exists():
        raise FHError("not_ready", "This movie isn't ready to play yet.", status=404)
    return FileResponse(path, media_type="video/mp4")


# jobs --------------------------------------------------------------------------------

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with db.db() as conn:
        j = db.row(conn, "SELECT * FROM jobs WHERE id=?", job_id)
        if j is None:
            raise _not_found("job")
        return _job_out(j, _stages(conn, job_id))


@app.post("/jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str):
    with db.db() as conn:
        j = db.row(conn, "SELECT * FROM jobs WHERE id=?", job_id)
        if j is None:
            raise _not_found("job")
        if j["state"] == "queued":
            # Not picked up yet: the worker sees the flag and skips it.
            conn.execute("UPDATE jobs SET cancel_requested=1, state='cancelled', finished_at=?, "
                         "seq=seq+1 WHERE id=?", (db.now(), job_id))
            if j["kind"] in ("ingest", "index"):
                conn.execute("UPDATE movies SET status='cancelled' WHERE id=?", (j["movie_id"],))
        elif j["state"] == "running":
            conn.execute("UPDATE jobs SET cancel_requested=1, seq=seq+1 WHERE id=?", (job_id,))
    return {"ok": True}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request,
                     last_event_id: str | None = Header(default=None)):
    def load():
        with db.db() as conn:
            j = db.row(conn, "SELECT * FROM jobs WHERE id=?", job_id)
            return (j, _stages(conn, job_id)) if j else (None, None)

    j, _ = await run_in_threadpool(load)
    if j is None:
        raise _not_found("job")

    async def gen():
        last = int(last_event_id) if last_event_id and last_event_id.isdigit() else -1
        quiet = 0.0
        yield "retry: 3000\n\n"
        while not await request.is_disconnected():
            j, stages = await run_in_threadpool(load)
            if j is None:
                yield 'event: gone\ndata: {}\n\n'
                return
            if j["seq"] != last:
                last = j["seq"]
                quiet = 0.0
                yield f"id: {last}\nevent: progress\ndata: {json.dumps(_job_out(j, stages))}\n\n"
            if j["state"] in jobs.TERMINAL:
                return
            quiet += 0.5
            if quiet >= 15:
                quiet = 0.0
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("framehound.api:app", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
