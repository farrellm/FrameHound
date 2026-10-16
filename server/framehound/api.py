"""FastAPI app: REST + SSE. Search runs in process against memory-mapped indexes."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
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
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from framehound_indexer import bundle, load_lock
from pydantic import BaseModel, Field

from . import bitmagnet, db, evict, jobs, queue, tasks
from .config import settings
from .errors import FHError
from .resolver import Collection, SourceResolver
from .search import dialogue, faces, visual
from .torrent import TORRENT_FILE

log = logging.getLogger("framehound.api")

_encoder_state = {"state": "loading", "error": None}
_visual_state = {"state": "loading", "error": None}
_face_state = {"state": "loading", "error": None}


def _warm_encoders() -> None:
    try:
        from framehound_indexer.encoders import sentence_encoder

        sentence_encoder()
        _encoder_state["state"] = "ready"
    except Exception as e:  # search degrades to lexical-only dialogue matching
        log.exception("sentence encoder failed to load")
        _encoder_state.update(state="error", error=f"{type(e).__name__}: {e}")
    try:
        from framehound_indexer.encoders import visual_encoder

        visual_encoder().text_session()
        visual_encoder().vision_session()
        _visual_state["state"] = "ready"
    except Exception as e:  # Describe it and Show a picture are unavailable until it loads
        log.exception("visual encoder failed to load")
        _visual_state.update(state="error", error=f"{type(e).__name__}: {e}")
    try:
        from framehound_indexer.encoders import face_encoder

        face_encoder().sessions()
        _face_state["state"] = "ready"
    except Exception as e:  # Show a picture degrades to looks only
        log.exception("face encoder failed to load")
        _face_state.update(state="error", error=f"{type(e).__name__}: {e}")


def _colab_forever() -> None:
    """Inline (no redis) mode with Colab: what worker-index does in compose."""
    from .worker import colab_housekeeping

    colab_housekeeping()


def _gc_clips_forever() -> None:
    time.sleep(60)  # let startup (and a worker's recovery) settle first
    while True:
        try:
            gc_clips()
        except Exception:
            log.exception("clip gc failed")
        try:
            evict.enforce_quota()
        except Exception:
            log.exception("eviction failed")
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
            if settings().index_backend == "colab":
                # Colab jobs keep running on their VM: re-attach instead of failing them.
                conn.execute("UPDATE jobs SET heartbeat_at=? WHERE state='running' AND "
                             "colab_session IS NOT NULL", (db.now(),))
            jobs.reap_stale(conn)
            for j in db.rows(conn, "SELECT id, kind FROM jobs WHERE state='queued'"):
                queue.enqueue(j["kind"], j["id"])
        if settings().index_backend == "colab":
            threading.Thread(target=_colab_forever, daemon=True).start()
    if os.environ.get("FH_SKIP_ENCODERS") != "1":
        threading.Thread(target=_warm_encoders, daemon=True).start()
    threading.Thread(target=_gc_clips_forever, daemon=True).start()
    yield


app = FastAPI(title="FrameHound", version="0.1.0", lifespan=lifespan)


@app.exception_handler(FHError)
async def fh_error(_: Request, exc: FHError):
    return JSONResponse(exc.to_dict(), status_code=exc.status)


# auth -------------------------------------------------------------------------------
#
# With FH_TOKEN set, every request needs it, as a bearer token or as the cookie
# POST /auth sets. The cookie is what lets <video>, <img>, EventSource and clip
# links through, since they can't send headers. It holds an HMAC of the token,
# so changing FH_TOKEN signs every browser out.

AUTH_COOKIE = "fh_auth"
OPEN_PATHS = {"/health", "/auth"}


def _cookie_value(token: str) -> str:
    return hmac.new(token.encode(), b"framehound-session", hashlib.sha256).hexdigest()


def _authorized(request: Request, token: str) -> bool:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer ") and \
            hmac.compare_digest(header[7:].strip().encode(), token.encode()):
        return True
    cookie = request.cookies.get(AUTH_COOKIE, "")
    return bool(cookie) and hmac.compare_digest(cookie.encode(),
                                                _cookie_value(token).encode())


AUTH_REQUIRED = {"code": "auth_required",
                 "message": "This FrameHound needs its access token to open."}


@app.middleware("http")
async def require_token(request: Request, call_next):
    token = settings().token
    if token and request.url.path not in OPEN_PATHS and not _authorized(request, token):
        return JSONResponse(AUTH_REQUIRED, status_code=401)
    return await call_next(request)


class SignIn(BaseModel):
    token: str = Field(min_length=1, max_length=1000)


@app.post("/auth")
async def sign_in(body: SignIn, request: Request):
    token = settings().token
    if not token:
        return {"ok": True, "auth_required": False}
    if not hmac.compare_digest(body.token.strip().encode(), token.encode()):
        await asyncio.sleep(1)  # slows guessing
        raise FHError("bad_token", "That token isn't right. It's FH_TOKEN in the server's "
                      ".env.", status=401)
    resp = JSONResponse({"ok": True, "auth_required": True})
    https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie(AUTH_COOKIE, _cookie_value(token), max_age=400 * 86400, httponly=True,
                    samesite="lax", secure=https, path="/")
    return resp


@app.delete("/auth", status_code=204)
def sign_out():
    resp = Response(status_code=204)
    resp.delete_cookie(AUTH_COOKIE, path="/")
    return resp


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
        "thumbs": _sample_thumbs(conn, m["id"]) if "visual" in contents else [],
    }


def _sample_thumbs(conn, movie_id: str, n: int = 5) -> list[str]:
    """Shot thumbnails spread across the movie, for the library's mini strip."""
    rows = conn.execute("SELECT idx FROM shots WHERE movie_id=? ORDER BY idx",
                        (movie_id,)).fetchall()
    if not rows:
        return []
    picks = sorted({round(i * (len(rows) - 1) / max(n - 1, 1)) for i in range(n)})
    return [visual.thumb_url(movie_id, rows[k][0]) for k in picks]


def _job_out(j: dict, stages: list[dict] | None = None) -> dict:
    out = {k: j[k] for k in ("id", "movie_id", "kind", "state", "stage", "pct", "eta_s",
                             "error", "error_code", "gpu", "compute_units_used", "seq")}
    out["detail"] = j["message"] if j["state"] == "failed" else None
    # What a running stage is waiting on ("3 peers", "Waiting for another movie's GPU").
    out["note"] = j["message"] if j["state"] == "running" else None
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
def health(request: Request):
    s = settings()
    if s.token and not _authorized(request, s.token):
        return {"ok": True, "auth_required": True, "signed_in": False}
    du = shutil.disk_usage(s.data_dir) if s.data_dir.exists() else None
    return {
        "ok": True,
        "redis": queue.redis_ok(),
        "disk_free_bytes": du.free if du else None,
        "index_backend": s.index_backend,
        "colab": _colab_health(),
        "bitmagnet": {"state": "ok" if s.bitmagnet_url else "not_configured"},
        "encoders": _encoder_state,
        "visual_encoder": _visual_state,
        "face_encoder": _face_state,
        "models_lock_hash": load_lock().hash,
        "auth_required": bool(s.token), "signed_in": True,
        "quota_bytes": int(s.max_data_gb * 1024**3) if s.max_data_gb > 0 else None,
    }


_colab_cache: dict = {"at": 0.0, "value": None}


def _colab_health() -> dict:
    """Colab auth and balance, cached for a minute (each call runs the CLI)."""
    if settings().index_backend != "colab":
        return {"state": "not_configured"}
    if time.monotonic() - _colab_cache["at"] < 60 and _colab_cache["value"]:
        return _colab_cache["value"]
    from .colab import ColabCli, ColabError

    try:
        u = ColabCli().usage()
        value = {"state": "ok", "balance_cu": u.balance, "rate_cu_per_hour": u.rate_per_hour}
    except ColabError as e:
        value = {"state": "auth_broken" if e.code == "colab_auth" else "error",
                 "error": e.message}
    _colab_cache.update(at=time.monotonic(), value=value)
    return value


# movies -------------------------------------------------------------------------

class AddMovie(BaseModel):
    source: str = Field(min_length=1, max_length=4096)
    # A display title to use over the source's own (e.g. a magnet's release name).
    title: str | None = Field(default=None, max_length=300)


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
        limit = settings().max_torrent_kb * 1024
        too_big = FHError("bad_torrent", "That file is too large to be a .torrent file.",
                          status=413)
        if int(request.headers.get("content-length") or 0) > limit + 65536:
            raise too_big
        form = await request.form(max_files=1)
        upload = form.get("torrent")
        if upload is None or isinstance(upload, str):
            raise FHError("bad_torrent", "Choose a .torrent file to add.")
        data = await upload.read(limit + 1)
        if len(data) > limit:
            raise too_big
        return await run_in_threadpool(_add_torrent_file, data)
    try:
        body = AddMovie.model_validate(await request.json())
    except Exception as e:
        raise FHError("unrecognized", "Paste a link to a movie.", detail=str(e)) from e
    title = (body.title or "").strip() or None
    return await run_in_threadpool(_add_movie, body.source.strip(), title)


def _add_torrent_file(data: bytes):
    res = SourceResolver.resolve_torrent_file(data)
    return _create_movie(res, torrent=data)


def _add_movie(source: str, title: str | None = None):
    resolver = SourceResolver()
    try:
        res = resolver.resolve(source)
    finally:
        resolver.client.close()
    if isinstance(res, Collection):
        return JSONResponse(_collection_out(res), status_code=200)
    if title:
        res.title = title
    return _create_movie(res)


def _create_movie(res, torrent: bytes | None = None):
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
        if torrent is not None:
            mdir = settings().movie_dir(movie_id)
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / TORRENT_FILE).write_bytes(torrent)
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


@app.get("/torrents/search")
async def find_torrents(q: str = Query(default="", max_length=200)):
    if not settings().bitmagnet_url:
        raise FHError("not_configured", "Finding movies by name needs Bitmagnet. "
                      "Set BITMAGNET_URL on the server.", status=404)
    q = q.strip()
    if len(q) < 2:
        raise FHError("unrecognized", "Type at least two letters of the movie's name.")
    return await run_in_threadpool(bitmagnet.search, q)


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
    visual.evict(movie_id)
    faces.evict(movie_id)
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
                # A re-download keeps its index: those stages are already done.
                done = tasks.has_current_index(movie_id)
                stages += [{"stage": st, "state": "done" if done else "pending",
                            "pct": 1 if done else 0, "eta_s": 0 if done else None,
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
    return [{**r, "thumb_url": visual.thumb_url(movie_id, r["idx"])} for r in rs]


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
    image: bytes | None = None
    if ctype.startswith("multipart/"):
        limit = int(settings().max_query_image_mb * 1024 * 1024)
        too_big = FHError("image_too_big", f"That picture is over "
                          f"{settings().max_query_image_mb:g} MB. Use a smaller one.",
                          status=413)
        if int(request.headers.get("content-length") or 0) > limit + 65536:
            raise too_big
        form = await request.form(max_files=1)
        raw = {k: v for k, v in form.items() if isinstance(v, str)}
        raw.setdefault("type", "image")
        upload = form.get("image")
        if upload is not None and not isinstance(upload, str):
            image = await upload.read(limit + 1)
            if len(image) > limit:
                raise too_big
    elif ctype.startswith("application/x-www-form-urlencoded"):
        raw = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    else:
        try:
            raw = await request.json()
        except ValueError as e:
            raise FHError("bad_query", "That search couldn't be read.", detail=str(e)) from e
    try:
        body = SearchBody.model_validate(raw)
    except Exception as e:
        raise FHError("bad_query", "That search couldn't be read.", detail=str(e)) from e
    return await run_in_threadpool(_search, movie_id, body, image)


MAX_IMAGE_PIXELS = 50_000_000


def _open_image(data: bytes):
    """A query picture as RGB, upright (EXIF), and no bigger than 2048 px a side."""
    import io

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > MAX_IMAGE_PIXELS:
            raise FHError("bad_image", "That picture is too large. Use a smaller one.")
        im = ImageOps.exif_transpose(im).convert("RGB")
    except FHError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise FHError("bad_image", "That file isn't a picture FrameHound can read. "
                      "Try a JPEG or PNG.", detail=str(e)) from e
    im.thumbnail((2048, 2048))
    return im


def _search(movie_id: str, body: SearchBody, image: bytes | None = None) -> dict:
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
    if body.type == "image":
        if "visual" not in contents:
            raise FHError("no_visual_index", "This movie was indexed for dialogue only, so "
                          "Show a picture isn't available. Re-index it, or quote a line "
                          "instead.", status=409)
        if not image:
            raise FHError("bad_query", "Choose a picture to find.")
    if body.type == "text" and "visual" not in contents and not use_dialogue:
        raise FHError("no_visual_index", "This movie was indexed for dialogue only, so "
                      "Describe it isn't available. Re-index it, or quote a line instead.",
                      status=409)
    if not q and body.type != "image":
        raise FHError("bad_query", "Type something to find.")
    if body.type == "dialogue" and "dialogue" not in contents:
        raise FHError("no_dialogue_index", "This movie has no dialogue index.", status=409)

    top_k = body.top_k or s.top_k
    bins = body.heat_bins or s.heat_bins
    duration = m["duration_s"] or 0.0
    index_dir = s.movie_dir(movie_id) / "index"
    results = []
    semantic = None
    face_found = None
    image_sha = None
    if body.type in ("text", "image") and "visual" in contents:
        if _visual_state["state"] == "error":
            raise FHError("encoder_unavailable", "Searching by what's on screen is "
                          "unavailable: the visual model didn't load on the server.",
                          detail=_visual_state["error"], status=503)
        vidx = visual.visual_index(movie_id, index_dir)
        fidx = faces.face_index(movie_id, index_dir, vidx.times.size) \
            if "faces" in contents else None
        if body.type == "text":
            from framehound_indexer.encoders import encode_text

            results.append(vidx.search(encode_text(q), duration, top_k, bins, s.z_min,
                                       faces=fidx))
        else:
            from framehound_indexer.encoders import encode_face, encode_image

            image_sha = hashlib.sha256(image or b"").hexdigest()
            im = _open_image(image or b"")
            face_vec = None
            if fidx is not None and _face_state["state"] != "error":
                found = encode_face(im)
                face_found = found is not None
                face_vec = found[0] if found else None
            results.append(vidx.search_image(
                encode_image(im), face_vec, duration, top_k, bins, s.z_min, fidx,
                match=float(load_lock().faces["match"]), bonus=s.face_bonus,
                scale=s.face_scale))
    if use_dialogue and "dialogue" in contents:
        qvec = None
        if _encoder_state["state"] != "error":
            try:
                from framehound_indexer.encoders import encode_sentence

                qvec = encode_sentence([q])[0]
            except Exception:
                log.exception("query encoding failed; lexical only")
        semantic = qvec is not None
        idx = dialogue.dialogue_index(movie_id, index_dir)
        results.append(idx.search(q, qvec, duration, top_k, bins, s.z_min))
    result = _merge(results, top_k) if len(results) > 1 else results[0]
    latency = (time.perf_counter() - t0) * 1000
    with db.db() as conn:
        conn.execute("INSERT INTO queries(id, movie_id, type, q, image_sha, created_at, "
                     "latency_ms) VALUES (?,?,?,?,?,?,?)",
                     (db.new_id(), movie_id, body.type, q or None, image_sha, db.now(),
                      latency))
    result["latency_ms"] = round(latency, 1)
    result["semantic"] = semantic
    if body.type == "image":
        result["face_found"] = face_found
    return result


def _merge(results: list[dict], top_k: int) -> dict:
    """Visual + dialogue: each list's scores scaled by its own best, interleaved by
    that, overlaps resolved by NMS; heat is the pointwise max (both are z)."""
    from .search.heat import nms

    cands = []
    for r in results:
        best = max((c["score"] for c in r["candidates"]), default=1.0) or 1.0
        cands += [{**c, "score": round(c["score"] / best, 4)} for c in r["candidates"]]
    heat = [max(vals) for vals in zip(*(r["heat"] for r in results), strict=True)]
    return {"candidates": nms(cands)[:top_k], "heat": heat,
            "heat_threshold": min(r["heat_threshold"] for r in results)}


# clips -----------------------------------------------------------------------------

class ClipBody(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    mode: str = Field(default="accurate", pattern="^(accurate|fast)$")
    format: str = Field(default="mp4", pattern="^(mp4|gif)$")
    subs: bool = False
    crop: bool = False
    # The face's box at the candidate's peak, as fractions of the frame (x, y, w, h).
    face_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)


@app.post("/movies/{movie_id}/clips", status_code=201)
def make_clip(movie_id: str, body: ClipBody):
    s = settings()
    if body.subs and body.format != "mp4":
        raise FHError("bad_format", "Subtitles are only for video clips. Choose Video with "
                      "subtitles.")
    crop_box = None
    if body.crop:
        b = body.face_bbox
        if not b or not all(0 <= v <= 1 for v in b) or b[2] <= 0 or b[3] <= 0 \
                or b[0] + b[2] > 1.001 or b[1] + b[3] > 1.001:
            raise FHError("bad_crop", "Crop to the face needs a face. Pick a Same face match.")
        crop_box = json.dumps([round(v, 5) for v in b])
    # Cropping, subtitles and GIF all re-encode, so they are always exact.
    mode = "accurate" if crop_box or body.subs or body.format == "gif" else body.mode
    dur = body.end - body.start
    if dur < 0.1:
        raise FHError("bad_span", "Set an out point after the in point.")
    if dur > s.max_clip_s + 1e-6:
        raise FHError("too_long", f"Clips are limited to {s.max_clip_s:g} seconds. "
                      "Move the in or out point.")
    if body.format == "gif" and dur > s.max_gif_s + 1e-6:
        raise FHError("too_long", f"GIF is limited to {s.max_gif_s:g} seconds. Shorten the "
                      "clip to use it.")
    if body.subs and "dialogue" not in set((_manifest(movie_id) or {}).get("contents", [])):
        raise FHError("no_dialogue_index", "This movie has no transcript, so it can't have "
                      "subtitles. Cut a plain video clip.", status=409)
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
                     "created_at, crop_box, subs) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (clip_id, movie_id, body.start, min(body.end, m["duration_s"] or body.end),
                      mode, body.format, job_id, db.now(), crop_box, int(body.subs)))
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
            "crop": bool(c["crop_box"]), "subs": bool(c["subs"]),
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


@app.get("/media/{movie_id}/thumbs/{idx}.jpg")
def thumb(movie_id: str, idx: int):
    if not re.fullmatch(r"[a-z0-9]{1,32}", movie_id) or idx < 0:
        raise _not_found()
    path = settings().movie_dir(movie_id) / "index" / bundle.THUMBS / f"{idx}.jpg"
    if not path.exists():
        raise FHError("not_found", "No such thumbnail.", status=404)
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


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
