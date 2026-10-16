"""Worker entry points: ingest and clip run on the ``io`` queue, index on ``index``."""

from __future__ import annotations

import json
import shutil
import time

from framehound_indexer import load_lock

from . import db, jobs, queue
from .config import settings
from .download import download, sha256_file
from .errors import Cancelled, FHError
from .importer import import_bundle
from .index_backend import get_backend
from .media import cut_clip, ffprobe, transcode_proxy, validate
from .safe_http import SafeClient

ACTIVE = ("ingesting", "indexing")


def _movie(conn, movie_id: str) -> dict:
    m = db.row(conn, "SELECT * FROM movies WHERE id=?", movie_id)
    if m is None:
        raise FHError("not_found", "This movie was removed.")
    return m


def _set_movie(conn, movie_id: str, **fields) -> None:
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE movies SET {cols} WHERE id=?", (*fields.values(), movie_id))


def start_ingest(conn, movie_id: str) -> str:
    job_id = jobs.create(conn, movie_id, "ingest", jobs.INGEST_STAGES)
    _set_movie(conn, movie_id, status="ingesting", error=None)
    return job_id


def start_index(conn, movie_id: str) -> str:
    attempt = conn.execute("SELECT COUNT(*) FROM jobs WHERE movie_id=? AND kind='index'",
                           (movie_id,)).fetchone()[0]
    job_id = jobs.create(conn, movie_id, "index", get_backend().planned_stages(), attempt)
    _set_movie(conn, movie_id, status="indexing", error=None)
    return job_id


# ingest ---------------------------------------------------------------------

def run_ingest(job_id: str) -> None:
    db.init()
    job = jobs.Job(job_id)
    conn = job.conn
    mid = job.movie_id
    try:
        if not job.start():
            return
        m = _movie(conn, mid)
        mdir = settings().movie_dir(mid)
        mdir.mkdir(parents=True, exist_ok=True)
        src = mdir / m["source_file"]

        if src.exists():
            job.skip("downloading")
        else:
            job.stage("downloading", 0.0, force=True)
            t0 = time.monotonic()

            def on_dl(frac: float, done: int, total: int) -> None:
                eta = None
                if frac > 0.01:
                    eta = round((time.monotonic() - t0) * (1 - frac) / frac)
                job.stage("downloading", frac, eta)

            with SafeClient() as client:
                download(client, m["source_url"], src, m["bytes"], m["source_md5"], on_dl,
                         job.check_cancel)
            job.stage_done("downloading")

        job.stage("probing", 0.0, force=True)
        size = src.stat().st_size
        probe = ffprobe(src)
        validate(probe, size)
        sha = sha256_file(src)
        job.check_cancel(force=True)
        target = _dedup(conn, mid, sha)
        if target:
            _set_movie(conn, mid, status="merged", merged_into=target)
            job.finish(message=f"merged into {target}")
            shutil.rmtree(mdir, ignore_errors=True)
            return
        _set_movie(conn, mid, sha256=sha, duration_s=probe.duration_s, width=probe.width,
                   height=probe.height, fps=probe.fps, bytes=size)
        job.stage_done("probing")

        proxy = mdir / "proxy.mp4"
        if proxy.exists():
            job.skip("transcoding_proxy")
        else:
            job.stage("transcoding_proxy", 0.0, force=True)
            t0 = time.monotonic()

            def on_tx(frac: float) -> None:
                eta = round((time.monotonic() - t0) * (1 - frac) / frac) if frac > 0.01 else None
                job.stage("transcoding_proxy", frac, eta)

            transcode_proxy(src, proxy, probe.duration_s, on_tx, job.check_cancel)
            job.stage_done("transcoding_proxy")

        with db.tx(conn):
            index_job = start_index(conn, mid)
        job.finish()
        queue.enqueue("index", index_job)
    except Cancelled:
        job.cancelled()
        _set_movie(conn, mid, status="cancelled", error=None)
    except Exception as e:  # noqa: BLE001 (recorded on the job)
        err = job.fail(e)
        _set_movie(conn, mid, status="failed", error=err.message)
    finally:
        job.close()


def _dedup(conn, movie_id: str, sha: str) -> str | None:
    """Return the id of an existing movie with the same bytes to merge into, if any.

    A ready movie with current models, or one still in progress, absorbs the
    new one. A stale or failed one gives up its claim on the hash instead.
    """
    other = db.row(conn, "SELECT * FROM movies WHERE sha256=? AND id<>?", sha, movie_id)
    if other is None:
        return None
    if (other["status"] == "ready" and other["models_lock_hash"] == load_lock().hash) \
            or other["status"] in ACTIVE:
        return other["id"]
    _set_movie(conn, other["id"], sha256=None)
    return None


# index ----------------------------------------------------------------------

def run_index(job_id: str) -> None:
    db.init()
    s = settings()
    job = jobs.Job(job_id)
    conn = job.conn
    mid = job.movie_id
    backend = get_backend()
    handle: dict | None = None
    failed = True
    try:
        if not job.start():
            return
        job.set(gpu=backend.gpu or "local")
        m = _movie(conn, mid)
        mdir = s.movie_dir(mid)
        src = mdir / (m["source_file"] or "")
        if not src.exists():
            raise FHError("source_missing", "The movie file is gone. Re-download it to index.")

        job.stage("staging_media", 0.0, force=True)
        spec = {"movie_id": mid, "media": {"path": str(src)}, "duration_s": m["duration_s"],
                "models_lock": load_lock().data}
        handle = backend.start(job_id, spec)
        job.set(backend_handle=json.dumps(handle))
        job.stage_done("staging_media")

        timeout = s.local_index_timeout_s if backend.name == "local" else s.index_timeout_s
        deadline = time.monotonic() + timeout
        last_progress = time.monotonic()
        finished = False
        while not finished:
            events, alive = backend.poll(handle)
            for ev in events:
                last_progress = time.monotonic()
                if ev["stage"] == "done":
                    finished = True
                elif ev["stage"] == "error":
                    raise FHError("index_failed", "Indexing failed.",
                                  detail=f"{ev.get('message')}\n{backend.log_tail(handle)}")
                else:
                    job.stage(ev["stage"], ev.get("pct", 0.0), ev.get("eta_s"),
                              state="done" if ev.get("state") == "done" else "running",
                              force=ev.get("state") == "done")
            if finished:
                break
            if not alive:
                # The process may have written its last lines just before exiting.
                events, _ = backend.poll(handle)
                if any(ev["stage"] == "done" for ev in events):
                    break
                raise FHError("index_failed", "Indexing stopped unexpectedly.",
                              detail=backend.log_tail(handle))
            now = time.monotonic()
            if now > deadline:
                raise FHError("index_timeout", f"Indexing took longer than "
                              f"{timeout // 60} minutes and was stopped.")
            if now - last_progress > s.stage_watchdog_s:
                raise FHError("index_stalled", "Indexing stopped making progress and was "
                              "stopped.", detail=backend.log_tail(handle))
            job.check_cancel()
            time.sleep(backend.poll_interval_s)

        job.stage("downloading_index", 0.0, force=True)
        bundle_path = backend.fetch(handle, mdir / "index.tar.zst")
        job.stage_done("downloading_index")
        job.stage("importing", 0.0, force=True)
        import_bundle(mid, bundle_path)
        job.stage_done("importing")
        _set_movie(conn, mid, status="ready", error=None)
        job.finish()
        failed = False
    except Cancelled:
        job.cancelled()
        _set_movie(conn, mid, status="cancelled", error=None)
        failed = False
    except Exception as e:  # noqa: BLE001 (recorded on the job)
        err = job.fail(e)
        _set_movie(conn, mid, status="failed", error=err.message)
    finally:
        if handle is not None:
            backend.cleanup(handle, failed=failed)
        job.close()


# clip -----------------------------------------------------------------------

def run_clip(job_id: str) -> None:
    db.init()
    job = jobs.Job(job_id)
    conn = job.conn
    try:
        if not job.start():
            return
        clip = db.row(conn, "SELECT * FROM clips WHERE job_id=?", job_id)
        if clip is None:
            raise FHError("not_found", "This clip was removed.")
        m = _movie(conn, clip["movie_id"])
        mdir = settings().movie_dir(m["id"])
        src = mdir / (m["source_file"] or "")
        if not src.exists():
            raise FHError("source_missing", "The movie file is gone. Re-download it to clip.")
        out = mdir / "clips" / f"{clip['id']}.{clip['format']}"
        out.parent.mkdir(parents=True, exist_ok=True)
        job.stage("cutting", 0.0, force=True)
        cut_clip(src, out, clip["start"], clip["end"], clip["mode"],
                 lambda f: job.stage("cutting", f), job.check_cancel)
        conn.execute("UPDATE clips SET path=? WHERE id=?", (str(out), clip["id"]))
        job.stage_done("cutting")
        job.finish()
    except Cancelled:
        job.cancelled()
    except Exception as e:  # noqa: BLE001 (recorded on the job)
        job.fail(e)
    finally:
        job.close()
