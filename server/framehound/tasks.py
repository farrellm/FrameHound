"""Worker entry points: ingest and clip run on the ``io`` queue, index on ``index``."""

from __future__ import annotations

import json
import logging
import shutil
import time

from framehound_indexer import load_lock

from . import db, evict, jobs, queue
from .colab import SESSION_PREFIX, ColabCli, ColabError
from .config import settings
from .download import download, sha256_file
from .errors import Cancelled, FHError
from .importer import import_bundle
from .index_backend import SessionLost, get_backend
from .media import cut_clip, face_crop, ffprobe, transcode_proxy, validate
from .safe_http import SafeClient
from .subtitles import clip_srt
from .torrent import download_torrent

log = logging.getLogger("framehound.tasks")

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
        # A movie whose source was evicted keeps its index: re-download, check it's
        # the same file, and return to ready without indexing again.
        redownload = bool(m["sha256"]) and has_current_index(mid)
        src = mdir / m["source_file"] if m["source_file"] else None

        if src is not None and src.exists():
            job.skip("downloading")
        else:
            job.stage("downloading", 0.0, force=True)
            evict.enforce_quota(protect={mid}, need=int(m["bytes"] or 0))
            t0 = time.monotonic()

            def eta(frac: float) -> float | None:
                return round((time.monotonic() - t0) * (1 - frac) / frac) if frac > 0.01 \
                    else None

            if m["source_kind"] == "torrent":
                def on_file(f) -> str:
                    name = f"source{f.ext}"
                    _set_movie(conn, mid, source_file=name, bytes=f.size)
                    return name

                def on_tor(frac: float, done: int, total: int, peers: int) -> None:
                    note = "Finding peers" if not total else "Waiting for peers" \
                        if not peers else f"{peers} peer{'' if peers == 1 else 's'}"
                    job.stage("downloading", frac, eta(frac), message=note)

                src = download_torrent(mdir, m["source_url"], on_file, on_tor,
                                       job.check_cancel)
            else:
                assert src is not None
                with SafeClient() as client:
                    download(client, m["source_url"], src, m["bytes"], m["source_md5"],
                             lambda frac, done, total: job.stage("downloading", frac,
                                                                 eta(frac)),
                             job.check_cancel)
            job.stage_done("downloading")

        job.stage("probing", 0.0, force=True)
        size = src.stat().st_size
        probe = ffprobe(src)
        validate(probe, size)
        sha = sha256_file(src)
        job.check_cancel(force=True)
        if redownload:
            if sha != m["sha256"]:
                src.unlink(missing_ok=True)
                raise FHError("download_failed", "The downloaded file isn't the one this "
                              "movie was indexed from, so FrameHound can't clip from it. "
                              "Remove the movie and add it again.",
                              detail=f"sha256 {sha} != {m['sha256']}")
        else:
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
        evict.enforce_quota(protect={mid})

        if redownload:
            _set_movie(conn, mid, status="ready", error=None, last_accessed_at=db.now())
            job.finish(message="re-downloaded")
            return
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


def has_current_index(movie_id: str) -> bool:
    from framehound_indexer import bundle

    p = settings().movie_dir(movie_id) / "index" / bundle.MANIFEST
    try:
        return json.loads(p.read_text()).get("models_lock_hash") == load_lock().hash
    except (OSError, ValueError):
        return False


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
    """Index a movie; a job already running with a backend handle is re-attached
    (the worker restarted while its Colab VM kept indexing)."""
    db.init()
    s = settings()
    job = jobs.Job(job_id)
    conn = job.conn
    mid = job.movie_id
    row = job.row()
    resume = row.get("state") == "running" and bool(row.get("backend_handle"))
    handle: dict | None = json.loads(row["backend_handle"]) if resume else None
    backend = get_backend(handle.get("backend", "local") if handle else None)
    failed = True
    try:
        if resume:
            if not job.resume():
                return
        elif not job.start():
            return
        else:
            job.set(gpu=backend.gpu or "local")
        m = _movie(conn, mid)
        mdir = s.movie_dir(mid)
        src = mdir / (m["source_file"] or "")
        if not src.exists():
            raise FHError("source_missing", "The movie file is gone. Re-download it to index.")
        spec = {"movie_id": mid, "media": {"path": str(src)}, "duration_s": m["duration_s"],
                "models_lock": load_lock().data,
                "source": {"kind": m["source_kind"], "url": m["source_url"],
                           "sha256": m["sha256"], "proxy": str(mdir / "proxy.mp4")}}
        attempt = int(row.get("attempt") or 0)
        if handle is None:
            handle = backend.start(job, spec, attempt)
            job.set(backend_handle=json.dumps(handle))
        retries = s.colab_retries
        while True:
            try:
                _wait_for_index(job, backend, handle)
                break
            except SessionLost as e:
                if retries <= 0:
                    raise FHError("colab_lost", "The Colab GPU went away during indexing "
                                  "too many times. Retry later.", detail=e.detail) from e
                retries -= 1
                attempt += 1
                log.warning("index %s: session lost, retrying on a fresh VM (%d)", job_id,
                            attempt)
                backend.cleanup(handle, failed=True)
                job.set(attempt=attempt, colab_session=None)
                handle = backend.start(job, spec, attempt)
                job.set(backend_handle=json.dumps(handle))

        job.stage("downloading_index", 0.0, force=True)
        bundle_path = backend.fetch(handle, mdir / "index.tar.zst")
        job.stage_done("downloading_index")
        job.stage("importing", 0.0, force=True)
        import_bundle(mid, bundle_path)
        job.stage_done("importing")
        _set_movie(conn, mid, status="ready", error=None)
        failed = False
    except Cancelled:
        if handle is not None:
            backend.cancel(handle)
        _finish_index(job, backend, handle, failed=False)
        handle = None
        job.cancelled()
        _set_movie(conn, mid, status="cancelled", error=None)
        return
    except Exception as e:  # noqa: BLE001 (recorded on the job)
        _finish_index(job, backend, handle, failed=True)
        handle = None
        err = job.fail(e)
        _set_movie(conn, mid, status="failed", error=err.message)
        return
    finally:
        if handle is not None:
            _finish_index(job, backend, handle, failed=failed)
        if not failed:
            job.finish()
        job.close()


def _finish_index(job: jobs.Job, backend, handle: dict | None, failed: bool) -> None:
    """Release the backend (stops the Colab VM) and record what it cost."""
    if handle is None:
        return
    try:
        backend.cleanup(handle, failed=failed)
    except Exception:
        log.exception("index cleanup failed")
    if handle.get("final_cu") is not None:
        job.set(compute_units_used=handle["final_cu"])


def _wait_for_index(job: jobs.Job, backend, handle: dict) -> None:
    """Poll the pipeline until it finishes, relaying its progress to the job."""
    s = settings()
    timeout = s.local_index_timeout_s if backend.name == "local" else s.index_timeout_s
    started = handle.get("started") or time.time()
    deadline = time.monotonic() + max(timeout - (time.time() - started), 60)
    last_progress = time.monotonic()
    while True:
        events, alive = backend.poll(handle)
        job.set(backend_handle=json.dumps(handle))
        for ev in events:
            last_progress = time.monotonic()
            if ev["stage"] == "done":
                return
            if ev["stage"] == "error":
                raise FHError("index_failed", "Indexing failed.",
                              detail=f"{ev.get('message')}\n{backend.log_tail(handle)}")
            job.stage(ev["stage"], ev.get("pct", 0.0), ev.get("eta_s"),
                      state="done" if ev.get("state") == "done" else "running",
                      force=ev.get("state") == "done")
        if not alive:
            # The process may have written its last lines just before exiting.
            events, _ = backend.poll(handle)
            if any(ev["stage"] == "done" for ev in events):
                return
            err = next((ev for ev in events if ev["stage"] == "error"), None)
            raise FHError("index_failed", "Indexing failed." if err else
                          "Indexing stopped unexpectedly.",
                          detail=f"{err.get('message') if err else ''}\n"
                                 f"{backend.log_tail(handle)}".strip())
        cu = backend.compute_units(handle) if hasattr(backend, "compute_units") else None
        if cu is not None:
            job.set(compute_units_used=cu)
        now = time.monotonic()
        if now > deadline:
            raise FHError("index_timeout", f"Indexing took longer than "
                          f"{timeout // 60} minutes and was stopped.")
        if now - last_progress > s.stage_watchdog_s:
            raise FHError("index_stalled", "Indexing stopped making progress and was "
                          "stopped.", detail=backend.log_tail(handle))
        job.check_cancel()
        time.sleep(backend.poll_interval_s)


def recover_index_jobs() -> list[str]:
    """After a restart: re-attach running Colab jobs whose VM is still indexing.

    Returns the job ids re-enqueued. Jobs whose VM is gone are failed now rather
    than waiting for the stale-job reaper.
    """
    db.init()
    cli = ColabCli()
    resumed = []
    with db.db() as conn:
        running = db.rows(conn, "SELECT id, movie_id, colab_session, backend_handle FROM jobs "
                          "WHERE kind='index' AND state='running' AND colab_session IS NOT NULL")
    for j in running:
        try:
            alive = cli.status(j["colab_session"])
        except ColabError as e:
            log.warning("recover %s: colab status failed: %s", j["id"], e)
            continue
        if alive and j["backend_handle"]:
            with db.db() as conn:
                conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (db.now(), j["id"]))
            queue.enqueue("index", j["id"])
            resumed.append(j["id"])
        else:
            msg = "The Colab GPU went away while the worker was restarting. Retry to index."
            with db.db() as conn:
                conn.execute("UPDATE jobs SET state='failed', error=?, error_code='colab_lost', "
                             "finished_at=?, seq=seq+1 WHERE id=?", (msg, db.now(), j["id"]))
                conn.execute("UPDATE movies SET status='failed', error=? WHERE id=? AND "
                             "status='indexing'", (msg, j["movie_id"]))
    return resumed


def reconcile_sessions(cli: ColabCli | None = None) -> list[str]:
    """Stop every ``fh-*`` Colab session that no running job owns (leak prevention).

    Sessions the CLI can't name (no local record) are only reported: they can't be
    addressed by name.
    """
    cli = cli or ColabCli()
    with db.db() as conn:
        jobs.reap_stale(conn)
        owned = {r[0] for r in conn.execute(
            "SELECT colab_session FROM jobs WHERE state='running' AND colab_session IS NOT NULL")}
    stopped = []
    for sess in cli.sessions():
        if sess.name == "?":
            log.warning("colab session %s has no local record; stop it from the Colab UI",
                        sess.endpoint)
        elif sess.name.startswith(SESSION_PREFIX) and sess.name not in owned:
            log.warning("stopping orphaned colab session %s", sess.name)
            cli.stop(sess.name)
            stopped.append(sess.name)
    return stopped


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
        crop = None
        if clip["crop_box"]:
            if not (m["width"] and m["height"]):
                raise FHError("bad_crop", "FrameHound doesn't know this movie's frame size, "
                              "so it can't crop. Cut without cropping.")
            crop = face_crop(json.loads(clip["crop_box"]), int(m["width"]), int(m["height"]))
        srt = None
        if clip["subs"]:
            srt = out.with_suffix(".srt")
            srt.write_text(clip_srt(conn, m["id"], clip["start"], clip["end"]))
        try:
            cut_clip(src, out, clip["start"], clip["end"], clip["mode"],
                     lambda f: job.stage("cutting", f), job.check_cancel, crop=crop,
                     fmt=clip["format"], srt=srt)
        finally:
            if srt is not None:
                srt.unlink(missing_ok=True)
        conn.execute("UPDATE clips SET path=? WHERE id=?", (str(out), clip["id"]))
        job.stage_done("cutting")
        job.finish()
    except Cancelled:
        job.cancelled()
    except Exception as e:  # noqa: BLE001 (recorded on the job)
        job.fail(e)
    finally:
        job.close()
