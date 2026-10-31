"""``framehound-worker QUEUE``: an RQ worker, plus Colab housekeeping for ``index``.

``io`` alone runs a pool of ``MAX_DOWNLOADS`` workers, so that many downloads
(and proxies and clips) run at once. The pool respawns a worker that dies.

Before taking jobs, the index worker re-attaches to Colab jobs whose VM kept
indexing while the worker was down, and it runs the session reconciler at
startup and every ``RECONCILE_S`` (design.md §5, Robustness).
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from .config import settings

log = logging.getLogger("framehound.worker")


def colab_housekeeping(forever: bool = True) -> None:
    from . import tasks

    try:
        resumed = tasks.recover_index_jobs()
        if resumed:
            log.info("re-attached index jobs %s", resumed)
    except Exception:
        log.exception("index job recovery failed")
    while True:
        try:
            stopped = tasks.reconcile_sessions()
            if stopped:
                log.warning("stopped orphaned colab sessions %s", stopped)
        except Exception:
            log.exception("colab reconcile failed")
        if not forever:
            return
        time.sleep(settings().reconcile_s)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("colabkit").setLevel(logging.DEBUG)  # every colab CLI call, timed
    queues = argv or ["io"]
    s = settings()
    if "index" in queues and s.index_backend == "colab":
        threading.Thread(target=colab_housekeeping, daemon=True, name="colab").start()
    from redis import Redis
    from rq import Queue, Worker
    from rq.worker_pool import WorkerPool

    conn = Redis.from_url(s.redis_url or "redis://redis:6379/0")
    if queues == ["io"] and s.max_downloads > 1:
        WorkerPool(queues, connection=conn, num_workers=s.max_downloads).start()
        return
    Worker([Queue(q, connection=conn) for q in queues], connection=conn).work()
