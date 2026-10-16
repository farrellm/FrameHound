"""Enqueue jobs on RQ, or run them in background threads when REDIS_URL is unset (dev)."""

from __future__ import annotations

import importlib
import threading

from .config import settings

QUEUES = {"ingest": "io", "clip": "io", "index": "index"}
TASKS = {"ingest": "framehound.tasks.run_ingest", "index": "framehound.tasks.run_index",
         "clip": "framehound.tasks.run_clip"}


_inline_locks = {name: threading.Lock() for name in set(QUEUES.values())}


def _timeout(kind: str) -> int:
    s = settings()
    if kind == "index":
        return max(s.index_timeout_s, s.local_index_timeout_s) + 900
    return 6 * 3600 if kind == "ingest" else 3600


def enqueue(kind: str, job_id: str) -> None:
    s = settings()
    if s.redis_url:
        from redis import Redis
        from rq import Queue

        q = Queue(QUEUES[kind], connection=Redis.from_url(s.redis_url))
        q.enqueue(TASKS[kind], job_id, job_id=f"fh-{job_id}", job_timeout=_timeout(kind),
                  result_ttl=3600, failure_ttl=7 * 86400)
        return
    mod, _, fn = TASKS[kind].rpartition(".")
    func = getattr(importlib.import_module(mod), fn)
    lock = _inline_locks[QUEUES[kind]]

    def run() -> None:
        with lock:  # one job per queue at a time, like one RQ worker per queue
            func(job_id)

    threading.Thread(target=run, name=f"fh-{kind}-{job_id}", daemon=True).start()


def redis_ok() -> bool | None:
    s = settings()
    if not s.redis_url:
        return None
    try:
        from redis import Redis

        return bool(Redis.from_url(s.redis_url, socket_timeout=2).ping())
    except Exception:  # noqa: BLE001 (health check reports any failure)
        return False
