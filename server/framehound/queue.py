"""Enqueue jobs on RQ, or run them in background threads when REDIS_URL is unset (dev)."""

from __future__ import annotations

import importlib
import threading

from .config import settings

QUEUES = {"ingest": "io", "clip": "io", "index": "index"}
TASKS = {"ingest": "framehound.tasks.run_ingest", "index": "framehound.tasks.run_index",
         "clip": "framehound.tasks.run_clip"}


_inline_slots: dict[str, threading.BoundedSemaphore] = {}
_inline_guard = threading.Lock()


def _inline_slot(queue_name: str) -> threading.BoundedSemaphore:
    """As many jobs at once per queue as RQ workers on it: MAX_DOWNLOADS for io."""
    with _inline_guard:
        if queue_name not in _inline_slots:
            n = settings().max_downloads if queue_name == "io" else 1
            _inline_slots[queue_name] = threading.BoundedSemaphore(n)
        return _inline_slots[queue_name]


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
    slot = _inline_slot(QUEUES[kind])

    def run() -> None:
        with slot:
            func(job_id)

    threading.Thread(target=run, name=f"fh-{kind}-{job_id}", daemon=True).start()


def requeue_lost(jobs: list[dict]) -> list[str]:
    """Enqueue each ``queued`` job (``id``, ``kind``) that RQ no longer holds.

    A Redis restart can drop its queues while the db still says ``queued``.
    Enqueueing a job twice is harmless: ``Job.start`` runs it only once.
    """
    from redis import Redis
    from rq.exceptions import NoSuchJobError
    from rq.job import Job, JobStatus

    conn = Redis.from_url(settings().redis_url)
    live = {JobStatus.QUEUED, JobStatus.STARTED, JobStatus.SCHEDULED, JobStatus.DEFERRED}
    lost = []
    for j in jobs:
        try:
            if Job.fetch(f"fh-{j['id']}", connection=conn).get_status() in live:
                continue
        except NoSuchJobError:
            pass
        enqueue(j["kind"], j["id"])
        lost.append(j["id"])
    return lost


def redis_ok() -> bool | None:
    s = settings()
    if not s.redis_url:
        return None
    try:
        from redis import Redis

        return bool(Redis.from_url(s.redis_url, socket_timeout=2).ping())
    except Exception:  # noqa: BLE001 (health check reports any failure)
        return False
