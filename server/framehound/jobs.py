"""Job rows and their per-stage progress, written by workers and read by the api.

Every write bumps ``jobs.seq``; the SSE endpoint watches it. A daemon thread
heartbeats while a job runs, so the api can tell a job whose worker died.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import traceback

from . import db
from .config import settings
from .errors import Cancelled, FHError

INGEST_STAGES = ["downloading", "probing", "transcoding_proxy"]
TERMINAL = {"done", "failed", "cancelled"}

log = logging.getLogger("framehound.jobs")


def create(conn, movie_id: str, kind: str, stages: list[str], attempt: int = 0) -> str:
    job_id = db.new_id()
    conn.execute(
        "INSERT INTO jobs(id, movie_id, kind, state, stage, pct, attempt, created_at, seq) "
        "VALUES (?,?,?,?,?,?,?,?,1)",
        (job_id, movie_id, kind, "queued", stages[0] if stages else None, 0.0, attempt, db.now()),
    )
    conn.executemany(
        "INSERT INTO job_stages(job_id, stage, ord) VALUES (?,?,?)",
        [(job_id, st, i) for i, st in enumerate(stages)],
    )
    return job_id


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.conn = db.connect()
        r = db.row(self.conn, "SELECT * FROM jobs WHERE id=?", job_id)
        if r is None:
            raise KeyError(job_id)
        self.movie_id: str = r["movie_id"]
        self.kind: str = r["kind"]
        self._last_write: dict[str, float] = {}
        self._last_cancel_check = 0.0
        self._stop = threading.Event()
        self._hb: threading.Thread | None = None

    def row(self) -> dict:
        return db.row(self.conn, "SELECT * FROM jobs WHERE id=?", self.id) or {}

    def start(self) -> bool:
        """Mark running; False if the job was cancelled (or taken) before a worker got it."""
        cur = self.conn.execute(
            "UPDATE jobs SET state='running', started_at=?, heartbeat_at=?, seq=seq+1 "
            "WHERE id=? AND state='queued' AND cancel_requested=0",
            (db.now(), db.now(), self.id),
        )
        if cur.rowcount == 0:
            return False
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()
        return True

    def resume(self) -> bool:
        """Re-claim a job left running by a worker that restarted (Colab re-attach)."""
        cur = self.conn.execute(
            "UPDATE jobs SET heartbeat_at=?, seq=seq+1 WHERE id=? AND state='running' "
            "AND cancel_requested=0", (db.now(), self.id))
        if cur.rowcount == 0:
            return False
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()
        return True

    def _heartbeat(self) -> None:
        conn = db.connect()
        try:
            while not self._stop.wait(15):
                # A busy disk can hold the lock past busy_timeout; try again next beat
                # rather than let the thread die and the job look lost.
                try:
                    conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?",
                                 (db.now(), self.id))
                except sqlite3.OperationalError as e:
                    log.warning("job %s heartbeat skipped: %s", self.id, e)
        finally:
            conn.close()

    def stage(self, stage: str, pct: float = 0.0, eta_s: float | None = None,
              state: str = "running", message: str | None = None, force: bool = False) -> None:
        """Record progress; non-forced running updates are throttled to 2 per second."""
        t = time.monotonic()
        if not force and state == "running" and t - self._last_write.get(stage, 0) < 0.5:
            return
        self._last_write[stage] = t
        pct = max(0.0, min(1.0, float(pct)))
        try:
            self._write_stage(stage, pct, eta_s, state, message)
        except sqlite3.OperationalError as e:
            # Losing one progress tick beats failing an hours-long download over it.
            if force or state != "running":
                raise
            log.warning("job %s progress skipped: %s", self.id, e)

    def _write_stage(self, stage: str, pct: float, eta_s: float | None, state: str,
                     message: str | None) -> None:
        with db.tx(self.conn):
            self.conn.execute(
                "INSERT INTO job_stages(job_id, stage, ord, state, pct, eta_s) "
                "VALUES (?,?,(SELECT COALESCE(MAX(ord),0)+1 FROM job_stages WHERE job_id=?),?,?,?) "
                "ON CONFLICT(job_id, stage) DO UPDATE SET state=excluded.state, pct=excluded.pct, "
                "eta_s=excluded.eta_s",
                (self.id, stage, self.id, state, pct, eta_s),
            )
            self.conn.execute(
                # A stage's note lasts until it gives a new one or another stage reports.
                "UPDATE jobs SET message=CASE WHEN ? IS NOT NULL THEN ? WHEN stage=? "
                "THEN message END, stage=?, pct=?, eta_s=?, heartbeat_at=?, seq=seq+1 "
                "WHERE id=?",
                (message, message, stage, stage, pct, eta_s, db.now(), self.id),
            )

    def stage_done(self, stage: str) -> None:
        self.stage(stage, 1.0, 0, state="done", force=True)

    def skip(self, stage: str) -> None:
        self.conn.execute("UPDATE job_stages SET state='skipped' WHERE job_id=? AND stage=?",
                          (self.id, stage))

    def set(self, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE jobs SET {cols}, seq=seq+1 WHERE id=?",
                          (*fields.values(), self.id))

    def check_cancel(self, force: bool = False) -> None:
        t = time.monotonic()
        if not force and t - self._last_cancel_check < 1.0:
            return
        self._last_cancel_check = t
        r = self.conn.execute("SELECT cancel_requested FROM jobs WHERE id=?", (self.id,)).fetchone()
        if r and r[0]:
            raise Cancelled()

    def _end(self, state: str, **fields) -> None:
        self._stop.set()
        with db.tx(self.conn):
            self.conn.execute(
                "UPDATE job_stages SET state=? WHERE job_id=? AND state='running'",
                ("failed" if state == "failed" else "pending" if state == "cancelled" else "done",
                 self.id),
            )
            cols = "".join(f", {k}=?" for k in fields)
            self.conn.execute(
                f"UPDATE jobs SET state=?, finished_at=?, seq=seq+1{cols} WHERE id=?",
                (state, db.now(), *fields.values(), self.id),
            )

    def finish(self, message: str | None = None) -> None:
        self._end("done", pct=1.0, message=message)

    def cancelled(self) -> None:
        self._end("cancelled", message="Cancelled")

    def fail(self, err: BaseException) -> FHError:
        if not isinstance(err, FHError):
            err = FHError("internal", "Something went wrong on the server.",
                          detail="".join(traceback.format_exception(err))[-2000:])
        self._end("failed", error=err.message, error_code=err.code, message=err.detail)
        return err

    def close(self) -> None:
        self._stop.set()
        self.conn.close()


def reap_stale(conn) -> None:
    """Fail running jobs whose worker stopped heartbeating."""
    cutoff = db.now() - settings().stale_job_s
    stale = db.rows(conn, "SELECT id, movie_id, kind FROM jobs WHERE state='running' "
                    "AND COALESCE(heartbeat_at, started_at, 0) < ?", cutoff)
    for j in stale:
        msg = "The worker stopped while this job was running. Retry to start it again."
        conn.execute("UPDATE jobs SET state='failed', error=?, error_code='worker_lost', "
                     "finished_at=?, seq=seq+1 WHERE id=?", (msg, db.now(), j["id"]))
        conn.execute("UPDATE job_stages SET state='failed' WHERE job_id=? AND state='running'",
                     (j["id"],))
        if j["kind"] in ("ingest", "index"):
            conn.execute("UPDATE movies SET status='failed', error=? WHERE id=? "
                         "AND status IN ('ingesting','indexing')", (msg, j["movie_id"]))
