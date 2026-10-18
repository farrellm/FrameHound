import sqlite3

import pytest
from framehound import db, jobs


def _job() -> jobs.Job:
    with db.db() as conn:
        job_id = jobs.create(conn, "m1", "ingest", jobs.INGEST_STAGES)
    job = jobs.Job(job_id)
    job.conn.execute("PRAGMA busy_timeout=50")
    return job


def test_progress_survives_a_locked_db():
    job = _job()
    other = db.connect()
    other.execute("BEGIN IMMEDIATE")
    try:
        job.stage("downloading", 0.3, message="4 peers")  # skipped, not raised
        with pytest.raises(sqlite3.OperationalError):
            job.stage("downloading", 0.3, force=True)
    finally:
        other.execute("ROLLBACK")
        other.close()
    job.stage("downloading", 0.5, force=True)
    assert job.row()["pct"] == 0.5
    job.close()
