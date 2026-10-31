"""ColabIndexBackend against a fake ``colab`` executable on PATH (design.md §11)."""

import json
import os
import sys
from pathlib import Path

import pytest
from framehound import colab, config, db, index_backend, jobs, queue, tasks
from test_pipeline_flow import add_local_movie

FAKE_COLAB = Path(__file__).with_name("fake_colab.py")
FAKE_INDEXER = [sys.executable, str(Path(__file__).with_name("fake_indexer.py"))]


@pytest.fixture
def fake(tmp_path, monkeypatch, data_dir):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    wrapper = bindir / "colab"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_COLAB}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_COLAB_DIR", str(tmp_path / "colab"))
    monkeypatch.setenv("INDEX_BACKEND", "colab")
    monkeypatch.setenv("COLAB_POLL_S", "0.05")
    config.reset_settings()
    enq: list[tuple[str, str]] = []
    monkeypatch.setattr(queue, "enqueue", lambda kind, job_id: enq.append((kind, job_id)))

    def backend(name=None):
        b = index_backend.ColabIndexBackend(command=FAKE_INDEXER)
        b.usage_every_s = 0
        return b

    monkeypatch.setattr(tasks, "get_backend", backend)

    class Fake:
        dir = tmp_path / "colab"
        enqueued = enq

        def calls(self, cmd: str | None = None) -> list[list[str]]:
            path = self.dir / "calls.jsonl"
            out = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
            return [c for c in out if cmd is None or c[0] == cmd]

        def state(self) -> dict:
            return json.loads((self.dir / "state.json").read_text())

    return Fake()


def ingest(data_dir, mid: str) -> str:
    tasks.run_ingest(add_local_movie(data_dir, mid))
    with db.db() as conn:
        return db.row(conn, "SELECT id FROM jobs WHERE movie_id=? AND kind='index'", mid)["id"]


def movie(mid: str) -> dict:
    with db.db() as conn:
        return db.row(conn, "SELECT * FROM movies WHERE id=?", mid)


def job_row(job_id: str) -> dict:
    with db.db() as conn:
        return db.row(conn, "SELECT * FROM jobs WHERE id=?", job_id)


def test_success_stops_the_session(data_dir, fake):
    job_id = ingest(data_dir, "c1")
    tasks.run_index(job_id)
    m = movie("c1")
    assert m["status"] == "ready", m["error"]
    j = job_row(job_id)
    assert j["colab_session"] == "fh-c1-0" and j["gpu"] == "L4"
    assert j["compute_units_used"] == pytest.approx(0.4)
    assert fake.state()["sessions"] == {}  # stopped
    assert [c[2] for c in fake.calls("new")] == ["fh-c1-0"]
    with db.db() as conn:
        stages = {r["stage"]: r["state"] for r in db.rows(
            conn, "SELECT stage, state FROM job_stages WHERE job_id=?", job_id)}
        assert db.rows(conn, "SELECT idx FROM shots WHERE movie_id='c1'")
    assert stages["provisioning_gpu"] == "done" and stages["visual"] == "done"
    assert (data_dir / "movies/c1/index/thumbs/1.jpg").exists()


def test_gpu_fallback_on_400(data_dir, fake, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_REJECT", "L4")
    job_id = ingest(data_dir, "c2")
    tasks.run_index(job_id)
    assert movie("c2")["status"] == "ready", movie("c2")["error"]
    assert job_row(job_id)["gpu"] == "T4"
    assert [c[c.index("--gpu") + 1] for c in fake.calls("new")] == ["L4", "T4"]


def test_no_gpu_at_all(data_dir, fake, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_REJECT", "L4,T4")
    job_id = ingest(data_dir, "c3")
    tasks.run_index(job_id)
    j = job_row(job_id)
    assert j["state"] == "failed" and j["error_code"] == "colab_no_gpu"
    assert "L4 or T4" in j["error"]


def test_auth_broken_fails_the_job_with_its_code(data_dir, fake, monkeypatch):
    job_id = ingest(data_dir, "c9")
    monkeypatch.setenv("FAKE_COLAB_AUTH_BROKEN", "1")
    tasks.run_index(job_id)
    j = job_row(job_id)
    assert j["state"] == "failed" and j["error_code"] == "colab_auth"
    assert "sign-in has expired" in j["error"]


def test_budget_refuses_before_provisioning(data_dir, fake, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_BALANCE", "1")
    job_id = ingest(data_dir, "c4")
    tasks.run_index(job_id)
    assert job_row(job_id)["error_code"] == "colab_budget"
    assert fake.calls("new") == []


def test_stalled_progress_trips_watchdog(data_dir, fake, monkeypatch):
    monkeypatch.setenv("FH_FAKE_STALL", "1")
    monkeypatch.setenv("STAGE_WATCHDOG", "1")
    config.reset_settings()
    job_id = ingest(data_dir, "c5")
    tasks.run_index(job_id)
    assert job_row(job_id)["error_code"] == "index_stalled"
    assert fake.state()["sessions"] == {}
    assert (data_dir / "logs" / "fh-c5-0.jsonl").exists()  # colab log saved on failure


def test_vanished_session_retries_on_a_fresh_vm(data_dir, fake, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_VANISH", "fh-c6-0")
    job_id = ingest(data_dir, "c6")
    tasks.run_index(job_id)
    assert movie("c6")["status"] == "ready", movie("c6")["error"]
    j = job_row(job_id)
    assert j["colab_session"] == "fh-c6-1" and j["attempt"] == 1
    assert [c[2] for c in fake.calls("new")] == ["fh-c6-0", "fh-c6-1"]


def test_reconciler_stops_orphans(data_dir, fake):
    cli = colab.ColabCli()
    cli.new("fh-orphan-0", "T4")
    cli.new("someone-else", "T4")
    assert tasks.reconcile_sessions(cli) == ["fh-orphan-0"]
    assert list(fake.state()["sessions"]) == ["someone-else"]


def test_worker_restart_reattaches(data_dir, fake):
    job_id = ingest(data_dir, "c7")
    # The first worker provisions and launches, then dies.
    job = jobs.Job(job_id)
    assert job.start()
    with db.db() as conn:
        m = db.row(conn, "SELECT * FROM movies WHERE id='c7'")
    backend = tasks.get_backend()
    spec = {"movie_id": "c7", "media": {"path": str(data_dir / "movies/c7/source.mp4")},
            "duration_s": m["duration_s"], "models_lock": __import__(
                "framehound_indexer").load_lock().data, "source": {"kind": "upload"}}
    handle = backend.start(job, spec)
    job.set(backend_handle=json.dumps(handle))
    job.close()
    # A new worker comes up: recovery re-enqueues it, and running it re-attaches.
    assert tasks.recover_index_jobs() == [job_id]
    assert fake.enqueued[-1] == ("index", job_id)
    tasks.run_index(job_id)
    assert movie("c7")["status"] == "ready", movie("c7")["error"]
    assert len(fake.calls("new")) == 1
    assert fake.state()["sessions"] == {}


def test_proxy_is_uploaded_in_parts(data_dir, fake, monkeypatch):
    monkeypatch.setattr(index_backend, "CHUNK", 4096)
    job_id = ingest(data_dir, "c8")
    with db.db() as conn:
        conn.execute("UPDATE movies SET source_kind='torrent' WHERE id='c8'")
    tasks.run_index(job_id)
    assert movie("c8")["status"] == "ready", movie("c8")["error"]
    ups = [c for c in fake.calls("upload") if "upload.mp4" in c[-1]]
    assert len(ups) > 1 and all(".part" in c[-1] for c in ups)


def test_auth_broken(data_dir, fake, monkeypatch):
    monkeypatch.setenv("FAKE_COLAB_AUTH_BROKEN", "1")
    with pytest.raises(colab.ColabError) as e:
        colab.ColabCli().usage()
    assert e.value.code == "colab_auth"
    from fastapi.testclient import TestClient
    from framehound.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["colab"]["state"] == "auth_broken"

