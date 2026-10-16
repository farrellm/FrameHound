"""Ingest → index (fake indexer) → import → search → clip, through the real job code."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from framehound import db, index_backend, queue, tasks
from framehound.media import ffprobe

FAKE = [sys.executable, str(Path(__file__).with_name("fake_indexer.py"))]


@pytest.fixture
def enqueued(monkeypatch):
    calls = []
    monkeypatch.setattr(queue, "enqueue", lambda kind, job_id: calls.append((kind, job_id)))
    monkeypatch.setattr(tasks, "get_backend", lambda name=None: _fast_backend())
    return calls


def _fast_backend():
    b = index_backend.LocalIndexBackend(FAKE)
    b.poll_interval_s = 0.05
    return b


@pytest.fixture
def client():
    from framehound.api import app

    with TestClient(app) as c:
        yield c


def make_video(path: Path, seconds: int = 4) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size=320x240:rate=25:duration={seconds}", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(path)], check=True)


def add_local_movie(data_dir: Path, mid: str = "mv1") -> str:
    mdir = data_dir / "movies" / mid
    mdir.mkdir(parents=True)
    make_video(mdir / "source.mp4")
    with db.db() as conn:
        conn.execute("INSERT INTO movies(id, title, source_kind, source_ref, source_file, status, "
                     "created_at) VALUES (?,?,?,?,?,?,?)",
                     (mid, "Test Movie", "http", f"https://example.com/{mid}.mp4", "source.mp4",
                      "ingesting", db.now()))
        job_id = tasks.start_ingest(conn, mid)
    return job_id


def test_full_flow(data_dir, enqueued, client, monkeypatch):
    ingest_job = add_local_movie(data_dir)
    tasks.run_ingest(ingest_job)
    with db.db() as conn:
        m = db.row(conn, "SELECT * FROM movies WHERE id='mv1'")
        st = db.rows(conn, "SELECT stage, state FROM job_stages WHERE job_id=? ORDER BY ord",
                     ingest_job)
    assert m["status"] == "indexing", m["error"]
    assert m["duration_s"] == pytest.approx(4.0, abs=0.1) and m["sha256"]
    assert [s["state"] for s in st] == ["skipped", "done", "done"]
    assert (data_dir / "movies/mv1/proxy.mp4").exists()
    assert enqueued[-1][0] == "index"

    # The snapshot mixes ingest and index stages in order.
    snap = client.get("/movies/mv1/index-progress").json()
    assert [s["stage"] for s in snap["stages"]][:4] == [
        "downloading", "probing", "transcoding_proxy", "staging_media"]

    tasks.run_index(enqueued[-1][1])
    movie = client.get("/movies/mv1").json()
    assert movie["status"] == "ready", movie["error"]
    assert movie["capabilities"] == {"dialogue": True, "visual": True, "faces": True}
    assert movie["thumbs"] == ["/media/mv1/thumbs/0.jpg", "/media/mv1/thumbs/1.jpg"]
    assert movie["models_ok"] is True
    assert not (data_dir / "work" / "mv1").exists()  # cleaned up

    r = client.post("/movies/mv1/search", json={"type": "dialogue", "q": "never come back",
                                                 "heat_bins": 64})
    assert r.status_code == 200, r.text
    top = r.json()["candidates"][0]
    assert [w["word"] for w in top["matched_words"]] == ["never", "come", "back"]
    assert len(r.json()["heat"]) == 64

    # Text search: the fake index's frame 5 (t=2.5 s, shot 1) is e_5.
    from framehound_indexer import encoders

    monkeypatch.setattr(encoders, "encode_text",
                        lambda q: np.eye(16, dtype=np.float32)[5])
    r = client.post("/movies/mv1/search", json={"type": "text", "q": "a red hat",
                                                 "heat_bins": 64})
    assert r.status_code == 200, r.text
    top = r.json()["candidates"][0]
    assert top["reason"] == "visual" and top["peak_t"] == 2.5
    assert top["start"] <= 2.0 <= top["end"] and top["thumbs"] == ["/media/mv1/thumbs/1.jpg"]
    assert client.get("/media/mv1/thumbs/1.jpg").status_code == 200
    shots = client.get("/movies/mv1/shots", params={"from": 2.5, "to": 9}).json()
    assert [s["idx"] for s in shots] == [1]

    r = client.post("/movies/mv1/clips", json={"start": 1.0, "end": 3.0})
    assert r.status_code == 201, r.text
    clip_id, job_id = r.json()["clip_id"], r.json()["job_id"]
    tasks.run_clip(job_id)
    meta = client.get(f"/clips/{clip_id}").json()
    assert meta["ready"]
    f = client.get(f"/clips/{clip_id}/file")
    assert f.status_code == 200 and f.headers["content-type"] == "video/mp4"
    out = data_dir / "movies/mv1/clips" / f"{clip_id}.mp4"
    assert ffprobe(out).duration_s == pytest.approx(2.0, abs=1 / 25 + 0.03)

    rng = client.get("/media/mv1/proxy.mp4", headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100

    ev = client.get(f"/jobs/{job_id}/events")
    assert "event: progress" in ev.text and '"state": "done"' in ev.text


def test_index_failure_is_recorded(data_dir, enqueued, client, monkeypatch):
    ingest_job = add_local_movie(data_dir, "mv2")
    tasks.run_ingest(ingest_job)
    index_job = enqueued[-1][1]
    orig = index_backend.LocalIndexBackend.start

    def start_failing(self, job, spec, attempt=0):
        return orig(self, job, {**spec, "fail": True}, attempt)

    monkeypatch.setattr(index_backend.LocalIndexBackend, "start", start_failing)
    tasks.run_index(index_job)
    movie = client.get("/movies/mv2").json()
    assert movie["status"] == "failed" and movie["error"] == "Indexing failed."
    assert movie["job"]["detail"] and "boom" in movie["job"]["detail"]

    r = client.post("/movies/mv2/retry")
    assert r.status_code == 202 and r.json()["kind"] == "index"


def test_dedup_merges_into_ready_movie(data_dir, enqueued, client):
    tasks.run_ingest(add_local_movie(data_dir, "a1"))
    tasks.run_index(enqueued[-1][1])
    # Same bytes under a different URL.
    mdir = data_dir / "movies" / "b2"
    mdir.mkdir(parents=True)
    (mdir / "source.mp4").write_bytes((data_dir / "movies/a1/source.mp4").read_bytes())
    with db.db() as conn:
        conn.execute("INSERT INTO movies(id, title, source_kind, source_ref, source_file, status, "
                     "created_at) VALUES ('b2','Copy','http','https://mirror/x.mp4','source.mp4',"
                     "'ingesting',?)", (db.now(),))
        job = tasks.start_ingest(conn, "b2")
    tasks.run_ingest(job)
    b2 = client.get("/movies/b2").json()
    assert b2["status"] == "merged" and b2["merged_into"] == "a1"
    assert [m["id"] for m in client.get("/movies").json()["movies"]] == ["a1"]


def test_cancel_queued_job(data_dir, enqueued, client):
    job = add_local_movie(data_dir, "c3")
    assert client.post(f"/jobs/{job}/cancel").status_code == 202
    tasks.run_ingest(job)  # worker picks it up late and must skip it
    assert client.get("/movies/c3").json()["status"] == "cancelled"
    assert not (data_dir / "movies/c3/proxy.mp4").exists()


def test_clip_validation(data_dir, client):
    add_local_movie(data_dir, "d4")
    with db.db() as conn:
        conn.execute("UPDATE movies SET duration_s=4 WHERE id='d4'")
    assert client.post("/movies/d4/clips", json={"start": 0, "end": 200}).json()["code"] == "too_long"
    assert client.post("/movies/d4/clips", json={"start": 2, "end": 2}).json()["code"] == "bad_span"
    assert client.post("/movies/d4/clips", json={"start": 0, "end": 2, "format": "gif"}
                       ).json()["code"] == "unsupported_format"
