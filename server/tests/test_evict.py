"""Disk quota: LRU eviction (clips, then sources, then whole movies) and Re-download."""

import shutil

import pytest
from framehound import config, db, evict, tasks
from test_face_search import ready_movie
from test_pipeline_flow import add_local_movie, client, enqueued, make_video  # noqa: F401


@pytest.fixture
def two_movies(data_dir, enqueued):  # noqa: F811
    ready_movie(data_dir, enqueued, "a1")
    job = add_local_movie(data_dir, "b2")
    make_video(data_dir / "movies/b2/source.mp4", seconds=5)  # other bytes: no dedup
    tasks.run_ingest(job)
    tasks.run_index(enqueued[-1][1])
    with db.db() as conn:  # a1 was used longest ago
        conn.execute("UPDATE movies SET last_accessed_at=100 WHERE id='a1'")
        conn.execute("UPDATE movies SET last_accessed_at=200 WHERE id='b2'")
    return data_dir


def quota(monkeypatch, nbytes: int) -> None:
    monkeypatch.setenv("MAX_DATA_GB", repr(nbytes / 1024**3))
    config.reset_settings()


def size(p):
    return p.stat().st_size


def test_no_quota_does_nothing(two_movies):
    assert evict.enforce_quota() == []


def test_clips_go_first(two_movies, monkeypatch):
    clip = two_movies / "movies/a1/clips/c1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"c" * 50_000)
    with db.db() as conn:
        conn.execute("INSERT INTO clips(id, movie_id, start, \"end\", path, created_at) "
                     "VALUES ('c1','a1',0,1,?,?)", (str(clip), db.now()))
    quota(monkeypatch, evict.usage() - 10_000)
    assert evict.enforce_quota() == ["clips:a1"]
    assert not clip.exists() and (two_movies / "movies/a1/source.mp4").exists()


def test_sources_before_indexes_in_lru_order(two_movies, monkeypatch, client):  # noqa: F811
    a_src = size(two_movies / "movies/a1/source.mp4")
    quota(monkeypatch, evict.usage() - a_src - 10)
    assert evict.enforce_quota() == ["source:a1", "source:b2"]
    movie = client.get("/movies/a1").json()
    assert movie["status"] == "ready" and not movie["has_source"] and movie["has_proxy"]
    assert client.post("/movies/a1/clips", json={"start": 0, "end": 1}).json()["code"] \
        == "source_missing"


def test_whole_movie_last_and_busy_ones_skipped(two_movies, monkeypatch, client):  # noqa: F811
    (two_movies / "movies/a1/source.torrent").write_bytes(b"d4:infoe")
    with db.db() as conn:
        conn.execute("INSERT INTO jobs(id, movie_id, kind, state, created_at) "
                     "VALUES ('j','b2','clip','running',?)", (db.now(),))
    quota(monkeypatch, 1)
    assert evict.enforce_quota() == ["source:a1", "all:a1"]
    a1 = client.get("/movies/a1").json()
    assert a1["status"] == "evicted" and not a1["has_proxy"]
    assert sorted(p.name for p in (two_movies / "movies/a1").iterdir()) == ["source.torrent"]
    with db.db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM shots WHERE movie_id='a1'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM shots WHERE movie_id='b2'").fetchone()[0] > 0
    # Protected movies are skipped too.
    assert evict.enforce_quota(protect={"b2"}) == []


def _fake_download(monkeypatch, src):
    def fake(http, url, dest, size, md5, on_progress, check_cancel):
        shutil.copyfile(src, dest)
        return dest.stat().st_size

    monkeypatch.setattr(tasks, "download", fake)


def test_redownload_skips_indexing(two_movies, monkeypatch, enqueued, client):  # noqa: F811
    keep = two_movies / "keep.mp4"
    shutil.copyfile(two_movies / "movies/a1/source.mp4", keep)
    (two_movies / "movies/a1/source.mp4").unlink()
    r = client.post("/movies/a1/retry")
    assert r.status_code == 202 and r.json()["kind"] == "ingest"
    # The kept index's stages show as done while the file comes back.
    snap = client.get("/movies/a1/index-progress").json()
    assert {st["state"] for st in snap["stages"] if st["job"] == "index"} == {"done"}
    _fake_download(monkeypatch, keep)
    n = len(enqueued)
    tasks.run_ingest(r.json()["job_id"])
    movie = client.get("/movies/a1").json()
    assert movie["status"] == "ready" and movie["has_source"], movie["error"]
    assert len(enqueued) == n  # no index job


def test_redownload_of_a_different_file_fails(two_movies, monkeypatch, client):  # noqa: F811
    other = two_movies / "other.mp4"
    make_video(other, seconds=3)
    (two_movies / "movies/a1/source.mp4").unlink()
    job = client.post("/movies/a1/retry").json()["job_id"]
    _fake_download(monkeypatch, other)
    tasks.run_ingest(job)
    movie = client.get("/movies/a1").json()
    assert movie["status"] == "failed" and "isn't the one" in movie["error"]
    assert not (two_movies / "movies/a1/source.mp4").exists()
