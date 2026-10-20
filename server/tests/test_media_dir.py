"""MEDIA_DIR: movie files on their own disk, the index (and db) in DATA_DIR."""

import pytest
from framehound import config, db, evict, tasks
from test_face_search import ready_movie
from test_pipeline_flow import client, enqueued  # noqa: F401


@pytest.fixture
def media_dir(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    config.reset_settings()
    return tmp_path / "media"


def test_files_and_index_land_apart(data_dir, media_dir, enqueued, client):  # noqa: F811
    ready_movie(data_dir, enqueued)
    assert (media_dir / "movies/mv1/source.mp4").exists()
    assert (media_dir / "movies/mv1/proxy.mp4").exists()
    assert not (media_dir / "movies/mv1/index").exists()
    assert (data_dir / "movies/mv1/index").is_dir()
    assert {p.name for p in (data_dir / "movies/mv1").iterdir()} == {"index"}

    movie = client.get("/movies/mv1").json()
    assert movie["status"] == "ready" and movie["has_source"] and movie["has_proxy"]
    assert client.get("/media/mv1/proxy.mp4").status_code == 200
    hit = client.post("/movies/mv1/search", json={"type": "dialogue", "q": "hello"}).json()
    assert hit["candidates"]


def test_clips_cut_into_the_media_dir(data_dir, media_dir, enqueued, client):  # noqa: F811
    ready_movie(data_dir, enqueued)
    r = client.post("/movies/mv1/clips", json={"start": 0.5, "end": 1.5})
    assert r.status_code == 201, r.json()
    tasks.run_clip(r.json()["job_id"])
    with db.db() as conn:
        path = db.row(conn, "SELECT path FROM clips WHERE id=?", r.json()["clip_id"])["path"]
    assert path.startswith(str(media_dir / "movies/mv1/clips"))


def test_delete_removes_both(data_dir, media_dir, enqueued, client):  # noqa: F811
    ready_movie(data_dir, enqueued)
    assert client.delete("/movies/mv1").status_code == 204
    assert not (media_dir / "movies/mv1").exists() and not (data_dir / "movies/mv1").exists()


def test_eviction_counts_and_clears_both(data_dir, media_dir, enqueued,  # noqa: F811
                                        monkeypatch):
    ready_movie(data_dir, enqueued)
    files = evict.tree_bytes(media_dir / "movies")
    index = evict.tree_bytes(data_dir / "movies")
    assert files and index and evict.usage() == files + index
    (media_dir / "movies/mv1/source.mp4").unlink()  # past sources: the whole movie goes
    monkeypatch.setenv("MAX_DATA_GB", repr(1 / 1024**3))
    config.reset_settings()
    assert evict.enforce_quota() == ["all:mv1"]
    assert evict.usage() == 0
