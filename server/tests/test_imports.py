"""The import folder: listing, path checks, adding a file, ingest by copy, no eviction."""

import os

import pytest
from framehound import config, db, evict, imports, tasks
from framehound.errors import FHError
from test_pipeline_flow import client, enqueued, make_video  # noqa: F401


@pytest.fixture
def import_dir(tmp_path, monkeypatch):
    d = tmp_path / "import"
    (d / "Noir").mkdir(parents=True)
    monkeypatch.setenv("IMPORT_DIR", str(d))
    config.reset_settings()
    return d


def test_listing_keeps_video_files_only(import_dir, tmp_path):
    (import_dir / "Noir/Detour (1945).mkv").write_bytes(b"x" * 10)
    (import_dir / "notes.txt").write_text("hi")
    (import_dir / ".hidden.mp4").write_bytes(b"x")
    (import_dir / ".trash").mkdir()
    (import_dir / ".trash/old.mp4").write_bytes(b"x")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    (import_dir / "link.mp4").symlink_to(outside)
    files = imports.list_files()
    assert [(f["path"], f["name"], f["bytes"]) for f in files] == [
        ("Noir/Detour (1945).mkv", "Detour (1945).mkv", 10)]


@pytest.mark.parametrize("rel", ["../outside.mp4", "/etc/passwd", "", "Noir", "notes.txt",
                                 "link.mp4", "missing.mp4"])
def test_resolve_refuses_paths_outside_or_not_video(import_dir, tmp_path, rel):
    (tmp_path / "outside.mp4").write_bytes(b"x")
    (import_dir / "link.mp4").symlink_to(tmp_path / "outside.mp4")
    (import_dir / "notes.txt").write_text("hi")
    with pytest.raises(FHError) as e:
        imports.resolve(rel)
    assert e.value.code == "not_found"


def test_off_without_import_dir(client):  # noqa: F811
    r = client.get("/imports")
    assert r.status_code == 404 and r.json()["code"] == "imports_off"
    assert client.get("/health").json()["imports"] == {"state": "not_configured"}


def test_add_file_then_duplicate(import_dir, enqueued, client):  # noqa: F811
    (import_dir / "Noir/Detour (1945).mkv").write_bytes(b"x" * 10)
    assert client.get("/health").json()["imports"] == {"state": "ok"}
    r = client.post("/movies", json={"file": "Noir/Detour (1945).mkv"})
    assert r.status_code == 201, r.text
    mid = r.json()["movie_id"]
    with db.db() as conn:
        m = db.row(conn, "SELECT * FROM movies WHERE id=?", mid)
    assert (m["source_kind"], m["source_url"], m["source_file"], m["title"], m["bytes"]) == (
        "local", "Noir/Detour (1945).mkv", "source.mkv", "Detour (1945)", 10)
    assert enqueued[-1][0] == "ingest"
    r = client.post("/movies", json={"file": "Noir/Detour (1945).mkv"})
    assert r.status_code == 409 and r.json()["movie_id"] == mid
    assert client.get("/imports").json()["files"][0]["movie_id"] == mid
    r = client.post("/movies", json={"file": "../../etc/passwd"})
    assert r.json()["code"] == "not_found"


def test_ingest_copies_when_it_cannot_link(import_dir, enqueued, client, monkeypatch):  # noqa: F811
    make_video(import_dir / "clip.mp4")

    def no_link(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", no_link)
    mid = client.post("/movies", json={"file": "clip.mp4"}).json()["movie_id"]
    tasks.run_ingest(enqueued[-1][1])
    with db.db() as conn:
        m = db.row(conn, "SELECT * FROM movies WHERE id=?", mid)
    assert m["status"] == "indexing", m["error"]
    src = config.settings().media_movie_dir(mid) / "source.mp4"
    assert src.read_bytes() == (import_dir / "clip.mp4").read_bytes()
    assert not src.is_symlink() and (import_dir / "clip.mp4").exists()


def test_local_movies_keep_source_and_proxy(import_dir, enqueued, client, monkeypatch):  # noqa: F811
    make_video(import_dir / "clip.mp4")
    mid = client.post("/movies", json={"file": "clip.mp4"}).json()["movie_id"]
    tasks.run_ingest(enqueued[-1][1])
    tasks.run_index(enqueued[-1][1])
    mdir = config.settings().media_movie_dir(mid)
    clip = mdir / "clips/c1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"c" * 50_000)
    with db.db() as conn:
        assert db.row(conn, "SELECT status FROM movies WHERE id=?", mid)["status"] == "ready"
        conn.execute("INSERT INTO clips(id, movie_id, start, \"end\", path, created_at) "
                     "VALUES ('c1',?,0,1,?,?)", (mid, str(clip), db.now()))
    monkeypatch.setenv("MAX_DATA_GB", repr(1 / 1024**3))
    config.reset_settings()
    assert evict.enforce_quota() == [f"clips:{mid}"]
    assert (mdir / "source.mp4").exists() and (mdir / "proxy.mp4").exists()
