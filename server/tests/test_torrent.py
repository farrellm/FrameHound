"""Torrent ingest against an in-process seeder on 127.0.0.1."""

import time

import libtorrent as lt
import pytest
from framehound import config, db, tasks
from framehound.errors import FHError
from framehound.torrent import download_torrent, listen_on, port_slot
from test_pipeline_flow import client, enqueued, make_video  # noqa: F401


@pytest.fixture
def seeder(tmp_path):
    """A seeding session for a two-file torrent: a movie and a text file."""
    root = tmp_path / "seed"
    (root / "Old Movie").mkdir(parents=True)
    make_video(root / "Old Movie" / "old.movie.mp4", seconds=2)
    (root / "Old Movie" / "readme.txt").write_bytes(b"r" * 40000)
    ct = lt.create_torrent(lt.list_files(str(root / "Old Movie")), 16384)
    lt.set_piece_hashes(ct, str(root))
    data = lt.bencode(ct.generate())
    sess = lt.session({"listen_interfaces": "127.0.0.1:0", "enable_dht": False,
                       "enable_lsd": False, "enable_upnp": False, "enable_natpmp": False,
                       "allow_multiple_connections_per_ip": True})
    params = lt.add_torrent_params()
    params.ti = lt.torrent_info(lt.bdecode(data))
    params.save_path = str(root)
    h = sess.add_torrent(params)
    for _ in range(100):
        if h.status().is_seeding:
            break
        time.sleep(0.05)
    assert h.status().is_seeding
    yield {"data": data, "magnet": lt.make_magnet_uri(params.ti), "port": sess.listen_port()}
    sess.pause()


@pytest.fixture
def local_peers(monkeypatch, seeder):
    monkeypatch.setenv("FH_ALLOW_PRIVATE_URLS", "1")
    monkeypatch.setenv("TORRENT_LISTEN", "127.0.0.1:0")
    monkeypatch.setenv("FH_TORRENT_PEERS", f"127.0.0.1:{seeder['port']}")
    config.reset_settings()
    return seeder


def _ingested(data_dir, client, enqueued, mid):  # noqa: F811
    tasks.run_ingest(enqueued[-1][1])
    movie = client.get(f"/movies/{mid}").json()
    assert movie["status"] == "indexing", movie["error"]
    mdir = data_dir / "movies" / mid
    assert (mdir / "source.mp4").exists() and (mdir / "proxy.mp4").exists()
    assert not (mdir / "torrent").exists()
    with db.db() as conn:
        m = db.row(conn, "SELECT * FROM movies WHERE id=?", mid)
    assert m["source_kind"] == "torrent" and m["source_file"] == "source.mp4"
    assert m["duration_s"] == pytest.approx(2.0, abs=0.1)
    return m


def test_magnet_ingest(data_dir, local_peers, client, enqueued):  # noqa: F811
    r = client.post("/movies", json={"source": local_peers["magnet"]})
    assert r.status_code == 201, r.text
    m = _ingested(data_dir, client, enqueued, r.json()["movie_id"])
    assert m["title"] == "Old Movie" and m["source_ref"].startswith("btih:")
    # The same torrent again is a duplicate, by its hash.
    again = client.post("/movies", files={"torrent": ("x.torrent", local_peers["data"])})
    assert again.status_code == 409 and again.json()["movie_id"] == m["id"]


def test_torrent_file_ingest(data_dir, local_peers, client, enqueued):  # noqa: F811
    r = client.post("/movies", files={"torrent": ("old.torrent", local_peers["data"],
                                                  "application/x-bittorrent")})
    assert r.status_code == 201, r.text
    mid = r.json()["movie_id"]
    assert (data_dir / "movies" / mid / "source.torrent").read_bytes() == local_peers["data"]
    _ingested(data_dir, client, enqueued, mid)


def test_bad_torrent_upload(client):  # noqa: F811
    r = client.post("/movies", files={"torrent": ("x.torrent", b"garbage")})
    assert r.status_code == 422 and r.json()["code"] == "bad_torrent"


def _download(tmp_path, magnet):
    return download_torrent(tmp_path / "m", magnet, lambda f: f"source{f.ext}",
                            lambda *a: None, lambda: None)


def test_stall_without_peers(tmp_path, monkeypatch, seeder):
    monkeypatch.setenv("FH_ALLOW_PRIVATE_URLS", "1")
    monkeypatch.setenv("TORRENT_LISTEN", "127.0.0.1:0")
    monkeypatch.setenv("TORRENT_STALL_S", "1.5")
    config.reset_settings()
    (tmp_path / "m").mkdir()
    with pytest.raises(FHError) as e:
        _download(tmp_path, seeder["magnet"])
    assert e.value.code == "torrent_stalled" and "no peers" in e.value.message


def test_private_peers_are_blocked(tmp_path, monkeypatch, seeder):
    # Without the dev override the IP filter refuses the seeder on 127.0.0.1.
    monkeypatch.setenv("TORRENT_LISTEN", "127.0.0.1:0")
    monkeypatch.setenv("TORRENT_STALL_S", "2")
    monkeypatch.setenv("FH_TORRENT_PEERS", f"127.0.0.1:{seeder['port']}")
    config.reset_settings()
    (tmp_path / "m").mkdir()
    with pytest.raises(FHError) as e:
        _download(tmp_path, seeder["magnet"])
    assert e.value.code == "torrent_stalled"


def test_listen_on_offsets_ports():
    listen = "0.0.0.0:6881,[::]:6881s,127.0.0.1:0"
    assert listen_on(listen, 0) == listen
    assert listen_on(listen, 2) == "0.0.0.0:6883,[::]:6883s,127.0.0.1:0"
    assert listen_on(listen, None) == "0.0.0.0:0,[::]:0s,127.0.0.1:0"


def test_port_slots_are_exclusive(monkeypatch):
    monkeypatch.setenv("MAX_DOWNLOADS", "2")
    config.reset_settings()
    with port_slot() as a, port_slot() as b, port_slot() as c:
        assert (a, b, c) == (0, 1, None)
    with port_slot() as a:
        assert a == 0  # freed again
