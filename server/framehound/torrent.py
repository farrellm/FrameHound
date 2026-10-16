"""BitTorrent ingest with libtorrent, on the host only (design.md §4).

Torrents never run on a Colab VM: Colab's policy forbids peer-to-peer sharing,
so a torrent movie's proxy is uploaded to the VM instead. Only the largest
video file is downloaded; every other file gets priority 0.
"""

from __future__ import annotations

import ipaddress
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import libtorrent as lt

from .config import settings
from .errors import FHError
from .media import VIDEO_EXTS

TORRENT_FILE = "source.torrent"
WORK_DIR = "torrent"

# Peers and trackers at these addresses are refused, like the HTTP SSRF guard.
BLOCKED_NETS = ["0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
                "172.16.0.0/12", "192.168.0.0/16", "198.18.0.0/15", "224.0.0.0/3",
                "::/127", "fc00::/7", "fe80::/10", "ff00::/8"]


@dataclass
class TorrentFile:
    index: int
    path: str  # inside the torrent
    size: int
    ext: str


def pick_video(ti: lt.torrent_info) -> TorrentFile | None:
    """The largest file with a video extension (pad files skipped)."""
    fs = ti.layout()
    best: TorrentFile | None = None
    for i in range(fs.num_files()):
        if fs.file_flags(i) & fs.flag_pad_file:
            continue
        path = fs.file_path(i)
        ext = PurePosixPath(path).suffix.lower()
        if ext in VIDEO_EXTS and (best is None or fs.file_size(i) > best.size):
            best = TorrentFile(i, path, fs.file_size(i), ext)
    return best


def parse_torrent(data: bytes) -> lt.torrent_info:
    try:
        return lt.torrent_info(lt.bdecode(data))
    except Exception as e:  # libtorrent raises RuntimeError or TypeError
        raise FHError("bad_torrent", "That isn't a .torrent file FrameHound can read.",
                      detail=str(e)) from e


def parse_magnet(uri: str) -> lt.add_torrent_params:
    try:
        p = lt.parse_magnet_uri(uri)
    except Exception as e:
        raise FHError("unrecognized", "That magnet link is missing its torrent hash.",
                      detail=str(e)) from e
    if p.info_hashes.get_best().is_all_zeros():
        raise FHError("unrecognized", "That magnet link is missing its torrent hash.")
    return p


def info_hash(ih: lt.info_hash_t) -> str:
    return str(ih.v1 if ih.has_v1() else ih.get_best())


def _ip_filter() -> lt.ip_filter:
    f = lt.ip_filter()
    for net in BLOCKED_NETS:
        n = ipaddress.ip_network(net)
        f.add_rule(str(n.network_address), str(n.broadcast_address), 1)
    return f


def _session() -> lt.session:
    s = settings()
    pack = {
        "listen_interfaces": s.torrent_listen,
        "enable_upnp": False, "enable_natpmp": False,
        "ssrf_mitigation": not s.allow_private_urls,
        "allow_multiple_connections_per_ip": s.allow_private_urls,
        "alert_mask": lt.alert_category.error,
        "user_agent": "FrameHound/0.1",
    }
    if s.allow_private_urls:  # dev/test: a local seeder; no DHT or LSD noise
        pack.update(enable_dht=False, enable_lsd=False)
    sess = lt.session(pack)
    if not s.allow_private_urls:
        sess.set_ip_filter(_ip_filter())
    return sess


Progress = Callable[[float, int, int, int], None]  # frac, done, total, peers


def download_torrent(movie_dir: Path, magnet: str | None, on_file: Callable[[TorrentFile], str],
                     on_progress: Progress, check_cancel: Callable[[], None]) -> Path:
    """Download the torrent's main video into ``movie_dir`` and return its path.

    The torrent comes from ``movie_dir/source.torrent`` if it exists, else from
    ``magnet``. ``on_file`` is called once the file is known and returns the
    server-derived name to store it under (``source.mp4``…). Partial data in
    ``movie_dir/torrent/`` is rechecked and resumed by libtorrent.
    """
    s = settings()
    work = movie_dir / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    tfile = movie_dir / TORRENT_FILE
    if tfile.exists():
        params = lt.add_torrent_params()
        params.ti = parse_torrent(tfile.read_bytes())
        # Known up front: never fetch the other files at all.
        picked = pick_video(params.ti)
        if picked is not None:
            params.file_priorities = [4 if i == picked.index else 0
                                      for i in range(params.ti.num_files())]
    elif magnet:
        params = parse_magnet(magnet)
    else:
        raise FHError("source_missing", "The torrent for this movie is gone. Add it again.")
    params.save_path = str(work)
    sess = _session()
    try:
        h = sess.add_torrent(params)
        for peer in s.torrent_peers:
            host, _, port = peer.rpartition(":")
            h.connect_peer((host, int(port)))
        return _run(sess, h, movie_dir, work, on_file, on_progress, check_cancel, s)
    finally:
        sess.pause()


def _run(sess, h, movie_dir: Path, work: Path, on_file, on_progress, check_cancel, s) -> Path:
    picked: TorrentFile | None = None
    dest: Path | None = None
    last_progress = time.monotonic()
    last_done = -1
    stall = s.torrent_stall_s
    seed_until: float | None = None
    while True:
        check_cancel()
        st = h.status()
        for a in sess.pop_alerts():
            if isinstance(a, lt.torrent_error_alert | lt.file_error_alert):
                raise FHError("download_failed", "The torrent download failed.",
                              detail=a.message())
        if picked is None and st.has_metadata:
            ti = h.torrent_file()
            picked = pick_video(ti)
            if picked is None:
                raise FHError("no_video", "This torrent has no movie file FrameHound can "
                              "index.", detail=f"files: {ti.num_files()}")
            if picked.size > s.max_bytes:
                raise FHError("too_big", "This file is larger than the size limit.",
                              detail=f"{picked.size} bytes")
            h.prioritize_files([4 if i == picked.index else 0
                                for i in range(ti.num_files())])
            dest = movie_dir / on_file(picked)
        if picked is not None:
            done = h.file_progress()[picked.index]
            on_progress(done / picked.size if picked.size else 0.0, done, picked.size,
                        st.num_peers)
            if done != last_done:
                last_done = done
                last_progress = time.monotonic()
            if done >= picked.size and st.is_finished:
                if seed_until is None:
                    seed_until = time.monotonic() + (s.seed_max_s if s.seed_ratio > 0 else 0)
                ratio = st.all_time_upload / max(picked.size, 1)
                if ratio >= s.seed_ratio or time.monotonic() >= seed_until:
                    break
        else:
            on_progress(0.0, 0, 0, st.num_peers)
        if st.num_peers > 0 and picked is None:
            last_progress = time.monotonic()  # fetching metadata from someone
        if seed_until is None and time.monotonic() - last_progress > stall:
            mins = stall / 60
            raise FHError("torrent_stalled", f"The torrent has had no peers for "
                          f"{mins:g} minute{'s' if mins != 1 else ''}." if not st.num_peers
                          else f"The torrent stopped downloading for {mins:g} minutes.",
                          detail=f"peers {st.num_peers}, state {st.state}")
        time.sleep(0.5)
    assert picked is not None and dest is not None
    sess.remove_torrent(h)
    time.sleep(0.2)
    (work / picked.path).replace(dest)
    shutil.rmtree(work, ignore_errors=True)
    return dest
