"""Streaming download with Range resume and checksum verification."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from .config import settings
from .errors import FHError
from .safe_http import SafeClient


def download(client: SafeClient, url: str, dest: Path, expected_size: int | None,
             md5: str | None, on_progress: Callable[[float, int, int], None],
             check_cancel: Callable[[], None]) -> int:
    """Download ``url`` to ``dest``; resume a ``.part`` file if one exists."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    limit = settings().max_bytes
    with client.stream(url, headers) as r:
        if r.status_code == 416 and have and expected_size == have:
            pass  # already complete
        elif r.status_code >= 400:
            raise FHError("download_failed", f"The download failed (the server said "
                          f"{r.status_code}).", detail=f"HTTP {r.status_code} for {url}")
        else:
            if r.status_code != 206:
                have = 0  # server ignored Range; start over
            total = expected_size or (have + int(r.headers.get("content-length") or 0)) or 0
            if total and total > limit:
                raise FHError("too_big", "This file is larger than the size limit.",
                              detail=f"{total} bytes")
            done = have
            with open(part, "ab" if have else "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if done > limit:
                        raise FHError("too_big", "This file is larger than the size limit.")
                    on_progress(done / total if total else 0.0, done, total)
                    check_cancel()
    size = part.stat().st_size
    if expected_size and size != expected_size:
        raise FHError("download_failed", "The download ended early. Retry to resume it.",
                      detail=f"got {size} of {expected_size} bytes")
    if md5:
        h = hashlib.md5()
        with open(part, "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        if h.hexdigest() != md5:
            part.unlink()
            raise FHError("download_failed", "The downloaded file was damaged. Retry to "
                          "download it again.", detail=f"md5 {h.hexdigest()} != {md5}")
    part.replace(dest)
    return size


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(4 << 20):
            h.update(chunk)
    return h.hexdigest()
