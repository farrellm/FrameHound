"""The import folder: movie files already on the server (IMPORT_DIR), added by path."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .config import settings
from .errors import FHError
from .media import VIDEO_EXTS

MAX_FILES = 2000

GONE = "That file isn't in the import folder any more."


def root() -> Path:
    d = settings().import_dir
    if d is None:
        raise FHError("imports_off", "No import folder is set up on the server.", status=404)
    return d.resolve()


def list_files() -> list[dict]:
    """Video files under the folder, newest first; dotfiles and symlinks out are skipped."""
    base = root()
    out: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            if name.startswith(".") or PurePosixPath(name).suffix.lower() not in VIDEO_EXTS:
                continue
            p = Path(dirpath) / name
            try:
                real = p.resolve(strict=True)
                st = real.stat()
            except OSError:
                continue
            if not real.is_relative_to(base) or not real.is_file():
                continue
            out.append({"path": p.relative_to(base).as_posix(), "name": name,
                        "bytes": st.st_size, "mtime": int(st.st_mtime)})
    out.sort(key=lambda f: (-f["mtime"], f["path"]))
    return out[:MAX_FILES]


def resolve(rel: str) -> Path:
    """The file ``rel`` names, only if it's a video file inside the folder."""
    base = root()
    pure = PurePosixPath(rel)
    if not rel or pure.is_absolute() or ".." in pure.parts:
        raise FHError("not_found", GONE, detail=f"refused path {rel!r}")
    try:
        p = (base / pure).resolve(strict=True)
    except OSError as e:
        raise FHError("not_found", GONE, detail=str(e)) from e
    if not p.is_relative_to(base) or not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
        raise FHError("not_found", GONE, detail=f"refused path {rel!r}")
    return p


def copy_in(src: Path, dest: Path, on_progress: Callable[[float, int, int], None],
            check_cancel: Callable[[], None]) -> int:
    """Hard-link ``src`` to ``dest`` when they share a filesystem, else copy it."""
    total = src.stat().st_size
    if total > settings().max_bytes:
        raise FHError("too_big", "This file is larger than the size limit.",
                      detail=f"{total} bytes")
    dest.unlink(missing_ok=True)
    try:
        os.link(src, dest)
        return total
    except OSError:
        pass  # another filesystem, or a read-only mount
    part = dest.with_name(dest.name + ".part")
    done = 0
    with open(src, "rb") as fin, open(part, "wb") as fout:
        while chunk := fin.read(4 << 20):
            fout.write(chunk)
            done += len(chunk)
            on_progress(done / total if total else 0.0, done, total)
            check_cancel()
    part.replace(dest)
    return done
