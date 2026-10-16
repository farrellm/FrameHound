"""IndexBackend: where the indexing pipeline runs.

``LocalIndexBackend`` runs ``python -m framehound_indexer`` in a detached
subprocess on this host. ``ColabIndexBackend`` (M2) drives the same package on
a Colab GPU through the ``colab`` CLI, behind the same interface.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from framehound_indexer import bundle
from framehound_indexer.pipeline import DEFAULT_STAGES

from .config import settings


class IndexBackend(Protocol):
    name: str
    gpu: str | None
    poll_interval_s: float

    def planned_stages(self) -> list[str]: ...
    def pipeline_stages(self) -> list[str]: ...
    def start(self, job_id: str, spec: dict) -> dict: ...
    def poll(self, handle: dict) -> tuple[list[dict], bool]: ...
    def fetch(self, handle: dict, dest: Path) -> Path: ...
    def cancel(self, handle: dict) -> None: ...
    def cleanup(self, handle: dict, failed: bool) -> None: ...
    def log_tail(self, handle: dict, n: int = 2000) -> str: ...


_procs: dict[int, subprocess.Popen] = {}


class LocalIndexBackend:
    name = "local"
    gpu = None  # shown as "this computer"
    poll_interval_s = 2.0

    def __init__(self, command: list[str] | None = None):
        self.command = command or [sys.executable, "-m", "framehound_indexer"]

    def pipeline_stages(self) -> list[str]:
        return list(DEFAULT_STAGES)

    def planned_stages(self) -> list[str]:
        return ["staging_media", *self.pipeline_stages(), "downloading_index", "importing"]

    def _work_dir(self, movie_id: str) -> Path:
        return settings().data_dir / "work" / movie_id

    def start(self, job_id: str, spec: dict) -> dict:
        work = self._work_dir(spec["movie_id"])
        # Fresh run per job; a crashed run in the same job would resume from markers.
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        spec = {**spec, "work_dir": str(work), "stages": self.pipeline_stages(),
                "zstd_level": settings().index_zstd_level, "device": "auto"}
        job_path = work / "job.json"
        job_path.write_text(json.dumps(spec, indent=1))
        log = open(work / "run.log", "ab")  # noqa: SIM115 (handed to the child)
        proc = subprocess.Popen(
            [*self.command, str(job_path)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env={**os.environ},
        )
        log.close()
        _procs[proc.pid] = proc
        return {"pid": proc.pid, "work_dir": str(work), "offset": 0}

    def poll(self, handle: dict) -> tuple[list[dict], bool]:
        alive = _alive(handle["pid"])
        path = Path(handle["work_dir"]) / "progress.jsonl"
        events: list[dict] = []
        if path.exists():
            with open(path, "rb") as fh:
                fh.seek(handle["offset"])
                data = fh.read()
            # Only consume complete lines; a partial last line is read next time.
            end = data.rfind(b"\n") + 1
            handle["offset"] += end
            for line in data[:end].splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events, alive

    def fetch(self, handle: dict, dest: Path) -> Path:
        src = Path(handle["work_dir"]) / bundle.BUNDLE_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dest)
        return dest

    def cancel(self, handle: dict) -> None:
        pid = handle.get("pid")
        if pid and _alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            proc = _procs.get(pid)
            if proc is not None:
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(pid, signal.SIGKILL)

    def log_tail(self, handle: dict, n: int = 2000) -> str:
        path = Path(handle["work_dir"]) / "run.log"
        if not path.exists():
            return ""
        data = path.read_bytes()[-n:]
        return data.decode(errors="replace")

    def cleanup(self, handle: dict, failed: bool) -> None:
        self.cancel(handle)
        _procs.pop(handle.get("pid"), None)
        work = Path(handle["work_dir"])
        if failed and (work / "run.log").exists():
            logs = settings().data_dir / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "run.log", logs / f"{work.name}-{handle['pid']}.log")
        shutil.rmtree(work, ignore_errors=True)


def _alive(pid: int) -> bool:
    proc = _procs.get(pid)
    if proc is not None:
        return proc.poll() is None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def get_backend() -> IndexBackend:
    name = settings().index_backend
    if name == "local":
        return LocalIndexBackend()
    raise RuntimeError(f"INDEX_BACKEND={name!r} is not available yet (colab arrives in M2)")
