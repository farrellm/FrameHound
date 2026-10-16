"""IndexBackend: where the indexing pipeline runs.

``LocalIndexBackend`` runs ``python -m framehound_indexer`` in a detached
subprocess on this host. ``ColabIndexBackend`` drives the same package on a
Colab GPU through the ``colab`` CLI (design.md §5), behind the same interface.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import framehound_indexer
from framehound_indexer import bundle
from framehound_indexer.pipeline import DEFAULT_STAGES

from . import db
from .colab import GPUS, SESSION_PREFIX, ColabCli, ColabError, SessionGone, marker_line
from .config import settings
from .errors import FHError

if TYPE_CHECKING:
    from .jobs import Job

log = logging.getLogger("framehound.index_backend")


class IndexBackend(Protocol):
    name: str
    gpu: str | None
    poll_interval_s: float

    def planned_stages(self) -> list[str]: ...
    def pipeline_stages(self) -> list[str]: ...
    def start(self, job: Job, spec: dict, attempt: int = 0) -> dict: ...
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

    def start(self, job: Job, spec: dict, attempt: int = 0) -> dict:
        job.stage("staging_media", 0.0, force=True)
        work = self._work_dir(spec["movie_id"])
        # Fresh run per job; a crashed run in the same job would resume from markers.
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        spec = {**_pipeline_spec(spec), "work_dir": str(work), "stages": self.pipeline_stages(),
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
        job.stage_done("staging_media")
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


def _pipeline_spec(spec: dict) -> dict:
    """What the pipeline's job.json needs (the host-only ``source`` stays here)."""
    return {k: v for k, v in spec.items() if k != "source"}


# Colab ------------------------------------------------------------------------

REMOTE_SCRIPTS = Path(framehound_indexer.__file__).parent / "remote"
CHUNK = 64 << 20  # the CLI moves a file as one base64 request; keep each part modest
PUBLIC_KINDS = ("http", "archive")


def package_tarball() -> tuple[Path, str]:
    """A reproducible tar of ``framehound_indexer/``, cached under DATA_DIR by hash."""
    root = Path(framehound_indexer.__file__).parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(p for p in root.rglob("*") if p.is_file()
                           and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")):
            data = path.read_bytes()
            info = tarfile.TarInfo(f"framehound_indexer/{path.relative_to(root).as_posix()}")
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()[:16]
    out = settings().data_dir / "cache" / f"framehound_indexer-{sha}.tar"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".tmp").write_bytes(data)
        out.with_suffix(".tmp").replace(out)
    return out, sha


def session_name(movie_id: str, attempt: int) -> str:
    return f"{SESSION_PREFIX}{movie_id}-{attempt}"


class SessionLost(FHError):
    """The VM was reclaimed or disconnected; the job may retry on a fresh one."""

    def __init__(self, detail: str | None = None):
        super().__init__("colab_lost", "The Colab GPU went away during indexing.", detail=detail)


class ColabIndexBackend:
    name = "colab"
    usage_every_s = 60.0

    def __init__(self, cli: ColabCli | None = None, command: list[str] | None = None):
        s = settings()
        self.cli = cli or ColabCli()
        self.gpu = s.colab_gpu
        self.poll_interval_s = s.colab_poll_s
        self.root = s.colab_remote_root.rstrip("/")
        self.command = command  # argv prefix on the VM (tests); default runs remote/run.py

    def pipeline_stages(self) -> list[str]:
        return list(DEFAULT_STAGES)

    def planned_stages(self) -> list[str]:
        return ["provisioning_gpu", "staging_media", *self.pipeline_stages(),
                "downloading_index", "importing"]

    # helpers ---------------------------------------------------------------

    def _exec(self, session: str, script: str, marker: str, timeout: float = 120,
              **env: str) -> str:
        out = self.cli.exec_file(session, REMOTE_SCRIPTS / script, env, timeout=timeout)
        found = marker_line(out, marker)
        if found is None:
            if "Session '" in out and ("not found" in out or "lost" in out):
                raise SessionGone(session, out[-1500:])
            raise ColabError(f"The {script} step failed on the Colab VM.", detail=out[-2000:])
        return found

    def _util(self, session: str, **env: str) -> dict:
        return json.loads(self._exec(session, "util.py", "@@fh-util", **env))

    def _upload(self, session: str, local: Path, remote: str) -> None:
        size = local.stat().st_size
        if size <= CHUNK:
            self.cli.upload(session, local, remote)
            return
        with open(local, "rb") as src, tempfile.TemporaryDirectory() as tmp:
            n = 0
            while data := src.read(CHUNK):
                part = Path(tmp) / f"part{n}"
                part.write_bytes(data)
                self.cli.upload(session, part, f"{remote}.part{n}")
                part.unlink()
                n += 1
        self._util(session, FH_OP="join", FH_DEST=remote, FH_PARTS=str(n))

    def _download(self, session: str, remote: str, local: Path) -> None:
        info = self._util(session, FH_OP="split", FH_SRC=remote, FH_CHUNK=str(CHUNK))
        tmp = local.with_name(local.name + ".part")
        with open(tmp, "wb") as fh:
            for i in range(info["parts"]):
                part = local.with_name(f"{local.name}.part{i}")
                self.cli.download(session, f"{remote}.part{i}", part)
                fh.write(part.read_bytes())
                part.unlink()
        if tmp.stat().st_size != info["bytes"]:
            raise ColabError("The index came back incomplete from Colab.",
                             detail=f"{tmp.stat().st_size} != {info['bytes']} bytes")
        tmp.replace(local)

    def _month_spend(self) -> float:
        t = time.gmtime()
        start = time.mktime((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        with db.db() as conn:
            r = conn.execute("SELECT COALESCE(SUM(compute_units_used), 0) FROM jobs "
                             "WHERE kind='index' AND created_at >= ?", (start,)).fetchone()
        return float(r[0] or 0)

    def _check_budget(self) -> float:
        s = settings()
        usage = self.cli.usage()
        if usage.balance < s.min_balance_cu:
            raise FHError("colab_budget", f"Colab has {usage.balance:.1f} compute units left, "
                          f"below the {s.min_balance_cu:g} FrameHound keeps in reserve. Add "
                          "compute units, or index on this computer.")
        spent = self._month_spend()
        if spent >= s.monthly_budget_cu:
            raise FHError("colab_budget", f"This would pass this month's GPU budget "
                          f"({s.monthly_budget_cu:g} compute units; {spent:.1f} used). Raise "
                          "the budget in settings or index on this computer.")
        return usage.balance

    def _wait_for_slot(self, job: Job) -> None:
        s = settings()
        while True:
            with db.db() as conn:
                busy = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE kind='index' AND state='running' "
                    "AND colab_session IS NOT NULL AND id<>?", (job.id,)).fetchone()[0]
            if busy < s.max_colab_sessions:
                return
            job.stage("provisioning_gpu", 0.0, message="Waiting for another movie's GPU",
                      force=True)
            job.check_cancel(force=True)
            time.sleep(15)

    def _provision(self, job: Job, movie_id: str, attempt: int) -> tuple[str, str]:
        s = settings()
        chain = [g for g in dict.fromkeys([self.gpu, *s.gpu_fallback])]
        bad = [g for g in chain if g not in GPUS]
        if bad:
            raise FHError("config", f"Unknown Colab GPU {bad[0]!r}. Use one of "
                          f"{', '.join(GPUS)}.")
        session = session_name(movie_id, attempt)
        errors = []
        for gpu in chain:
            job.check_cancel(force=True)
            try:
                self.cli.new(session, gpu)
                return session, gpu
            except ColabError as e:
                if e.code == "colab_auth":
                    raise
                errors.append(f"{gpu}: {e.detail or e.message}")
                self.cli.stop(session)  # in case it half-exists
        raise FHError("colab_no_gpu", f"Colab has no {' or '.join(chain)} available for this "
                      "account right now. Retry later, or index on this computer.",
                      detail="\n".join(errors)[-2000:])

    # IndexBackend ----------------------------------------------------------

    def start(self, job: Job, spec: dict, attempt: int = 0) -> dict:
        movie_id = spec["movie_id"]
        job.stage("provisioning_gpu", 0.0, force=True)
        balance = self._check_budget()
        self._wait_for_slot(job)
        session, gpu = self._provision(job, movie_id, attempt)
        self.gpu = gpu
        remote = f"{self.root}/{movie_id}"
        handle = {"backend": "colab", "session": session, "gpu": gpu, "attempt": attempt,
                  "movie_id": movie_id, "remote": remote, "offset": 0, "fails": 0,
                  "balance_before": balance, "started": time.time(), "usage_at": time.time(),
                  "log": None}
        job.set(colab_session=session, gpu=gpu, backend_handle=json.dumps(handle))
        job.stage_done("provisioning_gpu")
        try:
            self._stage(job, handle, spec)
        except BaseException:
            self.cleanup(handle, failed=True)
            raise
        return handle

    def _stage(self, job: Job, handle: dict, spec: dict) -> None:
        session, remote = handle["session"], handle["remote"]
        job.stage("staging_media", 0.0, force=True)
        tar, sha = package_tarball()
        pkg = f"{self.root}/pkg/{sha}"
        self._util(session, FH_OP="mkdir", FH_DIRS=f"{remote}:{self.root}/pkg")
        self._upload(session, tar, f"{pkg}.tar")
        self._util(session, FH_OP="unpack", FH_TAR=f"{pkg}.tar", FH_DEST=pkg)
        job.stage("staging_media", 0.2, force=True)
        job.check_cancel(force=True)
        self.cli.install(session, REMOTE_SCRIPTS / "requirements-colab.txt")
        job.stage("staging_media", 0.5, force=True)
        job.check_cancel(force=True)

        src = spec.get("source") or {}
        if src.get("kind") in PUBLIC_KINDS and src.get("url") and src.get("sha256"):
            # Datacenter bandwidth beats uploading from home; the VM checks the hash.
            media = {"url": src["url"], "sha256": src["sha256"]}
        else:
            proxy = Path(src.get("proxy") or spec["media"]["path"])
            dest = f"{remote}/upload{proxy.suffix or '.mp4'}"
            self._upload(session, proxy, dest)
            media = {"path": dest}
        pipeline = {**_pipeline_spec(spec), "media": media, "work_dir": remote,
                    "stages": self.pipeline_stages(), "device": "auto", "zstd_level": 19}
        with tempfile.TemporaryDirectory() as tmp:
            job_json = Path(tmp) / "job.json"
            job_json.write_text(json.dumps(pipeline, indent=1))
            self.cli.upload(session, job_json, f"{remote}/job.json")
        env = {"FH_JOB": f"{remote}/job.json", "FH_PKG": pkg}
        if self.command:
            env["FH_CMD"] = json.dumps(self.command)
        pid = self._exec(session, "launch.py", "@@fh-launched", **env).split()[0]
        handle["pid"] = int(pid)
        job.set(backend_handle=json.dumps(handle))
        job.stage_done("staging_media")

    def poll(self, handle: dict) -> tuple[list[dict], bool]:
        session = handle["session"]
        try:
            raw = self._exec(session, "poll.py", "@@fh-poll", timeout=60,
                             FH_WORK=handle["remote"], FH_OFFSET=str(handle["offset"]))
            out = json.loads(raw)
        except SessionGone as e:
            raise SessionLost(e.detail) from e
        except (ColabError, json.JSONDecodeError) as e:
            if getattr(e, "code", None) == "colab_auth":
                raise
            handle["fails"] = handle.get("fails", 0) + 1
            log.warning("colab poll failed (%d in a row): %s", handle["fails"], e)
            if handle["fails"] >= 3:
                if not self._alive(session):
                    raise SessionLost(str(e)) from e
                handle["fails"] = 0
            return [], True
        handle["fails"] = 0
        handle["offset"] = out["offset"]
        if out.get("log"):
            handle["log"] = out["log"]
        return out["lines"], bool(out["alive"])

    def _alive(self, session: str) -> bool:
        try:
            return self.cli.status(session)
        except ColabError as e:
            if e.code == "colab_auth":
                raise
            return True  # can't tell; keep polling

    def compute_units(self, handle: dict) -> float | None:
        """Units this job has used so far: the balance drop, else rate × elapsed."""
        if time.time() - handle.get("usage_at", 0) < self.usage_every_s:
            return None
        handle["usage_at"] = time.time()
        try:
            u = self.cli.usage()
        except ColabError:
            return None
        used = handle["balance_before"] - u.balance
        if used <= 0:
            used = u.rate_per_hour * (time.time() - handle["started"]) / 3600
        return round(max(used, 0.0), 2)

    def fetch(self, handle: dict, dest: Path) -> Path:
        self._download(handle["session"], f"{handle['remote']}/{bundle.BUNDLE_NAME}", dest)
        return dest

    def cancel(self, handle: dict) -> None:
        try:
            self._util(handle["session"], FH_OP="kill", FH_PIDFILE=f"{handle['remote']}/pid")
        except ColabError as e:
            log.warning("colab kill failed: %s", e)

    def log_tail(self, handle: dict, n: int = 2000) -> str:
        return (handle.get("log") or "")[-n:]

    def cleanup(self, handle: dict, failed: bool) -> None:
        session = handle["session"]
        if failed:
            try:
                self.cli.save_log(session, settings().data_dir / "logs" / f"{session}.jsonl")
            except ColabError as e:
                log.warning("colab log failed: %s", e)
        try:
            self.cli.stop(session)
        except ColabError as e:
            log.warning("colab stop failed: %s", e)
        try:
            u = self.cli.usage()
            handle["final_cu"] = round(max(handle["balance_before"] - u.balance, 0.0), 2)
        except ColabError:
            pass


def get_backend(name: str | None = None) -> IndexBackend:
    name = name or settings().index_backend
    if name == "local":
        return LocalIndexBackend()
    if name == "colab":
        return ColabIndexBackend()
    raise RuntimeError(f"INDEX_BACKEND={name!r} isn't one of local, colab")
