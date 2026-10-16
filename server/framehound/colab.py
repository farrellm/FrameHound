"""The one adapter around the ``colab`` CLI (v0.7.x); nothing else shells out to it.

Exit codes are not reliable (``exec`` exits 0 when the code raises, ``status``
and ``stop`` exit 0 for an unknown session), so results are judged from the
output text: marker lines printed by our own scripts, and the CLI's ``[colab]``
messages. Every call is logged with its argv, exit code and duration.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .errors import FHError

log = logging.getLogger("framehound.colab")

# `colab new --gpu X` silently falls back to A100 for an unknown name.
GPUS = ("T4", "L4", "G4", "H100", "A100")
SESSION_PREFIX = "fh-"

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SESSION_LINE = re.compile(r"^\[([^\]]+)\] (\S+)(?: \| (.*))?$")
_AUTH = re.compile(r"\b(401|403)\b|unauthenticated|invalid_grant|reauth|credentials|"
                   r"permission denied|refresh token", re.IGNORECASE)


class ColabError(FHError):
    def __init__(self, message: str, detail: str | None = None, code: str = "colab_error"):
        super().__init__(code, message, detail=detail)


class ColabAuthError(ColabError):
    def __init__(self, detail: str | None = None):
        super().__init__("Colab sign-in has expired. On the server, sign the colab CLI in "
                         "again (run any colab command, or `gcloud auth application-default "
                         "login` with the Colab scopes for ADC), then retry.", detail,
                         code="colab_auth")


class SessionGone(ColabError):
    def __init__(self, session: str, detail: str | None = None):
        super().__init__(f"The Colab session {session} is gone.", detail, code="colab_lost")


@dataclass(frozen=True)
class Usage:
    balance: float
    rate_per_hour: float
    assignments: int


@dataclass(frozen=True)
class SessionInfo:
    name: str  # "?" for a server-side assignment with no local record
    endpoint: str
    hardware: str | None = None


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def parse_usage(text: str) -> Usage:
    bal = re.search(r"Current balance:\s*(-?[\d.]+)", text)
    rate = re.search(r"Usage rate:\s*(-?[\d.]+)", text)
    n = re.search(r"Active assignments:\s*(\d+)", text)
    if not bal:
        raise ColabError("Couldn't read the Colab compute-unit balance.", detail=text[-1000:])
    return Usage(float(bal.group(1)), float(rate.group(1)) if rate else 0.0,
                 int(n.group(1)) if n else 0)


def parse_sessions(text: str) -> list[SessionInfo]:
    out = []
    for line in text.splitlines():
        m = _SESSION_LINE.match(line.strip())
        if not m or m.group(1) == "colab":
            continue
        hw = re.search(r"Hardware: ([^|]+)", m.group(3) or "")
        out.append(SessionInfo(m.group(1), m.group(2), hw.group(1).strip() if hw else None))
    return out


def marker_line(text: str, marker: str) -> str | None:
    """The rest of the last line starting with ``marker`` (our scripts print one)."""
    found = None
    for line in strip_ansi(text).splitlines():
        if line.startswith(marker):
            found = line[len(marker):].strip()
    return found


class ColabCli:
    def __init__(self, binary: str | None = None, auth: str | None = None):
        s = settings()
        self.binary = binary or s.colab_bin
        self.auth = s.colab_auth if auth is None else auth

    def _argv(self, *args: str) -> list[str]:
        return [self.binary, *(["--auth", self.auth] if self.auth else []), *args]

    def run(self, *args: str, timeout: float = 300, check: bool = True) -> str:
        argv = self._argv(*args)
        t0 = time.monotonic()
        try:
            p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as e:
            log.warning("colab %s", {"argv": argv, "timeout_s": timeout})
            raise ColabError(f"`colab {args[0]}` timed out after {timeout:.0f} s.") from e
        except FileNotFoundError as e:
            raise ColabError("The colab CLI isn't installed on the server.",
                             detail=str(e), code="colab_missing") from e
        out = strip_ansi(p.stdout.decode(errors="replace"))
        log.info("colab %s", {"argv": argv, "exit": p.returncode,
                              "duration_s": round(time.monotonic() - t0, 2)})
        if check and p.returncode != 0:
            raise self._error(args, out)
        return out

    def _error(self, args: tuple[str, ...], out: str) -> ColabError:
        # Checked first: "appears to be lost (404/401)" is not an auth problem.
        if re.search(r"Session '[^']*' (not found|appears to be lost)", out):
            sess = args[args.index("-s") + 1] if "-s" in args else "?"
            return SessionGone(sess, out[-1500:])
        if _AUTH.search(out):
            return ColabAuthError(out[-1500:])
        return ColabError(f"`colab {args[0]}` failed.", detail=out[-1500:])

    # commands ----------------------------------------------------------------

    def usage(self) -> Usage:
        return parse_usage(self.run("usage", timeout=60))

    def new(self, session: str, gpu: str) -> None:
        out = self.run("new", "-s", session, "--gpu", gpu, timeout=600, check=False)
        if "Session READY" not in out:
            if _AUTH.search(out):
                raise ColabAuthError(out[-1500:])
            raise ColabError(f"Colab wouldn't start a {gpu} session.", detail=out[-1500:],
                             code="colab_no_gpu")

    def install(self, session: str, requirements: Path) -> str:
        return self.run("install", "-s", session, "-r", str(requirements), timeout=1200)

    def upload(self, session: str, local: Path, remote: str, timeout: float = 3600) -> None:
        out = self.run("upload", "-s", session, str(local), remote, timeout=timeout)
        if "Uploaded" not in out:
            raise ColabError("Upload to Colab failed.", detail=out[-1500:])

    def download(self, session: str, remote: str, local: Path, timeout: float = 1800) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        out = self.run("download", "-s", session, remote, str(local), timeout=timeout)
        if "Downloaded" not in out or not local.exists():
            raise ColabError("Download from Colab failed.", detail=out[-1500:])

    def exec_file(self, session: str, script: Path, env: dict[str, str] | None = None,
                  timeout: float = 120) -> str:
        args = ["exec", "-s", session, "-f", str(script), "--timeout", f"{timeout:g}"]
        for k, v in (env or {}).items():
            args += ["--env", f"{k}={v}"]
        return self.run(*args, timeout=timeout + 60)

    def status(self, session: str) -> bool:
        """True while the backend still has the session (``status`` syncs first)."""
        out = self.run("status", "-s", session, timeout=60, check=False)
        if _AUTH.search(out) and "not found" not in out:
            raise ColabAuthError(out[-1500:])
        return any(s.name == session for s in parse_sessions(out))

    def sessions(self) -> list[SessionInfo]:
        return parse_sessions(self.run("sessions", timeout=60))

    def stop(self, session: str) -> None:
        self.run("stop", "-s", session, timeout=180, check=False)

    def save_log(self, session: str, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.run("log", "-s", session, "-o", str(out), timeout=120, check=False)
