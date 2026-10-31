"""The one adapter around the ``colab`` CLI; nothing else shells out to it.

The CLI wrapper itself, its output parsing and its errors live in ``colabkit`` (shared with
degas). Its errors carry the same ``code``/``message``/``detail`` as ``FHError``, and
``jobs.Job.fail`` records them as such.
"""

from __future__ import annotations

import colabkit
from colabkit import (
    GPUS,
    ColabAuthError,
    ColabError,
    SessionGone,
    SessionInfo,
    Usage,
    marker_line,
    parse_sessions,
    parse_usage,
    strip_ansi,
)

from .config import settings

__all__ = ["GPUS", "SESSION_PREFIX", "ColabAuthError", "ColabCli", "ColabError", "SessionGone",
           "SessionInfo", "Usage", "marker_line", "parse_sessions", "parse_usage", "strip_ansi"]

SESSION_PREFIX = "fh-"


class ColabCli(colabkit.ColabCli):
    def __init__(self, binary: str | None = None, auth: str | None = None):
        s = settings()
        super().__init__(binary or s.colab_bin, (s.colab_auth if auth is None else auth) or None)

    def stop(self, session: str, check: bool = False) -> None:
        """Best effort: ``stop`` exits 0 for an unknown session anyway."""
        super().stop(session, check=check)
