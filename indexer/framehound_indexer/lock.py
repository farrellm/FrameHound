"""models.lock: the single pin for every model and its preprocessing.

Query embeddings are only comparable to index embeddings when both come from
the exact same lock, so every bundle records ``hash`` and the api refuses to
search a bundle whose hash differs from its own.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

LOCK_ENV = "FH_MODELS_LOCK"


def canonical_hash(data: dict[str, Any]) -> str:
    """Hash the parsed lock, so whitespace and key order don't matter."""
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ModelsLock:
    data: dict[str, Any]

    @property
    def hash(self) -> str:
        return canonical_hash(self.data)

    @property
    def whisper(self) -> dict[str, Any]:
        return self.data["whisper"]

    @property
    def sentence(self) -> dict[str, Any]:
        return self.data["sentence"]


def load_lock(path: str | os.PathLike[str] | None = None) -> ModelsLock:
    """Load the lock from ``path``, ``$FH_MODELS_LOCK``, or the packaged default.

    ``$FH_MODELS_LOCK`` may also name a packaged lock (``models.tiny.lock``).
    """
    path = path or os.environ.get(LOCK_ENV)
    if path:
        p = Path(path)
        if not p.exists() and p.name == str(path):
            text = resources.files(__package__).joinpath(p.name).read_text()
        else:
            text = p.read_text()
    else:
        text = resources.files(__package__).joinpath("models.lock").read_text()
    return ModelsLock(json.loads(text))


def models_dir() -> str | None:
    """Weight cache directory (``$FH_MODELS_DIR``), shared by api and workers."""
    return os.environ.get("FH_MODELS_DIR") or None
