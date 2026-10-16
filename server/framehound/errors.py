"""Errors that carry a stable ``code`` for the UI, user-facing copy, and a technical detail."""

from __future__ import annotations


class FHError(Exception):
    status = 422

    def __init__(self, code: str, message: str, detail: str | None = None,
                 status: int | None = None, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.extra = extra
        if status is not None:
            self.status = status

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message, **self.extra}
        if self.detail:
            d["detail"] = self.detail
        return d


class Cancelled(Exception):
    """Raised inside a job when the user asked to stop it."""
