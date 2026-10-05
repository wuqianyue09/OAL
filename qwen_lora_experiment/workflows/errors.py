"""Shared exceptions and failure serialization for pilot workflows."""

from __future__ import annotations
import traceback


class OrchestrationError(RuntimeError):
    """Raised when durable pilot orchestration cannot safely proceed."""


class PreflightFailure(OrchestrationError):
    """Raised after an unsuccessful read-only preflight is persisted."""


def _exception_record(error: BaseException) -> dict[str, str]:
    """Serialize the root exception with its real, JSON-safe traceback."""
    return {
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }
