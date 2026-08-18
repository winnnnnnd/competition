from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class PipelineError(RuntimeError):
    message: str
    kind: ErrorKind = ErrorKind.PERMANENT
    stage: str = "unknown"

    def __str__(self) -> str:
        return self.message


def classify_error(exc: BaseException) -> ErrorKind:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if "cancel" in name or "cancel" in message:
        return ErrorKind.CANCELLED
    if "timeout" in name or "timed out" in message:
        return ErrorKind.TIMEOUT
    transient_tokens = ("connection", "temporarily", "unavailable", "reset", "broken pipe", "503", "429")
    if any(token in name or token in message for token in transient_tokens):
        return ErrorKind.TRANSIENT
    return ErrorKind.PERMANENT
