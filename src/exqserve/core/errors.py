"""Protocol-neutral serving error metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCategory(str, Enum):
    """Stable semantic categories that external protocols can map independently."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CONTEXT_LENGTH = "context_length"
    OVERLOADED = "overloaded"
    MODEL_FAILURE = "model_failure"
    RUNTIME_FAILURE = "runtime_failure"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class CanonicalError:
    """Safe error information crossing serving-layer boundaries."""

    category: ErrorCategory
    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.category, ErrorCategory):
            raise TypeError("category must be an ErrorCategory")
        if not self.code.strip():
            raise ValueError("code must not be empty")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a bool")
