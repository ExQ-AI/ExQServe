"""Protocol-neutral serving error metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureCause(str, Enum):
    """Stable protocol-neutral failure-origin facts for diagnostics and policy."""

    OUTPUT_EOS = "output_eos"
    OUTPUT_LENGTH = "output_length"
    PARSER_AMBIGUITY_LIMIT = "parser_ambiguity_limit"
    RUNTIME_RECOVERING = "runtime_recovering"
    RESTART_REQUIRED = "restart_required"
    CONSTRAINT_FAILURE = "constraint_failure"
    MODEL_TOOL_OUTPUT_INVALID = "model_tool_output_invalid"


class SemanticCommitClass(str, Enum):
    """Coarse semantic publication state used for replay-safety decisions."""

    NO_SEMANTIC_COMMIT = "no_semantic_commit"
    CONTENT_COMMITTED = "content_committed"
    PARTIAL_TOOL_COMMITTED = "partial_tool_committed"
    TOOL_COMPLETED = "tool_completed"


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
    cause: FailureCause | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.category, ErrorCategory):
            raise TypeError("category must be an ErrorCategory")
        if not self.code.strip():
            raise ValueError("code must not be empty")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a bool")
        if self.cause is not None and not isinstance(self.cause, FailureCause):
            raise TypeError("cause must be a FailureCause or None")


def public_error_code(error: CanonicalError) -> str | None:
    """Return a standards-oriented compatibility code only when normalization is required.

    FailureCause remains the neutral home for detailed failure origin.  Public clients must
    not depend on ExQServe-specific cause-derived machine codes for liveness or retry.
    """

    if not isinstance(error, CanonicalError):
        raise TypeError("error must be a CanonicalError")
    if error.category is ErrorCategory.CONTEXT_LENGTH:
        return "context_length_exceeded"
    return None


def commit_aware_error(error: CanonicalError, commit_class: SemanticCommitClass) -> CanonicalError:
    """Clear generic replay safety after any semantic stream commit."""

    if not isinstance(error, CanonicalError):
        raise TypeError("error must be a CanonicalError")
    if not isinstance(commit_class, SemanticCommitClass):
        raise TypeError("commit_class must be a SemanticCommitClass")
    if commit_class is SemanticCommitClass.NO_SEMANTIC_COMMIT or not error.retryable:
        return error
    return CanonicalError(error.category, error.code, error.message, False, error.cause)
