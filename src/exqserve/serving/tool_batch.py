"""Serving-owned Tool Call batch policy and publication transaction state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from exqserve.agent.tools import ToolPolicy
from exqserve.agent.validation import (
    ValidationCode,
    ValidationIssue,
    ValidationResult,
    validate_tool_calls_with_canonical_arguments,
)
from exqserve.core.events import (
    GenerationEvent,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import ToolCallItem
from exqserve.model.contracts import ToolConstraintGuarantee

_MODEL_TOOL_OUTPUT_INVALID_CODES = frozenset(
    {
        ValidationCode.INVALID_JSON,
        ValidationCode.DUPLICATE_JSON_KEY,
        ValidationCode.JSON_VALUE_NOT_OBJECT,
        ValidationCode.SCHEMA_VALIDATION_FAILED,
    }
)

_TOOL_CALL_INVALID_CODES = frozenset(
    {
        *_MODEL_TOOL_OUTPUT_INVALID_CODES,
        ValidationCode.DUPLICATE_TOOL_CALL_ID,
        ValidationCode.INVALID_TOOL_CALL_ORDER,
    }
)

_FORMAT_GUARANTEED_INVALID_CODES = frozenset(
    {
        ValidationCode.INVALID_JSON,
        ValidationCode.DUPLICATE_JSON_KEY,
        ValidationCode.JSON_VALUE_NOT_OBJECT,
    }
)


def tool_validation_failure(result: ValidationResult) -> tuple[str, str]:
    """Map canonical Tool Call validation failures to stable serving error semantics."""
    if result.is_valid:
        raise ValueError("tool validation result must contain at least one issue")
    if any(issue.code in _TOOL_CALL_INVALID_CODES for issue in result.issues):
        return "tool_call_invalid", "Model produced an invalid tool call."
    return "tool_policy_violation", "Model output violated the requested tool policy."


class _BatchLifecycle(Enum):
    OPEN = auto()
    COMMITTED = auto()
    ABORTED = auto()


@dataclass(frozen=True, slots=True)
class BatchFailure:
    code: str
    message: str
    validation_issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchDecision:
    events: tuple[GenerationEvent, ...] = ()
    failure: BatchFailure | None = None


def is_model_tool_output_invalid(
    failure: BatchFailure,
    guarantee: ToolConstraintGuarantee,
) -> bool:
    """Return whether a completed Tool failure is model-output invalid rather than integrity failure."""

    if not isinstance(failure, BatchFailure):
        raise TypeError("failure must be a BatchFailure")
    if not isinstance(guarantee, ToolConstraintGuarantee):
        raise TypeError("guarantee must be a ToolConstraintGuarantee")
    if failure.code != "tool_call_invalid" or not failure.validation_issues:
        return False
    issue_codes = tuple(issue.code for issue in failure.validation_issues)
    if any(code not in _MODEL_TOOL_OUTPUT_INVALID_CODES for code in issue_codes):
        return False
    if guarantee in {ToolConstraintGuarantee.SCHEMA, ToolConstraintGuarantee.UNKNOWN}:
        return False
    if guarantee is ToolConstraintGuarantee.FORMAT:
        return all(code not in _FORMAT_GUARANTEED_INVALID_CODES for code in issue_codes)
    return guarantee is ToolConstraintGuarantee.NONE

def violates_tool_constraint_guarantee(
    failure: BatchFailure,
    guarantee: ToolConstraintGuarantee,
) -> bool:
    """Return whether validation proves an active Tool generation guarantee contradicted itself."""

    if not isinstance(failure, BatchFailure):
        raise TypeError("failure must be a BatchFailure")
    if not isinstance(guarantee, ToolConstraintGuarantee):
        raise TypeError("guarantee must be a ToolConstraintGuarantee")
    if failure.code != "tool_call_invalid" or not failure.validation_issues:
        return False
    issue_codes = tuple(issue.code for issue in failure.validation_issues)
    if any(code not in _MODEL_TOOL_OUTPUT_INVALID_CODES for code in issue_codes):
        return False
    if guarantee is ToolConstraintGuarantee.SCHEMA:
        return True
    if guarantee is ToolConstraintGuarantee.FORMAT:
        return any(code in _FORMAT_GUARANTEED_INVALID_CODES for code in issue_codes)
    return False



class ToolCallBatchGate:
    """Own Tool stream validation, completion staging, atomic buffering, and commit/abort."""

    def __init__(
        self,
        policy: ToolPolicy,
        *,
        tool_call_fanout_limit: int,
        atomic_parallel_tools: bool,
        constrained_parallel_tool_call_limit: int,
    ) -> None:
        if not isinstance(policy, ToolPolicy):
            raise TypeError("policy must be a ToolPolicy")
        if not isinstance(tool_call_fanout_limit, int) or isinstance(tool_call_fanout_limit, bool):
            raise TypeError("tool_call_fanout_limit must be an integer")
        if tool_call_fanout_limit <= 0:
            raise ValueError("tool_call_fanout_limit must be positive")
        if not isinstance(atomic_parallel_tools, bool):
            raise TypeError("atomic_parallel_tools must be a bool")
        if not isinstance(constrained_parallel_tool_call_limit, int) or isinstance(
            constrained_parallel_tool_call_limit, bool
        ):
            raise TypeError("constrained_parallel_tool_call_limit must be an integer")
        if constrained_parallel_tool_call_limit <= 0:
            raise ValueError("constrained_parallel_tool_call_limit must be positive")

        self._policy = policy
        self._atomic_parallel_tools = atomic_parallel_tools
        self._effective_limit = (
            min(tool_call_fanout_limit, constrained_parallel_tool_call_limit)
            if atomic_parallel_tools
            else tool_call_fanout_limit
        )
        self._accepted_call_ids: set[str] = set()
        self._completed_calls: list[ToolCallItem] = []
        self._buffered_events: list[GenerationEvent] = []
        self._lifecycle = _BatchLifecycle.OPEN

    @property
    def completed_calls(self) -> tuple[ToolCallItem, ...]:
        return tuple(self._completed_calls)

    @property
    def has_buffered_events(self) -> bool:
        return bool(self._buffered_events)

    @property
    def buffered_event_count(self) -> int:
        return len(self._buffered_events)

    def _ensure_open(self) -> BatchFailure | None:
        if self._lifecycle is _BatchLifecycle.OPEN:
            return None
        return BatchFailure(
            "tool_call_stream_invalid",
            "Model produced Tool Call events after the Tool Call batch was finalized.",
        )

    def _accept_event(self, event: GenerationEvent) -> BatchDecision:
        if self._atomic_parallel_tools:
            self._buffered_events.append(event)
            return BatchDecision()
        return BatchDecision((event,))

    def on_started(self, event: ToolCallStarted) -> BatchDecision:
        lifecycle_failure = self._ensure_open()
        if lifecycle_failure is not None:
            return BatchDecision(failure=lifecycle_failure)
        if event.call_id in self._accepted_call_ids:
            return BatchDecision(
                failure=BatchFailure(
                    "tool_call_stream_invalid",
                    "Model produced a duplicate tool-call start event.",
                )
            )
        if len(self._accepted_call_ids) >= self._effective_limit:
            return BatchDecision(
                failure=BatchFailure(
                    "tool_policy_violation",
                    "Model output exceeded the server tool-call policy.",
                )
            )
        self._accepted_call_ids.add(event.call_id)
        return self._accept_event(event)

    def on_arguments_delta(self, event: ToolCallArgumentsDelta) -> BatchDecision:
        lifecycle_failure = self._ensure_open()
        if lifecycle_failure is not None:
            return BatchDecision(failure=lifecycle_failure)
        if event.call_id not in self._accepted_call_ids:
            return BatchDecision(
                failure=BatchFailure(
                    "tool_call_stream_invalid",
                    "Model produced tool arguments before an accepted tool call start.",
                )
            )
        return self._accept_event(event)

    def on_completed(self, event: ToolCallCompleted) -> BatchDecision:
        lifecycle_failure = self._ensure_open()
        if lifecycle_failure is not None:
            return BatchDecision(failure=lifecycle_failure)
        if event.call.call_id not in self._accepted_call_ids:
            return BatchDecision(
                failure=BatchFailure(
                    "tool_call_stream_invalid",
                    "Model completed a tool call that was not accepted.",
                )
            )

        candidate_calls = (*self._completed_calls, event.call)
        detailed_validation = validate_tool_calls_with_canonical_arguments(
            candidate_calls,
            self._policy,
        )
        validation = detailed_validation.result
        if not validation.is_valid:
            code, message = tool_validation_failure(validation)
            return BatchDecision(
                failure=BatchFailure(
                    code,
                    message,
                    validation.issues,
                )
            )

        self._completed_calls.append(event.call)
        self._buffered_events.append(event)
        return BatchDecision()

    def commit_events(self) -> tuple[GenerationEvent, ...]:
        if self._lifecycle is not _BatchLifecycle.OPEN:
            return ()
        self._lifecycle = _BatchLifecycle.COMMITTED
        events = tuple(self._buffered_events)
        self._buffered_events.clear()
        return events

    def abort(self) -> None:
        if self._lifecycle is _BatchLifecycle.COMMITTED:
            return
        self._lifecycle = _BatchLifecycle.ABORTED
        self._buffered_events.clear()
