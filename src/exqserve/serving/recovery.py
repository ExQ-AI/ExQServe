"""Bounded request-local attempt Recovery policy and wrapper."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Self

from exqserve.core.errors import CanonicalError, ErrorCategory, FailureCause
from exqserve.core.events import (
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    TimingUpdated,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import CompiledPrompt
from exqserve.runtime.contracts import (
    RuntimeCapabilities,
    RuntimeReadinessResult,
    RuntimeSamplingConfig,
)
from exqserve.serving.contracts import ServingRejected, ServingVisibilityMode
from exqserve.serving.terminal import TerminalDecision


class AttemptRecoveryKind(str, Enum):
    NONE = "none"
    REGENERATE_MODEL_OUTPUT = "regenerate_model_output"
    WAIT_RUNTIME_AND_RETRY = "wait_runtime_and_retry"


class PublicationState(str, Enum):
    UNPUBLISHED = "unpublished"
    PUBLISHED = "published"


class RecoverySkipReason(str, Enum):
    DISABLED = "disabled"
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    VISIBILITY_UNKNOWN = "visibility_unknown"
    SEMANTIC_ALREADY_PUBLISHED = "semantic_already_published"
    DETERMINISTIC_MODEL_OUTPUT = "deterministic_model_output"
    RUNTIME_REPLAY_UNSUPPORTED = "runtime_replay_unsupported"
    RUNTIME_READINESS_UNSUPPORTED = "runtime_readiness_unsupported"
    RUNTIME_NOT_READY = "runtime_not_ready"
    ATTACHMENT_REPLAY_UNSUPPORTED = "attachment_replay_unsupported"
    FAILURE_NOT_ALLOWLISTED = "failure_not_allowlisted"
    INJECTION_REPLAY_UNSUPPORTED = "injection_replay_unsupported"


@dataclass(frozen=True, slots=True)
class AttemptRecoveryEvidence:
    """Facts needed to decide whether one failed attempt may be discarded safely."""

    error: CanonicalError
    visibility_mode: ServingVisibilityMode
    publication_state: PublicationState
    attempt_ordinal: int
    max_extra_attempts: int
    seed: int | None = None
    sampling: RuntimeSamplingConfig | None = None
    fresh_attempt_replay: bool = False
    recovery_readiness: bool = False
    has_prompt_attachments: bool = False
    prompt_attachment_replay: bool = False
    request_replay_safe: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        if not isinstance(self.visibility_mode, ServingVisibilityMode):
            raise TypeError("visibility_mode must be a ServingVisibilityMode")
        if not isinstance(self.publication_state, PublicationState):
            raise TypeError("publication_state must be a PublicationState")
        if not isinstance(self.attempt_ordinal, int) or isinstance(self.attempt_ordinal, bool):
            raise TypeError("attempt_ordinal must be an integer")
        if self.attempt_ordinal < 1:
            raise ValueError("attempt_ordinal must be positive")
        if not isinstance(self.max_extra_attempts, int) or isinstance(self.max_extra_attempts, bool):
            raise TypeError("max_extra_attempts must be an integer")
        if self.max_extra_attempts not in {0, 1}:
            raise ValueError("max_extra_attempts must be 0 or 1")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("seed must be an integer or None")
        if self.sampling is not None and not isinstance(self.sampling, RuntimeSamplingConfig):
            raise TypeError("sampling must be RuntimeSamplingConfig or None")
        for name in (
            "fresh_attempt_replay",
            "recovery_readiness",
            "has_prompt_attachments",
            "prompt_attachment_replay",
            "request_replay_safe",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class AttemptRecoveryDecision:
    kind: AttemptRecoveryKind
    skip_reason: RecoverySkipReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AttemptRecoveryKind):
            raise TypeError("kind must be an AttemptRecoveryKind")
        if self.skip_reason is not None and not isinstance(self.skip_reason, RecoverySkipReason):
            raise TypeError("skip_reason must be RecoverySkipReason or None")
        if self.kind is AttemptRecoveryKind.NONE and self.skip_reason is None:
            raise ValueError("NONE recovery decisions require a skip_reason")
        if self.kind is not AttemptRecoveryKind.NONE and self.skip_reason is not None:
            raise ValueError("active recovery decisions must not expose a skip_reason")


class RecoveryAttemptStatus(str, Enum):
    SUBMISSION_FAILED = "submission_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RecoveryAttemptRecord:
    ordinal: int
    status: RecoveryAttemptStatus
    error_code: str | None = None
    failure_cause: FailureCause | None = None
    usage: TokenUsage | None = None
    timing: GenerationTiming | None = None
    discarded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool):
            raise TypeError("ordinal must be an integer")
        if self.ordinal < 1:
            raise ValueError("ordinal must be positive")
        if not isinstance(self.status, RecoveryAttemptStatus):
            raise TypeError("status must be a RecoveryAttemptStatus")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise TypeError("error_code must be a string or None")
        if self.failure_cause is not None and not isinstance(self.failure_cause, FailureCause):
            raise TypeError("failure_cause must be a FailureCause or None")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage or None")
        if self.timing is not None and not isinstance(self.timing, GenerationTiming):
            raise TypeError("timing must be GenerationTiming or None")
        if not isinstance(self.discarded, bool):
            raise TypeError("discarded must be a bool")


@dataclass(frozen=True, slots=True)
class RecoveryDiagnostics:
    attempts_started: int
    recovery_attempts: int
    recovered: bool
    final_skip_reason: RecoverySkipReason | None
    attempt_records: tuple[RecoveryAttemptRecord, ...] = ()
    attempt_ordinal: int = 1
    visibility_mode: ServingVisibilityMode = ServingVisibilityMode.UNKNOWN
    publication_state: PublicationState = PublicationState.UNPUBLISHED
    constraint_guarantee: GenerationGuarantee = GenerationGuarantee.UNKNOWN
    recovery_kind: AttemptRecoveryKind = AttemptRecoveryKind.NONE
    recovery_decision: str = "not_evaluated"
    runtime_state: str = "unknown"
    final_attempt: int = 1


def _attempt_record(
    ordinal: int,
    terminal: GenerationEvent,
    *,
    usage: TokenUsage | None,
    timing: GenerationTiming | None,
    discarded: bool,
) -> RecoveryAttemptRecord | None:
    if isinstance(terminal, GenerationCompleted):
        if terminal.usage is not None:
            usage = terminal.usage
        return RecoveryAttemptRecord(
            ordinal,
            RecoveryAttemptStatus.COMPLETED,
            usage=usage,
            timing=timing,
            discarded=discarded,
        )
    if isinstance(terminal, GenerationFailed):
        return RecoveryAttemptRecord(
            ordinal,
            RecoveryAttemptStatus.FAILED,
            error_code=terminal.error.code,
            failure_cause=terminal.error.cause,
            usage=usage,
            timing=timing,
            discarded=discarded,
        )
    if isinstance(terminal, GenerationCancelled):
        return RecoveryAttemptRecord(
            ordinal,
            RecoveryAttemptStatus.CANCELLED,
            usage=usage,
            timing=timing,
            discarded=discarded,
        )
    return None


class RecoveryAttemptSessionLike(Protocol):
    @property
    def compiled_prompt(self) -> CompiledPrompt:
        ...

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        ...

    async def cancel(self) -> None:
        ...


class RecoveryLeaseLike(Protocol):
    async def release(self) -> None:
        ...


AttemptFactory = Callable[[], Awaitable[RecoveryAttemptSessionLike]]
RuntimeReadinessWaiter = Callable[[float | None], Awaitable[RuntimeReadinessResult]]


def _model_reroll_is_deterministic(evidence: AttemptRecoveryEvidence) -> bool:
    if evidence.seed is not None:
        return True
    return evidence.sampling is not None and evidence.sampling.temperature == 0


def classify_attempt_recovery(evidence: AttemptRecoveryEvidence) -> AttemptRecoveryDecision:
    """Return the frozen V4 disposition from typed attempt facts only."""

    if not isinstance(evidence, AttemptRecoveryEvidence):
        raise TypeError("evidence must be AttemptRecoveryEvidence")
    if evidence.max_extra_attempts == 0:
        return AttemptRecoveryDecision(AttemptRecoveryKind.NONE, RecoverySkipReason.DISABLED)
    if evidence.attempt_ordinal > evidence.max_extra_attempts:
        return AttemptRecoveryDecision(
            AttemptRecoveryKind.NONE,
            RecoverySkipReason.ATTEMPT_BUDGET_EXHAUSTED,
        )
    if evidence.visibility_mode is ServingVisibilityMode.UNKNOWN:
        return AttemptRecoveryDecision(
            AttemptRecoveryKind.NONE,
            RecoverySkipReason.VISIBILITY_UNKNOWN,
        )
    if evidence.publication_state is PublicationState.PUBLISHED:
        return AttemptRecoveryDecision(
            AttemptRecoveryKind.NONE,
            RecoverySkipReason.SEMANTIC_ALREADY_PUBLISHED,
        )
    if evidence.has_prompt_attachments and not evidence.prompt_attachment_replay:
        return AttemptRecoveryDecision(
            AttemptRecoveryKind.NONE,
            RecoverySkipReason.ATTACHMENT_REPLAY_UNSUPPORTED,
        )
    if not evidence.request_replay_safe:
        return AttemptRecoveryDecision(
            AttemptRecoveryKind.NONE,
            RecoverySkipReason.INJECTION_REPLAY_UNSUPPORTED,
        )

    cause = evidence.error.cause
    if cause is FailureCause.RUNTIME_RECOVERING:
        if not evidence.fresh_attempt_replay:
            return AttemptRecoveryDecision(
                AttemptRecoveryKind.NONE,
                RecoverySkipReason.RUNTIME_REPLAY_UNSUPPORTED,
            )
        if not evidence.recovery_readiness:
            return AttemptRecoveryDecision(
                AttemptRecoveryKind.NONE,
                RecoverySkipReason.RUNTIME_READINESS_UNSUPPORTED,
            )
        return AttemptRecoveryDecision(AttemptRecoveryKind.WAIT_RUNTIME_AND_RETRY)

    if not evidence.fresh_attempt_replay:
        return AttemptRecoveryDecision(
            AttemptRecoveryKind.NONE,
            RecoverySkipReason.RUNTIME_REPLAY_UNSUPPORTED,
        )
    if _model_reroll_is_deterministic(evidence):
        return AttemptRecoveryDecision(
            AttemptRecoveryKind.NONE,
            RecoverySkipReason.DETERMINISTIC_MODEL_OUTPUT,
        )

    model_output_allowlisted = (
        cause is FailureCause.OUTPUT_EOS
        and evidence.error.code in {"tool_call_incomplete", "protocol_ambiguity"}
    ) or (
        cause is FailureCause.PARSER_AMBIGUITY_LIMIT
        and evidence.error.code == "protocol_ambiguity"
    ) or (
        cause is FailureCause.MODEL_TOOL_OUTPUT_INVALID
        and evidence.error.code == "tool_call_invalid"
    )
    if model_output_allowlisted:
        return AttemptRecoveryDecision(AttemptRecoveryKind.REGENERATE_MODEL_OUTPUT)
    return AttemptRecoveryDecision(
        AttemptRecoveryKind.NONE,
        RecoverySkipReason.FAILURE_NOT_ALLOWLISTED,
    )


def _is_public_semantic_event(event: GenerationEvent) -> bool:
    return isinstance(
        event,
        (
            TextStarted,
            TextDelta,
            TextCompleted,
            ReasoningStarted,
            ReasoningDelta,
            ReasoningCompleted,
            ToolCallCompleted,
        ),
    )


def _is_staged_tool_event(event: GenerationEvent) -> bool:
    return isinstance(event, (ToolCallStarted, ToolCallArgumentsDelta))


def _is_attempt_diagnostic_event(event: GenerationEvent) -> bool:
    return isinstance(event, (TimingUpdated, UsageUpdated))


def _observe_attempt_measurements(
    event: GenerationEvent,
    usage: TokenUsage | None,
    timing: GenerationTiming | None,
) -> tuple[TokenUsage | None, GenerationTiming | None]:
    if isinstance(event, UsageUpdated):
        usage = event.usage
    elif isinstance(event, TimingUpdated):
        timing = event.timing
    elif isinstance(event, GenerationCompleted) and event.usage is not None:
        usage = event.usage
    return usage, timing


def _drain_staged_for_publication(
    staged: deque[GenerationEvent],
    *,
    tool_validated: bool,
) -> tuple[GenerationEvent, ...]:
    publishable: list[GenerationEvent] = []
    while staged:
        event = staged.popleft()
        if _is_staged_tool_event(event) and not tool_validated:
            continue
        publishable.append(event)
    return tuple(publishable)


class RecoveringServingSession:
    """Outer request transaction that may discard at most one unpublished failed attempt."""

    def __init__(
        self,
        first_attempt: RecoveryAttemptSessionLike,
        *,
        attempt_factory: AttemptFactory,
        lease: RecoveryLeaseLike,
        visibility_mode: ServingVisibilityMode,
        max_extra_attempts: int,
        seed: int | None,
        sampling: RuntimeSamplingConfig | None,
        capabilities: RuntimeCapabilities | None,
        has_prompt_attachments: bool,
        runtime_waiter: RuntimeReadinessWaiter,
        initial_attempt_ordinal: int = 1,
        initial_attempts_started: int = 1,
        initial_recovery_attempts: int = 0,
        initial_attempt_records: tuple[RecoveryAttemptRecord, ...] = (),
    ) -> None:
        if not isinstance(visibility_mode, ServingVisibilityMode):
            raise TypeError("visibility_mode must be a ServingVisibilityMode")
        if max_extra_attempts not in {0, 1}:
            raise ValueError("max_extra_attempts must be 0 or 1")
        for name, value in (
            ("initial_attempt_ordinal", initial_attempt_ordinal),
            ("initial_attempts_started", initial_attempts_started),
            ("initial_recovery_attempts", initial_recovery_attempts),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if initial_attempt_ordinal < 1 or initial_attempts_started < 1:
            raise ValueError("initial attempt ordinal/count must be positive")
        if initial_recovery_attempts > max_extra_attempts:
            raise ValueError("initial recovery attempts exceed max_extra_attempts")
        if not isinstance(initial_attempt_records, tuple):
            raise TypeError("initial_attempt_records must be a tuple")
        if any(not isinstance(record, RecoveryAttemptRecord) for record in initial_attempt_records):
            raise TypeError("initial_attempt_records must contain RecoveryAttemptRecord values")
        self._current_attempt = first_attempt
        self._attempt_factory = attempt_factory
        self._lease = lease
        self._visibility_mode = visibility_mode
        self._max_extra_attempts = max_extra_attempts
        self._seed = seed
        self._sampling = sampling
        self._capabilities = capabilities
        self._has_prompt_attachments = has_prompt_attachments
        self._runtime_waiter = runtime_waiter
        self._attempt_ordinal = initial_attempt_ordinal
        self._attempts_started = initial_attempts_started
        self._recovery_attempts = initial_recovery_attempts
        self._attempt_records = list(initial_attempt_records)
        self._recovered = False
        self._final_skip_reason: RecoverySkipReason | None = None
        self._last_recovery_kind = (
            AttemptRecoveryKind.WAIT_RUNTIME_AND_RETRY
            if initial_recovery_attempts > 0
            else AttemptRecoveryKind.NONE
        )
        self._recovery_decision = "retried" if initial_recovery_attempts > 0 else "not_evaluated"
        self._runtime_state = "ready"
        self._released = False
        self._terminal = False
        self._cancelled = False
        self._trace_enabled = False
        self._published = False
        self._external_started = False
        self._iterator = self._run()

    @property
    def compiled_prompt(self) -> CompiledPrompt:
        return self._current_attempt.compiled_prompt

    @property
    def input_token_count(self) -> int:
        return len(self.compiled_prompt.input_ids)

    @property
    def diagnostics(self) -> RecoveryDiagnostics:
        constraint_guarantee = getattr(
            self._current_attempt,
            "constraint_guarantee",
            GenerationGuarantee.UNKNOWN,
        )
        if not isinstance(constraint_guarantee, GenerationGuarantee):
            constraint_guarantee = GenerationGuarantee.UNKNOWN
        return RecoveryDiagnostics(
            self._attempts_started,
            self._recovery_attempts,
            self._recovered,
            self._final_skip_reason,
            tuple(self._attempt_records),
            attempt_ordinal=self._attempt_ordinal,
            visibility_mode=self._visibility_mode,
            publication_state=(
                PublicationState.PUBLISHED if self._published else PublicationState.UNPUBLISHED
            ),
            constraint_guarantee=constraint_guarantee,
            recovery_kind=self._last_recovery_kind,
            recovery_decision=self._recovery_decision,
            runtime_state=self._runtime_state,
            final_attempt=self._attempt_ordinal,
        )

    @property
    def terminal_decision(self) -> TerminalDecision | None:
        decision = getattr(self._current_attempt, "terminal_decision", None)
        if decision is not None and not isinstance(decision, TerminalDecision):
            raise TypeError("attempt terminal_decision must be TerminalDecision or None")
        return decision

    def enable_runtime_trace(self) -> None:
        self._trace_enabled = True
        enable = getattr(self._current_attempt, "enable_runtime_trace", None)
        if callable(enable):
            enable()

    @property
    def runtime_trace(self) -> tuple[dict[str, object], ...]:
        trace = getattr(self._current_attempt, "runtime_trace", ())
        if not isinstance(trace, tuple):
            return ()
        return tuple(dict(entry) for entry in trace if isinstance(entry, dict))

    def __aiter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self._terminal:
            await self.cancel()
        return False

    async def __anext__(self) -> GenerationEvent:
        return await anext(self._iterator)

    async def _release_once(self) -> None:
        if self._released:
            return
        self._released = True
        await self._lease.release()

    async def cancel(self) -> None:
        if self._terminal or self._cancelled:
            return
        self._cancelled = True
        try:
            await self._current_attempt.cancel()
        finally:
            await self._release_once()

    def _evidence(self, error: CanonicalError) -> AttemptRecoveryEvidence:
        capabilities = self._capabilities
        return AttemptRecoveryEvidence(
            error=error,
            visibility_mode=self._visibility_mode,
            publication_state=(
                PublicationState.PUBLISHED if self._published else PublicationState.UNPUBLISHED
            ),
            attempt_ordinal=self._attempt_ordinal,
            max_extra_attempts=self._max_extra_attempts,
            seed=self._seed,
            sampling=self._sampling,
            fresh_attempt_replay=(
                False if capabilities is None else capabilities.fresh_attempt_replay
            ),
            recovery_readiness=(
                False if capabilities is None else capabilities.recovery_readiness
            ),
            has_prompt_attachments=self._has_prompt_attachments,
            prompt_attachment_replay=(
                False if capabilities is None else capabilities.prompt_attachment_replay
            ),
            request_replay_safe=bool(getattr(self._lease, "replay_safe", True)),
        )

    def _record_terminal_attempt(
        self,
        terminal: GenerationEvent,
        *,
        usage: TokenUsage | None,
        timing: GenerationTiming | None,
        discarded: bool,
    ) -> None:
        if usage is None:
            attempt_usage = getattr(self._current_attempt, "attempt_usage", None)
            if isinstance(attempt_usage, TokenUsage):
                usage = attempt_usage
        if timing is None:
            attempt_timing = getattr(self._current_attempt, "attempt_timing", None)
            if isinstance(attempt_timing, GenerationTiming):
                timing = attempt_timing
        record = _attempt_record(
            self._attempt_ordinal,
            terminal,
            usage=usage,
            timing=timing,
            discarded=discarded,
        )
        if record is not None:
            self._attempt_records.append(record)

    async def _start_retry_or_failure(
        self,
        decision: AttemptRecoveryDecision,
        request_id: str,
    ) -> GenerationFailed | GenerationCancelled | None:
        if decision.kind is AttemptRecoveryKind.NONE:
            if self._recovery_attempts == 0:
                self._last_recovery_kind = AttemptRecoveryKind.NONE
            self._recovery_decision = "skipped"
            self._final_skip_reason = decision.skip_reason
            return None
        self._last_recovery_kind = decision.kind
        self._recovery_decision = "retry_planned"
        if decision.kind is AttemptRecoveryKind.WAIT_RUNTIME_AND_RETRY:
            deadline = getattr(self._lease, "deadline", None)
            readiness = await self._runtime_waiter(
                deadline if isinstance(deadline, float | int) else None
            )
            if self._cancelled or getattr(self._lease, "is_active", True) is False:
                self._final_skip_reason = RecoverySkipReason.RUNTIME_NOT_READY
                return GenerationCancelled(request_id)
            self._runtime_state = readiness.value
            if readiness is RuntimeReadinessResult.CLOSED:
                self._recovery_decision = "runtime_closed"
                self._final_skip_reason = RecoverySkipReason.RUNTIME_NOT_READY
                return GenerationCancelled(request_id)
            if readiness is RuntimeReadinessResult.DEADLINE:
                self._recovery_decision = "deadline"
                self._final_skip_reason = RecoverySkipReason.RUNTIME_NOT_READY
                return GenerationFailed(
                    request_id,
                    CanonicalError(
                        category=ErrorCategory.RUNTIME_FAILURE,
                        code="request_timeout",
                        message="Inference request exceeded its serving deadline.",
                        retryable=True,
                    ),
                )
            if readiness is not RuntimeReadinessResult.READY:
                self._recovery_decision = "runtime_not_ready"
                self._final_skip_reason = RecoverySkipReason.RUNTIME_NOT_READY
                return GenerationFailed(
                    request_id,
                    CanonicalError(
                        category=ErrorCategory.RUNTIME_FAILURE,
                        code="runtime_recovery_unavailable",
                        message="Inference runtime did not become ready before request recovery ended.",
                        retryable=False,
                        cause=FailureCause.RESTART_REQUIRED,
                    ),
                )
        if self._cancelled or getattr(self._lease, "is_active", True) is False:
            self._final_skip_reason = RecoverySkipReason.RUNTIME_NOT_READY
            return GenerationCancelled(request_id)
        next_ordinal = self._attempt_ordinal + 1
        try:
            next_attempt = await self._attempt_factory()
        except ServingRejected as exc:
            if self._cancelled or getattr(self._lease, "is_active", True) is False:
                self._recovery_decision = "cancelled"
                return GenerationCancelled(request_id)
            if exc.attempt_started:
                self._recovery_decision = "retry_submission_failed"
                self._recovery_attempts += 1
                self._attempt_ordinal = next_ordinal
                self._attempts_started += 1
                self._attempt_records.append(
                    RecoveryAttemptRecord(
                        self._attempt_ordinal,
                        RecoveryAttemptStatus.SUBMISSION_FAILED,
                        error_code=exc.error.code,
                        failure_cause=exc.error.cause,
                    )
                )
            else:
                self._recovery_decision = "retry_blocked_before_submit"
            return GenerationFailed(request_id, exc.error)
        self._recovery_decision = "retried"
        self._recovery_attempts += 1
        self._attempt_ordinal = next_ordinal
        self._attempts_started += 1
        self._current_attempt = next_attempt
        if self._cancelled or getattr(self._lease, "is_active", True) is False:
            await self._current_attempt.cancel()
            return GenerationCancelled(request_id)
        if self._trace_enabled:
            enable = getattr(self._current_attempt, "enable_runtime_trace", None)
            if callable(enable):
                enable()
        return None

    async def _run_buffered_attempt(self) -> tuple[list[GenerationEvent], GenerationEvent | None]:
        events = [event async for event in self._current_attempt]
        return events, events[-1] if events else None

    async def _run(self) -> AsyncIterator[GenerationEvent]:
        request_id: str | None = None
        try:
            capabilities = self._capabilities
            recovery_unavailable = (
                self._max_extra_attempts == 0
                or self._visibility_mode is ServingVisibilityMode.UNKNOWN
                or capabilities is None
                or not capabilities.fresh_attempt_replay
                or (
                    self._has_prompt_attachments
                    and not capabilities.prompt_attachment_replay
                )
            )
            if recovery_unavailable:
                unavailable_usage: TokenUsage | None = None
                unavailable_timing: GenerationTiming | None = None
                async for event in self._current_attempt:
                    request_id = getattr(event, "request_id", request_id)
                    unavailable_usage, unavailable_timing = _observe_attempt_measurements(
                        event, unavailable_usage, unavailable_timing
                    )
                    if isinstance(event, GenerationFailed) and self._max_extra_attempts > 0:
                        self._final_skip_reason = classify_attempt_recovery(
                            self._evidence(event.error)
                        ).skip_reason
                    elif isinstance(event, GenerationCompleted) and self._recovery_attempts > 0:
                        self._recovered = True
                    if isinstance(event, (GenerationCompleted, GenerationFailed, GenerationCancelled)):
                        self._record_terminal_attempt(
                            event,
                            usage=unavailable_usage,
                            timing=unavailable_timing,
                            discarded=False,
                        )
                    yield event
                self._terminal = True
                return

            if self._visibility_mode is ServingVisibilityMode.BUFFERED:
                while True:
                    events, terminal = await self._run_buffered_attempt()
                    if events:
                        request_id = getattr(events[0], "request_id", request_id)
                    buffered_usage: TokenUsage | None = None
                    buffered_timing: GenerationTiming | None = None
                    for recorded_event in events:
                        buffered_usage, buffered_timing = _observe_attempt_measurements(
                            recorded_event, buffered_usage, buffered_timing
                        )
                    if isinstance(terminal, GenerationFailed):
                        decision = classify_attempt_recovery(self._evidence(terminal.error))
                        self._record_terminal_attempt(
                            terminal,
                            usage=buffered_usage,
                            timing=buffered_timing,
                            discarded=decision.kind is not AttemptRecoveryKind.NONE,
                        )
                        retry_failure = await self._start_retry_or_failure(
                            decision,
                            terminal.request_id,
                        )
                        if decision.kind is not AttemptRecoveryKind.NONE and retry_failure is None:
                            continue
                        yield retry_failure or terminal
                        self._terminal = True
                        return
                    if isinstance(terminal, GenerationCancelled):
                        self._record_terminal_attempt(
                            terminal,
                            usage=buffered_usage,
                            timing=buffered_timing,
                            discarded=False,
                        )
                        yield terminal
                        self._terminal = True
                        return
                    if isinstance(terminal, GenerationCompleted):
                        self._record_terminal_attempt(
                            terminal,
                            usage=buffered_usage,
                            timing=buffered_timing,
                            discarded=False,
                        )
                        if self._recovery_attempts > 0:
                            self._recovered = True
                    for event in events:
                        yield event
                    self._terminal = True
                    return

            while True:
                staged = deque[GenerationEvent]()
                retry_started = False
                streaming_usage: TokenUsage | None = None
                streaming_timing: GenerationTiming | None = None
                async for event in self._current_attempt:
                    request_id = getattr(event, "request_id", request_id)
                    streaming_usage, streaming_timing = _observe_attempt_measurements(
                        event, streaming_usage, streaming_timing
                    )
                    if isinstance(event, GenerationStarted):
                        if not self._external_started:
                            self._external_started = True
                            yield event
                        continue
                    if _is_staged_tool_event(event) and not self._published:
                        staged.append(event)
                        continue
                    if _is_attempt_diagnostic_event(event) and not self._published:
                        staged.append(event)
                        continue
                    if _is_public_semantic_event(event):
                        for staged_event in _drain_staged_for_publication(
                            staged,
                            tool_validated=isinstance(event, ToolCallCompleted),
                        ):
                            yield staged_event
                        self._published = True
                        yield event
                        continue
                    if isinstance(event, GenerationFailed):
                        decision = classify_attempt_recovery(self._evidence(event.error))
                        self._record_terminal_attempt(
                            event,
                            usage=streaming_usage,
                            timing=streaming_timing,
                            discarded=decision.kind is not AttemptRecoveryKind.NONE,
                        )
                        retry_failure = await self._start_retry_or_failure(
                            decision,
                            event.request_id,
                        )
                        if decision.kind is not AttemptRecoveryKind.NONE and retry_failure is None:
                            staged.clear()
                            retry_started = True
                            break
                        for staged_event in _drain_staged_for_publication(
                            staged,
                            tool_validated=False,
                        ):
                            yield staged_event
                        yield retry_failure or event
                        self._terminal = True
                        return
                    if isinstance(event, (GenerationCompleted, GenerationCancelled)):
                        self._record_terminal_attempt(
                            event,
                            usage=streaming_usage,
                            timing=streaming_timing,
                            discarded=False,
                        )
                        for staged_event in _drain_staged_for_publication(
                            staged,
                            tool_validated=False,
                        ):
                            yield staged_event
                        if isinstance(event, GenerationCompleted) and self._recovery_attempts > 0:
                            self._recovered = True
                        yield event
                        self._terminal = True
                        return
                    yield event
                if retry_started:
                    continue
                self._terminal = True
                return
        finally:
            await self._release_once()
