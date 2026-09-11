"""Serving-engine/session decorators that record protocol-neutral observability."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Protocol, Self

from exqserve.core.errors import CanonicalError
from exqserve.core.events import (
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
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
from exqserve.core.request import CanonicalRequest, RawPromptRequest
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage
from exqserve.observability.capture import CaptureManager, CaptureMode
from exqserve.observability.metrics import MetricsRegistry
from exqserve.serving.contracts import RawServingRequest, ServingRejected, ServingRequest


class ObservedCompiledPromptLike(Protocol):
    @property
    def prompt_hash(self) -> str:
        ...


class ObservedInnerSession(Protocol):
    @property
    def compiled_prompt(self) -> ObservedCompiledPromptLike:
        ...

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        ...

    async def cancel(self) -> None:
        ...


class ObservedInnerEngine(Protocol):
    async def submit(self, request: ServingRequest) -> ObservedInnerSession:
        ...

    async def count_input_tokens(self, request: ServingRequest) -> int:
        ...


class ObservedRawInnerEngine(Protocol):
    async def submit(self, request: RawServingRequest) -> ObservedInnerSession:
        ...


_SEMANTIC_EVENTS = (
    ReasoningStarted,
    ReasoningDelta,
    ReasoningCompleted,
    TextStarted,
    TextDelta,
    TextCompleted,
    ToolCallStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
)


def _enum_value(value: object, default: str) -> str:
    candidate = getattr(value, "value", None)
    return candidate if isinstance(candidate, str) else default


def _attempt_record_payloads(diagnostics: object | None) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    records = getattr(diagnostics, "attempt_records", ())
    if not isinstance(records, tuple):
        return payloads
    for record in records:
        ordinal = getattr(record, "ordinal", None)
        status_value = _enum_value(getattr(record, "status", None), "unknown")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool):
            continue
        error_code = getattr(record, "error_code", None)
        usage = getattr(record, "usage", None)
        timing = getattr(record, "timing", None)
        payloads.append(
            {
                "ordinal": ordinal,
                "status": status_value,
                "error_code": error_code if isinstance(error_code, str) else None,
                "failure_cause": _enum_value(getattr(record, "failure_cause", None), "") or None,
                "discarded": bool(getattr(record, "discarded", False)),
                "usage": None
                if not isinstance(usage, TokenUsage)
                else {
                    "input_tokens": usage.input_tokens,
                    "cached_input_tokens": usage.cached_input_tokens,
                    "output_tokens": usage.output_tokens,
                },
                "timing": None
                if not isinstance(timing, GenerationTiming)
                else {
                    "queue_seconds": timing.queue_seconds,
                    "prefill_seconds": timing.prefill_seconds,
                    "generation_seconds": timing.generation_seconds,
                },
            }
        )
    return payloads


def _execution_metadata(
    request: ServingRequest,
    diagnostics: object | None,
    *,
    publication_state: str,
    error: CanonicalError | None,
    constraint_guarantee: object | None = None,
) -> dict[str, object]:
    sampling = request.sampling
    attempts_started = getattr(diagnostics, "attempts_started", 1)
    recovery_attempts = getattr(diagnostics, "recovery_attempts", 0)
    recovered = getattr(diagnostics, "recovered", False)
    attempt_ordinal = getattr(diagnostics, "attempt_ordinal", attempts_started)
    final_attempt = getattr(diagnostics, "final_attempt", attempt_ordinal)
    skip_reason = getattr(diagnostics, "final_skip_reason", None)
    diagnostic_guarantee = getattr(diagnostics, "constraint_guarantee", None)
    if diagnostic_guarantee is not None:
        constraint_guarantee = diagnostic_guarantee
    return {
        "request_id": request.input.request_id,
        "attempt_count": attempts_started if isinstance(attempts_started, int) else 1,
        "attempt_ordinal": attempt_ordinal if isinstance(attempt_ordinal, int) else 1,
        "visibility_mode": request.visibility_mode.value,
        "publication_state": publication_state,
        "failure_code": None if error is None else error.code,
        "failure_cause": None if error is None or error.cause is None else error.cause.value,
        "constraint_guarantee": _enum_value(constraint_guarantee, "unknown"),
        "recovery_kind": _enum_value(getattr(diagnostics, "recovery_kind", None), "none"),
        "recovery_decision": getattr(diagnostics, "recovery_decision", "not_evaluated"),
        "recovery_skip_reason": _enum_value(skip_reason, "") or None,
        "runtime_state": getattr(diagnostics, "runtime_state", "unknown"),
        "final_attempt": final_attempt if isinstance(final_attempt, int) else 1,
        "seed": request.seed,
        "temperature": None if sampling is None else sampling.temperature,
        "recovery_attempts": recovery_attempts if isinstance(recovery_attempts, int) else 0,
        "recovered": recovered if isinstance(recovered, bool) else False,
        "attempts": _attempt_record_payloads(diagnostics),
    }


def _observe_recovery_metrics(
    metrics: MetricsRegistry,
    diagnostics: object | None,
    error: CanonicalError | None,
    *,
    surface: str,
    status: str,
) -> None:
    if diagnostics is not None:
        recovery_attempts = getattr(diagnostics, "recovery_attempts", 0)
        recovered = getattr(diagnostics, "recovered", False)
        kind = _enum_value(getattr(diagnostics, "recovery_kind", None), "none")
        records = getattr(diagnostics, "attempt_records", ())
        original_cause = "unknown"
        if isinstance(records, tuple):
            for record in records:
                if bool(getattr(record, "discarded", False)):
                    original_cause = _enum_value(
                        getattr(record, "failure_cause", None), "unknown"
                    )
                    break
        if isinstance(recovery_attempts, int) and recovery_attempts > 0:
            metrics.recovery_attempt(kind, original_cause)
            if recovered is True:
                metrics.recovery_success(kind, original_cause)
        skip = _enum_value(getattr(diagnostics, "final_skip_reason", None), "")
        if skip:
            metrics.recovery_skipped(skip)
            if skip == "attempt_budget_exhausted":
                metrics.recovery_exhausted(kind, original_cause)
    if status == "failed" and error is not None and error.cause is None:
        metrics.unclassified_terminal(surface)


class ObservedServingEngine:
    def __init__(
        self,
        engine: ObservedInnerEngine,
        metrics: MetricsRegistry,
        *,
        clock: Callable[[], float] = time.perf_counter,
        capture: CaptureManager | None = None,
    ) -> None:
        self._engine = engine
        self._metrics = metrics
        self._clock = clock
        self._capture = capture

    async def count_input_tokens(self, request: ServingRequest) -> int:
        return await self._engine.count_input_tokens(request)

    async def submit(self, request: ServingRequest) -> ObservedServingSession:
        started_at = self._clock()
        try:
            session = await self._engine.submit(request)
        except ServingRejected as exc:
            diagnostics = exc.execution_diagnostics
            _observe_recovery_metrics(
                self._metrics,
                diagnostics,
                exc.error,
                surface="submission",
                status="rejected",
            )
            self._metrics.request_rejected()
            await self._capture_unaccepted(
                request,
                started_at,
                "rejected",
                exc.error,
                diagnostics,
            )
            raise
        except Exception:
            self._metrics.request_failed_before_start()
            await self._capture_unaccepted(request, started_at, "failed", None, None)
            raise

        self._metrics.request_started()
        return ObservedServingSession(
            session,
            self._metrics,
            request=request.input,
            serving_request=request if isinstance(request, ServingRequest) else None,
            started_at=started_at,
            clock=self._clock,
            capture=self._capture,
        )

    async def _capture_unaccepted(
        self,
        request: ServingRequest,
        started_at: float,
        status: str,
        error: CanonicalError | None,
        diagnostics: object | None,
    ) -> None:
        if self._capture is None or not self._capture.enabled:
            return
        try:
            await self._capture.record_terminal(
                request=request.input,
                prompt_hash=None,
                status=status,
                elapsed_seconds=self._clock() - started_at,
                usage=TokenUsage(),
                timing=GenerationTiming(),
                error=error,
                events=(),
                execution=_execution_metadata(
                    request,
                    diagnostics,
                    publication_state="unpublished",
                    error=error,
                ),
            )
        except Exception:  # noqa: BLE001 - observability must not break serving
            self._metrics.capture_failed()


class ObservedRawServingEngine:
    def __init__(
        self,
        engine: ObservedRawInnerEngine,
        metrics: MetricsRegistry,
        *,
        clock: Callable[[], float] = time.perf_counter,
        capture: CaptureManager | None = None,
    ) -> None:
        self._engine = engine
        self._metrics = metrics
        self._clock = clock
        self._capture = capture

    async def submit(self, request: RawServingRequest) -> ObservedServingSession:
        started_at = self._clock()
        try:
            session = await self._engine.submit(request)
        except ServingRejected as exc:
            self._metrics.request_rejected()
            await self._capture_unaccepted(request.input, started_at, "rejected", exc.error)
            raise
        except Exception:
            self._metrics.request_failed_before_start()
            await self._capture_unaccepted(request.input, started_at, "failed", None)
            raise

        self._metrics.request_started()
        return ObservedServingSession(
            session,
            self._metrics,
            request=request.input,
            serving_request=request if isinstance(request, ServingRequest) else None,
            started_at=started_at,
            clock=self._clock,
            capture=self._capture,
        )

    async def _capture_unaccepted(
        self,
        request: RawPromptRequest,
        started_at: float,
        status: str,
        error: CanonicalError | None,
    ) -> None:
        if self._capture is None or not self._capture.enabled:
            return
        try:
            await self._capture.record_terminal(
                request=request,
                prompt_hash=None,
                status=status,
                elapsed_seconds=self._clock() - started_at,
                usage=TokenUsage(),
                timing=GenerationTiming(),
                error=error,
                events=(),
            )
        except Exception:  # noqa: BLE001 - observability must not break serving
            self._metrics.capture_failed()


class ObservedServingSession:
    def __init__(
        self,
        session: ObservedInnerSession,
        metrics: MetricsRegistry,
        *,
        request: CanonicalRequest | RawPromptRequest,
        started_at: float,
        clock: Callable[[], float],
        capture: CaptureManager | None,
        serving_request: ServingRequest | None = None,
    ) -> None:
        if serving_request is not None and not isinstance(serving_request, ServingRequest):
            raise TypeError("serving_request must be ServingRequest or None")
        self._session = session
        self._iterator = session.__aiter__()
        self._metrics = metrics
        self._request = request
        self._serving_request = serving_request
        self._started_at = started_at
        self._clock = clock
        self._capture = capture
        self._captured_events: list[GenerationEvent] | None = (
            [] if capture is not None and capture.mode is CaptureMode.FULL else None
        )
        if self._captured_events is not None:
            enable_runtime_trace = getattr(session, "enable_runtime_trace", None)
            if callable(enable_runtime_trace):
                enable_runtime_trace()
        self._first_semantic_seen = False
        self._tool_start_seen = False
        self._terminal = False
        self._capture_written = False
        self._terminal_status: str | None = None
        self._terminal_error: CanonicalError | None = None
        self._elapsed_seconds: float | None = None
        self._timing = GenerationTiming()
        self._usage = TokenUsage()

    @property
    def compiled_prompt(self) -> ObservedCompiledPromptLike:
        return self._session.compiled_prompt

    @property
    def input_token_count(self) -> int:
        value = getattr(self._session, "input_token_count", None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError("observed serving session does not expose input token count")
        return value

    def __aiter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self._terminal:
            await self.cancel()
        return False

    async def __anext__(self) -> GenerationEvent:
        try:
            event = await anext(self._iterator)
        except StopAsyncIteration:
            if not self._terminal:
                self._finalize("failed")
                await self._write_capture()
            raise
        except Exception:
            if not self._terminal:
                self._finalize("failed")
                await self._write_capture()
            raise

        if self._captured_events is not None:
            self._captured_events.append(event)
        self._observe(event)
        if self._terminal:
            await self._write_capture()
        return event

    async def cancel(self) -> None:
        if self._terminal:
            return
        try:
            await self._session.cancel()
        except Exception:
            self._finalize("failed")
            await self._write_capture()
            raise
        self._finalize("cancelled")
        await self._write_capture()

    def _observe(self, event: GenerationEvent) -> None:
        semantic_now: float | None = None
        if isinstance(event, _SEMANTIC_EVENTS) and not self._first_semantic_seen:
            semantic_now = self._clock()
            self._first_semantic_seen = True
            self._metrics.observe_ttfe(semantic_now - self._started_at)

        if isinstance(event, ToolCallStarted) and not self._tool_start_seen:
            tool_now = semantic_now if semantic_now is not None else self._clock()
            self._tool_start_seen = True
            self._metrics.observe_tool_start(tool_now - self._started_at)

        if isinstance(event, TimingUpdated):
            self._timing = event.timing
        elif isinstance(event, UsageUpdated):
            self._usage = event.usage
        elif isinstance(event, GenerationCompleted):
            if event.usage is not None:
                self._usage = event.usage
            self._finalize("completed")
        elif isinstance(event, GenerationFailed):
            self._finalize("failed", event.error)
        elif isinstance(event, GenerationCancelled):
            self._finalize("cancelled")

    def _finalize(self, status: str, error: CanonicalError | None = None) -> None:
        if self._terminal:
            return
        self._terminal = True
        self._terminal_status = status
        self._terminal_error = error
        self._elapsed_seconds = self._clock() - self._started_at
        self._metrics.observe_backend(self._timing, self._usage)
        diagnostics = getattr(self._session, "diagnostics", None)
        _observe_recovery_metrics(
            self._metrics,
            diagnostics,
            error,
            surface="serving",
            status=status,
        )
        self._metrics.request_finished(status, self._elapsed_seconds)

    def _execution_capture(self) -> dict[str, object] | None:
        serving = self._serving_request
        if serving is None:
            return None
        diagnostics = getattr(self._session, "diagnostics", None)
        return _execution_metadata(
            serving,
            diagnostics,
            publication_state="published" if self._first_semantic_seen else "unpublished",
            error=self._terminal_error,
            constraint_guarantee=getattr(self._session, "constraint_guarantee", None),
        )

    async def _write_capture(self) -> None:
        if self._capture_written:
            return
        self._capture_written = True
        if self._capture is None or not self._capture.enabled:
            return
        assert self._terminal_status is not None
        assert self._elapsed_seconds is not None
        events = tuple(self._captured_events) if self._captured_events is not None else ()
        runtime_trace = getattr(self._session, "runtime_trace", ())
        if not isinstance(runtime_trace, tuple):
            runtime_trace = ()
        try:
            await self._capture.record_terminal(
                request=self._request,
                prompt_hash=self._session.compiled_prompt.prompt_hash,
                status=self._terminal_status,
                elapsed_seconds=self._elapsed_seconds,
                usage=self._usage,
                timing=self._timing,
                error=self._terminal_error,
                events=events,
                runtime_trace=runtime_trace,
                execution=self._execution_capture(),
            )
        except Exception:  # noqa: BLE001 - capture failure must not break serving
            self._metrics.capture_failed()
