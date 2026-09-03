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
            started_at=started_at,
            clock=self._clock,
            capture=self._capture,
        )

    async def _capture_unaccepted(
        self,
        request: CanonicalRequest,
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
    ) -> None:
        self._session = session
        self._iterator = session.__aiter__()
        self._metrics = metrics
        self._request = request
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
        self._metrics.request_finished(status, self._elapsed_seconds)

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
            )
        except Exception:  # noqa: BLE001 - capture failure must not break serving
            self._metrics.capture_failed()
