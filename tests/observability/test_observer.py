from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.errors import CanonicalError, ErrorCategory, FailureCause
from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    TextDelta,
    TextStarted,
    TimingUpdated,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import CompiledPrompt, TemplateRequest
from exqserve.observability.capture import (
    CaptureManager,
    CaptureMode,
    MemoryCaptureSink,
    replay_events,
)
from exqserve.observability.metrics import MetricsRegistry
from exqserve.observability.observer import ObservedServingEngine
from exqserve.runtime.contracts import RuntimeSamplingConfig
from exqserve.serving.contracts import ServingRejected, ServingRequest, ServingVisibilityMode
from exqserve.serving.recovery import (
    AttemptRecoveryKind,
    PublicationState,
    RecoveryAttemptRecord,
    RecoveryAttemptStatus,
    RecoveryDiagnostics,
    RecoverySkipReason,
)


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _Session:
    def __init__(
        self,
        events: list[GenerationEvent],
        runtime_trace: tuple[dict[str, object], ...] = (),
        diagnostics: RecoveryDiagnostics | None = None,
    ) -> None:
        self._events = iter(events)
        self.cancel_calls = 0
        self.runtime_trace = runtime_trace
        self.diagnostics = diagnostics
        self.runtime_trace_enabled = False
        self.compiled_prompt = CompiledPrompt(
            "prompt",
            (1,),
            "a" * 64,
            (),
            TemplateRequest((), (), ()),
        )

    def enable_runtime_trace(self) -> None:
        self.runtime_trace_enabled = True

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Engine:
    def __init__(self, session: _Session | None = None) -> None:
        self.session = session
        self.rejection: CanonicalError | None = None
        self.rejection_diagnostics: RecoveryDiagnostics | None = None

    async def submit(self, request: ServingRequest) -> _Session:
        if self.rejection is not None:
            raise ServingRejected(
                self.rejection,
                execution_diagnostics=self.rejection_diagnostics,
            )
        assert self.session is not None
        return self.session


def _request() -> ServingRequest:
    return ServingRequest(
        CanonicalRequest("r", "m", (MessageItem(MessageRole.USER, "hello"),)),
        ReasoningPolicy(),
        ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True),
        8,
    )


def _metric(text: str, prefix: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_observer_records_semantic_latency_tool_latency_backend_and_terminal_once() -> None:
    async def scenario() -> None:
        usage = TokenUsage(input_tokens=10, cached_input_tokens=6, output_tokens=2)
        session = _Session(
            [
                GenerationStarted("r"),
                TextStarted("r"),
                TextDelta("r", "x"),
                ToolCallStarted("r", "c", "lookup", 0),
                TimingUpdated("r", GenerationTiming(0.1, 0.2, 0.25)),
                UsageUpdated("r", usage),
                GenerationCompleted("r", CompletionReason.TOOL_CALLS, usage),
            ]
        )
        metrics = MetricsRegistry()
        observer = ObservedServingEngine(_Engine(session), metrics, clock=_Clock([0.0, 0.2, 0.4, 1.0]))

        observed = await observer.submit(_request())
        events = [event async for event in observed]

        assert len(events) == 7
        text = metrics.render_text()
        assert _metric(text, "exqserve_active_requests ") == 0.0
        assert _metric(text, 'exqserve_requests_total{status="completed"}') == 1.0
        assert _metric(text, "exqserve_time_to_first_semantic_event_seconds_sum ") == pytest.approx(0.2)
        assert _metric(text, "exqserve_time_to_tool_call_start_seconds_sum ") == pytest.approx(0.4)
        assert _metric(text, "exqserve_request_latency_seconds_sum ") == pytest.approx(1.0)
        assert _metric(text, "exqserve_backend_prefill_seconds_sum ") == pytest.approx(0.2)
        assert _metric(text, "exqserve_input_tokens_total ") == 10.0

    asyncio.run(scenario())


def test_observer_early_cancel_releases_active_request_exactly_once() -> None:
    async def scenario() -> None:
        session = _Session([GenerationStarted("r"), TextStarted("r")])
        metrics = MetricsRegistry()
        observer = ObservedServingEngine(_Engine(session), metrics, clock=_Clock([0.0, 0.1]))
        observed = await observer.submit(_request())

        await observed.cancel()
        await observed.cancel()

        assert session.cancel_calls == 1
        text = metrics.render_text()
        assert _metric(text, "exqserve_active_requests ") == 0.0
        assert _metric(text, 'exqserve_requests_total{status="cancelled"}') == 1.0

    asyncio.run(scenario())


def test_rejected_request_counts_rejected_without_touching_active_gauge() -> None:
    async def scenario() -> None:
        engine = _Engine()
        engine.rejection = CanonicalError(
            ErrorCategory.OVERLOADED,
            "overloaded",
            "Busy.",
            retryable=True,
        )
        metrics = MetricsRegistry()
        observer = ObservedServingEngine(engine, metrics, clock=_Clock([0.0]))

        with pytest.raises(ServingRejected):
            await observer.submit(_request())

        text = metrics.render_text()
        assert _metric(text, "exqserve_active_requests ") == 0.0
        assert _metric(text, 'exqserve_requests_total{status="rejected"}') == 1.0

    asyncio.run(scenario())


def test_observer_full_capture_records_terminal_trace_for_replay() -> None:
    async def scenario() -> None:
        events: list[GenerationEvent] = [
            GenerationStarted("r"),
            TextStarted("r"),
            TextDelta("r", "ok"),
            GenerationCompleted("r", CompletionReason.STOP),
        ]
        runtime_trace = (
            {"type": "text_delta", "text": "raw </think>", "token_ids": [1, 2]},
            {
                "type": "finished",
                "reason": "eos",
                "backend_reason": "stop_token",
                "stop_sequence": None,
                "eos_token_id": 2,
                "eos_token_text": "</think>",
            },
        )
        session = _Session(events, runtime_trace)
        sink = MemoryCaptureSink()
        metrics = MetricsRegistry()
        observer = ObservedServingEngine(
            _Engine(session),
            metrics,
            clock=_Clock([0.0, 0.1, 0.5]),
            capture=CaptureManager(CaptureMode.FULL, sink),
        )

        observed = await observer.submit(_request())
        seen = [event async for event in observed]

        assert seen == events
        assert session.runtime_trace_enabled is True
        assert len(sink.records) == 1
        assert sink.records[0]["status"] == "completed"
        assert sink.records[0]["runtime_trace"] == list(runtime_trace)
        assert replay_events(sink.records[0]) == tuple(events)

    asyncio.run(scenario())


def test_capture_records_recovery_census_inputs_without_full_payload() -> None:
    async def scenario() -> None:
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            seed=17,
            sampling=RuntimeSamplingConfig(temperature=0.7),
            visibility_mode=ServingVisibilityMode.STREAMING,
        )
        sink = MemoryCaptureSink()
        observer = ObservedServingEngine(
            _Engine(_Session([GenerationStarted("r"), GenerationCompleted("r", CompletionReason.STOP)])),
            MetricsRegistry(),
            clock=_Clock([0.0, 0.5]),
            capture=CaptureManager(CaptureMode.METADATA, sink),
        )

        observed = await observer.submit(request)
        _ = [event async for event in observed]

        assert len(sink.records) == 1
        execution = sink.records[0]["execution"]
        assert execution == {
            "request_id": "r",
            "attempt_count": 1,
            "attempt_ordinal": 1,
            "visibility_mode": "streaming",
            "publication_state": "unpublished",
            "failure_code": None,
            "failure_cause": None,
            "constraint_guarantee": "unknown",
            "recovery_kind": "none",
            "recovery_decision": "not_evaluated",
            "recovery_skip_reason": None,
            "runtime_state": "unknown",
            "final_attempt": 1,
            "seed": 17,
            "temperature": 0.7,
            "recovery_attempts": 0,
            "recovered": False,
            "attempts": [],
        }
        assert "request" not in sink.records[0]

    asyncio.run(scenario())


def test_capture_records_per_attempt_recovery_cost_without_discarded_payload() -> None:
    async def scenario() -> None:
        diagnostics = RecoveryDiagnostics(
            attempts_started=2,
            recovery_attempts=1,
            recovered=True,
            final_skip_reason=None,
            attempt_records=(
                RecoveryAttemptRecord(
                    1,
                    RecoveryAttemptStatus.FAILED,
                    error_code="tool_call_incomplete",
                    failure_cause=FailureCause.OUTPUT_EOS,
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                    timing=GenerationTiming(0.1, 0.2, 0.3),
                    discarded=True,
                ),
                RecoveryAttemptRecord(
                    2,
                    RecoveryAttemptStatus.COMPLETED,
                    usage=TokenUsage(input_tokens=10, cached_input_tokens=8, output_tokens=3),
                    timing=GenerationTiming(0.0, 0.05, 0.2),
                ),
            ),
            attempt_ordinal=2,
            visibility_mode=ServingVisibilityMode.BUFFERED,
            publication_state=PublicationState.UNPUBLISHED,
            recovery_kind=AttemptRecoveryKind.REGENERATE_MODEL_OUTPUT,
            recovery_decision="retried",
            runtime_state="ready",
            final_attempt=2,
        )
        sink = MemoryCaptureSink()
        observer = ObservedServingEngine(
            _Engine(
                _Session(
                    [GenerationStarted("r"), GenerationCompleted("r", CompletionReason.STOP)],
                    diagnostics=diagnostics,
                )
            ),
            MetricsRegistry(),
            clock=_Clock([0.0, 0.5]),
            capture=CaptureManager(CaptureMode.METADATA, sink),
        )

        observed = await observer.submit(_request())
        _ = [event async for event in observed]

        execution = sink.records[0]["execution"]
        assert execution["attempt_count"] == 2
        assert execution["recovery_attempts"] == 1
        assert execution["recovered"] is True
        assert execution["attempt_ordinal"] == 2
        assert execution["final_attempt"] == 2
        assert execution["recovery_kind"] == "regenerate_model_output"
        assert execution["recovery_decision"] == "retried"
        assert execution["runtime_state"] == "ready"
        assert execution["attempts"] == [
            {
                "ordinal": 1,
                "status": "failed",
                "error_code": "tool_call_incomplete",
                "failure_cause": "output_eos",
                "discarded": True,
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": None,
                    "output_tokens": 5,
                },
                "timing": {
                    "queue_seconds": 0.1,
                    "prefill_seconds": 0.2,
                    "generation_seconds": 0.3,
                },
            },
            {
                "ordinal": 2,
                "status": "completed",
                "error_code": None,
                "failure_cause": None,
                "discarded": False,
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 8,
                    "output_tokens": 3,
                },
                "timing": {
                    "queue_seconds": 0.0,
                    "prefill_seconds": 0.05,
                    "generation_seconds": 0.2,
                },
            },
        ]
        assert "events" not in sink.records[0]

    asyncio.run(scenario())


def test_rejected_pre_session_recovery_preserves_execution_census_and_metrics() -> None:
    async def scenario() -> None:
        error = CanonicalError(
            ErrorCategory.RUNTIME_FAILURE,
            "restart_required",
            "restart",
            False,
            FailureCause.RESTART_REQUIRED,
        )
        diagnostics = RecoveryDiagnostics(
            attempts_started=2,
            recovery_attempts=1,
            recovered=False,
            final_skip_reason=RecoverySkipReason.ATTEMPT_BUDGET_EXHAUSTED,
            attempt_records=(
                RecoveryAttemptRecord(
                    1,
                    RecoveryAttemptStatus.SUBMISSION_FAILED,
                    error_code="runtime_recovering",
                    failure_cause=FailureCause.RUNTIME_RECOVERING,
                    discarded=True,
                ),
                RecoveryAttemptRecord(
                    2,
                    RecoveryAttemptStatus.SUBMISSION_FAILED,
                    error_code="restart_required",
                    failure_cause=FailureCause.RESTART_REQUIRED,
                ),
            ),
            attempt_ordinal=2,
            visibility_mode=ServingVisibilityMode.BUFFERED,
            publication_state=PublicationState.UNPUBLISHED,
            recovery_kind=AttemptRecoveryKind.WAIT_RUNTIME_AND_RETRY,
            recovery_decision="retry_submission_failed",
            runtime_state="restart_required",
            final_attempt=2,
        )
        engine = _Engine()
        engine.rejection = error
        engine.rejection_diagnostics = diagnostics
        sink = MemoryCaptureSink()
        metrics = MetricsRegistry()
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            visibility_mode=ServingVisibilityMode.BUFFERED,
        )
        observer = ObservedServingEngine(
            engine,
            metrics,
            clock=_Clock([0.0, 0.25]),
            capture=CaptureManager(CaptureMode.METADATA, sink),
        )

        with pytest.raises(ServingRejected):
            await observer.submit(request)

        execution = sink.records[0]["execution"]
        assert execution["attempt_count"] == 2
        assert execution["attempt_ordinal"] == 2
        assert execution["final_attempt"] == 2
        assert execution["recovery_attempts"] == 1
        assert execution["recovery_kind"] == "wait_runtime_and_retry"
        assert execution["recovery_decision"] == "retry_submission_failed"
        assert execution["runtime_state"] == "restart_required"
        assert execution["failure_code"] == "restart_required"
        assert execution["failure_cause"] == "restart_required"
        assert len(execution["attempts"]) == 2
        text = metrics.render_text()
        assert _metric(
            text,
            "exqserve_recovery_attempts_total{cause=\"runtime_recovering\",kind=\"wait_runtime_and_retry\"}",
        ) == 1.0
        assert _metric(
            text,
            "exqserve_recovery_exhausted_total{cause=\"runtime_recovering\",kind=\"wait_runtime_and_retry\"}",
        ) == 1.0

    asyncio.run(scenario())


def test_unknown_serving_terminal_increments_unclassified_metric() -> None:
    async def scenario() -> None:
        error = CanonicalError(
            ErrorCategory.MODEL_FAILURE,
            "unknown_model_failure",
            "unknown",
            False,
        )
        session = _Session([GenerationStarted("r"), GenerationFailed("r", error)])
        metrics = MetricsRegistry()
        observer = ObservedServingEngine(
            _Engine(session),
            metrics,
            clock=_Clock([0.0, 0.5]),
        )

        observed = await observer.submit(_request())
        _ = [event async for event in observed]

        assert _metric(
            metrics.render_text(),
            "exqserve_unclassified_terminal_total{surface=\"serving\"}",
        ) == 1.0

    asyncio.run(scenario())


def test_capture_sink_failure_is_counted_without_breaking_serving_stream() -> None:
    class BrokenSink:
        async def write(self, record: dict[str, object]) -> None:
            raise OSError("disk unavailable")

    async def scenario() -> None:
        events: list[GenerationEvent] = [
            GenerationStarted("r"),
            TextStarted("r"),
            GenerationCompleted("r", CompletionReason.STOP),
        ]
        metrics = MetricsRegistry()
        observer = ObservedServingEngine(
            _Engine(_Session(events)),
            metrics,
            clock=_Clock([0.0, 0.1, 0.5]),
            capture=CaptureManager(CaptureMode.METADATA, BrokenSink()),
        )

        observed = await observer.submit(_request())
        assert [event async for event in observed] == events
        assert _metric(metrics.render_text(), "exqserve_capture_failures_total ") == 1.0

    asyncio.run(scenario())
