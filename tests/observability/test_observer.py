from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
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
from exqserve.serving.contracts import ServingRejected, ServingRequest


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
    ) -> None:
        self._events = iter(events)
        self.cancel_calls = 0
        self.runtime_trace = runtime_trace
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

    async def submit(self, request: ServingRequest) -> _Session:
        if self.rejection is not None:
            raise ServingRejected(self.rejection)
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
