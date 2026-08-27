from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
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
    UsageUpdated,
)
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import CompiledPrompt, TemplateRequest
from exqserve.runtime.contracts import (
    RuntimeCancelled,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.serving.contracts import ServingRequest
from exqserve.serving.engine import ServingEngine


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool = False


class _Parser:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.feed_calls: list[str] = []
        self.finish_calls = 0

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        self.feed_calls.append(chunk)
        if chunk == "<think>why</think>answer":
            return (
                ReasoningStarted(self.request_id),
                ReasoningDelta(self.request_id, "why"),
                ReasoningCompleted(self.request_id, "why"),
                TextStarted(self.request_id),
                TextDelta(self.request_id, "answer"),
            )
        return ()

    def finish(self) -> _Finish:
        self.finish_calls += 1
        return _Finish((TextCompleted(self.request_id, "answer"),))


class _Compiler:
    def compile(self, request: object, reasoning: object, tool_policy: object) -> CompiledPrompt:
        return CompiledPrompt(
            text="prompt",
            input_ids=(1, 2, 3),
            prompt_hash="a" * 64,
            stop_conditions=("<stop>",),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
        )


class _Controlled:
    def __init__(self, events: list[RuntimeEvent]) -> None:
        self.events = list(events)
        self.terminal_reason: RequestTerminalReason | None = None
        self.cancel_calls: list[RequestTerminalReason] = []

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def cancel(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED,
    ) -> None:
        self.cancel_calls.append(reason)
        if self.terminal_reason is None:
            self.terminal_reason = reason


class _Controller:
    def __init__(self, controlled: _Controlled) -> None:
        self.controlled = controlled
        self.requests: list[RuntimeGenerationRequest] = []

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        self.requests.append(request)
        return self.controlled


def _serving_request(stop_conditions: tuple[str | int, ...] = ()) -> ServingRequest:
    return ServingRequest(
        CanonicalRequest(
            "req",
            "model",
            (MessageItem(MessageRole.USER, "hello"),),
        ),
        ReasoningPolicy(),
        ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True),
        max_output_tokens=16,
        stop_conditions=stop_conditions,
    )


def test_runtime_text_is_parsed_and_stop_completion_emits_usage_then_terminal() -> None:
    async def scenario() -> None:
        usage = TokenUsage(input_tokens=3, output_tokens=4, cached_input_tokens=2)
        controlled = _Controlled(
            [
                RuntimeStarted("req"),
                RuntimeTextDelta("req", "<think>why</think>answer"),
                RuntimeFinished("req", RuntimeStopReason.EOS, usage, RuntimeTiming()),
            ]
        )
        parser = _Parser("req")
        engine = ServingEngine(_Compiler(), lambda request_id, reasoning: parser, _Controller(controlled))

        events = [event async for event in await engine.submit(_serving_request())]

        assert events == [
            GenerationStarted("req"),
            ReasoningStarted("req"),
            ReasoningDelta("req", "why"),
            ReasoningCompleted("req", "why"),
            TextStarted("req"),
            TextDelta("req", "answer"),
            TextCompleted("req", "answer"),
            UsageUpdated("req", usage),
            GenerationCompleted("req", CompletionReason.STOP, usage),
        ]
        assert parser.feed_calls == ["<think>why</think>answer"]
        assert parser.finish_calls == 1

    asyncio.run(scenario())


def test_only_user_requested_stop_sequence_is_exposed_canonically() -> None:
    async def scenario() -> None:
        usage = TokenUsage(input_tokens=3, output_tokens=4)
        internal = _Controlled(
            [
                RuntimeStarted("req"),
                RuntimeFinished(
                    "req",
                    RuntimeStopReason.STOP_STRING,
                    usage,
                    RuntimeTiming(),
                    "<|im_end|>",
                ),
            ]
        )
        internal_engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning: _Parser(request_id),
            _Controller(internal),
        )
        internal_events = [event async for event in await internal_engine.submit(_serving_request(("END",)))]
        assert internal_events[-1] == GenerationCompleted("req", CompletionReason.STOP, usage)

        requested = _Controlled(
            [
                RuntimeStarted("req"),
                RuntimeFinished("req", RuntimeStopReason.STOP_STRING, usage, RuntimeTiming(), "END"),
            ]
        )
        requested_engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning: _Parser(request_id),
            _Controller(requested),
        )
        requested_events = [
            event async for event in await requested_engine.submit(_serving_request(("END",)))
        ]
        assert requested_events[-1] == GenerationCompleted(
            "req",
            CompletionReason.STOP,
            usage,
            "END",
        )

    asyncio.run(scenario())


def test_measured_runtime_timing_is_copied_before_usage_without_estimation() -> None:
    async def scenario() -> None:
        usage = TokenUsage(input_tokens=3, output_tokens=4, cached_input_tokens=2)
        controlled = _Controlled(
            [
                RuntimeStarted("req"),
                RuntimeFinished(
                    "req",
                    RuntimeStopReason.EOS,
                    usage,
                    RuntimeTiming(queue_seconds=0.1, prefill_seconds=0.2, generation_seconds=0.3),
                ),
            ]
        )
        engine = ServingEngine(_Compiler(), lambda request_id, reasoning: _Parser(request_id), _Controller(controlled))

        events = [event async for event in await engine.submit(_serving_request())]

        assert events[-3:] == [
            TimingUpdated("req", GenerationTiming(0.1, 0.2, 0.3)),
            UsageUpdated("req", usage),
            GenerationCompleted("req", CompletionReason.STOP, usage),
        ]

    asyncio.run(scenario())


def test_length_stop_maps_to_length_completion() -> None:
    async def scenario() -> None:
        usage = TokenUsage(input_tokens=3, output_tokens=16)
        controlled = _Controlled(
            [RuntimeStarted("req"), RuntimeFinished("req", RuntimeStopReason.LENGTH, usage, RuntimeTiming())]
        )
        parser = _Parser("req")
        engine = ServingEngine(_Compiler(), lambda request_id, reasoning: parser, _Controller(controlled))

        events = [event async for event in await engine.submit(_serving_request())]

        assert events[-2:] == [
            UsageUpdated("req", usage),
            GenerationCompleted("req", CompletionReason.LENGTH, usage),
        ]

    asyncio.run(scenario())


def test_runtime_failure_closes_parser_channels_then_preserves_error() -> None:
    async def scenario() -> None:
        error = CanonicalError(
            ErrorCategory.RUNTIME_FAILURE,
            "backend_failed",
            "Runtime failed.",
            retryable=False,
        )
        controlled = _Controlled([RuntimeStarted("req"), RuntimeFailed("req", error)])
        parser = _Parser("req")
        engine = ServingEngine(_Compiler(), lambda request_id, reasoning: parser, _Controller(controlled))

        events = [event async for event in await engine.submit(_serving_request())]

        assert events == [
            GenerationStarted("req"),
            TextCompleted("req", "answer"),
            GenerationFailed("req", error),
        ]
        assert parser.finish_calls == 1

    asyncio.run(scenario())


def test_timeout_runtime_cancel_maps_to_safe_timeout_failure() -> None:
    async def scenario() -> None:
        controlled = _Controlled([RuntimeCancelled("req")])
        controlled.terminal_reason = RequestTerminalReason.TIMEOUT
        parser = _Parser("req")
        engine = ServingEngine(_Compiler(), lambda request_id, reasoning: parser, _Controller(controlled))

        events = [event async for event in await engine.submit(_serving_request())]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.category is ErrorCategory.RUNTIME_FAILURE
        assert events[-1].error.code == "request_timeout"
        assert events[-1].error.retryable is True

    asyncio.run(scenario())


def test_ordinary_runtime_cancel_maps_to_generation_cancelled() -> None:
    async def scenario() -> None:
        controlled = _Controlled([RuntimeCancelled("req")])
        controlled.terminal_reason = RequestTerminalReason.CLIENT_CANCELLED
        parser = _Parser("req")
        engine = ServingEngine(_Compiler(), lambda request_id, reasoning: parser, _Controller(controlled))

        events = [event async for event in await engine.submit(_serving_request())]

        assert events[-1] == GenerationCancelled("req")

    asyncio.run(scenario())
