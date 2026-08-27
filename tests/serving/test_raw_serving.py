from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    TimingUpdated,
    UsageUpdated,
)
from exqserve.core.items import RawPromptItem
from exqserve.core.request import RawPromptRequest
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import (
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeRenderedPrompt,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.serving.contracts import RawServingRequest
from exqserve.serving.raw import RawServingEngine


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        self.calls.append(text)
        return RuntimeRenderedPrompt(text, (101, 102, 103))


class _ControlledSession:
    def __init__(self, events: list[object]) -> None:
        self.events = list(events)
        self.cancel_calls = 0
        self.terminal_reason = None

    def __aiter__(self) -> AsyncIterator[object]:  # type: ignore[type-arg]
        return self  # type: ignore[return-value]

    async def __anext__(self):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def cancel(self, reason=None) -> None:  # type: ignore[no-untyped-def]
        self.cancel_calls += 1


class _Controller:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.requests: list[RuntimeGenerationRequest] = []
        self.sessions: list[_ControlledSession] = []

    async def submit(self, request: RuntimeGenerationRequest) -> _ControlledSession:
        self.requests.append(request)
        session = _ControlledSession(list(self.events))
        self.sessions.append(session)
        return session


def _raw(prompt: RawPromptItem) -> RawServingRequest:
    return RawServingRequest(
        RawPromptRequest("req_raw", "m", (prompt,)),
        5,
        ("STOP",),
        7,
    )


def test_raw_serving_text_prompt_tokenizes_directly_and_emits_plain_text_events() -> None:
    async def scenario() -> None:
        usage = TokenUsage(input_tokens=3, output_tokens=2, cached_input_tokens=1)
        timing = RuntimeTiming(queue_seconds=0.1, prefill_seconds=0.2, generation_seconds=0.3)
        tokenizer = _Tokenizer()
        controller = _Controller(
            [
                RuntimeStarted("req_raw"),
                RuntimeTextDelta("req_raw", "hello"),
                RuntimeTextDelta("req_raw", " world"),
                RuntimeFinished("req_raw", RuntimeStopReason.EOS, usage, timing),
            ]
        )
        engine = RawServingEngine(tokenizer, controller)

        session = await engine.submit(_raw(RawPromptItem(text="PROMPT")))
        events: list[GenerationEvent] = [event async for event in session]

        assert tokenizer.calls == ["PROMPT"]
        assert controller.requests == [
            RuntimeGenerationRequest(
                "req_raw",
                (101, 102, 103),
                5,
                7,
                ("STOP",),
            )
        ]
        assert events == [
            GenerationStarted("req_raw"),
            TextStarted("req_raw"),
            TextDelta("req_raw", "hello"),
            TextDelta("req_raw", " world"),
            TextCompleted("req_raw", "hello world"),
            TimingUpdated(
                "req_raw",
                GenerationTiming(queue_seconds=0.1, prefill_seconds=0.2, generation_seconds=0.3),
            ),
            UsageUpdated("req_raw", usage),
            GenerationCompleted("req_raw", CompletionReason.STOP, usage),
        ]
        assert session.compiled_prompt.text == "PROMPT"
        assert session.compiled_prompt.input_ids == (101, 102, 103)
        assert session.compiled_prompt.stop_conditions == ("STOP",)

    asyncio.run(scenario())


def test_raw_serving_token_ids_never_calls_text_tokenizer() -> None:
    async def scenario() -> None:
        tokenizer = _Tokenizer()
        usage = TokenUsage(input_tokens=2, output_tokens=0)
        controller = _Controller(
            [RuntimeStarted("req_raw"), RuntimeFinished("req_raw", RuntimeStopReason.LENGTH, usage, RuntimeTiming())]
        )
        engine = RawServingEngine(tokenizer, controller)

        session = await engine.submit(_raw(RawPromptItem(token_ids=(9, 8))))
        events = [event async for event in session]

        assert tokenizer.calls == []
        assert controller.requests[0].input_ids == (9, 8)
        assert TextStarted("req_raw") in events
        assert TextCompleted("req_raw", "") in events
        assert GenerationCompleted("req_raw", CompletionReason.LENGTH, usage) in events

    asyncio.run(scenario())


def test_raw_serving_close_cancels_unfinished_runtime_once() -> None:
    async def scenario() -> None:
        tokenizer = _Tokenizer()
        controller = _Controller([RuntimeStarted("req_raw"), RuntimeTextDelta("req_raw", "partial")])
        engine = RawServingEngine(tokenizer, controller)
        session = await engine.submit(_raw(RawPromptItem(text="PROMPT")))

        first = await anext(session)
        assert first == GenerationStarted("req_raw")
        await session.cancel()
        await session.cancel()
        assert controller.sessions[0].cancel_calls == 1

    asyncio.run(scenario())
