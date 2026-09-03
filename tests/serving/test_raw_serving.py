from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import cast

import pytest

from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
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
from exqserve.model.contracts import PromptCompilerLike
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
from exqserve.serving.preprocessing import RendererLane, RendererLanePool
from exqserve.serving.raw import RawServingEngine
from exqserve.serving.terminal import TerminalPrimaryOwner


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

    async def acquire(self, request_id: str):  # type: ignore[no-untyped-def]
        del request_id
        return self

    async def release(self) -> None:
        return None

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


def test_raw_text_tokenization_uses_shared_pool_off_event_loop() -> None:
    async def scenario() -> None:
        main_thread = threading.get_ident()
        started = threading.Event()
        release = threading.Event()
        worker_threads: list[int] = []

        class BlockingTokenizer(_Tokenizer):
            def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
                worker_threads.append(threading.get_ident())
                started.set()
                release.wait(timeout=2)
                return super().tokenize_text(text)

        tokenizer = BlockingTokenizer()
        lane = RendererLane(tokenizer, cast(PromptCompilerLike, object()))
        pool = RendererLanePool((lane,))
        controller = _Controller([])
        engine = RawServingEngine(None, controller, preprocessing_pool=pool)

        submit = asyncio.create_task(engine.submit(_raw(RawPromptItem(text="PROMPT"))))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        assert started.is_set()
        assert not submit.done()
        assert worker_threads == [worker_threads[0]]
        assert worker_threads[0] != main_thread

        # The event loop remains responsive while the worker thread is blocked.
        progressed = False
        await asyncio.sleep(0)
        progressed = True
        assert progressed
        release.set()
        session = await submit
        assert session.compiled_prompt.input_ids == (101, 102, 103)

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


def test_raw_serving_resolves_auto_output_limit_after_tokenization() -> None:
    async def scenario() -> None:
        tokenizer = _Tokenizer()
        controller = _Controller([])
        engine = RawServingEngine(
            tokenizer,
            controller,
            output_limit_resolver=lambda prompt_tokens, requested: 96 - prompt_tokens
            if requested is None
            else requested,
        )
        request = RawServingRequest(
            RawPromptRequest("req_raw", "m", (RawPromptItem(text="PROMPT"),)),
            None,
        )

        await engine.submit(request)

        assert controller.requests[0].max_new_tokens == 93

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


def test_raw_serving_b1_loop_and_other_fail_closed() -> None:
    async def run(reason: RuntimeStopReason) -> tuple[list[GenerationEvent], object]:
        usage = TokenUsage(input_tokens=2, output_tokens=1)
        controller = _Controller(
            [RuntimeFinished("req_raw", reason, usage, RuntimeTiming())]
        )
        session = await RawServingEngine(_Tokenizer(), controller).submit(
            _raw(RawPromptItem(text="PROMPT"))
        )
        return [event async for event in session], session

    loop_events, loop_session = asyncio.run(run(RuntimeStopReason.LOOP))
    assert isinstance(loop_events[-1], GenerationFailed)
    assert loop_events[-1].error.code == "runtime_loop_detected"
    assert loop_session.terminal_decision is not None
    assert loop_session.terminal_decision.primary_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP

    other_events, other_session = asyncio.run(run(RuntimeStopReason.OTHER))
    assert isinstance(other_events[-1], GenerationFailed)
    assert other_events[-1].error.code == "runtime_terminal_unknown"
    assert other_session.terminal_decision is not None
    assert other_session.terminal_decision.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL


def test_raw_serving_b1_authority_commit_fault_rolls_back_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        usage = TokenUsage(input_tokens=2, output_tokens=1)
        controller = _Controller(
            [RuntimeFinished("req_raw", RuntimeStopReason.EOS, usage, RuntimeTiming())]
        )
        session = await RawServingEngine(_Tokenizer(), controller).submit(
            _raw(RawPromptItem(text="PROMPT"))
        )
        evidence = session._terminal_evidence
        original = evidence.commit_decision
        first = True

        def fail_once(decision):
            nonlocal first
            if first:
                first = False
                raise RuntimeError("authority commit fault")
            return original(decision)

        monkeypatch.setattr(evidence, "commit_decision", fail_once)
        events = [event async for event in session]

        assert not any(isinstance(event, GenerationCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "terminal_processing_failed"
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL

    asyncio.run(scenario())
