from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    ReasoningCompleted,
    TextCompleted,
    ToolCallCompleted,
)
from exqserve.core.items import MessageItem, MessageRole, ReasoningItem, ToolCallItem
from exqserve.model.contracts import CompiledPrompt, TemplateRequest
from exqserve.state.session import StatefulServingSession
from exqserve.state.store import InMemoryResponseStore


class _Session:
    def __init__(self, events: list[GenerationEvent]) -> None:
        self._events = iter(events)
        self.cancel_calls = 0
        self.compiled_prompt = CompiledPrompt("p", (1,), "a" * 64, (), TemplateRequest((), (), ()))

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def cancel(self) -> None:
        self.cancel_calls += 1


def test_state_session_persists_completed_outputs_before_terminal_delivery() -> None:
    async def scenario() -> None:
        call = ToolCallItem("c", "lookup", "{}", 0)
        terminal = GenerationCompleted("r", CompletionReason.TOOL_CALLS)
        inner = _Session(
            [
                ReasoningCompleted("r", "think"),
                TextCompleted("r", "answer"),
                ToolCallCompleted("r", call),
                terminal,
            ]
        )
        store = InMemoryResponseStore()
        base = (MessageItem(MessageRole.USER, "old"),)
        current = (MessageItem(MessageRole.USER, "new"),)
        session = StatefulServingSession(
            inner,
            store,
            response_id="resp_1",
            model="m",
            base_context=base,
            current_input=current,
            store_response=True,
        )

        seen: list[GenerationEvent] = []
        async for event in session:
            if event is terminal:
                record = await store.get("resp_1")
                assert record is not None
            seen.append(event)

        assert seen[-1] is terminal
        record = await store.get("resp_1")
        assert record is not None
        assert record.context_items == (
            *base,
            *current,
            ReasoningItem("think"),
            MessageItem(MessageRole.ASSISTANT, "answer"),
            call,
        )

    asyncio.run(scenario())


def test_state_session_does_not_store_when_disabled_failed_or_cancelled() -> None:
    async def collect(events: list[GenerationEvent], response_id: str, enabled: bool) -> None:
        store = InMemoryResponseStore()
        session = StatefulServingSession(
            _Session(events),
            store,
            response_id=response_id,
            model="m",
            base_context=(),
            current_input=(MessageItem(MessageRole.USER, "x"),),
            store_response=enabled,
        )
        _ = [event async for event in session]
        assert await store.get(response_id) is None

    asyncio.run(collect([GenerationCompleted("r", CompletionReason.STOP)], "off", False))
    error = CanonicalError(ErrorCategory.MODEL_FAILURE, "bad", "Bad.", False)
    asyncio.run(collect([GenerationFailed("r", error)], "failed", True))
    asyncio.run(collect([GenerationCancelled("r")], "cancelled", True))


def test_state_session_cancel_delegates_and_does_not_store() -> None:
    async def scenario() -> None:
        inner = _Session([])
        store = InMemoryResponseStore()
        session = StatefulServingSession(
            inner,
            store,
            response_id="resp_cancel",
            model="m",
            base_context=(),
            current_input=(),
            store_response=True,
        )
        await session.cancel()
        await session.cancel()
        assert inner.cancel_calls == 1
        assert await store.get("resp_cancel") is None

    asyncio.run(scenario())
