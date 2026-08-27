"""State-aware serving-session wrapper over canonical generation events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, Self

from exqserve.core.events import (
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    ReasoningCompleted,
    TextCompleted,
    ToolCallCompleted,
)
from exqserve.core.items import CanonicalItem, MessageItem, MessageRole, ReasoningItem
from exqserve.state.store import ResponseRecord, ResponseStore


class StatefulInnerSession(Protocol):
    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        ...

    async def cancel(self) -> None:
        ...


class StatefulServingSession:
    """Persist only successfully completed canonical output for later response continuation."""

    def __init__(
        self,
        session: StatefulInnerSession,
        store: ResponseStore,
        *,
        response_id: str,
        model: str,
        base_context: tuple[CanonicalItem, ...],
        current_input: tuple[CanonicalItem, ...],
        store_response: bool,
    ) -> None:
        if not response_id.strip():
            raise ValueError("response_id must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if not isinstance(store_response, bool):
            raise TypeError("store_response must be a bool")
        self._session = session
        self._iterator = session.__aiter__()
        self._store = store
        self._response_id = response_id
        self._model = model
        self._base_context = base_context
        self._current_input = current_input
        self._store_response = store_response
        self._outputs: list[CanonicalItem] = []
        self._terminal = False
        self._cancelled = False

    def __aiter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self._terminal:
            await self.cancel()
        return False

    async def __anext__(self) -> GenerationEvent:
        event = await anext(self._iterator)
        if isinstance(event, ReasoningCompleted):
            self._outputs.append(ReasoningItem(event.text))
        elif isinstance(event, TextCompleted):
            self._outputs.append(MessageItem(MessageRole.ASSISTANT, event.text))
        elif isinstance(event, ToolCallCompleted):
            self._outputs.append(event.call)
        elif isinstance(event, GenerationCompleted):
            if self._store_response:
                await self._store.put(
                    ResponseRecord(
                        self._response_id,
                        self._model,
                        (*self._base_context, *self._current_input, *self._outputs),
                    )
                )
            self._terminal = True
        elif isinstance(event, GenerationFailed | GenerationCancelled):
            self._terminal = True
        return event

    async def cancel(self) -> None:
        if self._terminal or self._cancelled:
            return
        self._cancelled = True
        await self._session.cancel()
