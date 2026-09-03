from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exqserve.control.request import (
    RequestControlConfig,
    RequestController,
    RequestTerminalReason,
)
from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import (
    RuntimeCancelled,
    RuntimeEvent,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStopReason,
    RuntimeTiming,
)


class _BlockingSession:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.release = asyncio.Event()
        self.cancel_calls = 0
        self.next_was_cancelled = False
        self._cancelled = False
        self._terminal_sent = False

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        if self._terminal_sent:
            raise StopAsyncIteration
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.next_was_cancelled = True
            raise
        self._terminal_sent = True
        if self._cancelled:
            return RuntimeCancelled(self.request_id)
        return RuntimeFinished(
            self.request_id,
            RuntimeStopReason.EOS,
            TokenUsage(input_tokens=1, output_tokens=1),
            RuntimeTiming(),
        )

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled = True
        self.release.set()


class _Runtime:
    def __init__(self) -> None:
        self.sessions: list[_BlockingSession] = []

    def submit(self, request: RuntimeGenerationRequest) -> _BlockingSession:
        session = _BlockingSession(request.request_id)
        self.sessions.append(session)
        return session


def _request() -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest("deadline", (1,), 4)


def test_deadline_cancels_backend_without_cancelling_runtime_iterator_task() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(
            runtime,
            RequestControlConfig(max_in_flight=1, timeout_seconds=0.01),
        )
        session = await controller.submit(_request())
        raw = runtime.sessions[0]

        events = [event async for event in session]

        assert events == [RuntimeCancelled("deadline")]
        assert raw.cancel_calls == 1
        assert raw.next_was_cancelled is False
        assert session.terminal_reason is RequestTerminalReason.TIMEOUT
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_external_consumer_cancellation_cleans_deadline_sleep_without_cancelling_runtime_next() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(
            runtime,
            RequestControlConfig(max_in_flight=1, timeout_seconds=3600.0),
        )
        session = await controller.submit(_request())
        raw = runtime.sessions[0]
        consumer = asyncio.create_task(anext(session))
        await asyncio.sleep(0)

        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)

        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert pending == []
        assert raw.cancel_calls == 1
        assert raw.next_was_cancelled is False
        assert session.terminal_reason is RequestTerminalReason.CLIENT_CANCELLED
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_completion_before_deadline_wins_without_cancellation() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(
            runtime,
            RequestControlConfig(max_in_flight=1, timeout_seconds=1.0),
        )
        session = await controller.submit(_request())
        raw = runtime.sessions[0]
        raw.release.set()

        events = [event async for event in session]

        assert isinstance(events[-1], RuntimeFinished)
        assert raw.cancel_calls == 0
        assert session.terminal_reason is RequestTerminalReason.COMPLETED
        assert controller.in_flight == 0

    asyncio.run(scenario())
