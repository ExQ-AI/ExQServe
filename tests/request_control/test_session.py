from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exqserve.control.request import (
    RequestControlConfig,
    RequestController,
    RequestTerminalReason,
)
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import (
    RuntimeCancelled,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTiming,
)


class _FakeSession:
    def __init__(self, request_id: str, events: list[RuntimeEvent]) -> None:
        self.request_id = request_id
        self.events = list(events)
        self.cancel_calls = 0
        self._cancelled = False

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        await asyncio.sleep(0)
        if self.events:
            return self.events.pop(0)
        if self._cancelled:
            self._cancelled = False
            return RuntimeCancelled(self.request_id)
        raise StopAsyncIteration

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled = True


class _FakeRuntime:
    def __init__(self, factory) -> None:  # type: ignore[no-untyped-def]
        self.factory = factory
        self.sessions: list[_FakeSession] = []

    def submit(self, request: RuntimeGenerationRequest) -> _FakeSession:
        session = self.factory(request)
        self.sessions.append(session)
        return session


class _BlockingSession:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.cancel_calls = 0
        self._cancelled = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        await self._cancelled.wait()
        return RuntimeCancelled(self.request_id)

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled.set()


def _request(request_id: str = "req") -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest(request_id, (1, 2, 3), 4)


def _finished(request_id: str) -> RuntimeFinished:
    return RuntimeFinished(
        request_id,
        RuntimeStopReason.EOS,
        TokenUsage(input_tokens=3, output_tokens=1),
        RuntimeTiming(),
    )


def test_normal_completion_and_runtime_failure_release_capacity_once() -> None:
    async def scenario() -> None:
        def factory(request: RuntimeGenerationRequest) -> _FakeSession:
            if request.request_id == "ok":
                return _FakeSession(request.request_id, [RuntimeStarted(request.request_id), _finished(request.request_id)])
            return _FakeSession(
                request.request_id,
                [
                    RuntimeFailed(
                        request.request_id,
                        CanonicalError(
                            ErrorCategory.RUNTIME_FAILURE,
                            "failed",
                            "Runtime failed.",
                            retryable=False,
                        ),
                    )
                ],
            )

        runtime = _FakeRuntime(factory)
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))

        ok = await controller.submit(_request("ok"))
        assert controller.in_flight == 1
        ok_events = [event async for event in ok]
        assert isinstance(ok_events[-1], RuntimeFinished)
        assert ok.terminal_reason is RequestTerminalReason.COMPLETED
        assert controller.in_flight == 0

        failed = await controller.submit(_request("bad"))
        failed_events = [event async for event in failed]
        assert isinstance(failed_events[-1], RuntimeFailed)
        assert failed.terminal_reason is RequestTerminalReason.RUNTIME_FAILED
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_unexpected_runtime_exhaustion_releases_capacity() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime(lambda request: _FakeSession(request.request_id, []))
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        session = await controller.submit(_request())

        assert [event async for event in session] == []
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_unsolicited_runtime_cancel_sets_runtime_cancelled_reason() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime(
            lambda request: _FakeSession(request.request_id, [RuntimeCancelled(request.request_id)])
        )
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        session = await controller.submit(_request())

        assert [event async for event in session] == [RuntimeCancelled("req")]
        assert session.terminal_reason is RequestTerminalReason.RUNTIME_CANCELLED
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_explicit_client_cancel_is_idempotent_releases_slot_and_preserves_reason() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime(lambda request: _FakeSession(request.request_id, []))
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        session = await controller.submit(_request("first"))
        raw = runtime.sessions[0]

        await session.cancel()
        await session.cancel()

        assert raw.cancel_calls == 1
        assert controller.in_flight == 0
        assert session.terminal_reason is RequestTerminalReason.CLIENT_CANCELLED
        assert await anext(session) == RuntimeCancelled("first")
        assert session.terminal_reason is RequestTerminalReason.CLIENT_CANCELLED

        second = await controller.submit(_request("second"))
        assert controller.in_flight == 1
        await second.cancel()

    asyncio.run(scenario())


def test_application_cancel_is_idempotent_releases_slot_and_preserves_reason() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime(lambda request: _FakeSession(request.request_id, []))
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        session = await controller.submit(_request("semantic"))
        raw = runtime.sessions[0]

        await session.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
        await session.cancel(RequestTerminalReason.APPLICATION_CANCELLED)

        assert raw.cancel_calls == 1
        assert controller.in_flight == 0
        assert session.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED
        assert await anext(session) == RuntimeCancelled("semantic")
        assert session.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED

    asyncio.run(scenario())


def test_async_context_exit_cancels_unfinished_session() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime(lambda request: _FakeSession(request.request_id, []))
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        session = await controller.submit(_request())
        raw = runtime.sessions[0]

        async with session:
            assert controller.in_flight == 1

        assert raw.cancel_calls == 1
        assert session.terminal_reason is RequestTerminalReason.CLIENT_CANCELLED
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_external_consumer_task_cancel_cancels_backend_before_releasing_capacity() -> None:
    async def scenario() -> None:
        blocking = _BlockingSession("external")

        class _BlockingRuntime:
            def submit(self, request: RuntimeGenerationRequest) -> _BlockingSession:
                assert request.request_id == "external"
                return blocking

        controller = RequestController(_BlockingRuntime(), RequestControlConfig(max_in_flight=1))
        session = await controller.submit(_request("external"))
        waiting = asyncio.create_task(anext(session))
        await asyncio.sleep(0)

        waiting.cancel()
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("consumer cancellation must propagate CancelledError")

        assert blocking.cancel_calls == 1
        assert controller.in_flight == 0
        assert session.terminal_reason is RequestTerminalReason.CLIENT_CANCELLED

    asyncio.run(scenario())
