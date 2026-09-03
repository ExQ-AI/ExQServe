from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from exqserve.control.request import (
    RequestControlConfig,
    RequestController,
    RequestRejected,
    RequestTerminalReason,
)
from exqserve.runtime.contracts import RuntimeCancelled, RuntimeEvent, RuntimeGenerationRequest


class _BlockingSession:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.cancel_calls = 0
        self._cancelled = False

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        if self._cancelled:
            self._cancelled = False
            return RuntimeCancelled(self.request_id)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled = True


class _Runtime:
    def __init__(self) -> None:
        self.sessions: list[_BlockingSession] = []
        self.requests: list[RuntimeGenerationRequest] = []

    def submit(self, request: RuntimeGenerationRequest) -> _BlockingSession:
        self.requests.append(request)
        session = _BlockingSession(request.request_id)
        self.sessions.append(session)
        return session


def _request(index: int) -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest(f"req-{index}", (1,), 2)


def test_controller_close_cancels_all_tracked_sessions_and_rejects_new_work() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=2))
        first = await controller.submit(_request(1))
        second = await controller.submit(_request(2))
        assert controller.in_flight == 2

        await controller.close()
        await controller.close()

        assert [session.cancel_calls for session in runtime.sessions] == [1, 1]
        assert first.terminal_reason is RequestTerminalReason.SERVER_SHUTDOWN
        assert second.terminal_reason is RequestTerminalReason.SERVER_SHUTDOWN
        assert controller.in_flight == 0

        with pytest.raises(RequestRejected) as exc_info:
            await controller.submit(_request(3))
        assert exc_info.value.error.code == "server_shutting_down"
        assert exc_info.value.error.retryable is True
        assert len(runtime.requests) == 2

    asyncio.run(scenario())


def test_concurrent_admission_never_exceeds_capacity_and_never_queues() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=3))

        async def attempt(index: int) -> object:
            try:
                return await controller.submit(_request(index))
            except RequestRejected as exc:
                return exc

        results = await asyncio.gather(*(attempt(index) for index in range(20)))
        accepted = [result for result in results if not isinstance(result, RequestRejected)]
        rejected = [result for result in results if isinstance(result, RequestRejected)]

        assert len(accepted) == 3
        assert len(rejected) == 17
        assert controller.in_flight == 3
        assert len(runtime.requests) == 3
        assert all(result.error.code == "server_overloaded" for result in rejected)

        await controller.close()
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_controller_close_can_claim_model_switch_origin() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        session = await controller.submit(_request(1))

        await controller.close(RequestTerminalReason.MODEL_SWITCH)

        assert runtime.sessions[0].cancel_calls == 1
        assert session.terminal_reason is RequestTerminalReason.MODEL_SWITCH
        assert controller.in_flight == 0

    asyncio.run(scenario())
