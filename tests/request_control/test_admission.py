from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from exqserve.control.request import RequestControlConfig, RequestController, RequestRejected
from exqserve.core.errors import ErrorCategory
from exqserve.runtime.contracts import RuntimeEvent, RuntimeGenerationRequest


class _IdleSession:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        async def stream() -> AsyncIterator[RuntimeEvent]:
            if False:
                yield  # pragma: no cover

        return stream()

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _FakeRuntime:
    def __init__(self) -> None:
        self.requests: list[RuntimeGenerationRequest] = []
        self.sessions: list[_IdleSession] = []
        self.fail_submit = False

    def submit(self, request: RuntimeGenerationRequest) -> _IdleSession:
        self.requests.append(request)
        if self.fail_submit:
            raise RuntimeError("backend submit failed")
        session = _IdleSession()
        self.sessions.append(session)
        return session


def _request(prompt: int = 3, output: int = 4) -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest(
        request_id=f"req-{prompt}-{output}",
        input_ids=tuple(range(prompt)),
        max_new_tokens=output,
    )


def test_prompt_output_and_total_limits_reject_before_runtime_submit() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        controller = RequestController(
            runtime,
            RequestControlConfig(
                max_in_flight=2,
                max_prompt_tokens=3,
                max_output_tokens=4,
                max_total_tokens=6,
            ),
        )

        cases = [
            (_request(prompt=4, output=1), ErrorCategory.CONTEXT_LENGTH, "prompt_limit_exceeded"),
            (_request(prompt=1, output=5), ErrorCategory.INVALID_REQUEST, "output_limit_exceeded"),
            (_request(prompt=3, output=4), ErrorCategory.CONTEXT_LENGTH, "total_context_limit_exceeded"),
        ]
        for request, category, code in cases:
            with pytest.raises(RequestRejected) as exc_info:
                await controller.submit(request)
            assert exc_info.value.error.category is category
            assert exc_info.value.error.code == code
            assert exc_info.value.error.retryable is False

        assert runtime.requests == []
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_explicit_32k_output_at_codex_compaction_threshold_is_rejected_by_total_context() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        controller = RequestController(
            runtime,
            RequestControlConfig(
                max_in_flight=1,
                max_output_tokens=32768,
                max_total_tokens=262144,
            ),
        )

        with pytest.raises(RequestRejected) as exc_info:
            await controller.submit(_request(prompt=235929, output=32768))

        assert exc_info.value.error.category is ErrorCategory.CONTEXT_LENGTH
        assert exc_info.value.error.code == "total_context_limit_exceeded"
        assert runtime.requests == []
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_capacity_full_rejects_immediately_without_second_runtime_submit() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))

        await controller.submit(_request())
        assert controller.in_flight == 1
        assert len(runtime.requests) == 1

        with pytest.raises(RequestRejected) as exc_info:
            await controller.submit(_request(prompt=2))

        assert exc_info.value.error.category is ErrorCategory.OVERLOADED
        assert exc_info.value.error.code == "server_overloaded"
        assert exc_info.value.error.retryable is True
        assert len(runtime.requests) == 1
        assert controller.in_flight == 1

    asyncio.run(scenario())


def test_submitted_lease_cannot_be_released_by_preprocessing_owner() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        request = _request()
        lease = await controller.acquire(request.request_id)
        session = await lease.submit(request)

        await lease.release()
        assert controller.in_flight == 1
        assert runtime.sessions[0].cancel_calls == 0

        await session.cancel()
        assert controller.in_flight == 0
        assert runtime.sessions[0].cancel_calls == 1

    asyncio.run(scenario())


def test_controlled_session_setup_failure_cancels_created_runtime_and_releases_capacity() -> None:
    class _BrokenSession(_IdleSession):
        def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
            raise RuntimeError("iterator setup failed")

    class _BrokenRuntime(_FakeRuntime):
        def submit(self, request: RuntimeGenerationRequest) -> _IdleSession:
            self.requests.append(request)
            session = _BrokenSession()
            self.sessions.append(session)
            return session

    async def scenario() -> None:
        runtime = _BrokenRuntime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))

        with pytest.raises(RuntimeError, match="iterator setup failed"):
            await controller.submit(_request())
        assert runtime.sessions[0].cancel_calls == 1
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_runtime_submit_failure_releases_reserved_capacity() -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        runtime.fail_submit = True
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))

        with pytest.raises(RuntimeError, match="backend submit failed"):
            await controller.submit(_request())
        assert controller.in_flight == 0

        runtime.fail_submit = False
        await controller.submit(_request(prompt=2))
        assert controller.in_flight == 1
        assert len(runtime.requests) == 2

    asyncio.run(scenario())
