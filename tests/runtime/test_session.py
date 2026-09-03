from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace

from exqserve.core.errors import ErrorCategory, FailureCause
from exqserve.runtime.contracts import (
    RuntimeCancelled,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
    RuntimeTextDelta,
)
from exqserve.runtime.exllamav3 import RuntimeSession


class _FakeJob:
    def __init__(self, results: list[Mapping[str, object] | BaseException]) -> None:
        self.results = list(results)
        self.cancel_calls = 0
        self.injected: list[str] = []
        self.job = SimpleNamespace(is_finished=False)
        self.generator = SimpleNamespace(error=None)

    def __aiter__(self) -> _FakeJob:
        return self

    async def __anext__(self) -> Mapping[str, object]:
        if not self.results:
            raise StopAsyncIteration
        value = self.results.pop(0)
        if isinstance(value, BaseException):
            raise value
        await asyncio.sleep(0)
        return value

    def constrain_output_now(self, output: str) -> None:
        self.injected.append(output)

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _GeneratorStyleJob:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def __aiter__(self):  # type: ignore[no-untyped-def]
        async def stream():  # type: ignore[no-untyped-def]
            yield {"stage": "started", "eos": False}
            yield {
                "stage": "streaming",
                "eos": True,
                "prompt_tokens": 3,
                "new_tokens": 1,
                "cached_tokens": 0,
            }

        return stream()

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _BlockingJob:
    def __init__(self) -> None:
        self.cancel_calls = 0
        self.release = asyncio.Event()

    def __aiter__(self) -> _BlockingJob:
        return self

    async def __anext__(self) -> Mapping[str, object]:
        await self.release.wait()
        return {
            "stage": "streaming",
            "text": "too late",
            "eos": True,
            "prompt_tokens": 3,
            "new_tokens": 2,
            "cached_tokens": 0,
        }

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self.release.set()


def _request() -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest("req-1", (1, 2, 3), 8)


async def _collect(session: RuntimeSession) -> list[object]:
    return [event async for event in session]


def test_session_accepts_exllama_style_async_iterable_without_direct_anext() -> None:
    async def scenario() -> None:
        events = await _collect(RuntimeSession(_request(), _GeneratorStyleJob()))
        assert isinstance(events[0], RuntimeStarted)
        assert isinstance(events[1], RuntimeFinished)
        assert len(events) == 2

    asyncio.run(scenario())


def test_session_flattens_backend_results_in_order_and_stops_after_finished() -> None:
    async def scenario() -> None:
        job = _FakeJob(
            [
                {"stage": "started", "eos": False},
                {"stage": "streaming", "text": "hel", "eos": False},
                {
                    "stage": "streaming",
                    "text": "lo",
                    "eos": True,
                    "eos_reason": "stop_string",
                    "prompt_tokens": 3,
                    "new_tokens": 2,
                    "cached_tokens": 1,
                },
                {"stage": "streaming", "text": "must not appear", "eos": False},
            ]
        )
        events = await _collect(RuntimeSession(_request(), job))

        assert isinstance(events[0], RuntimeStarted)
        assert events[1] == RuntimeTextDelta("req-1", "hel")
        assert events[2] == RuntimeTextDelta("req-1", "lo")
        assert isinstance(events[3], RuntimeFinished)
        assert len(events) == 4

    asyncio.run(scenario())


def test_cancel_is_idempotent_and_emits_cancelled_not_finished() -> None:
    async def scenario() -> None:
        job = _BlockingJob()
        session = RuntimeSession(_request(), job)
        waiting = asyncio.create_task(anext(session))
        await asyncio.sleep(0)

        await session.cancel()
        await session.cancel()

        event = await asyncio.wait_for(waiting, timeout=1)
        assert event == RuntimeCancelled("req-1")
        assert job.cancel_calls == 1

        try:
            await anext(session)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("cancelled session emitted another event")

    asyncio.run(scenario())


def test_cancel_before_iteration_still_cancels_backend_once() -> None:
    async def scenario() -> None:
        job = _FakeJob([])
        session = RuntimeSession(_request(), job)
        await session.cancel()
        events = await _collect(session)
        assert events == [RuntimeCancelled("req-1")]
        assert job.cancel_calls == 1

    asyncio.run(scenario())


def test_external_anext_task_cancel_cleans_internal_waiters_and_backend() -> None:
    async def scenario() -> None:
        job = _BlockingJob()
        session = RuntimeSession(_request(), job)
        waiting = asyncio.create_task(anext(session))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        waiting.cancel()
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("external cancellation must propagate CancelledError")

        await asyncio.sleep(0)
        assert job.cancel_calls == 1
        current = asyncio.current_task()
        owned = [
            task
            for task in asyncio.all_tasks()
            if task is not current
            and not task.done()
            and getattr(task.get_coro(), "__qualname__", "")
            in {"RuntimeSession._next_backend_result", "Event.wait"}
        ]
        assert owned == []

    asyncio.run(scenario())


def test_backend_exception_becomes_safe_runtime_failure_and_marks_backend_unhealthy() -> None:
    async def scenario() -> None:
        job = _FakeJob([RuntimeError("secret backend detail /local/path")])
        failures: list[bool] = []
        events = await _collect(RuntimeSession(_request(), job, lambda: failures.append(True)))

        assert len(events) == 1
        failure = events[0]
        assert isinstance(failure, RuntimeFailed)
        assert failure.error.category is ErrorCategory.RUNTIME_FAILURE
        assert failure.error.code == "generation_failed"
        assert "secret backend detail" not in failure.error.message
        assert failures == [True]

    asyncio.run(scenario())



def test_backend_failure_callback_projects_transient_recovery_cause() -> None:
    async def scenario() -> None:
        job = _FakeJob([RuntimeError("backend exploded")])
        events = await _collect(
            RuntimeSession(_request(), job, lambda: FailureCause.RUNTIME_RECOVERING)
        )

        assert len(events) == 1
        failure = events[0]
        assert isinstance(failure, RuntimeFailed)
        assert failure.error.code == "generation_failed"
        assert failure.error.retryable is True
        assert failure.error.cause is FailureCause.RUNTIME_RECOVERING

    asyncio.run(scenario())

def test_unexpected_backend_exhaustion_is_failure_not_silent_success() -> None:
    async def scenario() -> None:
        events = await _collect(RuntimeSession(_request(), _FakeJob([])))
        assert len(events) == 1
        failure = events[0]
        assert isinstance(failure, RuntimeFailed)
        assert failure.error.code == "backend_ended_early"

    asyncio.run(scenario())


def test_midstream_text_injection_forwards_to_active_backend_job_only() -> None:
    async def scenario() -> None:
        job = _FakeJob(
            [
                {"stage": "started", "eos": False},
                {
                    "stage": "streaming",
                    "text": "done",
                    "eos": True,
                    "prompt_tokens": 3,
                    "new_tokens": 1,
                    "cached_tokens": 0,
                },
            ]
        )
        session = RuntimeSession(_request(), job)

        session.inject_text("\nSTEER")
        session.inject_text(" ")
        assert job.injected == ["\nSTEER", " "]

        await _collect(session)
        try:
            session.inject_text("too late")
        except RuntimeError as exc:
            assert "no longer active" in str(exc)
        else:
            raise AssertionError("terminal generation accepted text injection")

    asyncio.run(scenario())


def test_midstream_injection_rejects_backend_eos_before_consumer_reads_terminal_event() -> None:
    job = _FakeJob([])
    session = RuntimeSession(_request(), job)
    job.job.is_finished = True

    try:
        session.inject_text("too late")
    except RuntimeError as exc:
        assert "no longer active" in str(exc)
    else:
        raise AssertionError("finished backend job accepted text injection")

    assert job.injected == []
