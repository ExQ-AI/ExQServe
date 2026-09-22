from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from starlette.requests import Request

from exqserve.protocol.disconnect import race_nonstream_disconnect


class _Session:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.terminal = False
        self.cancel_calls = 0

    async def _events(self) -> AsyncIterator[str]:
        self.started.set()
        await self.release.wait()
        if self.terminal:
            yield "terminal"

    def __aiter__(self) -> AsyncIterator[str]:
        return self._events()

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self.release.set()


async def _consume(session: _Session) -> str:
    terminal = False
    try:
        async for event in session:
            terminal = terminal or event == "terminal"
        return "ok"
    finally:
        if not terminal:
            await session.cancel()


def _request(receive):  # type: ignore[no-untyped-def]
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/test",
            "raw_path": b"/v1/test",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        },
        receive,
    )


def test_nonstream_disconnect_before_first_generation_event_cancels_once() -> None:
    async def scenario() -> None:
        session = _Session()

        async def receive() -> dict[str, object]:
            return {"type": "http.disconnect"}

        with pytest.raises(asyncio.CancelledError):
            await race_nonstream_disconnect(_request(receive), _consume(session))
        assert session.started.is_set()
        assert session.cancel_calls == 1

    asyncio.run(scenario())


def test_nonstream_disconnect_during_generation_cancels_once() -> None:
    async def scenario() -> None:
        session = _Session()
        disconnect = asyncio.Event()

        async def receive() -> dict[str, object]:
            await disconnect.wait()
            return {"type": "http.disconnect"}

        task = asyncio.create_task(race_nonstream_disconnect(_request(receive), _consume(session)))
        await session.started.wait()
        disconnect.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.cancel_calls == 1

    asyncio.run(scenario())


def test_nonstream_normal_completion_wins_and_does_not_cancel() -> None:
    async def scenario() -> None:
        session = _Session()
        never = asyncio.Event()

        async def receive() -> dict[str, object]:
            await never.wait()
            return {"type": "http.disconnect"}

        task = asyncio.create_task(race_nonstream_disconnect(_request(receive), _consume(session)))
        await session.started.wait()
        session.terminal = True
        session.release.set()
        assert await task == "ok"
        assert session.cancel_calls == 0

    asyncio.run(scenario())


def test_nonstream_terminal_disconnect_race_keeps_terminal_winner() -> None:
    async def scenario() -> None:
        session = _Session()
        race = asyncio.Event()

        async def receive() -> dict[str, object]:
            await race.wait()
            return {"type": "http.disconnect"}

        task = asyncio.create_task(race_nonstream_disconnect(_request(receive), _consume(session)))
        await session.started.wait()
        session.terminal = True
        session.release.set()
        race.set()
        assert await task == "ok"
        assert session.cancel_calls == 0

    asyncio.run(scenario())


def test_nonstream_outer_cancellation_cleans_result_from_first_scheduling_yield() -> None:
    async def scenario() -> None:
        never = asyncio.Event()
        cleaned = asyncio.Event()
        outer_task: asyncio.Task[str] | None = None

        async def receive() -> dict[str, object]:
            await never.wait()
            return {"type": "http.disconnect"}

        async def result() -> str:
            assert outer_task is not None
            outer_task.cancel()
            try:
                await never.wait()
            finally:
                cleaned.set()
            return "unreachable"

        outer_task = asyncio.create_task(race_nonstream_disconnect(_request(receive), result()))
        with pytest.raises(asyncio.CancelledError):
            await outer_task
        await asyncio.wait_for(cleaned.wait(), timeout=1)

    asyncio.run(scenario())
