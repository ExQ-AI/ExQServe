from __future__ import annotations

import asyncio
import json

import pytest

from exqserve.core.events import GenerationEvent, GenerationStarted
from exqserve.protocol.anthropic.api import create_anthropic_app
from exqserve.protocol.openai.api import create_openai_app
from exqserve.serving.contracts import RawServingRequest, ServingRequest


class _BlockingEngine:
    def __init__(self) -> None:
        self.submit_started = asyncio.Event()
        self.submit_cancelled = asyncio.Event()
        self.release_submit = asyncio.Event()

    async def count_input_tokens(self, request: ServingRequest) -> int:
        del request
        return 1

    async def submit(self, request: ServingRequest | RawServingRequest) -> object:
        del request
        self.submit_started.set()
        try:
            await self.release_submit.wait()
        except asyncio.CancelledError:
            self.submit_cancelled.set()
            raise
        raise AssertionError("submit should be cancelled by client disconnect")


class _TieSession:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.input_token_count = 1
        self.cancel_calls = 0
        self._started = False
        self._release = asyncio.Event()

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self) -> GenerationEvent:
        if not self._started:
            self._started = True
            return GenerationStarted(self.request_id)
        await self._release.wait()
        raise StopAsyncIteration

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self._release.set()


class _TieEngine:
    def __init__(self, gate: asyncio.Event) -> None:
        self.gate = gate
        self.submit_started = asyncio.Event()
        self.session: _TieSession | None = None

    async def count_input_tokens(self, request: ServingRequest) -> int:
        del request
        return 1

    async def submit(self, request: ServingRequest | RawServingRequest) -> _TieSession:
        self.submit_started.set()
        await self.gate.wait()
        self.session = _TieSession(request.input.request_id)
        return self.session


async def _exercise_disconnect(
    *,
    app: object,
    engine: _BlockingEngine,
    path: str,
    body: dict[str, object],
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
) -> None:
    encoded = json.dumps(body).encode()
    receives: list[dict[str, object]] = [
        {"type": "http.request", "body": encoded, "more_body": False},
        {"type": "http.disconnect"},
    ]

    async def receive() -> dict[str, object]:
        if receives:
            return receives.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": ((b"content-type", b"application/json"), *extra_headers),
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "root_path": "",
    }

    request_task = asyncio.create_task(app(scope, receive, send))  # type: ignore[operator]
    await asyncio.wait_for(engine.submit_started.wait(), timeout=1)
    await asyncio.wait_for(engine.submit_cancelled.wait(), timeout=1)
    await asyncio.wait_for(asyncio.gather(request_task, return_exceptions=True), timeout=1)

    assert request_task.done()
    assert sent == []

    engine.release_submit.set()


async def _exercise_stream_setup_tie(
    *,
    app: object,
    engine: _TieEngine,
    path: str,
    body: dict[str, object],
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
) -> None:
    encoded = json.dumps(body).encode()
    receive_calls = 0
    setup_disconnect_waiting = asyncio.Event()
    framework_receive_after_handoff = asyncio.Event()
    never = asyncio.Event()

    async def receive() -> dict[str, object]:
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls == 1:
            return {"type": "http.request", "body": encoded, "more_body": False}
        if receive_calls == 2:
            setup_disconnect_waiting.set()
            await engine.gate.wait()
            return {"type": "http.disconnect"}
        framework_receive_after_handoff.set()
        await never.wait()
        raise AssertionError("unreachable")

    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": ((b"content-type", b"application/json"), *extra_headers),
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
        "root_path": "",
    }

    request_task = asyncio.create_task(app(scope, receive, send))  # type: ignore[operator]
    await asyncio.wait_for(engine.submit_started.wait(), timeout=1)
    await asyncio.wait_for(setup_disconnect_waiting.wait(), timeout=1)
    engine.gate.set()
    await asyncio.wait_for(asyncio.gather(request_task, return_exceptions=True), timeout=1)

    assert request_task.done()
    assert receive_calls == 2
    assert not framework_receive_after_handoff.is_set()
    assert engine.session is not None
    assert engine.session.cancel_calls == 1
    assert sent == []


@pytest.mark.parametrize(
    ("path", "body"),
    (
        (
            "/v1/completions",
            {"model": "m", "prompt": "hello", "max_tokens": 8},
        ),
        (
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8},
        ),
        (
            "/v1/responses",
            {"model": "m", "input": "hello", "max_output_tokens": 8},
        ),
    ),
)
def test_openai_nonstream_disconnect_covers_submit(
    path: str,
    body: dict[str, object],
) -> None:
    async def scenario() -> None:
        engine = _BlockingEngine()
        app = create_openai_app(
            engine,  # type: ignore[arg-type]
            default_max_output_tokens=8,
            completion_engine=engine,  # type: ignore[arg-type]
        )
        await _exercise_disconnect(app=app, engine=engine, path=path, body=body)

    asyncio.run(scenario())


def test_anthropic_nonstream_disconnect_covers_submit() -> None:
    async def scenario() -> None:
        engine = _BlockingEngine()
        app = create_anthropic_app(engine)  # type: ignore[arg-type]
        await _exercise_disconnect(
            app=app,
            engine=engine,
            path="/v1/messages",
            body={
                "model": "m",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
            },
            extra_headers=((b"anthropic-version", b"2023-06-01"),),
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("path", "body"),
    (
        (
            "/v1/chat/completions",
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "stream": True,
            },
        ),
        (
            "/v1/responses",
            {"model": "m", "input": "hello", "max_output_tokens": 8, "stream": True},
        ),
    ),
)
def test_openai_stream_setup_disconnect_wins_simultaneous_handoff(
    path: str,
    body: dict[str, object],
) -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        engine = _TieEngine(gate)
        app = create_openai_app(engine, default_max_output_tokens=8)  # type: ignore[arg-type]
        await _exercise_stream_setup_tie(
            app=app,
            engine=engine,
            path=path,
            body=body,
        )

    asyncio.run(scenario())


def test_anthropic_stream_setup_disconnect_wins_simultaneous_handoff() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        engine = _TieEngine(gate)
        app = create_anthropic_app(engine)  # type: ignore[arg-type]
        await _exercise_stream_setup_tie(
            app=app,
            engine=engine,
            path="/v1/messages",
            body={
                "model": "m",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            extra_headers=((b"anthropic-version", b"2023-06-01"),),
        )

    asyncio.run(scenario())
