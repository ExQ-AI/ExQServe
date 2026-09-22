from __future__ import annotations

import asyncio
import json

import pytest

from exqserve.protocol.anthropic.api import create_anthropic_app
from exqserve.protocol.openai.api import create_openai_app
from exqserve.protocol.openai.responses import build_response_object
from exqserve.serving.contracts import RawServingRequest, ServingRequest
from exqserve.state.response_lifecycle import InMemoryResponseLifecycleStore
from exqserve.state.store import InMemoryResponseStore, ResponseRecord, ResponseStoreDisposition


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
        raise AssertionError("submit should be cancelled before response establishment")


class _ParentSession:
    async def cancel(self) -> None:
        return None


async def _exercise_queued_disconnect(
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


@pytest.mark.parametrize(
    ("path", "body"),
    (
        (
            "/v1/completions",
            {"model": "m", "prompt": "hello", "max_tokens": 8, "stream": True},
        ),
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
def test_openai_stream_disconnect_covers_submit_before_response_establishment(
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
        await _exercise_queued_disconnect(app=app, engine=engine, path=path, body=body)

    asyncio.run(scenario())


def test_anthropic_stream_disconnect_covers_submit_before_response_establishment() -> None:
    async def scenario() -> None:
        engine = _BlockingEngine()
        app = create_anthropic_app(engine)  # type: ignore[arg-type]
        await _exercise_queued_disconnect(
            app=app,
            engine=engine,
            path="/v1/messages",
            body={
                "model": "m",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            extra_headers=((b"anthropic-version", b"2023-06-01"),),
        )

    asyncio.run(scenario())


def test_responses_stream_setup_application_cancellation_releases_parent_pins() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore()
        lifecycle = InMemoryResponseLifecycleStore()
        assert (
            await state.put(ResponseRecord("resp_parent", "m", None, ()))
            is ResponseStoreDisposition.STORED
        )
        initial = build_response_object(
            response_id="resp_parent",
            created_at=1,
            model="m",
            status="in_progress",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            usage=None,
            previous_response_id=None,
            store=True,
        )
        await lifecycle.register_active(initial, _ParentSession(), retain=True)
        terminal = dict(initial)
        terminal["status"] = "completed"
        assert await lifecycle.finish("resp_parent", terminal)

        engine = _BlockingEngine()
        app = create_openai_app(
            engine,  # type: ignore[arg-type]
            default_max_output_tokens=8,
            response_store=state,
            response_lifecycle_store=lifecycle,
        )
        body = json.dumps(
            {
                "model": "m",
                "input": "child",
                "previous_response_id": "resp_parent",
                "max_output_tokens": 8,
                "stream": True,
            }
        ).encode()
        receives: list[dict[str, object]] = [
            {"type": "http.request", "body": body, "more_body": False},
        ]
        never = asyncio.Event()

        async def receive() -> dict[str, object]:
            if receives:
                return receives.pop(0)
            await never.wait()
            raise AssertionError("unreachable")

        async def send(message: dict[str, object]) -> None:
            del message

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": ((b"content-type", b"application/json"),),
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
            "root_path": "",
        }

        request_task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(engine.submit_started.wait(), timeout=1)
        assert state._pins == {"resp_parent": 1}
        assert lifecycle._pins == {"resp_parent": 1}

        request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        engine.release_submit.set()

        assert state._pins == {}
        assert lifecycle._pins == {}

    asyncio.run(scenario())
