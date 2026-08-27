from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
)
from exqserve.core.usage import TokenUsage
from exqserve.protocol.anthropic.api import create_anthropic_app
from exqserve.serving.contracts import ServingRequest


class _Session:
    def __init__(self, request_id: str) -> None:
        self._events: list[GenerationEvent] = [
            GenerationStarted(request_id),
            TextStarted(request_id),
            TextDelta(request_id, "hello"),
            TextCompleted(request_id, "hello"),
            GenerationCompleted(
                request_id,
                CompletionReason.STOP,
                TokenUsage(input_tokens=4, cached_input_tokens=1, output_tokens=1),
            ),
        ]
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Engine:
    def __init__(self) -> None:
        self.requests: list[ServingRequest] = []
        self.count_requests: list[ServingRequest] = []

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        return _Session(request.input.request_id)

    async def count_input_tokens(self, request: ServingRequest) -> int:
        self.count_requests.append(request)
        return 42


async def _request(app: object, method: str, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


def _headers() -> dict[str, str]:
    return {"anthropic-version": "2023-06-01"}


def test_messages_nonstream_and_stream_use_anthropic_wire_shape() -> None:
    async def scenario() -> None:
        engine = _Engine()
        app = create_anthropic_app(engine)
        body = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

        response = await _request(app, "POST", "/v1/messages", headers=_headers(), json=body)
        assert response.status_code == 200
        assert response.headers["request-id"].startswith("req_")
        assert response.json()["type"] == "message"
        assert response.json()["role"] == "assistant"
        assert response.json()["content"] == [{"type": "text", "text": "hello"}]
        assert response.json()["stop_reason"] == "end_turn"
        assert response.json()["usage"] == {
            "input_tokens": 3,
            "output_tokens": 1,
            "cache_read_input_tokens": 1,
            "cache_creation_input_tokens": 0,
        }

        streamed = await _request(
            app,
            "POST",
            "/v1/messages",
            headers=_headers(),
            json={**body, "stream": True},
        )
        assert streamed.status_code == 200
        assert streamed.headers["content-type"].startswith("text/event-stream")
        assert "event: message_start" in streamed.text
        assert '"type":"text_delta","text":"hello"' in streamed.text
        assert "event: message_stop" in streamed.text
        assert len(engine.requests) == 2

    asyncio.run(scenario())


def test_messages_requires_supported_anthropic_version_and_shapes_errors() -> None:
    async def scenario() -> None:
        engine = _Engine()
        app = create_anthropic_app(engine)
        body = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

        missing = await _request(app, "POST", "/v1/messages", json=body)
        assert missing.status_code == 400
        assert missing.json()["type"] == "error"
        assert missing.json()["error"]["type"] == "invalid_request_error"
        assert missing.json()["request_id"] == missing.headers["request-id"]

        unsupported = await _request(
            app,
            "POST",
            "/v1/messages",
            headers={"anthropic-version": "1900-01-01"},
            json=body,
        )
        assert unsupported.status_code == 400
        assert engine.requests == []

    asyncio.run(scenario())


def test_messages_dynamic_model_binding_and_body_limit() -> None:
    class _Model:
        def __init__(self, model_id: str) -> None:
            self.id = model_id

    async def scenario() -> None:
        engine = _Engine()
        current: list[_Model | None] = [_Model("first")]
        app = create_anthropic_app(
            engine,
            served_model=lambda: current[0],
            max_request_body_bytes=128,
        )

        first = await _request(
            app,
            "POST",
            "/v1/messages",
            headers=_headers(),
            json={"model": "first", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert first.status_code == 200

        current[0] = _Model("second")
        old = await _request(
            app,
            "POST",
            "/v1/messages",
            headers=_headers(),
            json={"model": "first", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert old.status_code == 404
        assert old.json()["error"]["type"] == "not_found_error"

        oversized = await _request(
            app,
            "POST",
            "/v1/messages",
            headers=_headers(),
            json={
                "model": "second",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "x" * 256}],
            },
        )
        assert oversized.status_code == 413
        assert oversized.json()["error"]["type"] == "request_too_large"
        assert len(engine.requests) == 1

    asyncio.run(scenario())


def test_count_tokens_uses_anthropic_request_shape_without_generation() -> None:
    async def scenario() -> None:
        engine = _Engine()
        app = create_anthropic_app(engine)
        response = await _request(
            app,
            "POST",
            "/v1/messages/count_tokens",
            headers=_headers(),
            json={
                "model": "m",
                "system": "Be concise.",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "ping", "input_schema": {"type": "object"}}],
            },
        )

        assert response.status_code == 200, response.text
        assert response.json() == {"input_tokens": 42}
        assert response.headers["request-id"].startswith("req_")
        assert engine.requests == []
        assert len(engine.count_requests) == 1
        counted = engine.count_requests[0]
        assert counted.input.model == "m"
        assert counted.max_output_tokens == 1

    asyncio.run(scenario())
