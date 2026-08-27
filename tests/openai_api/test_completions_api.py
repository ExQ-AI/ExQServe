from __future__ import annotations

import asyncio
import json
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
    UsageUpdated,
)
from exqserve.core.items import RawPromptItem
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.api import create_openai_app
from exqserve.serving.contracts import RawServingRequest


class _Session:
    def __init__(self, events: list[GenerationEvent]) -> None:
        self.events = list(events)
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        await asyncio.sleep(0)
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _RawEngine:
    def __init__(self) -> None:
        self.requests: list[RawServingRequest] = []
        self.sessions: list[_Session] = []

    async def submit(self, request: RawServingRequest) -> _Session:
        self.requests.append(request)
        request_id = request.input.request_id
        usage = TokenUsage(input_tokens=3, output_tokens=1, cached_input_tokens=2)
        session = _Session(
            [
                GenerationStarted(request_id),
                TextStarted(request_id),
                TextDelta(request_id, " continuation"),
                TextCompleted(request_id, " continuation"),
                UsageUpdated(request_id, usage),
                GenerationCompleted(request_id, CompletionReason.STOP, usage),
            ]
        )
        self.sessions.append(session)
        return session


class _UnusedChatEngine:
    async def submit(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("legacy completions must not submit through the Chat/Responses engine")


async def _request(app, method: str, url: str, **kwargs: object) -> httpx.Response:  # type: ignore[no-untyped-def]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def test_completions_nonstream_route_uses_raw_engine_and_legacy_wire_shape() -> None:
    async def scenario() -> None:
        raw = _RawEngine()
        app = create_openai_app(
            _UnusedChatEngine(),
            completion_engine=raw,
        )
        response = await _request(
            app,
            "POST",
            "/v1/completions",
            json={"model": "m", "prompt": "RAW", "max_tokens": 4, "echo": True, "stop": "END"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["x-request-id"].startswith("req_")
        body = response.json()
        assert body["object"] == "text_completion"
        assert body["choices"] == [
            {
                "text": "RAW continuation",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ]
        assert body["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
        request = raw.requests[0]
        assert request.input.request_id == response.headers["x-request-id"]
        assert request.input.items == (RawPromptItem(text="RAW"),)
        assert request.stop_conditions == ("END",)
        assert raw.sessions[0].cancel_calls == 0

    asyncio.run(scenario())


def test_completions_stream_route_emits_legacy_sse_and_done() -> None:
    async def scenario() -> None:
        raw = _RawEngine()
        app = create_openai_app(_UnusedChatEngine(), completion_engine=raw)
        response = await _request(
            app,
            "POST",
            "/v1/completions",
            json={
                "model": "m",
                "prompt": "RAW",
                "stream": True,
                "echo": True,
                "stream_options": {"include_usage": True},
            },
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text.endswith("data: [DONE]\n\n")
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert payloads[0]["object"] == "text_completion"
        assert payloads[0]["choices"][0]["text"] == "RAW"
        assert any(payload["choices"] and payload["choices"][0]["text"] == " continuation" for payload in payloads)
        assert any(payload.get("choices") == [] and "usage" in payload for payload in payloads)
        assert raw.sessions[0].cancel_calls == 0

    asyncio.run(scenario())


def test_completions_without_raw_engine_is_not_silently_routed_to_chat() -> None:
    async def scenario() -> None:
        app = create_openai_app(_UnusedChatEngine())
        response = await _request(
            app,
            "POST",
            "/v1/completions",
            json={"model": "m", "prompt": "RAW"},
        )
        assert response.status_code == 404

    asyncio.run(scenario())
