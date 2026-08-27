from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationStarted,
)
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.api import create_openai_app
from exqserve.protocol.openai.models import OpenAIModelInfo
from exqserve.serving.contracts import ServingRequest


class _Session:
    def __init__(self) -> None:
        usage = TokenUsage(input_tokens=1, cached_input_tokens=0, output_tokens=0)
        self.events: list[GenerationEvent] = [
            GenerationStarted("wire"),
            GenerationCompleted("wire", CompletionReason.STOP, usage),
        ]

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def cancel(self) -> None:
        return None


class _Engine:
    def __init__(self) -> None:
        self.requests: list[ServingRequest] = []

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        return _Session()


async def _request(app: object, method: str, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


def test_models_list_and_retrieve_share_one_public_model_object() -> None:
    async def scenario() -> None:
        engine = _Engine()
        info = OpenAIModelInfo("local-qwen", created=123, context_length=65536)
        app = create_openai_app(engine, default_max_output_tokens=8, served_model=info)

        listed = await _request(app, "GET", "/v1/models")
        assert listed.status_code == 200
        assert listed.json() == {
            "object": "list",
            "data": [
                {
                    "id": "local-qwen",
                    "object": "model",
                    "created": 123,
                    "owned_by": "exqserve",
                    "context_length": 65536,
                }
            ],
        }

        retrieved = await _request(app, "GET", "/v1/models/local-qwen")
        assert retrieved.status_code == 200
        assert retrieved.json() == listed.json()["data"][0]

    asyncio.run(scenario())


def test_unknown_model_retrieve_and_generation_are_rejected_before_submit() -> None:
    async def scenario() -> None:
        engine = _Engine()
        info = OpenAIModelInfo("local-qwen", created=123, context_length=65536)
        app = create_openai_app(engine, default_max_output_tokens=8, served_model=info)

        missing = await _request(app, "GET", "/v1/models/other")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "model_not_found"
        assert missing.json()["error"]["param"] == "model"

        chat = await _request(
            app,
            "POST",
            "/v1/chat/completions",
            json={"model": "other", "messages": [{"role": "user", "content": "hi"}]},
        )
        responses = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "other", "input": "hi"},
        )
        assert chat.status_code == 404
        assert responses.status_code == 404
        assert chat.json()["error"]["code"] == "model_not_found"
        assert responses.json()["error"]["code"] == "model_not_found"
        assert engine.requests == []

    asyncio.run(scenario())


def test_bound_model_accepts_matching_chat_and_responses_requests() -> None:
    async def scenario() -> None:
        engine = _Engine()
        info = OpenAIModelInfo("local-qwen", created=123, context_length=65536)
        app = create_openai_app(engine, default_max_output_tokens=8, served_model=info)

        chat = await _request(
            app,
            "POST",
            "/v1/chat/completions",
            json={"model": "local-qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        responses = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "local-qwen", "input": "hi"},
        )
        assert chat.status_code == 200
        assert responses.status_code == 200
        assert len(engine.requests) == 2

    asyncio.run(scenario())


def test_dynamic_model_source_updates_list_retrieve_and_generation_binding() -> None:
    async def scenario() -> None:
        engine = _Engine()
        current: list[OpenAIModelInfo | None] = [
            OpenAIModelInfo("first", created=1, context_length=4096)
        ]
        app = create_openai_app(
            engine,
            default_max_output_tokens=8,
            served_model=lambda: current[0],
        )

        assert (await _request(app, "GET", "/v1/models")).json()["data"][0]["id"] == "first"
        current[0] = OpenAIModelInfo("second", created=2, context_length=8192)
        assert (await _request(app, "GET", "/v1/models")).json()["data"][0]["id"] == "second"
        assert (await _request(app, "GET", "/v1/models/first")).status_code == 404
        assert (await _request(app, "GET", "/v1/models/second")).status_code == 200

        old = await _request(
            app,
            "POST",
            "/v1/chat/completions",
            json={"model": "first", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert old.status_code == 404
        current[0] = None
        assert (await _request(app, "GET", "/v1/models")).json() == {"object": "list", "data": []}
        unloaded = await _request(
            app,
            "POST",
            "/v1/chat/completions",
            json={"model": "second", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert unloaded.status_code == 404
        assert engine.requests == []

    asyncio.run(scenario())
