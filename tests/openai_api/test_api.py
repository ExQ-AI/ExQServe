from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.errors import CanonicalError, ErrorCategory
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
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.api import _iter_chat_sse, create_openai_app
from exqserve.protocol.openai.chat import ChatStreamSerializer
from exqserve.serving.contracts import ServingRejected, ServingRequest
from exqserve.state.store import InMemoryResponseStore, ResponseRecord


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


class _Engine:
    def __init__(self, events: list[GenerationEvent]) -> None:
        self.events = events
        self.requests: list[ServingRequest] = []
        self.count_requests: list[ServingRequest] = []
        self.sessions: list[_Session] = []
        self.reject: CanonicalError | None = None
        self.count_result = 42

    async def count_input_tokens(self, request: ServingRequest) -> int:
        self.count_requests.append(request)
        if self.reject is not None:
            raise ServingRejected(self.reject)
        return self.count_result

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        if self.reject is not None:
            raise ServingRejected(self.reject)
        session = _Session(list(self.events))
        self.sessions.append(session)
        return session


def _events() -> list[GenerationEvent]:
    usage = TokenUsage(input_tokens=3, output_tokens=1, cached_input_tokens=2)
    return [
        GenerationStarted("wire"),
        TextStarted("wire"),
        TextDelta("wire", "ok"),
        TextCompleted("wire", "ok"),
        UsageUpdated("wire", usage),
        GenerationCompleted("wire", CompletionReason.STOP, usage),
    ]


async def _request(app, method: str, url: str, **kwargs: object) -> httpx.Response:  # type: ignore[no-untyped-def]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def test_chat_nonstream_route_returns_chat_completion_json() -> None:
    async def scenario() -> None:
        engine = _Engine(_events())
        app = create_openai_app(engine, default_max_output_tokens=8)
        response = await _request(
            app,
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "ok"
        assert body["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
        assert engine.requests[0].input.items == (MessageItem(MessageRole.USER, "hi"),)
        assert engine.sessions[0].cancel_calls == 0

    asyncio.run(scenario())


def test_chat_stream_route_emits_sse_and_done_sentinel() -> None:
    async def scenario() -> None:
        engine = _Engine(_events())
        app = create_openai_app(engine, default_max_output_tokens=8)
        response = await _request(
            app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]\n\n" in response.text
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert payloads[0]["object"] == "chat.completion.chunk"
        assert any(payload.get("choices") == [] and "usage" in payload for payload in payloads)
        assert engine.sessions[0].cancel_calls == 0

    asyncio.run(scenario())


def test_responses_nonstream_route_returns_item_native_response() -> None:
    async def scenario() -> None:
        engine = _Engine(_events())
        app = create_openai_app(engine, default_max_output_tokens=8)
        response = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "qwen", "input": "hi"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == "ok"
        assert body["previous_response_id"] is None

    asyncio.run(scenario())


def test_responses_input_tokens_counts_without_generation() -> None:
    async def scenario() -> None:
        engine = _Engine(_events())
        app = create_openai_app(engine, default_max_output_tokens=8)
        response = await _request(
            app,
            "POST",
            "/v1/responses/input_tokens",
            json={
                "model": "qwen",
                "instructions": "Be concise.",
                "input": "hello",
                "reasoning": {"effort": "disabled"},
            },
        )

        assert response.status_code == 200, response.text
        assert response.json() == {"object": "response.input_tokens", "input_tokens": 42}
        assert response.headers["x-request-id"].startswith("req_")
        assert engine.requests == []
        assert engine.sessions == []
        assert len(engine.count_requests) == 1
        assert engine.count_requests[0].input.items == (
            MessageItem(MessageRole.DEVELOPER, "Be concise."),
            MessageItem(MessageRole.USER, "hello"),
        )

    asyncio.run(scenario())


def test_responses_input_tokens_reuses_previous_response_context_and_errors() -> None:
    async def scenario() -> None:
        engine = _Engine(_events())
        store = InMemoryResponseStore()
        await store.put(
            ResponseRecord(
                "resp-parent",
                "qwen",
                (
                    MessageItem(MessageRole.USER, "old question"),
                    MessageItem(MessageRole.ASSISTANT, "old answer"),
                ),
            )
        )
        await store.put(
            ResponseRecord(
                "resp-other",
                "other-model",
                (MessageItem(MessageRole.USER, "other history"),),
            )
        )
        app = create_openai_app(
            engine,
            default_max_output_tokens=8,
            response_store=store,
        )

        counted = await _request(
            app,
            "POST",
            "/v1/responses/input_tokens",
            json={
                "model": "qwen",
                "instructions": "new rules",
                "previous_response_id": "resp-parent",
                "input": "next",
            },
        )
        assert counted.status_code == 200, counted.text
        assert engine.count_requests[-1].input.items == (
            MessageItem(MessageRole.DEVELOPER, "new rules"),
            MessageItem(MessageRole.USER, "old question"),
            MessageItem(MessageRole.ASSISTANT, "old answer"),
            MessageItem(MessageRole.USER, "next"),
        )

        missing = await _request(
            app,
            "POST",
            "/v1/responses/input_tokens",
            json={"model": "qwen", "previous_response_id": "resp-missing", "input": "next"},
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "response_not_found"

        mismatch = await _request(
            app,
            "POST",
            "/v1/responses/input_tokens",
            json={"model": "qwen", "previous_response_id": "resp-other", "input": "next"},
        )
        assert mismatch.status_code == 400
        assert mismatch.json()["error"]["code"] == "response_model_mismatch"
        assert engine.requests == []
        assert engine.sessions == []
        assert len(engine.count_requests) == 1

    asyncio.run(scenario())


def test_responses_stream_route_uses_named_events_and_no_chat_done_sentinel() -> None:
    async def scenario() -> None:
        engine = _Engine(_events())
        app = create_openai_app(engine, default_max_output_tokens=8)
        response = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "qwen", "input": "hi", "stream": True},
        )

        assert response.status_code == 200
        assert "event: response.created" in response.text
        assert "event: response.output_text.delta" in response.text
        assert "event: response.completed" in response.text
        assert "[DONE]" not in response.text

    asyncio.run(scenario())


def test_protocol_and_serving_rejections_map_to_openai_error_statuses() -> None:
    async def scenario() -> None:
        engine = _Engine(_events())
        app = create_openai_app(engine, default_max_output_tokens=8)
        bad = await _request(app, "POST", "/v1/chat/completions", json={"model": "qwen", "messages": []})
        assert bad.status_code == 400
        assert bad.json()["error"]["code"] == "invalid_messages"

        engine.reject = CanonicalError(
            ErrorCategory.OVERLOADED,
            "server_overloaded",
            "Server is at capacity.",
            retryable=True,
        )
        overloaded = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "qwen", "input": "hi"},
        )
        assert overloaded.status_code == 429
        assert overloaded.json()["error"]["code"] == "server_overloaded"

        engine.reject = CanonicalError(
            ErrorCategory.CONTEXT_LENGTH,
            "total_context_limit_exceeded",
            "Total requested context limit exceeded.",
            retryable=False,
        )
        context_overflow = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "qwen", "input": "hi"},
        )
        assert context_overflow.status_code == 400
        assert context_overflow.json()["error"] == {
            "message": "Request exceeds the model context window.",
            "type": "invalid_request_error",
            "param": None,
            "code": "context_length_exceeded",
        }

    asyncio.run(scenario())


def test_closing_chat_stream_generator_cancels_unfinished_session_once() -> None:
    async def scenario() -> None:
        request = ServingRequest(
            CanonicalRequest("r", "m", (MessageItem(MessageRole.USER, "hi"),)),
            ReasoningPolicy(),
            ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True),
            max_output_tokens=4,
        )
        session = _Session([GenerationStarted("r"), TextStarted("r"), TextDelta("r", "x")])
        serializer = ChatStreamSerializer("m", response_id="chatcmpl-test", created=1)
        stream = _iter_chat_sse(session, serializer)

        first = await anext(stream)
        assert first.startswith("data: ")
        await stream.aclose()

        assert session.cancel_calls == 1
        assert request.max_output_tokens == 4

    asyncio.run(scenario())
