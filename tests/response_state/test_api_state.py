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
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import MessageItem, MessageRole, ToolCallItem, ToolResultItem
from exqserve.model.contracts import CompiledPrompt, TemplateRequest
from exqserve.protocol.openai.api import create_openai_app
from exqserve.serving.contracts import ServingRequest
from exqserve.state.store import InMemoryResponseStore


class _Session:
    def __init__(self, events: list[GenerationEvent]) -> None:
        self._events = iter(events)
        self.cancel_calls = 0
        self.compiled_prompt = CompiledPrompt("p", (1,), "a" * 64, (), TemplateRequest((), (), ()))

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Engine:
    def __init__(self, generations: list[list[GenerationEvent]]) -> None:
        self._generations = iter(generations)
        self.requests: list[ServingRequest] = []

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        return _Session(next(self._generations))


async def _post(app, body: dict[str, object]) -> httpx.Response:  # type: ignore[no-untyped-def]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/responses", json=body)


def _text_generation(text: str) -> list[GenerationEvent]:
    return [
        GenerationStarted("r"),
        TextStarted("r"),
        TextDelta("r", text),
        TextCompleted("r", text),
        GenerationCompleted("r", CompletionReason.STOP),
    ]


def test_previous_response_restores_context_but_previous_instructions_do_not_leak() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore()
        engine = _Engine([_text_generation("first answer"), _text_generation("second answer")])
        app = create_openai_app(engine, default_max_output_tokens=8, response_store=store)

        first = await _post(
            app,
            {"model": "m", "instructions": "OLD RULES", "input": "first question"},
        )
        assert first.status_code == 200
        first_id = first.json()["id"]
        assert (await store.get(first_id)) is not None

        second = await _post(
            app,
            {
                "model": "m",
                "instructions": "NEW RULES",
                "input": "second question",
                "previous_response_id": first_id,
            },
        )
        assert second.status_code == 200
        assert second.json()["previous_response_id"] == first_id

        assert engine.requests[1].input.items == (
            MessageItem(MessageRole.DEVELOPER, "NEW RULES"),
            MessageItem(MessageRole.USER, "first question"),
            MessageItem(MessageRole.ASSISTANT, "first answer"),
            MessageItem(MessageRole.USER, "second question"),
        )
        assert all(
            not (isinstance(item, MessageItem) and item.text == "OLD RULES")
            for item in engine.requests[1].input.items
        )

    asyncio.run(scenario())


def test_function_call_then_function_output_continuation_preserves_order() -> None:
    async def scenario() -> None:
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        first_events: list[GenerationEvent] = [
            GenerationStarted("r"),
            ToolCallStarted("r", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("r", "call-1", '{"id":1}', 0),
            ToolCallCompleted("r", call),
            GenerationCompleted("r", CompletionReason.TOOL_CALLS),
        ]
        engine = _Engine([first_events, _text_generation("done")])
        store = InMemoryResponseStore()
        app = create_openai_app(engine, default_max_output_tokens=8, response_store=store)

        first = await _post(
            app,
            {
                "model": "m",
                "input": "lookup",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        )
        first_id = first.json()["id"]

        second = await _post(
            app,
            {
                "model": "m",
                "input": [
                    {"type": "function_call_output", "call_id": "call-1", "output": "RESULT"}
                ],
                "previous_response_id": first_id,
            },
        )
        assert second.status_code == 200
        assert engine.requests[1].input.items == (
            MessageItem(MessageRole.USER, "lookup"),
            call,
            ToolResultItem("call-1", "RESULT"),
        )

    asyncio.run(scenario())


def test_store_false_and_missing_previous_id_behave_explicitly() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore()
        engine = _Engine([_text_generation("answer")])
        app = create_openai_app(engine, default_max_output_tokens=8, response_store=store)

        response = await _post(app, {"model": "m", "input": "x", "store": False})
        assert response.status_code == 200
        response_id = response.json()["id"]
        assert response.json()["store"] is False
        assert await store.get(response_id) is None

        missing = await _post(
            app,
            {"model": "m", "input": "next", "previous_response_id": response_id},
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "response_not_found"
        assert len(engine.requests) == 1

    asyncio.run(scenario())


def test_evicted_previous_response_fails_before_engine_submit() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore(max_records=1)
        engine = _Engine([_text_generation("one"), _text_generation("two")])
        app = create_openai_app(engine, default_max_output_tokens=8, response_store=store)

        first = await _post(app, {"model": "m", "input": "first"})
        first_id = first.json()["id"]
        second = await _post(app, {"model": "m", "input": "second"})
        assert second.status_code == 200
        assert await store.get(first_id) is None

        missing = await _post(
            app,
            {"model": "m", "input": "third", "previous_response_id": first_id},
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "response_not_found"
        assert len(engine.requests) == 2

    asyncio.run(scenario())


def test_stream_response_id_is_store_key_and_state_is_available_after_terminal_event() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore()
        engine = _Engine([_text_generation("streamed")])
        app = create_openai_app(engine, default_max_output_tokens=8, response_store=store)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/responses",
                json={"model": "m", "input": "stream", "stream": True},
            )

        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        created = next(item for item in payloads if item["type"] == "response.created")
        terminal = next(item for item in payloads if item["type"] == "response.completed")
        response_id = created["response"]["id"]
        assert terminal["response"]["id"] == response_id
        record = await store.get(response_id)
        assert record is not None
        assert record.context_items[-1] == MessageItem(MessageRole.ASSISTANT, "streamed")

    asyncio.run(scenario())


def test_previous_response_cannot_cross_model_boundary() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore()
        engine = _Engine([_text_generation("old answer")])
        app = create_openai_app(engine, default_max_output_tokens=8, response_store=store)

        first = await _post(app, {"model": "old-model", "input": "first"})
        assert first.status_code == 200
        response_id = first.json()["id"]

        mismatch = await _post(
            app,
            {
                "model": "new-model",
                "input": "continue",
                "previous_response_id": response_id,
            },
        )
        assert mismatch.status_code == 400
        assert mismatch.json()["error"]["code"] == "response_model_mismatch"
        assert mismatch.json()["error"]["param"] == "previous_response_id"
        assert len(engine.requests) == 1

    asyncio.run(scenario())
