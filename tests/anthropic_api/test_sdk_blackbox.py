from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

import pytest
import uvicorn

anthropic = pytest.importorskip("anthropic")

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
from exqserve.core.items import ToolCallItem, ToolResultItem
from exqserve.core.usage import TokenUsage
from exqserve.protocol.anthropic.api import create_anthropic_app
from exqserve.serving.contracts import ServingRequest


class _Session:
    def __init__(self, events: list[GenerationEvent]) -> None:
        self._events = list(events)
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        await asyncio.sleep(0)
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Engine:
    def __init__(self) -> None:
        self.requests: list[ServingRequest] = []
        self.count_requests: list[ServingRequest] = []

    async def count_input_tokens(self, request: ServingRequest) -> int:
        self.count_requests.append(request)
        return 17

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        request_id = request.input.request_id
        usage = TokenUsage(input_tokens=9, cached_input_tokens=2, output_tokens=3)
        if request.tools.tools and not any(isinstance(item, ToolResultItem) for item in request.input.items):
            call = ToolCallItem("toolu_sdk", request.tools.tools[0].name, '{"id":1}', 0)
            return _Session(
                [
                    GenerationStarted(request_id),
                    ToolCallStarted(request_id, call.call_id, call.name, call.index),
                    ToolCallArgumentsDelta(request_id, call.call_id, call.arguments_json, call.index),
                    ToolCallCompleted(request_id, call),
                    GenerationCompleted(request_id, CompletionReason.TOOL_CALLS, usage),
                ]
            )
        text = "tool result accepted" if any(
            isinstance(item, ToolResultItem) for item in request.input.items
        ) else "sdk hello"
        return _Session(
            [
                GenerationStarted(request_id),
                TextStarted(request_id),
                TextDelta(request_id, text),
                TextCompleted(request_id, text),
                GenerationCompleted(request_id, CompletionReason.STOP, usage),
            ]
        )


@contextmanager
def _serve(app: object) -> Iterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive():
        if time.monotonic() >= deadline:
            server.should_exit = True
            thread.join(timeout=2)
            raise AssertionError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        if thread.is_alive():
            raise AssertionError("uvicorn did not stop")


def test_current_anthropic_sdk_create_stream_count_and_tool_roundtrip() -> None:
    engine = _Engine()
    app = create_anthropic_app(engine)

    async def scenario(base_url: str) -> None:
        async with anthropic.AsyncAnthropic(
            api_key="test",
            base_url=base_url,
            max_retries=0,
        ) as client:
            counted = await client.messages.count_tokens(
                model="local-qwen",
                system="Be concise.",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert counted.input_tokens == 17

            message = await client.messages.create(
                model="local-qwen",
                max_tokens=16,
                messages=[{"role": "user", "content": "hello"}],
            )
            assert message.type == "message"
            assert message.role == "assistant"
            assert message.content[0].type == "text"
            assert message.content[0].text == "sdk hello"
            assert message.stop_reason == "end_turn"
            assert message.usage.input_tokens == 7
            assert message.usage.cache_read_input_tokens == 2

            async with client.messages.stream(
                model="local-qwen",
                max_tokens=16,
                messages=[{"role": "user", "content": "stream"}],
            ) as stream:
                streamed_text = await stream.get_final_text()
                final = await stream.get_final_message()
            assert streamed_text == "sdk hello"
            assert final.stop_reason == "end_turn"

            tool_message = await client.messages.create(
                model="local-qwen",
                max_tokens=32,
                messages=[{"role": "user", "content": "lookup"}],
                tools=[
                    {
                        "name": "lookup",
                        "description": "Lookup an id",
                        "input_schema": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                            "required": ["id"],
                        },
                    }
                ],
                tool_choice={"type": "tool", "name": "lookup"},
            )
            assert tool_message.stop_reason == "tool_use"
            tool_use = next(block for block in tool_message.content if block.type == "tool_use")
            assert tool_use.name == "lookup"
            assert tool_use.input == {"id": 1}

            continuation = await client.messages.create(
                model="local-qwen",
                max_tokens=16,
                messages=[
                    {"role": "user", "content": "lookup"},
                    {"role": "assistant", "content": [tool_use.model_dump()]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": "7",
                            }
                        ],
                    },
                ],
                tools=[
                    {
                        "name": "lookup",
                        "description": "Lookup an id",
                        "input_schema": {"type": "object"},
                    }
                ],
                tool_choice={"type": "auto"},
            )
            assert continuation.content[0].type == "text"
            assert continuation.content[0].text == "tool result accepted"

    with _serve(app) as base_url:
        asyncio.run(scenario(base_url))

    assert len(engine.count_requests) == 1
    assert len(engine.requests) == 4
