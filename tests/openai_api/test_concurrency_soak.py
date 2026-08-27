from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
)
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.api import create_openai_app
from exqserve.protocol.openai.lifecycle import InMemoryResponseLifecycleStore
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


class _EchoEngine:
    def __init__(self) -> None:
        self.requests: list[ServingRequest] = []

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        latest_user = next(
            (
                item.text
                for item in reversed(request.input.items)
                if isinstance(item, MessageItem) and item.role is MessageRole.USER
            ),
            "",
        )
        request_id = request.input.request_id
        if latest_user == "FAIL":
            return _Session(
                [
                    GenerationStarted(request_id),
                    GenerationFailed(
                        request_id,
                        CanonicalError(
                            ErrorCategory.MODEL_FAILURE,
                            "synthetic_failure",
                            "Synthetic request failure.",
                            False,
                        ),
                    ),
                ]
            )
        text = f"reply:{latest_user}"
        usage = TokenUsage(input_tokens=len(request.input.items), output_tokens=1)
        return _Session(
            [
                GenerationStarted(request_id),
                TextStarted(request_id),
                TextDelta(request_id, text),
                TextCompleted(request_id, text),
                GenerationCompleted(request_id, CompletionReason.STOP, usage),
            ]
        )


async def _client(app):  # type: ignore[no-untyped-def]
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _response_text(body: dict[str, object]) -> str:
    output = body["output"]
    assert isinstance(output, list) and output
    message = output[0]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, list) and content
    part = content[0]
    assert isinstance(part, dict)
    text = part["text"]
    assert isinstance(text, str)
    return text


def test_parallel_branches_from_one_parent_do_not_mutate_or_cross_contaminate() -> None:
    async def scenario() -> None:
        engine = _EchoEngine()
        lifecycle = InMemoryResponseLifecycleStore(max_records=32)
        app = create_openai_app(
            engine,
            default_max_output_tokens=8,
            response_lifecycle_store=lifecycle,
        )
        async with await _client(app) as client:
            root = await client.post("/v1/responses", json={"model": "m", "input": "root"})
            assert root.status_code == 200
            root_id = root.json()["id"]

            branch_a, branch_b = await asyncio.gather(
                client.post(
                    "/v1/responses",
                    json={"model": "m", "input": "branch-a", "previous_response_id": root_id},
                ),
                client.post(
                    "/v1/responses",
                    json={"model": "m", "input": "branch-b", "previous_response_id": root_id},
                ),
            )
            assert branch_a.status_code == branch_b.status_code == 200
            assert _response_text(branch_a.json()) == "reply:branch-a"
            assert _response_text(branch_b.json()) == "reply:branch-b"

            root_after = await client.get(f"/v1/responses/{root_id}")
            assert root_after.status_code == 200
            assert root_after.json() == root.json()

        branch_requests = engine.requests[-2:]
        branch_inputs = {
            request.input.items[-1].text: request.input.items  # type: ignore[union-attr]
            for request in branch_requests
        }
        assert set(branch_inputs) == {"branch-a", "branch-b"}
        for marker, items in branch_inputs.items():
            assert items == (
                MessageItem(MessageRole.USER, "root"),
                MessageItem(MessageRole.ASSISTANT, "reply:root"),
                MessageItem(MessageRole.USER, marker),
            )
        stats = await lifecycle.stats()
        assert stats.active == 0
        assert stats.retained == 3

    asyncio.run(scenario())


def test_repeated_multi_client_soak_has_no_cross_talk_or_active_leaks() -> None:
    async def scenario() -> None:
        engine = _EchoEngine()
        lifecycle = InMemoryResponseLifecycleStore(max_records=64, max_total_bytes=1024 * 1024)
        app = create_openai_app(
            engine,
            default_max_output_tokens=8,
            response_lifecycle_store=lifecycle,
        )

        async def worker(worker_id: int) -> None:
            async with await _client(app) as client:
                for turn in range(20):
                    marker = f"w{worker_id}-t{turn}"
                    store = turn % 3 != 0
                    response = await client.post(
                        "/v1/responses",
                        json={"model": "m", "input": marker, "store": store},
                    )
                    assert response.status_code == 200
                    assert _response_text(response.json()) == f"reply:{marker}"
                    request_id = response.headers["x-request-id"]
                    assert request_id.startswith("req_")
                    if store and turn % 5 == 0:
                        retrieved = await client.get(f"/v1/responses/{response.json()['id']}")
                        assert retrieved.status_code == 200
                        assert _response_text(retrieved.json()) == f"reply:{marker}"

        await asyncio.gather(*(worker(worker_id) for worker_id in range(8)))
        stats = await lifecycle.stats()
        assert stats.active == 0
        assert 0 < stats.retained <= 64
        assert stats.estimated_bytes <= 1024 * 1024
        assert len(engine.requests) == 160

    asyncio.run(scenario())


def test_request_failure_cleans_lifecycle_and_next_request_recovers() -> None:
    async def scenario() -> None:
        engine = _EchoEngine()
        lifecycle = InMemoryResponseLifecycleStore()
        app = create_openai_app(
            engine,
            default_max_output_tokens=8,
            response_lifecycle_store=lifecycle,
        )
        async with await _client(app) as client:
            failed = await client.post("/v1/responses", json={"model": "m", "input": "FAIL"})
            assert failed.status_code == 500
            assert failed.json()["error"]["code"] == "synthetic_failure"
            assert failed.headers["x-request-id"].startswith("req_")
            assert (await lifecycle.stats()).active == 0
            assert (await lifecycle.stats()).retained == 0

            recovered = await client.post(
                "/v1/responses",
                json={"model": "m", "input": "after-failure"},
            )
            assert recovered.status_code == 200
            assert _response_text(recovered.json()) == "reply:after-failure"
            assert (await lifecycle.stats()).active == 0

    asyncio.run(scenario())


def test_x_request_id_matches_canonical_request_for_chat_and_responses() -> None:
    async def scenario() -> None:
        engine = _EchoEngine()
        app = create_openai_app(engine, default_max_output_tokens=8)
        async with await _client(app) as client:
            chat = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "chat"}]},
            )
            assert chat.status_code == 200
            chat_request_id = chat.headers["x-request-id"]
            assert engine.requests[-1].input.request_id == chat_request_id

            response = await client.post("/v1/responses", json={"model": "m", "input": "responses"})
            assert response.status_code == 200
            response_request_id = response.headers["x-request-id"]
            assert engine.requests[-1].input.request_id == response_request_id
            assert response_request_id != chat_request_id

    asyncio.run(scenario())
