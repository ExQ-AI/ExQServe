from __future__ import annotations

import asyncio

import httpx

from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
)
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.api import _iter_responses_sse, create_openai_app
from exqserve.protocol.openai.lifecycle import InMemoryResponseLifecycleStore
from exqserve.protocol.openai.responses import ResponsesStreamSerializer, build_response_object
from exqserve.serving.contracts import ServingRequest


class _Session:
    def __init__(self, events: list[GenerationEvent] | None = None) -> None:
        self.events = list(events or [])
        self.cancel_calls = 0

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self) -> GenerationEvent:
        await asyncio.sleep(0)
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _BlockingSession:
    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._started = False
        self._cancelled = asyncio.Event()
        self._cancel_emitted = False
        self.cancel_calls = 0

    def __aiter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __anext__(self) -> GenerationEvent:
        if not self._started:
            self._started = True
            return GenerationStarted(self._request_id)
        if not self._cancelled.is_set():
            await self._cancelled.wait()
        if not self._cancel_emitted:
            self._cancel_emitted = True
            return GenerationCancelled(self._request_id)
        raise StopAsyncIteration

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled.set()


class _Engine:
    def __init__(self) -> None:
        self.requests: list[ServingRequest] = []

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        usage = TokenUsage(input_tokens=2, output_tokens=1)
        request_id = request.input.request_id
        return _Session(
            [
                GenerationStarted(request_id),
                TextStarted(request_id),
                TextDelta(request_id, "ok"),
                TextCompleted(request_id, "ok"),
                GenerationCompleted(request_id, CompletionReason.STOP, usage),
            ]
        )


async def _request(app, method: str, url: str, **kwargs: object) -> httpx.Response:  # type: ignore[no-untyped-def]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def _initial(response_id: str, *, store: bool = True, text: str = "") -> dict[str, object]:
    return build_response_object(
        response_id=response_id,
        created_at=1,
        model="m",
        status="in_progress",
        output=[] if not text else [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        parallel_tool_calls=True,
        tool_choice="auto",
        usage=None,
        previous_response_id=None,
        store=store,
    )


def test_lifecycle_store_retains_with_sliding_ttl_and_bounded_lru() -> None:
    async def scenario() -> None:
        now = [0.0]
        store = InMemoryResponseLifecycleStore(
            max_records=2,
            ttl_seconds=10,
            max_total_bytes=1024 * 1024,
            clock=lambda: now[0],
        )
        sessions = [_Session(), _Session(), _Session()]
        for index, session in enumerate(sessions, 1):
            response = _initial(f"resp_{index}")
            await store.register_active(response, session, retain=True)
            final = dict(response)
            final["status"] = "completed"
            await store.finish(f"resp_{index}", final)

        assert await store.retrieve("resp_1") is None
        assert await store.retrieve("resp_2") is not None
        now[0] = 9
        assert await store.retrieve("resp_2") is not None
        now[0] = 18
        assert await store.retrieve("resp_2") is not None
        now[0] = 29
        assert await store.retrieve("resp_2") is None
        stats = await store.stats()
        assert stats.active == 0
        assert stats.retained == 0
        assert stats.estimated_bytes == 0

    asyncio.run(scenario())


def test_lifecycle_cancel_isolated_and_retained_when_requested() -> None:
    async def scenario() -> None:
        store = InMemoryResponseLifecycleStore()
        first = _Session()
        second = _Session()
        await store.register_active(_initial("resp_a"), first, retain=True)
        await store.register_active(_initial("resp_b", store=False), second, retain=False)

        cancelled = await store.cancel("resp_a")
        assert cancelled["status"] == "cancelled"
        assert first.cancel_calls == 1
        assert second.cancel_calls == 0
        assert (await store.retrieve("resp_a"))["status"] == "cancelled"  # type: ignore[index]
        stats = await store.stats()
        assert stats.active == 1
        assert stats.retained == 1

        await store.abandon("resp_b")
        assert (await store.stats()).active == 0

    asyncio.run(scenario())


def test_cancel_endpoint_terminates_an_active_response_stream() -> None:
    async def scenario() -> None:
        lifecycle = InMemoryResponseLifecycleStore()
        session = _BlockingSession("req_stream")
        response_id = "resp_stream"
        await lifecycle.register_active(_initial(response_id), session, retain=True)
        serializer = ResponsesStreamSerializer(
            "m",
            response_id=response_id,
            created_at=1,
        )
        stream = _iter_responses_sse(session, serializer, lifecycle, response_id)

        first = await anext(stream)
        assert "event: response.created" in first
        app = create_openai_app(_Engine(), response_lifecycle_store=lifecycle)
        cancelled = await _request(app, "POST", f"/v1/responses/{response_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert session.cancel_calls == 1

        terminal = await anext(stream)
        assert "event: response.incomplete" in terminal
        try:
            await anext(stream)
        except StopAsyncIteration:
            pass
        else:  # pragma: no cover - terminal stream invariant
            raise AssertionError("cancelled response stream must terminate")

        retrieved = await lifecycle.retrieve(response_id)
        assert retrieved is not None
        assert retrieved["status"] == "cancelled"
        assert (await lifecycle.stats()).active == 0

    asyncio.run(scenario())


def test_responses_create_retrieve_store_false_and_terminal_cancel_contract() -> None:
    async def scenario() -> None:
        lifecycle = InMemoryResponseLifecycleStore()
        app = create_openai_app(
            _Engine(),
            default_max_output_tokens=8,
            response_lifecycle_store=lifecycle,
        )

        created = await _request(app, "POST", "/v1/responses", json={"model": "m", "input": "hi"})
        assert created.status_code == 200
        assert created.headers["x-request-id"].startswith("req_")
        response_id = created.json()["id"]

        retrieved = await _request(app, "GET", f"/v1/responses/{response_id}")
        assert retrieved.status_code == 200
        assert retrieved.json() == created.json()
        assert retrieved.headers["x-request-id"].startswith("req_")

        terminal_cancel = await _request(app, "POST", f"/v1/responses/{response_id}/cancel")
        assert terminal_cancel.status_code == 400
        assert terminal_cancel.json()["error"]["code"] == "response_not_cancellable"

        transient = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "m", "input": "hi", "store": False},
        )
        assert transient.status_code == 200
        transient_get = await _request(app, "GET", f"/v1/responses/{transient.json()['id']}")
        assert transient_get.status_code == 404
        assert transient_get.json()["error"]["code"] == "response_not_found"
        assert (await lifecycle.stats()).active == 0

    asyncio.run(scenario())


def test_cancel_endpoint_cancels_registered_active_response_and_keeps_auth_shape() -> None:
    async def scenario() -> None:
        lifecycle = InMemoryResponseLifecycleStore()
        session = _Session()
        await lifecycle.register_active(_initial("resp_live"), session, retain=True)
        app = create_openai_app(_Engine(), response_lifecycle_store=lifecycle)

        response = await _request(app, "POST", "/v1/responses/resp_live/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        assert session.cancel_calls == 1
        assert response.headers["x-request-id"].startswith("req_")

        retrieved = await _request(app, "GET", "/v1/responses/resp_live")
        assert retrieved.status_code == 200
        assert retrieved.json()["status"] == "cancelled"

        unknown = await _request(app, "POST", "/v1/responses/resp_missing/cancel")
        assert unknown.status_code == 404
        assert unknown.json()["error"]["code"] == "response_not_found"

    asyncio.run(scenario())
