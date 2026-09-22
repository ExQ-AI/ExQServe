from __future__ import annotations

import asyncio

import httpx
import pytest

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
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.api import _iter_responses_sse, create_openai_app
from exqserve.protocol.openai.responses import ResponsesStreamSerializer, build_response_object
from exqserve.serving.contracts import ServingRequest
from exqserve.state.response_authority import ResponseStateAuthority, ResponseStateNotFound
from exqserve.state.response_lifecycle import (
    InMemoryResponseLifecycleStore,
    ResponseLifecycleRetentionRefused,
)
from exqserve.state.session import StatefulServingSession
from exqserve.state.store import InMemoryResponseStore, ResponseRecord, ResponseStoreDisposition


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


class _DelayedCancelSession(_Session):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self.cancel_started.set()
        await self.release_cancel.wait()


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


def _initial(
    response_id: str,
    *,
    store: bool = True,
    text: str = "",
    parent: str | None = None,
) -> dict[str, object]:
    return build_response_object(
        response_id=response_id,
        created_at=1,
        model="m",
        status="in_progress",
        output=[] if not text else [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        parallel_tool_calls=True,
        tool_choice="auto",
        usage=None,
        previous_response_id=parent,
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


def test_lifecycle_cancel_returns_completed_when_finish_wins_during_cancel() -> None:
    async def scenario() -> None:
        store = InMemoryResponseLifecycleStore()
        session = _DelayedCancelSession()
        initial = _initial("resp_race")
        await store.register_active(initial, session, retain=True)

        cancel_task = asyncio.create_task(store.cancel("resp_race"))
        await session.cancel_started.wait()
        completed = dict(initial)
        completed["status"] = "completed"
        await store.finish("resp_race", completed)
        session.release_cancel.set()

        cancelled = await cancel_task
        assert cancelled["status"] == "completed"
        retrieved = await store.retrieve("resp_race")
        assert retrieved is not None
        assert retrieved["status"] == "completed"

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
        authority = ResponseStateAuthority(InMemoryResponseStore(), lifecycle)
        stream = _iter_responses_sse(session, serializer, authority, response_id)

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


def test_lifecycle_retention_budget_refusal_is_not_published_as_success() -> None:
    async def scenario() -> None:
        lifecycle = InMemoryResponseLifecycleStore(max_total_bytes=1)
        state = InMemoryResponseStore()
        app = create_openai_app(
            _Engine(),
            default_max_output_tokens=8,
            response_store=state,
            response_lifecycle_store=lifecycle,
        )

        response = await _request(
            app,
            "POST",
            "/v1/responses",
            json={"model": "m", "input": "hi", "store": True},
        )
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "response_store_refused"
        assert (await state.stats()).records == 0
        stats = await lifecycle.stats()
        assert stats.active == 0
        assert stats.retained == 0

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


def test_response_authority_stages_completion_until_lifecycle_finish() -> None:
    async def scenario() -> None:
        response_id = "resp_staged"
        state = InMemoryResponseStore()
        lifecycle = InMemoryResponseLifecycleStore()
        authority = ResponseStateAuthority(state, lifecycle)
        session = _Session()
        initial = _initial(response_id)
        await authority.register_active(initial, session, retain=True)
        record = ResponseRecord(
            response_id,
            "m",
            None,
            (
                MessageItem(MessageRole.USER, "go"),
                MessageItem(MessageRole.ASSISTANT, "done"),
            ),
        )

        assert await authority.put(record) is ResponseStoreDisposition.STORED
        assert await state.get(response_id) is None
        try:
            await authority.resolve_previous(response_id, "m")
        except ResponseStateNotFound:
            pass
        else:  # pragma: no cover - atomicity invariant
            raise AssertionError("staged completion must not be visible to continuation")

        completed = dict(initial)
        completed["status"] = "completed"
        assert await authority.finish(response_id, completed) is True
        assert await state.get(response_id) == record
        assert await authority.resolve_previous(response_id, "m") == record.delta_items

    asyncio.run(scenario())


def test_response_authority_cancel_winner_cannot_split_terminal_identity() -> None:
    async def scenario() -> None:
        for store in (True, False):
            for terminal_status in ("completed", "incomplete", "failed"):
                response_id = f"resp_cancel_wins_{store}_{terminal_status}"
                state = InMemoryResponseStore()
                lifecycle = InMemoryResponseLifecycleStore()
                authority = ResponseStateAuthority(state, lifecycle)
                session = _Session()
                initial = _initial(response_id, store=store)
                await authority.register_active(initial, session, retain=store)
                if store and terminal_status in {"completed", "incomplete"}:
                    assert await authority.put(
                        ResponseRecord(
                            response_id,
                            "m",
                            None,
                            (MessageItem(MessageRole.ASSISTANT, "done"),),
                        )
                    ) is ResponseStoreDisposition.STORED

                cancelled = await authority.cancel(response_id)
                assert cancelled["status"] == "cancelled"
                terminal = dict(initial)
                terminal["status"] = terminal_status
                assert await authority.finish(response_id, terminal) is True

                assert terminal["status"] == "cancelled"
                assert await state.get(response_id) is None
                retained = await authority.retrieve(response_id)
                if store:
                    assert retained is not None
                    assert retained["status"] == "cancelled"
                else:
                    assert retained is None
                assert await authority.is_terminal(response_id) is True

                await authority.abandon(response_id)
                assert await authority.is_terminal(response_id) is False
                assert not authority._pending_records
                assert not authority._terminal_winners
                assert not authority._transitions

    asyncio.run(scenario())


def test_response_authority_finish_winner_cannot_be_replaced_by_cancel() -> None:
    async def scenario() -> None:
        for store in (True, False):
            for terminal_status in ("completed", "incomplete", "failed"):
                response_id = f"resp_finish_wins_{store}_{terminal_status}"
                state = InMemoryResponseStore()
                lifecycle = InMemoryResponseLifecycleStore()
                authority = ResponseStateAuthority(state, lifecycle)
                session = _Session()
                initial = _initial(response_id, store=store)
                await authority.register_active(initial, session, retain=store)
                if store and terminal_status in {"completed", "incomplete"}:
                    assert await authority.put(
                        ResponseRecord(
                            response_id,
                            "m",
                            None,
                            (MessageItem(MessageRole.ASSISTANT, "done"),),
                        )
                    ) is ResponseStoreDisposition.STORED

                terminal = dict(initial)
                terminal["status"] = terminal_status
                assert await authority.finish(response_id, terminal) is True
                winner = await authority.cancel(response_id)

                assert winner["status"] == terminal_status
                assert terminal["status"] == terminal_status
                assert session.cancel_calls == 0
                retained = await authority.retrieve(response_id)
                if store:
                    assert retained is not None
                    assert retained["status"] == terminal_status
                else:
                    assert retained is None

                await authority.abandon(response_id)
                if store:
                    retained = await authority.retrieve(response_id)
                    assert retained is not None
                    assert retained["status"] == terminal_status
                assert not authority._pending_records
                assert not authority._terminal_winners
                assert not authority._transitions

    asyncio.run(scenario())


def test_responses_cancel_then_immediate_close_preserves_terminal_identity() -> None:
    async def scenario() -> None:
        response_id = "resp_cancel_then_close"
        state = InMemoryResponseStore()
        lifecycle = InMemoryResponseLifecycleStore()
        authority = ResponseStateAuthority(state, lifecycle)
        session = _BlockingSession("req-cancel-then-close")
        initial = _initial(response_id)
        await authority.register_active(initial, session, retain=True)
        stream = _iter_responses_sse(
            session,
            ResponsesStreamSerializer("m", response_id=response_id, created_at=1, store=True),
            authority,
            response_id,
        )

        assert "event: response.created" in await anext(stream)
        cancelled = await authority.cancel(response_id)
        assert cancelled["status"] == "cancelled"
        assert session.cancel_calls == 1
        retained = await authority.retrieve(response_id)
        assert retained is not None
        assert retained["status"] == "cancelled"

        await stream.aclose()

        retained = await authority.retrieve(response_id)
        assert retained is not None
        assert retained["status"] == "cancelled"
        assert session.cancel_calls == 1
        assert not authority._pending_records
        assert not authority._terminal_winners
        assert not authority._transitions

    asyncio.run(scenario())


def test_responses_terminal_close_does_not_abandon_committed_state() -> None:
    async def scenario() -> None:
        response_id = "resp_terminal_close"
        state = InMemoryResponseStore()
        lifecycle = InMemoryResponseLifecycleStore()
        authority = ResponseStateAuthority(state, lifecycle)
        inner = _Session(
            [
                GenerationStarted("req-terminal-close"),
                TextStarted("req-terminal-close"),
                TextDelta("req-terminal-close", "done"),
                TextCompleted("req-terminal-close", "done"),
                GenerationCompleted("req-terminal-close", CompletionReason.STOP),
            ]
        )
        session = StatefulServingSession(
            inner,
            authority,
            response_id=response_id,
            model="m",
            base_context=(),
            current_input=(MessageItem(MessageRole.USER, "go"),),
            store_response=True,
        )
        initial = _initial(response_id)
        await authority.register_active(initial, session, retain=True)
        stream = _iter_responses_sse(
            session,
            ResponsesStreamSerializer("m", response_id=response_id, created_at=1, store=True),
            authority,
            response_id,
        )

        while True:
            chunk = await anext(stream)
            if "event: response.completed" in chunk:
                break
        await stream.aclose()

        retained = await authority.retrieve(response_id)
        assert retained is not None
        assert retained["status"] == "completed"
        assert await state.materialize(response_id) is not None
        assert inner.cancel_calls == 0

    asyncio.run(scenario())

@pytest.mark.parametrize("status", ("completed", "incomplete", "failed"))
def test_retained_terminal_capacity_refusal_is_uniform_for_finish_statuses(status: str) -> None:
    async def scenario() -> None:
        lifecycle = InMemoryResponseLifecycleStore(max_records=1)
        authority = ResponseStateAuthority(InMemoryResponseStore(max_records=10), lifecycle)

        parent_session = _Session()
        parent = _initial("resp_parent")
        await lifecycle.register_active(parent, parent_session, retain=True)
        parent_terminal = dict(parent)
        parent_terminal["status"] = "completed"
        assert await lifecycle.finish("resp_parent", parent_terminal)

        child_session = _Session()
        child = _initial("resp_child", parent="resp_parent")
        await authority.register_active(child, child_session, retain=True)
        terminal = dict(child)
        terminal["status"] = status

        assert not await authority.finish("resp_child", terminal)
        assert await authority.retrieve("resp_child") is None
        stats = await lifecycle.stats()
        assert stats.active == 0
        assert stats.retained == 1
        retained_parent = await lifecycle.retrieve("resp_parent")
        assert retained_parent is not None
        assert retained_parent["status"] == "completed"

    asyncio.run(scenario())


def test_retained_cancel_capacity_refusal_is_explicit_and_does_not_leave_active_state() -> None:
    async def scenario() -> None:
        lifecycle = InMemoryResponseLifecycleStore(max_records=1)
        authority = ResponseStateAuthority(InMemoryResponseStore(max_records=10), lifecycle)

        parent_session = _Session()
        parent = _initial("resp_parent")
        await lifecycle.register_active(parent, parent_session, retain=True)
        parent_terminal = dict(parent)
        parent_terminal["status"] = "completed"
        assert await lifecycle.finish("resp_parent", parent_terminal)

        child_session = _Session()
        child = _initial("resp_child", parent="resp_parent")
        await authority.register_active(child, child_session, retain=True)

        with pytest.raises(ResponseLifecycleRetentionRefused):
            await authority.cancel("resp_child")

        assert child_session.cancel_calls == 1
        assert await authority.retrieve("resp_child") is None
        stats = await lifecycle.stats()
        assert stats.active == 0
        assert stats.retained == 1
        retained_parent = await lifecycle.retrieve("resp_parent")
        assert retained_parent is not None
        assert retained_parent["status"] == "completed"

    asyncio.run(scenario())


def test_cancel_endpoint_maps_retention_refusal_to_response_store_refused() -> None:
    async def scenario() -> None:
        lifecycle = InMemoryResponseLifecycleStore(max_records=1)
        parent_session = _Session()
        parent = _initial("resp_parent")
        await lifecycle.register_active(parent, parent_session, retain=True)
        parent_terminal = dict(parent)
        parent_terminal["status"] = "completed"
        assert await lifecycle.finish("resp_parent", parent_terminal)

        child_session = _Session()
        child = _initial("resp_child", parent="resp_parent")
        await lifecycle.register_active(child, child_session, retain=True)
        app = create_openai_app(_Engine(), response_lifecycle_store=lifecycle)

        response = await _request(app, "POST", "/v1/responses/resp_child/cancel")
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "response_store_refused"
        assert child_session.cancel_calls == 1

        child_get = await _request(app, "GET", "/v1/responses/resp_child")
        assert child_get.status_code == 404
        stats = await lifecycle.stats()
        assert stats.active == 0
        assert stats.retained == 1
        retained_parent = await lifecycle.retrieve("resp_parent")
        assert retained_parent is not None
        assert retained_parent["status"] == "completed"

    asyncio.run(scenario())
