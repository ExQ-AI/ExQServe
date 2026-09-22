from __future__ import annotations

import asyncio
import json

import pytest

from exqserve.core.items import MessageItem, MessageRole
from exqserve.protocol.openai.responses import build_response_object
from exqserve.state.response_authority import ResponseStateAuthority, ResponseStateNotFound
from exqserve.state.response_lifecycle import InMemoryResponseLifecycleStore
from exqserve.state.store import InMemoryResponseStore, ResponseRecord, ResponseStoreDisposition


class _Session:
    def __init__(self) -> None:
        self.cancel_calls = 0

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _BlockingCancelSession:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_calls = 0

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self.started.set()
        await self.release.wait()


class _PausingDiscardStateStore(InMemoryResponseStore):
    def __init__(self) -> None:
        super().__init__(max_records=10)
        self.discard_started = asyncio.Event()
        self.release_discard = asyncio.Event()

    async def discard(self, response_id: str) -> None:
        if response_id == "resp_a":
            self.discard_started.set()
            await self.release_discard.wait()
        await super().discard(response_id)


def _wire(
    response_id: str,
    *,
    parent: str | None = None,
    status: str = "in_progress",
    blob: str | None = None,
    store: bool = True,
) -> dict[str, object]:
    response = build_response_object(
        response_id=response_id,
        created_at=1,
        model="m",
        status=status,
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        usage=None,
        previous_response_id=parent,
        store=store,
    )
    if blob is not None:
        response["test_blob"] = blob
    return response


async def _commit(
    authority: ResponseStateAuthority,
    response_id: str,
    parent: str | None = None,
) -> None:
    assert (
        await authority.put(
            ResponseRecord(
                response_id,
                "m",
                parent,
                (MessageItem(MessageRole.USER, response_id),),
            )
        )
        is ResponseStoreDisposition.STORED
    )
    initial = _wire(response_id, parent=parent)
    await authority.register_active(initial, _Session(), retain=True)
    terminal = dict(initial)
    terminal["status"] = "completed"
    assert await authority.finish(response_id, terminal)


def test_partial_continuation_pin_acquire_cancellation_rolls_back_lifecycle_pin() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore()
        lifecycle = InMemoryResponseLifecycleStore()
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_parent")

        await state._lock.acquire()
        task = asyncio.create_task(
            authority.resolve_previous("resp_parent", "m", pin_owner="resp_child")
        )
        try:
            for _ in range(100):
                if lifecycle._pins.get("resp_parent") == 1:
                    break
                await asyncio.sleep(0)
            assert lifecycle._pins == {"resp_parent": 1}
            assert state._pins == {}
            task.cancel()
        finally:
            state._lock.release()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert lifecycle._pins == {}
        assert state._pins == {}

    asyncio.run(scenario())


def test_continuation_release_completes_both_unpins_despite_caller_cancellation() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore()
        lifecycle = InMemoryResponseLifecycleStore()
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_parent")
        await authority.resolve_previous("resp_parent", "m", pin_owner="resp_child")
        assert state._pins == {"resp_parent": 1}
        assert lifecycle._pins == {"resp_parent": 1}

        await state._lock.acquire()
        task = asyncio.create_task(authority.release_continuation("resp_child"))
        try:
            for _ in range(100):
                if "resp_child" not in authority._continuation_pins:
                    break
                await asyncio.sleep(0)
            assert "resp_child" not in authority._continuation_pins
            task.cancel()
        finally:
            state._lock.release()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert state._pins == {}
        assert lifecycle._pins == {}

    asyncio.run(scenario())


def test_lifecycle_record_budget_evicts_dependency_subtree_in_both_stores() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore(max_records=10)
        lifecycle = InMemoryResponseLifecycleStore(max_records=2)
        authority = ResponseStateAuthority(state, lifecycle)

        await _commit(authority, "resp_a")
        await _commit(authority, "resp_b", "resp_a")
        await _commit(authority, "resp_c")

        life_a = await lifecycle.retrieve("resp_a")
        life_b = await lifecycle.retrieve("resp_b")
        life_c = await lifecycle.retrieve("resp_c")
        state_a = await state.get("resp_a")
        state_b = await state.get("resp_b")
        state_c = await state.get("resp_c")

        assert life_c is not None
        assert state_c is not None
        assert (life_a is None) == (state_a is None)
        assert (life_b is None) == (state_b is None)
        if life_b is not None:
            assert life_a is not None
            assert state_a is not None
        assert (await lifecycle.stats()).retained <= 2
        assert (await state.stats()).records == (await lifecycle.stats()).retained

    asyncio.run(scenario())


def test_lifecycle_byte_budget_evicts_parent_and_child_as_one_subtree() -> None:
    async def scenario() -> None:
        blob = "x" * 256
        a_terminal = _wire("resp_a", status="completed", blob=blob)
        b_terminal = _wire("resp_b", parent="resp_a", status="completed", blob=blob)
        byte_limit = sum(
            len(
                json.dumps(
                    response,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            for response in (a_terminal, b_terminal)
        )
        lifecycle = InMemoryResponseLifecycleStore(max_records=10, max_total_bytes=byte_limit)

        async def retain(response_id: str, parent: str | None) -> None:
            initial = _wire(response_id, parent=parent, blob=blob)
            await lifecycle.register_active(initial, _Session(), retain=True)
            terminal = dict(initial)
            terminal["status"] = "completed"
            assert await lifecycle.finish(response_id, terminal)

        await retain("resp_a", None)
        await retain("resp_b", "resp_a")
        await retain("resp_c", None)

        assert await lifecycle.retrieve("resp_a") is None
        assert await lifecycle.retrieve("resp_b") is None
        assert await lifecycle.retrieve("resp_c") is not None
        assert (await lifecycle.stats()).retained == 1

    asyncio.run(scenario())


def test_retained_commit_serializes_continuation_pin_acquisition() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore(max_records=1)
        lifecycle = InMemoryResponseLifecycleStore(max_records=1)
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_a")

        assert (
            await authority.put(
                ResponseRecord(
                    "resp_c",
                    "m",
                    None,
                    (MessageItem(MessageRole.USER, "resp_c"),),
                )
            )
            is ResponseStoreDisposition.STORED
        )
        initial_c = _wire("resp_c")
        await authority.register_active(initial_c, _Session(), retain=True)
        terminal_c = dict(initial_c)
        terminal_c["status"] = "completed"

        await state._lock.acquire()
        finish_task = asyncio.create_task(authority.finish("resp_c", terminal_c))
        try:
            for _ in range(100):
                waiters = getattr(state._lock, "_waiters", None)
                if waiters:
                    break
                await asyncio.sleep(0)
            resolve_task = asyncio.create_task(
                authority.resolve_previous("resp_a", "m", pin_owner="resp_child")
            )
            await asyncio.sleep(0)
            assert lifecycle._pins == {}
            assert state._pins == {}
        finally:
            state._lock.release()

        assert await finish_task is True
        with pytest.raises(ResponseStateNotFound):
            await resolve_task
        assert await state.get("resp_a") is None
        assert await lifecycle.retrieve("resp_a") is None
        assert await state.get("resp_c") is not None
        assert await lifecycle.retrieve("resp_c") is not None

    asyncio.run(scenario())


def test_retained_commit_finishes_cross_store_transaction_before_propagating_cancel() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore(max_records=1)
        lifecycle = InMemoryResponseLifecycleStore(max_records=1)
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_a")

        await authority.put(
            ResponseRecord(
                "resp_c",
                "m",
                None,
                (MessageItem(MessageRole.USER, "resp_c"),),
            )
        )
        initial_c = _wire("resp_c")
        await authority.register_active(initial_c, _Session(), retain=True)
        terminal_c = dict(initial_c)
        terminal_c["status"] = "completed"

        await state._lock.acquire()
        finish_task = asyncio.create_task(authority.finish("resp_c", terminal_c))
        try:
            for _ in range(100):
                waiters = getattr(state._lock, "_waiters", None)
                if waiters:
                    break
                await asyncio.sleep(0)
            await lifecycle._lock.acquire()
        finally:
            state._lock.release()

        try:
            for _ in range(100):
                if "resp_c" in state._records and "resp_a" not in state._records:
                    break
                await asyncio.sleep(0)
            finish_task.cancel()
        finally:
            lifecycle._lock.release()

        with pytest.raises(asyncio.CancelledError):
            await finish_task
        assert await state.get("resp_a") is None
        assert await lifecycle.retrieve("resp_a") is None
        assert await state.get("resp_c") is not None
        retained_c = await lifecycle.retrieve("resp_c")
        assert retained_c is not None
        assert retained_c["status"] == "completed"
        assert (await lifecycle.stats()).active == 0

    asyncio.run(scenario())


def test_eviction_sync_completes_before_finish_cancellation_returns() -> None:
    async def scenario() -> None:
        state = _PausingDiscardStateStore()
        lifecycle = InMemoryResponseLifecycleStore(max_records=1)
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_a")

        await authority.put(
            ResponseRecord(
                "resp_c",
                "m",
                None,
                (MessageItem(MessageRole.USER, "resp_c"),),
            )
        )
        initial_c = _wire("resp_c")
        await authority.register_active(initial_c, _Session(), retain=True)
        terminal_c = dict(initial_c)
        terminal_c["status"] = "completed"
        finish_task = asyncio.create_task(authority.finish("resp_c", terminal_c))

        await asyncio.wait_for(state.discard_started.wait(), timeout=1)
        finish_task.cancel()
        await asyncio.sleep(0)
        assert not finish_task.done()
        state.release_discard.set()

        with pytest.raises(asyncio.CancelledError):
            await finish_task
        assert await state.get("resp_a") is None
        assert await lifecycle.retrieve("resp_a") is None
        assert await state.get("resp_c") is not None
        assert await lifecycle.retrieve("resp_c") is not None
        assert (await state.stats()).records == (await lifecycle.stats()).retained == 1

    asyncio.run(scenario())


def test_eviction_sync_completes_before_retained_cancel_cancellation_returns() -> None:
    async def scenario() -> None:
        state = _PausingDiscardStateStore()
        lifecycle = InMemoryResponseLifecycleStore(max_records=1)
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_a")

        await authority.put(
            ResponseRecord(
                "resp_c",
                "m",
                None,
                (MessageItem(MessageRole.USER, "resp_c"),),
            )
        )
        session_c = _Session()
        await authority.register_active(_wire("resp_c"), session_c, retain=True)
        cancel_task = asyncio.create_task(authority.cancel("resp_c"))

        await asyncio.wait_for(state.discard_started.wait(), timeout=1)
        assert await lifecycle.retrieve("resp_a") is None
        retained_c = await lifecycle.retrieve("resp_c")
        assert retained_c is not None
        assert retained_c["status"] == "cancelled"
        assert session_c.cancel_calls == 1

        cancel_task.cancel()
        await asyncio.sleep(0)
        assert not cancel_task.done()
        state.release_discard.set()

        with pytest.raises(asyncio.CancelledError):
            await cancel_task
        assert await state.get("resp_a") is None
        assert await lifecycle.retrieve("resp_a") is None
        assert await state.get("resp_c") is None
        retained_c = await lifecycle.retrieve("resp_c")
        assert retained_c is not None
        assert retained_c["status"] == "cancelled"
        assert (await state.stats()).records == 0
        assert (await lifecycle.stats()).retained == 1

    asyncio.run(scenario())


def test_blocked_retained_cancel_does_not_block_unrelated_continuation() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore(max_records=10)
        lifecycle = InMemoryResponseLifecycleStore(max_records=10)
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_parent")

        blocking = _BlockingCancelSession()
        await authority.register_active(_wire("resp_cancel"), blocking, retain=True)
        cancel_task = asyncio.create_task(authority.cancel("resp_cancel"))
        await asyncio.wait_for(blocking.started.wait(), timeout=1)

        resolve_task = asyncio.create_task(
            authority.resolve_previous("resp_parent", "m", pin_owner="resp_child")
        )
        resolved = await asyncio.wait_for(resolve_task, timeout=1)
        assert len(resolved) == 1
        assert not cancel_task.done()
        assert not authority._retained_commit_lock.locked()

        blocking.release.set()
        await asyncio.wait_for(cancel_task, timeout=1)
        await authority.release_continuation("resp_child")

    asyncio.run(scenario())


def test_blocked_nonretained_cancel_never_holds_retained_commit_gate() -> None:
    async def scenario() -> None:
        state = InMemoryResponseStore(max_records=10)
        lifecycle = InMemoryResponseLifecycleStore(max_records=10)
        authority = ResponseStateAuthority(state, lifecycle)
        await _commit(authority, "resp_parent")

        blocking = _BlockingCancelSession()
        await authority.register_active(
            _wire("resp_ephemeral", store=False),
            blocking,
            retain=False,
        )
        cancel_task = asyncio.create_task(authority.cancel("resp_ephemeral"))
        await asyncio.wait_for(blocking.started.wait(), timeout=1)

        resolved = await asyncio.wait_for(
            authority.resolve_previous("resp_parent", "m", pin_owner="resp_child"),
            timeout=1,
        )
        assert len(resolved) == 1
        assert not cancel_task.done()
        assert not authority._retained_commit_lock.locked()

        blocking.release.set()
        await asyncio.wait_for(cancel_task, timeout=1)
        await authority.release_continuation("resp_child")

    asyncio.run(scenario())
