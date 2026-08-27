from __future__ import annotations

import asyncio

from exqserve.core.items import MessageItem, MessageRole
from exqserve.state.store import (
    InMemoryResponseStore,
    ResponseRecord,
    estimate_response_record_bytes,
)


def _record(response_id: str, text: str) -> ResponseRecord:
    return ResponseRecord(
        response_id,
        "model",
        (MessageItem(MessageRole.USER, text),),
    )


def test_store_put_get_and_lru_eviction() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore(max_records=2)
        await store.put(_record("r1", "one"))
        await store.put(_record("r2", "two"))
        assert (await store.get("r1")).response_id == "r1"  # refresh r1
        await store.put(_record("r3", "three"))

        assert await store.get("r2") is None
        assert (await store.get("r1")).context_items == (MessageItem(MessageRole.USER, "one"),)
        assert (await store.get("r3")).response_id == "r3"
        assert await store.size() == 2

    asyncio.run(scenario())


def test_descendant_flattened_context_survives_parent_eviction() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore(max_records=1)
        parent = ResponseRecord("parent", "m", (MessageItem(MessageRole.USER, "first"),))
        await store.put(parent)

        child_context = parent.context_items + (MessageItem(MessageRole.ASSISTANT, "answer"),)
        child = ResponseRecord("child", "m", child_context)
        await store.put(child)

        assert await store.get("parent") is None
        restored = await store.get("child")
        assert restored is not None
        assert restored.context_items == child_context

    asyncio.run(scenario())


def test_response_record_is_immutable_and_validates_identity() -> None:
    record = _record("r", "x")
    assert record.response_id == "r"
    assert record.model == "model"


def test_store_idle_ttl_expires_and_reads_refresh_expiry() -> None:
    async def scenario() -> None:
        now = [0.0]
        store = InMemoryResponseStore(
            ttl_seconds=10.0,
            max_total_bytes=1024 * 1024,
            clock=lambda: now[0],
        )
        await store.put(_record("r1", "one"))
        now[0] = 9.0
        assert await store.get("r1") is not None
        now[0] = 18.0
        assert await store.get("r1") is not None
        now[0] = 29.0
        assert await store.get("r1") is None
        assert await store.size() == 0

    asyncio.run(scenario())


def test_store_evicts_lru_by_aggregate_byte_budget() -> None:
    async def scenario() -> None:
        first = _record("r1", "x" * 64)
        second = _record("r2", "y" * 64)
        one_record_budget = max(
            estimate_response_record_bytes(first),
            estimate_response_record_bytes(second),
        ) + 8
        store = InMemoryResponseStore(max_records=10, max_total_bytes=one_record_budget)
        await store.put(first)
        await store.put(second)

        assert await store.get("r1") is None
        assert await store.get("r2") is not None
        stats = await store.stats()
        assert stats.records == 1
        assert stats.estimated_bytes <= one_record_budget

    asyncio.run(scenario())


def test_store_does_not_retain_single_record_larger_than_budget() -> None:
    async def scenario() -> None:
        record = _record("large", "z" * 512)
        size = estimate_response_record_bytes(record)
        store = InMemoryResponseStore(max_total_bytes=size - 1)
        await store.put(record)
        assert await store.get("large") is None
        assert (await store.stats()).estimated_bytes == 0

    asyncio.run(scenario())
