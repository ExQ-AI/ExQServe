from __future__ import annotations

import asyncio

from exqserve.core.items import MessageItem, MessageRole
from exqserve.state.store import (
    InMemoryResponseStore,
    ResponseRecord,
    ResponseStoreDisposition,
    estimate_response_record_bytes,
)


def _record(
    response_id: str,
    text: str,
    *,
    parent: str | None = None,
    model: str = "model",
) -> ResponseRecord:
    return ResponseRecord(
        response_id,
        model,
        parent,
        (MessageItem(MessageRole.USER, text),),
    )


def test_store_put_get_materialize_and_lru_eviction() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore(max_records=2)
        assert await store.put(_record("r1", "one")) is ResponseStoreDisposition.STORED
        assert await store.put(_record("r2", "two")) is ResponseStoreDisposition.STORED
        assert (await store.get("r1")).response_id == "r1"  # refresh r1
        assert await store.put(_record("r3", "three")) is ResponseStoreDisposition.STORED

        assert await store.get("r2") is None
        assert await store.materialize("r1") == (MessageItem(MessageRole.USER, "one"),)
        assert (await store.get("r3")).response_id == "r3"
        assert await store.size() == 2

    asyncio.run(scenario())


def test_parent_eviction_cascades_descendants_instead_of_leaving_dangling_child() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore(max_records=2)
        assert await store.put(_record("parent", "first", model="m")) is ResponseStoreDisposition.STORED
        assert (
            await store.put(_record("child", "answer", parent="parent", model="m"))
            is ResponseStoreDisposition.STORED
        )
        assert await store.materialize("child") == (
            MessageItem(MessageRole.USER, "first"),
            MessageItem(MessageRole.USER, "answer"),
        )

        # Adding a third root evicts the oldest parent. Dependency-safe eviction
        # invalidates its child in the same operation.
        assert await store.put(_record("other", "third", model="m")) is ResponseStoreDisposition.STORED
        assert await store.get("parent") is None
        assert await store.get("child") is None
        assert await store.materialize("child") is None
        assert await store.get("other") is not None

    asyncio.run(scenario())


def test_response_record_is_immutable_and_validates_identity() -> None:
    record = _record("r", "x")
    assert record.response_id == "r"
    assert record.model == "model"
    assert record.parent_response_id is None
    assert record.delta_items == (MessageItem(MessageRole.USER, "x"),)


def test_store_idle_ttl_parent_expiry_cascades_child() -> None:
    async def scenario() -> None:
        now = [0.0]
        store = InMemoryResponseStore(
            ttl_seconds=10.0,
            max_total_bytes=1024 * 1024,
            clock=lambda: now[0],
        )
        assert await store.put(_record("r1", "one")) is ResponseStoreDisposition.STORED
        now[0] = 9.0
        assert await store.get("r1") is not None
        now[0] = 10.0
        assert (
            await store.put(_record("r2", "two", parent="r1"))
            is ResponseStoreDisposition.STORED
        )
        now[0] = 18.0
        assert await store.materialize("r2") is not None  # refreshes the whole chain
        now[0] = 29.0
        assert await store.get("r1") is None
        assert await store.get("r2") is None
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
        assert await store.put(first) is ResponseStoreDisposition.STORED
        assert await store.put(second) is ResponseStoreDisposition.STORED

        assert await store.get("r1") is None
        assert await store.get("r2") is not None
        stats = await store.stats()
        assert stats.records == 1
        assert stats.estimated_bytes <= one_record_budget

    asyncio.run(scenario())


def test_store_refuses_child_at_record_budget_without_destroying_committed_ancestors() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore(max_records=3)
        assert await store.put(_record("a", "a", model="m")) is ResponseStoreDisposition.STORED
        assert await store.put(_record("b", "b", parent="a", model="m")) is ResponseStoreDisposition.STORED
        assert await store.put(_record("c", "c", parent="b", model="m")) is ResponseStoreDisposition.STORED

        assert (
            await store.put(_record("d", "d", parent="c", model="m"))
            is ResponseStoreDisposition.REFUSED_BUDGET
        )
        assert await store.size() == 3
        assert await store.materialize("c") == (
            MessageItem(MessageRole.USER, "a"),
            MessageItem(MessageRole.USER, "b"),
            MessageItem(MessageRole.USER, "c"),
        )
        assert await store.get("d") is None

    asyncio.run(scenario())


def test_store_evicts_unrelated_root_instead_of_new_child_ancestors() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore(max_records=4)
        assert await store.put(_record("other", "other", model="m")) is ResponseStoreDisposition.STORED
        assert await store.put(_record("a", "a", model="m")) is ResponseStoreDisposition.STORED
        assert await store.put(_record("b", "b", parent="a", model="m")) is ResponseStoreDisposition.STORED
        assert await store.put(_record("c", "c", parent="b", model="m")) is ResponseStoreDisposition.STORED

        assert await store.put(_record("d", "d", parent="c", model="m")) is ResponseStoreDisposition.STORED
        assert await store.get("other") is None
        assert await store.materialize("d") == (
            MessageItem(MessageRole.USER, "a"),
            MessageItem(MessageRole.USER, "b"),
            MessageItem(MessageRole.USER, "c"),
            MessageItem(MessageRole.USER, "d"),
        )

    asyncio.run(scenario())


def test_store_refuses_child_at_byte_budget_without_destroying_committed_ancestors() -> None:
    async def scenario() -> None:
        records = (
            _record("a", "a" * 64, model="m"),
            _record("b", "b" * 64, parent="a", model="m"),
            _record("c", "c" * 64, parent="b", model="m"),
        )
        budget = sum(estimate_response_record_bytes(record) for record in records)
        store = InMemoryResponseStore(max_records=10, max_total_bytes=budget)
        for record in records:
            assert await store.put(record) is ResponseStoreDisposition.STORED

        assert (
            await store.put(_record("d", "d" * 64, parent="c", model="m"))
            is ResponseStoreDisposition.REFUSED_BUDGET
        )
        assert (await store.stats()).estimated_bytes == budget
        assert await store.materialize("c") is not None
        assert await store.get("d") is None

    asyncio.run(scenario())


def test_store_refuses_single_record_larger_than_budget_explicitly() -> None:
    async def scenario() -> None:
        record = _record("large", "z" * 512)
        size = estimate_response_record_bytes(record)
        store = InMemoryResponseStore(max_total_bytes=size - 1)
        assert await store.put(record) is ResponseStoreDisposition.REFUSED_TOO_LARGE
        assert await store.get("large") is None
        assert (await store.stats()).estimated_bytes == 0

    asyncio.run(scenario())


def test_store_refuses_missing_parent_model_mismatch_and_cycle_without_mutation() -> None:
    async def scenario() -> None:
        store = InMemoryResponseStore()
        assert (
            await store.put(_record("orphan", "x", parent="missing"))
            is ResponseStoreDisposition.REFUSED_MISSING_PARENT
        )

        assert await store.put(_record("a", "a", model="m")) is ResponseStoreDisposition.STORED
        assert (
            await store.put(_record("bad-model", "b", parent="a", model="other"))
            is ResponseStoreDisposition.REFUSED_MODEL_MISMATCH
        )
        assert (
            await store.put(_record("b", "b", parent="a", model="m"))
            is ResponseStoreDisposition.STORED
        )
        assert (
            await store.put(_record("a", "replacement", parent="b", model="m"))
            is ResponseStoreDisposition.REFUSED_INVALID_GRAPH
        )
        assert await store.materialize("b") == (
            MessageItem(MessageRole.USER, "a"),
            MessageItem(MessageRole.USER, "b"),
        )

    asyncio.run(scenario())


def test_store_1000_hop_chain_materializes_iteratively_with_linear_retained_payload() -> None:
    async def scenario() -> None:
        count = 1000
        store = InMemoryResponseStore(max_records=count + 10, max_total_bytes=16 * 1024 * 1024)
        records: list[ResponseRecord] = []
        parent: str | None = None
        for index in range(count):
            response_id = f"r{index:04d}"
            record = _record(response_id, f"v{index:04d}", parent=parent)
            records.append(record)
            assert await store.put(record) is ResponseStoreDisposition.STORED
            parent = response_id

        assert parent is not None
        materialized = await store.materialize(parent)
        assert materialized is not None
        assert len(materialized) == count
        assert materialized[0] == MessageItem(MessageRole.USER, "v0000")
        assert materialized[-1] == MessageItem(MessageRole.USER, "v0999")

        stats = await store.stats()
        assert stats.records == count
        # The budget is exactly the sum of unique response deltas; no ancestor
        # context is retained again inside each child record.
        assert stats.estimated_bytes == sum(estimate_response_record_bytes(record) for record in records)

    asyncio.run(scenario())

def test_store_pinned_chain_survives_ttl_and_capacity_until_all_pins_release() -> None:
    async def scenario() -> None:
        now = [0.0]
        store = InMemoryResponseStore(
            max_records=2,
            ttl_seconds=1.0,
            clock=lambda: now[0],
        )
        assert await store.put(_record("parent", "p", model="m")) is ResponseStoreDisposition.STORED
        assert (
            await store.put(_record("child", "c", parent="parent", model="m"))
            is ResponseStoreDisposition.STORED
        )

        first_pin = await store.pin_chain("child")
        second_pin = await store.pin_chain("child")
        assert first_pin == ("child", "parent")
        assert second_pin == first_pin

        now[0] = 2.0
        assert await store.get("parent") is not None
        assert (
            await store.put(_record("other", "o", model="m"))
            is ResponseStoreDisposition.REFUSED_BUDGET
        )

        await store.unpin_chain(first_pin)
        now[0] = 4.0
        assert await store.materialize("child") is not None

        await store.unpin_chain(second_pin)
        now[0] = 6.0
        assert await store.get("parent") is None
