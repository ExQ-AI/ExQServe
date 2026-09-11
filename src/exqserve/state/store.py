"""Protocol-neutral response records and bounded parent-linked in-memory state store."""

from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Protocol

from exqserve.core.items import CanonicalItem


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    response_id: str
    model: str
    parent_response_id: str | None
    delta_items: tuple[CanonicalItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.response_id, str) or not self.response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.parent_response_id is not None and (
            not isinstance(self.parent_response_id, str) or not self.parent_response_id.strip()
        ):
            raise ValueError("parent_response_id must be None or a non-empty string")
        if not isinstance(self.delta_items, tuple):
            raise TypeError("delta_items must be a tuple")


class ResponseStoreDisposition(str, Enum):
    STORED = "stored"
    REFUSED_TOO_LARGE = "refused_too_large"
    REFUSED_MISSING_PARENT = "refused_missing_parent"
    REFUSED_MODEL_MISMATCH = "refused_model_mismatch"
    REFUSED_INVALID_GRAPH = "refused_invalid_graph"
    REFUSED_BUDGET = "refused_budget"


class ResponseStoreInvariantError(RuntimeError):
    """Raised when injected/corrupt response graph state violates store invariants."""


@dataclass(frozen=True, slots=True)
class ResponseStoreStats:
    records: int
    estimated_bytes: int


class ResponseStore(Protocol):
    async def get(self, response_id: str) -> ResponseRecord | None:
        ...

    async def materialize(self, response_id: str) -> tuple[CanonicalItem, ...] | None:
        ...

    async def put(self, record: ResponseRecord) -> ResponseStoreDisposition:
        ...

    async def discard(self, response_id: str) -> None:
        ...


@dataclass(slots=True)
class _StoredRecord:
    record: ResponseRecord
    estimated_bytes: int
    expires_at: float


def _estimate_value_bytes(value: object) -> int:
    """Deterministic retained-state estimate used only for the store budget."""

    if value is None:
        return 1
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, Enum):
        return _estimate_value_bytes(value.value)
    if isinstance(value, bool | int | float):
        return 16
    if isinstance(value, tuple | list):
        return 16 + sum(_estimate_value_bytes(item) for item in value)
    if isinstance(value, dict):
        return 32 + sum(
            _estimate_value_bytes(key) + _estimate_value_bytes(item) for key, item in value.items()
        )
    if is_dataclass(value) and not isinstance(value, type):
        return 32 + sum(
            len(field.name.encode("utf-8")) + _estimate_value_bytes(getattr(value, field.name))
            for field in fields(value)
        )
    return len(repr(value).encode("utf-8"))


def estimate_response_record_bytes(record: ResponseRecord) -> int:
    if not isinstance(record, ResponseRecord):
        raise TypeError("record must be a ResponseRecord")
    return _estimate_value_bytes(record)


class InMemoryResponseStore:
    """Bound response deltas by dependency-safe LRU, sliding TTL, and byte budget."""

    def __init__(
        self,
        max_records: int = 1024,
        *,
        ttl_seconds: float = 3600.0,
        max_total_bytes: int = 64 * 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool):
            raise TypeError("max_records must be an integer")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        if not isinstance(ttl_seconds, int | float) or isinstance(ttl_seconds, bool):
            raise TypeError("ttl_seconds must be a number")
        if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive and finite")
        if not isinstance(max_total_bytes, int) or isinstance(max_total_bytes, bool):
            raise TypeError("max_total_bytes must be an integer")
        if max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._max_records = max_records
        self._ttl_seconds = float(ttl_seconds)
        self._max_total_bytes = max_total_bytes
        self._clock = clock
        self._records: OrderedDict[str, _StoredRecord] = OrderedDict()
        self._children: dict[str, set[str]] = {}
        self._estimated_bytes = 0
        self._lock = asyncio.Lock()

    @staticmethod
    def _validate_response_id(response_id: str) -> None:
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string")

    def _remove_one_locked(self, response_id: str) -> None:
        stored = self._records.pop(response_id, None)
        if stored is None:
            self._children.pop(response_id, None)
            return
        self._estimated_bytes -= stored.estimated_bytes
        parent_id = stored.record.parent_response_id
        if parent_id is not None:
            siblings = self._children.get(parent_id)
            if siblings is not None:
                siblings.discard(response_id)
                if not siblings:
                    self._children.pop(parent_id, None)
        self._children.pop(response_id, None)

    def _remove_subtree_locked(self, response_id: str) -> None:
        stack = [response_id]
        ordered: list[str] = []
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            ordered.append(current)
            stack.extend(self._children.get(current, ()))
        for current in reversed(ordered):
            self._remove_one_locked(current)

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            response_id
            for response_id, stored in self._records.items()
            if stored.expires_at <= now
        ]
        for response_id in expired:
            if response_id in self._records:
                self._remove_subtree_locked(response_id)

    def _evict_to_budget_locked(self) -> None:
        while self._records and (
            len(self._records) > self._max_records
            or self._estimated_bytes > self._max_total_bytes
        ):
            oldest = next(iter(self._records))
            self._remove_subtree_locked(oldest)

    def _parent_chain_is_valid_locked(self, parent_id: str, child_id: str, model: str) -> bool:
        current: str | None = parent_id
        seen: set[str] = set()
        while current is not None:
            if current == child_id or current in seen:
                return False
            seen.add(current)
            stored = self._records.get(current)
            if stored is None or stored.record.model != model:
                return False
            current = stored.record.parent_response_id
        return True

    async def get(self, response_id: str) -> ResponseRecord | None:
        self._validate_response_id(response_id)
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            stored = self._records.get(response_id)
            if stored is None:
                return None
            stored.expires_at = now + self._ttl_seconds
            self._records.move_to_end(response_id)
            return stored.record

    async def materialize(self, response_id: str) -> tuple[CanonicalItem, ...] | None:
        self._validate_response_id(response_id)
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            if response_id not in self._records:
                return None

            chain: list[_StoredRecord] = []
            seen: set[str] = set()
            current: str | None = response_id
            model: str | None = None
            while current is not None:
                if current in seen:
                    raise ResponseStoreInvariantError("response parent chain contains a cycle")
                seen.add(current)
                stored = self._records.get(current)
                if stored is None:
                    self._remove_subtree_locked(response_id)
                    return None
                if model is None:
                    model = stored.record.model
                elif stored.record.model != model:
                    raise ResponseStoreInvariantError("response parent chain crosses model identity")
                chain.append(stored)
                current = stored.record.parent_response_id

            items: list[CanonicalItem] = []
            for stored in reversed(chain):
                items.extend(stored.record.delta_items)
                stored.expires_at = now + self._ttl_seconds
                self._records.move_to_end(stored.record.response_id)
            return tuple(items)

    async def put(self, record: ResponseRecord) -> ResponseStoreDisposition:
        if not isinstance(record, ResponseRecord):
            raise TypeError("record must be a ResponseRecord")
        estimated_bytes = estimate_response_record_bytes(record)
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            if estimated_bytes > self._max_total_bytes:
                return ResponseStoreDisposition.REFUSED_TOO_LARGE

            parent_id = record.parent_response_id
            if parent_id is not None:
                parent = self._records.get(parent_id)
                if parent is None:
                    return ResponseStoreDisposition.REFUSED_MISSING_PARENT
                if parent.record.model != record.model:
                    return ResponseStoreDisposition.REFUSED_MODEL_MISMATCH
                if not self._parent_chain_is_valid_locked(parent_id, record.response_id, record.model):
                    return ResponseStoreDisposition.REFUSED_INVALID_GRAPH

            if record.response_id in self._records:
                self._remove_subtree_locked(record.response_id)
            self._records[record.response_id] = _StoredRecord(
                record,
                estimated_bytes,
                now + self._ttl_seconds,
            )
            self._estimated_bytes += estimated_bytes
            if parent_id is not None:
                self._children.setdefault(parent_id, set()).add(record.response_id)
            self._records.move_to_end(record.response_id)
            self._evict_to_budget_locked()
            if record.response_id not in self._records:
                return ResponseStoreDisposition.REFUSED_BUDGET
            return ResponseStoreDisposition.STORED

    async def discard(self, response_id: str) -> None:
        self._validate_response_id(response_id)
        async with self._lock:
            self._purge_expired_locked(self._clock())
            self._remove_subtree_locked(response_id)

    async def size(self) -> int:
        async with self._lock:
            self._purge_expired_locked(self._clock())
            return len(self._records)

    async def stats(self) -> ResponseStoreStats:
        async with self._lock:
            self._purge_expired_locked(self._clock())
            return ResponseStoreStats(len(self._records), self._estimated_bytes)
