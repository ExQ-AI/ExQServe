"""Protocol-neutral response records and bounded in-memory state store."""

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
    context_items: tuple[CanonicalItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.response_id, str) or not self.response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.context_items, tuple):
            raise TypeError("context_items must be a tuple")


@dataclass(frozen=True, slots=True)
class ResponseStoreStats:
    records: int
    estimated_bytes: int


class ResponseStore(Protocol):
    async def get(self, response_id: str) -> ResponseRecord | None:
        ...

    async def put(self, record: ResponseRecord) -> None:
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
    """Bound response history by LRU order, sliding TTL, and estimated retained bytes."""

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
        self._estimated_bytes = 0
        self._lock = asyncio.Lock()

    def _remove_locked(self, response_id: str) -> None:
        stored = self._records.pop(response_id, None)
        if stored is not None:
            self._estimated_bytes -= stored.estimated_bytes

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            response_id
            for response_id, stored in self._records.items()
            if stored.expires_at <= now
        ]
        for response_id in expired:
            self._remove_locked(response_id)

    def _evict_to_budget_locked(self) -> None:
        while len(self._records) > self._max_records or self._estimated_bytes > self._max_total_bytes:
            _, stored = self._records.popitem(last=False)
            self._estimated_bytes -= stored.estimated_bytes

    async def get(self, response_id: str) -> ResponseRecord | None:
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            stored = self._records.get(response_id)
            if stored is None:
                return None
            stored.expires_at = now + self._ttl_seconds
            self._records.move_to_end(response_id)
            return stored.record

    async def put(self, record: ResponseRecord) -> None:
        if not isinstance(record, ResponseRecord):
            raise TypeError("record must be a ResponseRecord")
        estimated_bytes = estimate_response_record_bytes(record)
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            self._remove_locked(record.response_id)
            if estimated_bytes > self._max_total_bytes:
                return
            self._records[record.response_id] = _StoredRecord(
                record,
                estimated_bytes,
                now + self._ttl_seconds,
            )
            self._estimated_bytes += estimated_bytes
            self._records.move_to_end(record.response_id)
            self._evict_to_budget_locked()

    async def size(self) -> int:
        async with self._lock:
            self._purge_expired_locked(self._clock())
            return len(self._records)

    async def stats(self) -> ResponseStoreStats:
        async with self._lock:
            self._purge_expired_locked(self._clock())
            return ResponseStoreStats(len(self._records), self._estimated_bytes)
