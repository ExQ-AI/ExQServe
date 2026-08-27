"""Bounded in-memory lifecycle registry for OpenAI Response resources."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class CancellableResponseSession(Protocol):
    async def cancel(self) -> None:
        ...


class ResponseLifecycleNotFound(LookupError):
    pass


class ResponseLifecycleNotCancellable(RuntimeError):
    pass


@dataclass(slots=True)
class _ActiveResponse:
    response: dict[str, object]
    session: CancellableResponseSession
    retain: bool


@dataclass(slots=True)
class _RetainedResponse:
    response: dict[str, object]
    estimated_bytes: int
    expires_at: float


@dataclass(frozen=True, slots=True)
class ResponseLifecycleStats:
    active: int
    retained: int
    estimated_bytes: int


def _response_id(response: dict[str, object]) -> str:
    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise ValueError("response resource must contain a non-empty id")
    return response_id


def _estimate_wire_bytes(response: dict[str, object]) -> int:
    try:
        encoded = json.dumps(
            response,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("response resource must be JSON serializable") from exc
    return len(encoded)


class InMemoryResponseLifecycleStore:
    """Tracks active Responses and bounded retained wire resources separately."""

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
        self._active: dict[str, _ActiveResponse] = {}
        self._retained: OrderedDict[str, _RetainedResponse] = OrderedDict()
        self._estimated_bytes = 0
        self._lock = asyncio.Lock()

    def _remove_retained_locked(self, response_id: str) -> None:
        retained = self._retained.pop(response_id, None)
        if retained is not None:
            self._estimated_bytes -= retained.estimated_bytes

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            response_id
            for response_id, retained in self._retained.items()
            if retained.expires_at <= now
        ]
        for response_id in expired:
            self._remove_retained_locked(response_id)

    def _retain_locked(self, response: dict[str, object], now: float) -> None:
        response_id = _response_id(response)
        cloned = copy.deepcopy(response)
        estimated_bytes = _estimate_wire_bytes(cloned)
        self._remove_retained_locked(response_id)
        if estimated_bytes > self._max_total_bytes:
            return
        self._retained[response_id] = _RetainedResponse(
            cloned,
            estimated_bytes,
            now + self._ttl_seconds,
        )
        self._estimated_bytes += estimated_bytes
        self._retained.move_to_end(response_id)
        while len(self._retained) > self._max_records or self._estimated_bytes > self._max_total_bytes:
            _, evicted = self._retained.popitem(last=False)
            self._estimated_bytes -= evicted.estimated_bytes

    async def register_active(
        self,
        response: dict[str, object],
        session: CancellableResponseSession,
        *,
        retain: bool,
    ) -> None:
        if not isinstance(response, dict):
            raise TypeError("response must be a dictionary")
        if not isinstance(retain, bool):
            raise TypeError("retain must be a boolean")
        response_id = _response_id(response)
        async with self._lock:
            self._purge_expired_locked(self._clock())
            if response_id in self._active:
                raise RuntimeError("response id is already active")
            self._active[response_id] = _ActiveResponse(copy.deepcopy(response), session, retain)

    async def update_active(self, response_id: str, response: dict[str, object]) -> None:
        async with self._lock:
            active = self._active.get(response_id)
            if active is not None:
                active.response = copy.deepcopy(response)

    async def finish(self, response_id: str, response: dict[str, object]) -> None:
        if not isinstance(response, dict):
            raise TypeError("response must be a dictionary")
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            active = self._active.pop(response_id, None)
            if active is not None:
                if active.retain:
                    self._retain_locked(response, now)
                return

            retained = self._retained.get(response_id)
            if (
                retained is not None
                and retained.response.get("status") == "cancelled"
                and response.get("status") == "cancelled"
            ):
                self._retain_locked(response, now)

    async def abandon(self, response_id: str) -> None:
        async with self._lock:
            self._active.pop(response_id, None)

    async def retrieve(self, response_id: str) -> dict[str, object] | None:
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            active = self._active.get(response_id)
            if active is not None and active.retain:
                return copy.deepcopy(active.response)
            retained = self._retained.get(response_id)
            if retained is None:
                return None
            retained.expires_at = now + self._ttl_seconds
            self._retained.move_to_end(response_id)
            return copy.deepcopy(retained.response)

    async def cancel(self, response_id: str) -> dict[str, object]:
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        async with self._lock:
            self._purge_expired_locked(self._clock())
            active = self._active.get(response_id)
            if active is None:
                if response_id in self._retained:
                    raise ResponseLifecycleNotCancellable(response_id)
                raise ResponseLifecycleNotFound(response_id)
            session = active.session
            retain = active.retain
            cancelled = copy.deepcopy(active.response)
            cancelled["status"] = "cancelled"
            cancelled["error"] = None
            cancelled["incomplete_details"] = None

        await session.cancel()

        async with self._lock:
            current = self._active.get(response_id)
            if current is active:
                self._active.pop(response_id, None)
                if retain:
                    self._retain_locked(cancelled, self._clock())
        return cancelled

    async def stats(self) -> ResponseLifecycleStats:
        async with self._lock:
            self._purge_expired_locked(self._clock())
            return ResponseLifecycleStats(
                active=len(self._active),
                retained=len(self._retained),
                estimated_bytes=self._estimated_bytes,
            )
