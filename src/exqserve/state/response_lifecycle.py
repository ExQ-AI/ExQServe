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


class ResponseLifecycleRetentionRefused(RuntimeError):
    pass


@dataclass(slots=True)
class _ActiveResponse:
    response: dict[str, object]
    session: CancellableResponseSession
    retain: bool
    terminal_response: dict[str, object] | None = None


@dataclass(slots=True)
class PreparedResponseCancellation:
    response_id: str
    active: _ActiveResponse
    cancelled: dict[str, object]


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
        self._children: dict[str, set[str]] = {}
        self._pins: dict[str, int] = {}
        self._estimated_bytes = 0
        self._lock = asyncio.Lock()

    def _remove_retained_one_locked(self, response_id: str) -> None:
        retained = self._retained.pop(response_id, None)
        self._pins.pop(response_id, None)
        if retained is None:
            self._children.pop(response_id, None)
            return
        self._estimated_bytes -= retained.estimated_bytes
        parent_id = retained.response.get("previous_response_id")
        if isinstance(parent_id, str):
            siblings = self._children.get(parent_id)
            if siblings is not None:
                siblings.discard(response_id)
                if not siblings:
                    self._children.pop(parent_id, None)
        self._children.pop(response_id, None)

    def _subtree_ids_locked(self, response_id: str) -> tuple[str, ...]:
        stack = [response_id]
        ordered: list[str] = []
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen or current not in self._retained:
                continue
            seen.add(current)
            ordered.append(current)
            stack.extend(self._children.get(current, ()))
        return tuple(ordered)

    def _remove_subtree_locked(self, response_id: str) -> tuple[str, ...]:
        subtree = self._subtree_ids_locked(response_id)
        for current in reversed(subtree):
            self._remove_retained_one_locked(current)
        return subtree

    def _replace_retained_locked(
        self,
        response_id: str,
        retained: _RetainedResponse,
    ) -> None:
        existing = self._retained.pop(response_id, None)
        if existing is not None:
            self._estimated_bytes -= existing.estimated_bytes
            old_parent = existing.response.get("previous_response_id")
            if isinstance(old_parent, str):
                siblings = self._children.get(old_parent)
                if siblings is not None:
                    siblings.discard(response_id)
                    if not siblings:
                        self._children.pop(old_parent, None)

        self._retained[response_id] = retained
        self._estimated_bytes += retained.estimated_bytes
        parent_id = retained.response.get("previous_response_id")
        if isinstance(parent_id, str):
            self._children.setdefault(parent_id, set()).add(response_id)
        self._retained.move_to_end(response_id)

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            response_id
            for response_id, retained in self._retained.items()
            if retained.expires_at <= now
        ]
        for response_id in expired:
            if response_id not in self._retained:
                continue
            subtree = self._subtree_ids_locked(response_id)
            if any(self._pins.get(current, 0) > 0 for current in subtree):
                continue
            self._remove_subtree_locked(response_id)

    def _retained_ancestor_ids_locked(self, response: dict[str, object]) -> set[str] | None:
        protected = {_response_id(response)}
        current = response.get("previous_response_id")
        seen: set[str] = set()
        while current is not None:
            if not isinstance(current, str) or not current or current in seen:
                return None
            seen.add(current)
            protected.add(current)
            parent = self._retained.get(current)
            if parent is None:
                return None
            current = parent.response.get("previous_response_id")
        return protected

    def _retention_victims_locked(
        self,
        response: dict[str, object],
        estimated_bytes: int,
    ) -> tuple[str, ...] | None:
        if estimated_bytes > self._max_total_bytes:
            return None
        response_id = _response_id(response)
        protected = self._retained_ancestor_ids_locked(response)
        if protected is None:
            return None

        existing = self._retained.get(response_id)
        projected_records = len(self._retained) + (0 if existing is not None else 1)
        projected_bytes = self._estimated_bytes + estimated_bytes
        if existing is not None:
            projected_bytes -= existing.estimated_bytes

        victims: list[str] = []
        victim_ids: set[str] = set()
        for candidate in self._retained:
            if projected_records <= self._max_records and projected_bytes <= self._max_total_bytes:
                break
            if candidate in victim_ids:
                continue
            subtree = self._subtree_ids_locked(candidate)
            if any(current in protected or self._pins.get(current, 0) > 0 for current in subtree):
                continue
            victims.append(candidate)
            for current in subtree:
                if current in victim_ids:
                    continue
                retained = self._retained.get(current)
                if retained is None:
                    continue
                victim_ids.add(current)
                projected_records -= 1
                projected_bytes -= retained.estimated_bytes
        if projected_records > self._max_records or projected_bytes > self._max_total_bytes:
            return None
        return tuple(victims)

    def _retain_locked(self, response: dict[str, object], now: float) -> tuple[str, ...] | None:
        response_id = _response_id(response)
        cloned = copy.deepcopy(response)
        estimated_bytes = _estimate_wire_bytes(cloned)
        victims = self._retention_victims_locked(cloned, estimated_bytes)
        if victims is None:
            return None

        evicted: set[str] = set()
        for victim in victims:
            evicted.update(self._remove_subtree_locked(victim))
        self._replace_retained_locked(
            response_id,
            _RetainedResponse(
                cloned,
                estimated_bytes,
                now + self._ttl_seconds,
            ),
        )
        return tuple(sorted(evicted))

    async def pin_chain(self, response_id: str) -> tuple[str, ...] | None:
        """Pin one retained lifecycle resource and all ancestors during continuation."""

        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            chain: list[str] = []
            seen: set[str] = set()
            current: str | None = response_id
            while current is not None:
                if current in seen:
                    return None
                seen.add(current)
                retained = self._retained.get(current)
                if retained is None:
                    return None
                chain.append(current)
                parent = retained.response.get("previous_response_id")
                if parent is not None and (not isinstance(parent, str) or not parent):
                    return None
                current = parent

            for current in chain:
                self._pins[current] = self._pins.get(current, 0) + 1
                retained = self._retained[current]
                retained.expires_at = now + self._ttl_seconds
                self._retained.move_to_end(current)
            return tuple(chain)

    async def unpin_chain(self, response_ids: tuple[str, ...]) -> None:
        if not isinstance(response_ids, tuple):
            raise TypeError("response_ids must be a tuple")
        async with self._lock:
            for response_id in response_ids:
                count = self._pins.get(response_id, 0)
                if count <= 1:
                    self._pins.pop(response_id, None)
                else:
                    self._pins[response_id] = count - 1

    async def can_retain(self, response: dict[str, object]) -> bool:
        if not isinstance(response, dict):
            raise TypeError("response must be a dictionary")
        cloned = copy.deepcopy(response)
        estimated_bytes = _estimate_wire_bytes(cloned)
        async with self._lock:
            self._purge_expired_locked(self._clock())
            return self._retention_victims_locked(cloned, estimated_bytes) is not None

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

    async def finish_with_evictions(
        self,
        response_id: str,
        response: dict[str, object],
    ) -> tuple[bool, tuple[str, ...]]:
        if not isinstance(response, dict):
            raise TypeError("response must be a dictionary")
        async with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            active = self._active.get(response_id)
            if active is not None:
                evicted: tuple[str, ...] = ()
                if active.retain:
                    retained_result = self._retain_locked(response, now)
                    if retained_result is None:
                        self._active.pop(response_id, None)
                        active.terminal_response = copy.deepcopy(response)
                        return False, ()
                    evicted = retained_result
                self._active.pop(response_id, None)
                active.terminal_response = copy.deepcopy(response)
                return True, evicted

            retained = self._retained.get(response_id)
            if (
                retained is not None
                and retained.response.get("status") == "cancelled"
                and response.get("status") == "cancelled"
            ):
                retained_result = self._retain_locked(response, now)
                if retained_result is None:
                    return False, ()
                return True, retained_result
            return True, ()

    async def finish(self, response_id: str, response: dict[str, object]) -> bool:
        finished, _ = await self.finish_with_evictions(response_id, response)
        return finished

    async def abandon(self, response_id: str) -> None:
        async with self._lock:
            self._active.pop(response_id, None)

    async def discard(self, response_id: str) -> None:
        """Remove any active or retained identity without invoking cancellation."""

        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        async with self._lock:
            self._active.pop(response_id, None)
            self._remove_subtree_locked(response_id)

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

    async def prepare_cancel(self, response_id: str) -> tuple[PreparedResponseCancellation, bool]:
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string")
        async with self._lock:
            self._purge_expired_locked(self._clock())
            active = self._active.get(response_id)
            if active is None:
                if response_id in self._retained:
                    raise ResponseLifecycleNotCancellable(response_id)
                raise ResponseLifecycleNotFound(response_id)
            cancelled = copy.deepcopy(active.response)
            cancelled["status"] = "cancelled"
            cancelled["error"] = None
            cancelled["incomplete_details"] = None
            return PreparedResponseCancellation(response_id, active, cancelled), active.retain

    async def cancel_prepared_session(self, prepared: PreparedResponseCancellation) -> None:
        await prepared.active.session.cancel()

    async def commit_prepared_cancel_with_evictions(
        self,
        prepared: PreparedResponseCancellation,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        response_id = prepared.response_id
        active = prepared.active
        cancelled = prepared.cancelled
        retain = active.retain
        async with self._lock:
            current = self._active.get(response_id)
            if current is active:
                self._active.pop(response_id, None)
                evicted: tuple[str, ...] = ()
                if retain:
                    retained_result = self._retain_locked(cancelled, self._clock())
                    if retained_result is None:
                        active.terminal_response = copy.deepcopy(cancelled)
                        raise ResponseLifecycleRetentionRefused(response_id)
                    evicted = retained_result
                return cancelled, evicted
            if active.terminal_response is not None:
                return copy.deepcopy(active.terminal_response), ()
        return cancelled, ()

    async def cancel_with_evictions(
        self,
        response_id: str,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        prepared, _ = await self.prepare_cancel(response_id)
        await self.cancel_prepared_session(prepared)
        return await self.commit_prepared_cancel_with_evictions(prepared)

    async def cancel(self, response_id: str) -> dict[str, object]:
        response, _ = await self.cancel_with_evictions(response_id)
        return response

    async def stats(self) -> ResponseLifecycleStats:
        async with self._lock:
            self._purge_expired_locked(self._clock())
            return ResponseLifecycleStats(
                active=len(self._active),
                retained=len(self._retained),
                estimated_bytes=self._estimated_bytes,
            )
