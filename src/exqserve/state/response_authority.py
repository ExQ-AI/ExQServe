"""Single Response identity authority over lifecycle and continuation state."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from exqserve.core.items import CanonicalItem
from exqserve.state.response_lifecycle import (
    CancellableResponseSession,
    InMemoryResponseLifecycleStore,
)
from exqserve.state.store import (
    ResponseRecord,
    ResponseStore,
    ResponseStoreDisposition,
)


class ResponseStateNotFound(LookupError):
    pass


class ResponseStateModelMismatch(ValueError):
    pass


@dataclass(slots=True)
class _ResponseTransition:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass(frozen=True, slots=True)
class ResponseStateAuthority:
    """Own lifecycle and continuation identity as one serialized state transition."""

    state_store: ResponseStore
    lifecycle_store: InMemoryResponseLifecycleStore
    _pending_records: dict[str, ResponseRecord] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _terminal_winners: dict[str, dict[str, object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _transition_registry_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _transitions: dict[str, _ResponseTransition] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @staticmethod
    def _response_id(response: dict[str, object]) -> str:
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response resource must contain a non-empty id")
        return response_id

    @asynccontextmanager
    async def _transition(self, response_id: str) -> AsyncIterator[None]:
        async with self._transition_registry_lock:
            transition = self._transitions.get(response_id)
            if transition is None:
                transition = _ResponseTransition()
                self._transitions[response_id] = transition
            transition.users += 1
        try:
            async with transition.lock:
                yield
        finally:
            async with self._transition_registry_lock:
                transition.users -= 1
                if transition.users == 0 and self._transitions.get(response_id) is transition:
                    self._transitions.pop(response_id, None)

    async def _retained_chain_is_consistent(self, response_id: str, model: str) -> bool:
        """Require lifecycle and continuation identity for every retained ancestor."""

        current: str | None = response_id
        descendants: list[str] = []
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                await self.state_store.discard(response_id)
                for descendant in descendants:
                    await self.lifecycle_store.discard(descendant)
                return False
            seen.add(current)

            record = await self.state_store.get(current)
            if record is None or record.model != model:
                await self.state_store.discard(response_id)
                for descendant in descendants:
                    await self.lifecycle_store.discard(descendant)
                return False

            response = await self.lifecycle_store.retrieve(current)
            if (
                response is None
                or response.get("status") != "completed"
                or response.get("store") is not True
                or response.get("model") != model
            ):
                await self.state_store.discard(current)
                await self.lifecycle_store.discard(current)
                for descendant in descendants:
                    await self.lifecycle_store.discard(descendant)
                return False

            descendants.append(current)
            current = record.parent_response_id
        return True

    async def _discard_locked(self, response_id: str) -> None:
        self._pending_records.pop(response_id, None)
        self._terminal_winners.pop(response_id, None)
        await self.state_store.discard(response_id)
        await self.lifecycle_store.discard(response_id)

    async def get(self, response_id: str) -> ResponseRecord | None:
        """Return only committed continuation state; staged completion is private."""

        async with self._transition(response_id):
            return await self.state_store.get(response_id)

    async def materialize(self, response_id: str) -> tuple[CanonicalItem, ...] | None:
        """Return only committed continuation state; staged completion is private."""

        async with self._transition(response_id):
            return await self.state_store.materialize(response_id)

    async def put(self, record: ResponseRecord) -> ResponseStoreDisposition:
        """Stage canonical completion until its lifecycle terminal can commit atomically."""

        if not isinstance(record, ResponseRecord):
            raise TypeError("record must be a ResponseRecord")
        async with self._transition(record.response_id):
            self._pending_records[record.response_id] = record
        return ResponseStoreDisposition.STORED

    async def discard(self, response_id: str) -> None:
        async with self._transition(response_id):
            await self._discard_locked(response_id)

    async def register_active(
        self,
        response: dict[str, object],
        session: CancellableResponseSession,
        *,
        retain: bool,
    ) -> None:
        response_id = self._response_id(response)
        async with self._transition(response_id):
            self._terminal_winners.pop(response_id, None)
            await self.lifecycle_store.register_active(response, session, retain=retain)

    async def update_active(self, response_id: str, response: dict[str, object]) -> None:
        async with self._transition(response_id):
            await self.lifecycle_store.update_active(response_id, response)

    async def finish(self, response_id: str, response: dict[str, object]) -> bool:
        """Commit the first terminal outcome and project that winner onto later arrivals."""

        async with self._transition(response_id):
            pending = self._pending_records.pop(response_id, None)
            winner = self._terminal_winners.get(response_id)
            if winner is not None:
                await self.state_store.discard(response_id)
                response.clear()
                response.update(copy.deepcopy(winner))
                return True

            status = response.get("status")
            retained = response.get("store") is True
            if status == "completed" and retained:
                if pending is None:
                    record = await self.state_store.get(response_id)
                    if record is None:
                        await self.lifecycle_store.discard(response_id)
                        return False
                else:
                    disposition = await self.state_store.put(pending)
                    if disposition is not ResponseStoreDisposition.STORED:
                        await self.lifecycle_store.discard(response_id)
                        return False
                    record = pending

                wire_model = response.get("model")
                if not isinstance(wire_model, str) or record.model != wire_model:
                    await self._discard_locked(response_id)
                    return False

                await self.lifecycle_store.finish(response_id, response)
                if not await self._retained_chain_is_consistent(response_id, wire_model):
                    await self._discard_locked(response_id)
                    return False
                self._terminal_winners[response_id] = copy.deepcopy(response)
                return True

            await self.state_store.discard(response_id)
            await self.lifecycle_store.finish(response_id, response)
            self._terminal_winners[response_id] = copy.deepcopy(response)
            return True

    async def is_terminal(self, response_id: str) -> bool:
        async with self._transition(response_id):
            return response_id in self._terminal_winners

    async def abandon(self, response_id: str) -> None:
        """Drop active identity, or only release the tombstone for an established terminal."""

        async with self._transition(response_id):
            if self._terminal_winners.pop(response_id, None) is not None:
                self._pending_records.pop(response_id, None)
                return
            await self._discard_locked(response_id)

    async def retrieve(self, response_id: str) -> dict[str, object] | None:
        async with self._transition(response_id):
            response = await self.lifecycle_store.retrieve(response_id)
            if response is None:
                self._pending_records.pop(response_id, None)
                await self.state_store.discard(response_id)
                return None

            if response.get("status") == "completed" and response.get("store") is True:
                wire_model = response.get("model")
                if not isinstance(wire_model, str) or not await self._retained_chain_is_consistent(
                    response_id,
                    wire_model,
                ):
                    await self._discard_locked(response_id)
                    return None
            return response

    async def cancel(self, response_id: str) -> dict[str, object]:
        async with self._transition(response_id):
            winner = self._terminal_winners.get(response_id)
            if winner is not None:
                return copy.deepcopy(winner)

            response = await self.lifecycle_store.cancel(response_id)
            if response.get("status") == "cancelled":
                self._pending_records.pop(response_id, None)
                await self.state_store.discard(response_id)
            if response.get("status") in {"completed", "incomplete", "failed", "cancelled"}:
                self._terminal_winners[response_id] = copy.deepcopy(response)
            return response

    async def resolve_previous(
        self,
        response_id: str | None,
        model: str,
    ) -> tuple[CanonicalItem, ...]:
        if response_id is None:
            return ()

        async with self._transition(response_id):
            response = await self.lifecycle_store.retrieve(response_id)
            if response is None:
                self._pending_records.pop(response_id, None)
                await self.state_store.discard(response_id)
                raise ResponseStateNotFound(response_id)
            if response.get("status") != "completed" or response.get("store") is not True:
                raise ResponseStateNotFound(response_id)

            wire_model = response.get("model")
            if isinstance(wire_model, str) and wire_model != model:
                raise ResponseStateModelMismatch(response_id)

            record = await self.state_store.get(response_id)
            if record is None:
                await self.lifecycle_store.discard(response_id)
                raise ResponseStateNotFound(response_id)
            if record.model != model:
                raise ResponseStateModelMismatch(response_id)

            if not await self._retained_chain_is_consistent(response_id, model):
                raise ResponseStateNotFound(response_id)

            materialized = await self.state_store.materialize(response_id)
            if materialized is None:
                await self.lifecycle_store.discard(response_id)
                raise ResponseStateNotFound(response_id)
            return materialized
