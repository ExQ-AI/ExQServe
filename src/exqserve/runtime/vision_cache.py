"""Bounded runtime-owned cache for CPU-resident multimodal embeddings."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_WARNING_INTERVAL_SECONDS = 60.0


class _VisionCommitCancelled(RuntimeError):
    """Internal control flow when cancellation wins before persistent publication."""


@dataclass(frozen=True, slots=True)
class VisionEmbeddingCacheRequestStats:
    unique_media_count: int
    unique_media_bytes: int
    retained_media_bytes: int
    protected_prefix_entries: int
    protected_prefix_bytes: int
    admission_skipped: int
    first_unretained_media_ordinal: int | None


@dataclass(frozen=True, slots=True)
class VisionEmbeddingCacheStats:
    queries: int
    hits: int
    misses: int
    evictions: int
    admission_skipped: int
    over_budget_requests: int
    incomplete_prefix_retention_requests: int
    entries: int
    retained_tensor_bytes: int
    max_retained_bytes: int
    last_request_unique_media_count: int
    last_request_unique_media_bytes: int
    last_request_retained_media_bytes: int
    last_request_protected_prefix_entries: int
    last_request_protected_prefix_bytes: int
    last_request_first_unretained_media_ordinal: int | None


def _tensor_retained_bytes(tensor: object) -> int | None:
    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if not callable(numel) or not callable(element_size):
        return None

    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", None)
    if device_type != "cpu":
        return None

    retained = int(numel()) * int(element_size())
    if retained < 0:
        return None
    return retained


def embedding_retained_tensor_bytes(embedding: object) -> int | None:
    """Return retained CPU tensor bytes, or None when the embedding is unsafe to cache."""

    tensor = getattr(embedding, "embeddings", None)
    if tensor is None:
        return None
    retained = _tensor_retained_bytes(tensor)
    if retained is None:
        return None

    deepstack = getattr(embedding, "deepstack_embeddings", None)
    if deepstack is None:
        return retained
    if not isinstance(deepstack, (list, tuple)):
        return None
    for deepstack_tensor in deepstack:
        tensor_bytes = _tensor_retained_bytes(deepstack_tensor)
        if tensor_bytes is None:
            return None
        retained += tensor_bytes
    return retained


class VisionEmbeddingCache:
    """Thread-safe byte-bounded cache with request-commit prefix-priority retention."""

    def __init__(self, max_retained_bytes: int) -> None:
        if not isinstance(max_retained_bytes, int) or isinstance(max_retained_bytes, bool):
            raise TypeError("max_retained_bytes must be an integer")
        if max_retained_bytes < 0:
            raise ValueError("max_retained_bytes must be non-negative")
        self._max_retained_bytes = max_retained_bytes
        self._entries: OrderedDict[str, tuple[object, int]] = OrderedDict()
        self._retained_tensor_bytes = 0
        self._queries = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._admission_skipped = 0
        self._over_budget_requests = 0
        self._incomplete_prefix_retention_requests = 0
        self._last_request = VisionEmbeddingCacheRequestStats(0, 0, 0, 0, 0, 0, None)
        self._last_warning_at = float("-inf")
        self._lock = threading.Lock()
        self._commit_cancellation = threading.local()

    @property
    def enabled(self) -> bool:
        return self._max_retained_bytes > 0

    @property
    def max_retained_bytes(self) -> int:
        return self._max_retained_bytes

    def get(self, key: str) -> object | None:
        """Legacy single-entry LRU lookup; prompt compilation uses snapshot_request()."""
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        with self._lock:
            self._queries += 1
            entry = self._entries.pop(key, None)
            if entry is None:
                self._misses += 1
                return None
            self._entries[key] = entry
            self._hits += 1
            return entry[0]

    def peek(self, key: str) -> object | None:
        """Return a cached embedding without changing LRU or hit/miss accounting."""
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry[0]

    def snapshot_request(self, keys: tuple[str, ...]) -> dict[str, object]:
        """Snapshot unique request keys without changing cache recency or retention layout."""
        if not isinstance(keys, tuple) or not all(isinstance(key, str) for key in keys):
            raise TypeError("keys must be a tuple of strings")
        if len(set(keys)) != len(keys):
            raise ValueError("keys must be unique")
        with self._lock:
            self._queries += len(keys)
            resolved: dict[str, object] = {}
            for key in keys:
                entry = self._entries.get(key)
                if entry is None:
                    self._misses += 1
                    continue
                self._hits += 1
                resolved[key] = entry[0]
            return resolved

    def commit_request(
        self,
        resolved_items: tuple[tuple[str, object], ...],
    ) -> VisionEmbeddingCacheRequestStats:
        """Commit one successfully compiled request using prefix-priority bounded retention."""
        if not isinstance(resolved_items, tuple):
            raise TypeError("resolved_items must be a tuple")
        keys: list[str] = []
        sizes: list[int | None] = []
        for item in resolved_items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("resolved_items entries must be (key, embedding) tuples")
            key, embedding = item
            if not isinstance(key, str):
                raise TypeError("resolved item key must be a string")
            keys.append(key)
            sizes.append(embedding_retained_tensor_bytes(embedding))
        request_key_set = set(keys)
        if len(request_key_set) != len(keys):
            raise ValueError("resolved_items keys must be unique")

        budget = self._max_retained_bytes
        unique_media_bytes = sum(size for size in sizes if size is not None)

        protected_prefix_entries = 0
        protected_prefix_bytes = 0
        for size in sizes:
            if size is None or protected_prefix_bytes + size > budget:
                break
            protected_prefix_entries += 1
            protected_prefix_bytes += size

        first_unretained = (
            protected_prefix_entries + 1
            if protected_prefix_entries < len(resolved_items)
            else None
        )
        over_budget = bool(
            resolved_items
            and budget > 0
            and all(size is not None for size in sizes)
            and unique_media_bytes > budget
        )

        retained_current: list[tuple[str, object, int]] = []
        retained_current_keys: set[str] = set()
        retained_bytes = 0
        admission_skipped = 0

        for index, ((key, embedding), size) in enumerate(zip(resolved_items, sizes, strict=True)):
            if index < protected_prefix_entries:
                assert size is not None
                retained_current.append((key, embedding, size))
                retained_current_keys.add(key)
                retained_bytes += size
                continue
            if size is None or retained_bytes + size > budget:
                admission_skipped += 1
                continue
            retained_current.append((key, embedding, size))
            retained_current_keys.add(key)
            retained_bytes += size

        cancellation = getattr(self._commit_cancellation, "authority", None)
        if cancellation is None:
            commit_guard = nullcontext(True)
        else:
            guard_factory = getattr(cancellation, "commit_guard", None)
            if not callable(guard_factory):
                raise TypeError("cancellation authority must provide commit_guard()")
            commit_guard = guard_factory()

        with commit_guard as commit_allowed:
            if not commit_allowed:
                raise _VisionCommitCancelled
            with self._lock:
                old_entries = self._entries
                available_for_old = budget - retained_bytes
                selected_old_newest_first: list[tuple[str, object, int]] = []
                for key, (embedding, size) in reversed(old_entries.items()):
                    if key in request_key_set:
                        continue
                    if size <= available_for_old:
                        selected_old_newest_first.append((key, embedding, size))
                        available_for_old -= size

                new_entries: OrderedDict[str, tuple[object, int]] = OrderedDict()
                for key, embedding, size in reversed(selected_old_newest_first):
                    new_entries[key] = (embedding, size)
                for key, embedding, size in retained_current:
                    new_entries[key] = (embedding, size)

                evicted = 0
                for key, (old_embedding, _) in old_entries.items():
                    new_entry = new_entries.get(key)
                    if new_entry is None or new_entry[0] is not old_embedding:
                        evicted += 1

                self._entries = new_entries
                self._retained_tensor_bytes = sum(size for _, size in new_entries.values())
                self._evictions += evicted
                self._admission_skipped += admission_skipped
                if over_budget:
                    self._over_budget_requests += 1
                if first_unretained is not None:
                    self._incomplete_prefix_retention_requests += 1

                retained_media_bytes = sum(
                    size for key, (_, size) in new_entries.items() if key in retained_current_keys
                )
                request_stats = VisionEmbeddingCacheRequestStats(
                    unique_media_count=len(resolved_items),
                    unique_media_bytes=unique_media_bytes,
                    retained_media_bytes=retained_media_bytes,
                    protected_prefix_entries=protected_prefix_entries,
                    protected_prefix_bytes=protected_prefix_bytes,
                    admission_skipped=admission_skipped,
                    first_unretained_media_ordinal=first_unretained,
                )
                self._last_request = request_stats
                now = time.monotonic()
                should_warn = (
                    over_budget
                    and first_unretained is not None
                    and now - self._last_warning_at >= _WARNING_INTERVAL_SECONDS
                )
                if should_warn:
                    self._last_warning_at = now

        if should_warn:
            logger.warning(
                "multimodal working set exceeds vision cache budget; inference remains correct, "
                "but prefix reuse beyond the first unretained media item is not guaranteed "
                "(unique_media=%d unique_bytes=%d budget_bytes=%d protected_prefix=%d "
                "first_unretained_ordinal=%d)",
                request_stats.unique_media_count,
                request_stats.unique_media_bytes,
                budget,
                request_stats.protected_prefix_entries,
                request_stats.first_unretained_media_ordinal,
            )
        return request_stats

    def commit_request_if_active(
        self,
        resolved_items: tuple[tuple[str, object], ...],
        cancellation: object | None,
    ) -> VisionEmbeddingCacheRequestStats | None:
        """Commit unless caller cancellation linearizes before persistent publication."""
        if cancellation is None:
            return self.commit_request(resolved_items)
        if not callable(getattr(cancellation, "commit_guard", None)):
            raise TypeError("cancellation authority must provide commit_guard()")
        if getattr(self._commit_cancellation, "authority", None) is not None:
            raise RuntimeError("nested vision cache commit cancellation authority")
        self._commit_cancellation.authority = cancellation
        try:
            try:
                return self.commit_request(resolved_items)
            except _VisionCommitCancelled:
                return None
        finally:
            del self._commit_cancellation.authority

    def put(self, key: str, embedding: object) -> bool:
        """Legacy single-entry LRU insertion; prompt compilation uses commit_request()."""
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        if not self.enabled:
            return False
        retained = embedding_retained_tensor_bytes(embedding)
        if retained is None or retained > self._max_retained_bytes:
            return False

        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._retained_tensor_bytes -= previous[1]
            self._entries[key] = (embedding, retained)
            self._retained_tensor_bytes += retained
            while self._retained_tensor_bytes > self._max_retained_bytes and self._entries:
                _, (_, evicted_bytes) = self._entries.popitem(last=False)
                self._retained_tensor_bytes -= evicted_bytes
                self._evictions += 1
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._retained_tensor_bytes = 0
            self._last_request = VisionEmbeddingCacheRequestStats(0, 0, 0, 0, 0, 0, None)

    def stats(self) -> VisionEmbeddingCacheStats:
        with self._lock:
            last = self._last_request
            return VisionEmbeddingCacheStats(
                queries=self._queries,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                admission_skipped=self._admission_skipped,
                over_budget_requests=self._over_budget_requests,
                incomplete_prefix_retention_requests=self._incomplete_prefix_retention_requests,
                entries=len(self._entries),
                retained_tensor_bytes=self._retained_tensor_bytes,
                max_retained_bytes=self._max_retained_bytes,
                last_request_unique_media_count=last.unique_media_count,
                last_request_unique_media_bytes=last.unique_media_bytes,
                last_request_retained_media_bytes=last.retained_media_bytes,
                last_request_protected_prefix_entries=last.protected_prefix_entries,
                last_request_protected_prefix_bytes=last.protected_prefix_bytes,
                last_request_first_unretained_media_ordinal=last.first_unretained_media_ordinal,
            )
