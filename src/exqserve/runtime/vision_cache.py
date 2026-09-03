"""Bounded runtime-owned cache for CPU-resident multimodal embeddings."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisionEmbeddingCacheStats:
    queries: int
    hits: int
    misses: int
    evictions: int
    entries: int
    retained_tensor_bytes: int


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
    """Thread-safe LRU bounded by retained MMEmbedding tensor bytes."""

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
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._max_retained_bytes > 0

    def get(self, key: str) -> object | None:
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
        """Return a cached embedding without changing public hit/miss accounting."""
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry[0]

    def put(self, key: str, embedding: object) -> bool:
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

    def stats(self) -> VisionEmbeddingCacheStats:
        with self._lock:
            return VisionEmbeddingCacheStats(
                queries=self._queries,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                entries=len(self._entries),
                retained_tensor_bytes=self._retained_tensor_bytes,
            )
