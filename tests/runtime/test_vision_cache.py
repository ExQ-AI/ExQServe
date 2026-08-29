from __future__ import annotations

from types import SimpleNamespace

from exqserve.runtime.vision_cache import VisionEmbeddingCache, embedding_retained_tensor_bytes


class _Tensor:
    def __init__(self, numel: int, element_size: int, device_type: str = "cpu") -> None:
        self._numel = numel
        self._element_size = element_size
        self.device = SimpleNamespace(type=device_type)

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


def _embedding(size: int, *, deepstack: tuple[int, ...] = (), device_type: str = "cpu") -> object:
    return SimpleNamespace(
        embeddings=_Tensor(size, 1, device_type),
        deepstack_embeddings=[_Tensor(value, 1, device_type) for value in deepstack],
    )


def test_embedding_retained_tensor_bytes_counts_base_and_deepstack() -> None:
    assert embedding_retained_tensor_bytes(_embedding(20, deepstack=(3, 7))) == 30


def test_embedding_retained_tensor_bytes_rejects_non_cpu_tensor() -> None:
    assert embedding_retained_tensor_bytes(_embedding(20, device_type="cuda")) is None


def test_cache_hits_refresh_lru_and_evict_by_tensor_bytes() -> None:
    cache = VisionEmbeddingCache(20)
    first = _embedding(10)
    second = _embedding(10)
    third = _embedding(10)

    assert cache.get("first") is None
    assert cache.put("first", first) is True
    assert cache.put("second", second) is True
    assert cache.get("first") is first
    assert cache.put("third", third) is True

    assert cache.get("second") is None
    assert cache.get("first") is first
    assert cache.get("third") is third
    stats = cache.stats()
    assert stats.evictions == 1
    assert stats.entries == 2
    assert stats.retained_tensor_bytes == 20


def test_cache_skips_oversized_or_non_cpu_entries() -> None:
    cache = VisionEmbeddingCache(10)

    assert cache.put("oversized", _embedding(11)) is False
    assert cache.put("cuda", _embedding(5, device_type="cuda")) is False
    assert cache.stats().entries == 0


def test_zero_budget_disables_retention_without_breaking_queries() -> None:
    cache = VisionEmbeddingCache(0)

    assert cache.enabled is False
    assert cache.put("image", _embedding(1)) is False
    assert cache.get("image") is None
    stats = cache.stats()
    assert stats.queries == 1
    assert stats.misses == 1
    assert stats.entries == 0


def test_clear_releases_entries_but_preserves_cumulative_counters() -> None:
    cache = VisionEmbeddingCache(10)
    assert cache.put("image", _embedding(4)) is True
    assert cache.get("image") is not None

    cache.clear()

    stats = cache.stats()
    assert stats.entries == 0
    assert stats.retained_tensor_bytes == 0
    assert stats.queries == 1
    assert stats.hits == 1
