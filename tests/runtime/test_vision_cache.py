from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_legacy_cache_hits_refresh_lru_and_evict_by_tensor_bytes() -> None:
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


def test_legacy_cache_skips_oversized_or_non_cpu_entries() -> None:
    cache = VisionEmbeddingCache(10)

    assert cache.put("oversized", _embedding(11)) is False
    assert cache.put("cuda", _embedding(5, device_type="cuda")) is False
    assert cache.stats().entries == 0


def test_request_snapshot_does_not_refresh_lru_or_mutate_layout() -> None:
    cache = VisionEmbeddingCache(20)
    first = _embedding(10)
    second = _embedding(10)
    cache.put("first", first)
    cache.put("second", second)

    snapshot = cache.snapshot_request(("first", "missing"))

    assert snapshot == {"first": first}
    assert cache.peek("first") is first
    assert cache.peek("second") is second
    stats = cache.stats()
    assert (stats.queries, stats.hits, stats.misses) == (2, 1, 1)
    assert stats.entries == 2
    assert stats.retained_tensor_bytes == 20


def test_request_commit_preserves_longest_prefix_when_working_set_is_slightly_over_budget() -> None:
    cache = VisionEmbeddingCache(30)
    first = _embedding(10)
    second = _embedding(10)
    third = _embedding(10)
    fourth = _embedding(10)

    request = (("a", first), ("b", second), ("c", third), ("d", fourth))
    commit = cache.commit_request(request)

    assert commit.unique_media_count == 4
    assert commit.unique_media_bytes == 40
    assert commit.protected_prefix_entries == 3
    assert commit.protected_prefix_bytes == 30
    assert commit.admission_skipped == 1
    assert commit.first_unretained_media_ordinal == 4
    assert cache.peek("a") is first
    assert cache.peek("b") is second
    assert cache.peek("c") is third
    assert cache.peek("d") is None
    stats = cache.stats()
    assert stats.entries == 3
    assert stats.retained_tensor_bytes == 30
    assert stats.max_retained_bytes == 30
    assert stats.over_budget_requests == 1
    assert stats.incomplete_prefix_retention_requests == 1


def test_over_budget_repeated_scan_hits_prefix_instead_of_zero_hit_miss_wave() -> None:
    cache = VisionEmbeddingCache(30)
    first = _embedding(10)
    second = _embedding(10)
    third = _embedding(10)
    fourth = _embedding(10)

    cache.commit_request((("a", first), ("b", second), ("c", third), ("d", fourth)))

    snapshot = cache.snapshot_request(("a", "b", "c", "d"))

    assert snapshot == {"a": first, "b": second, "c": third}
    stats = cache.stats()
    assert (stats.hits, stats.misses) == (3, 1)

    fourth_recomputed = _embedding(10)
    cache.commit_request(
        (("a", first), ("b", second), ("c", third), ("d", fourth_recomputed))
    )
    second_snapshot = cache.snapshot_request(("a", "b", "c", "d"))
    assert second_snapshot == {"a": first, "b": second, "c": third}
    stats = cache.stats()
    assert (stats.hits, stats.misses) == (6, 2)
    assert stats.entries == 3
    assert stats.retained_tensor_bytes == 30


def test_request_commit_recovers_prefix_from_bad_previous_lru_layout() -> None:
    cache = VisionEmbeddingCache(30)
    old_b = _embedding(10)
    old_c = _embedding(10)
    old_d = _embedding(10)
    cache.put("b", old_b)
    cache.put("c", old_c)
    cache.put("d", old_d)

    snapshot = cache.snapshot_request(("a", "b", "c", "d"))
    assert snapshot == {"b": old_b, "c": old_c, "d": old_d}

    new_a = _embedding(10)
    cache.commit_request(
        (("a", new_a), ("b", old_b), ("c", old_c), ("d", old_d))
    )

    assert cache.peek("a") is new_a
    assert cache.peek("b") is old_b
    assert cache.peek("c") is old_c
    assert cache.peek("d") is None
    stats = cache.stats()
    assert stats.evictions == 1
    assert stats.last_request_protected_prefix_entries == 3
    assert stats.last_request_first_unretained_media_ordinal == 4


def test_residual_entries_never_displace_protected_prefix() -> None:
    cache = VisionEmbeddingCache(35)
    first = _embedding(15)
    second = _embedding(15)
    too_large_for_residual = _embedding(10)
    later_small = _embedding(5)

    commit = cache.commit_request(
        (
            ("a", first),
            ("b", second),
            ("c", too_large_for_residual),
            ("d", later_small),
        )
    )

    assert commit.protected_prefix_entries == 2
    assert commit.first_unretained_media_ordinal == 3
    assert cache.peek("a") is first
    assert cache.peek("b") is second
    assert cache.peek("c") is None
    assert cache.peek("d") is later_small
    assert cache.stats().retained_tensor_bytes == 35


def test_oversized_first_embedding_is_used_but_not_persisted() -> None:
    cache = VisionEmbeddingCache(10)
    oversized = _embedding(11)
    later = _embedding(5)

    commit = cache.commit_request((("oversized", oversized), ("later", later)))

    assert commit.protected_prefix_entries == 0
    assert commit.first_unretained_media_ordinal == 1
    assert commit.admission_skipped == 1
    assert cache.peek("oversized") is None
    assert cache.peek("later") is later
    stats = cache.stats()
    assert stats.entries == 1
    assert stats.retained_tensor_bytes == 5
    assert stats.incomplete_prefix_retention_requests == 1


def test_zero_budget_disables_persistent_retention_but_request_commit_is_bounded() -> None:
    cache = VisionEmbeddingCache(0)
    embedding = _embedding(1)

    assert cache.enabled is False
    assert cache.snapshot_request(("image",)) == {}
    commit = cache.commit_request((("image", embedding),))

    assert commit.protected_prefix_entries == 0
    assert commit.admission_skipped == 1
    assert commit.first_unretained_media_ordinal == 1
    stats = cache.stats()
    assert stats.queries == 1
    assert stats.misses == 1
    assert stats.entries == 0
    assert stats.retained_tensor_bytes == 0
    assert stats.incomplete_prefix_retention_requests == 1


def test_commit_rejects_duplicate_request_keys() -> None:
    cache = VisionEmbeddingCache(10)
    embedding = _embedding(5)

    with pytest.raises(ValueError, match="unique"):
        cache.commit_request((("same", embedding), ("same", embedding)))


def test_over_budget_warning_is_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    cache = VisionEmbeddingCache(10)

    with caplog.at_level("WARNING"):
        cache.commit_request((("a", _embedding(8)), ("b", _embedding(8))))
        cache.commit_request((("a", _embedding(8)), ("b", _embedding(8))))

    warnings = [
        record
        for record in caplog.records
        if "multimodal working set exceeds vision cache budget" in record.getMessage()
    ]
    assert len(warnings) == 1


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
    assert stats.last_request_unique_media_count == 0
