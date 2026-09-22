from __future__ import annotations

from types import SimpleNamespace

from exqserve.core.engine_stats import RuntimeEngineState
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime, _GeneratorLifecycleState


class _SizedCache:
    def __init__(self, size: int, **attributes: object) -> None:
        self._size = size
        for name, value in attributes.items():
            setattr(self, name, value)

    def __len__(self) -> int:
        return self._size


def test_engine_stats_before_lazy_generator_uses_loaded_cache_without_creating_generator() -> None:
    runtime = ExLlamaV3Runtime()
    runtime._resources = SimpleNamespace(  # type: ignore[assignment]
        config=SimpleNamespace(max_batch_size=4),
        cache=SimpleNamespace(max_num_tokens=2048),
    )

    stats = runtime.engine_stats

    assert runtime._generator is None
    assert stats.state is RuntimeEngineState.UNINITIALIZED
    assert stats.active_jobs == 0
    assert stats.pending_jobs == 0
    assert stats.max_batch_size == 4
    assert stats.kv_pages_total == 8
    assert stats.kv_pages_referenced == 0
    assert stats.kv_pages_unreferenced == 8
    assert stats.kv_pages_evicted_since_generator_start is None


def test_engine_stats_maps_current_generator_page_table_semantics() -> None:
    page_table = SimpleNamespace(
        max_pages=16,
        num_unreferenced_pages=lambda: 6,
        metrics={
            "evictions": 7,
            "evictions_live": 3,
            "stashes_stranded": 2,
            "alloc_pages": 20,
            "alloc_cached_pages": 9,
            "alloc_tier_pages": 4,
            "alloc_kv_only_pages": 1,
        },
    )
    backend_generator = SimpleNamespace(
        max_batch_size=8,
        num_active_jobs=lambda: 2,
        num_pending_jobs=lambda: 5,
        pagetable=page_table,
        cpu_page_cache=_SizedCache(
            5,
            max_slots=12,
            metrics={"pushes": 11, "restores": 4, "evictions": 3, "dedup_hits": 2},
        ),
        recurrent_cache=_SizedCache(
            2,
            current_size=65536,
            metrics={"stash_evictions": 6, "stash_pruned": 1},
        ),
    )
    runtime = ExLlamaV3Runtime()
    runtime._resources = SimpleNamespace(  # type: ignore[assignment]
        config=SimpleNamespace(max_batch_size=8),
        cache=SimpleNamespace(max_num_tokens=4096),
    )
    runtime._generator = SimpleNamespace(generator=backend_generator)

    stats = runtime.engine_stats

    assert stats.state is RuntimeEngineState.READY
    assert (stats.active_jobs, stats.pending_jobs, stats.max_batch_size) == (2, 5, 8)
    assert (stats.kv_pages_total, stats.kv_pages_referenced, stats.kv_pages_unreferenced) == (16, 10, 6)
    assert stats.kv_pages_evicted_since_generator_start == 7
    assert stats.kv_cached_pages_evicted_since_generator_start == 3
    assert stats.kv_recurrent_checkpoints_stranded_since_generator_start == 2
    assert stats.kv_pages_allocated_since_generator_start == 20
    assert stats.kv_cached_pages_reused_since_generator_start == 9
    assert stats.kv_pages_restored_from_cpu_tier_since_generator_start == 4
    assert stats.kv_cached_kv_only_pages_since_generator_start == 1
    assert stats.cpu_kv_cached_pages == 5
    assert stats.cpu_kv_cache_max_pages == 12
    assert stats.cpu_kv_cache_pushes_since_generator_start == 11
    assert stats.cpu_kv_cache_restores_since_generator_start == 4
    assert stats.cpu_kv_cache_evictions_since_generator_start == 3
    assert stats.cpu_kv_cache_dedup_hits_since_generator_start == 2
    assert stats.recurrent_cache_entries == 2
    assert stats.recurrent_cache_bytes == 65536
    assert stats.recurrent_cache_evictions_since_generator_start == 6
    assert stats.recurrent_cache_pruned_since_generator_start == 1


def test_recovering_engine_stats_never_inspects_quarantined_generator() -> None:
    runtime = ExLlamaV3Runtime()
    runtime._resources = SimpleNamespace(  # type: ignore[assignment]
        config=SimpleNamespace(max_batch_size=8),
        cache=SimpleNamespace(max_num_tokens=4096),
    )
    runtime._generator_state = _GeneratorLifecycleState.RECOVERING
    runtime._quarantined_generator = object()

    stats = runtime.engine_stats

    assert stats.state is RuntimeEngineState.RECOVERING
    assert stats.active_jobs is None
    assert stats.kv_pages_total is None
