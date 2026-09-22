"""Protocol-neutral inference-engine state snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class RuntimeEngineState(str, Enum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RECOVERING = "recovering"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RuntimeEngineStats:
    state: RuntimeEngineState
    active_jobs: int | None = None
    pending_jobs: int | None = None
    max_batch_size: int | None = None
    kv_pages_total: int | None = None
    kv_pages_referenced: int | None = None
    kv_pages_unreferenced: int | None = None
    kv_pages_evicted_since_generator_start: int | None = None
    kv_cached_pages_evicted_since_generator_start: int | None = None
    kv_recurrent_checkpoints_stranded_since_generator_start: int | None = None
    kv_pages_allocated_since_generator_start: int | None = None
    kv_cached_pages_reused_since_generator_start: int | None = None
    kv_pages_restored_from_cpu_tier_since_generator_start: int | None = None
    kv_cached_kv_only_pages_since_generator_start: int | None = None
    cpu_kv_cached_pages: int | None = None
    cpu_kv_cache_max_pages: int | None = None
    cpu_kv_cache_pushes_since_generator_start: int | None = None
    cpu_kv_cache_restores_since_generator_start: int | None = None
    cpu_kv_cache_evictions_since_generator_start: int | None = None
    cpu_kv_cache_dedup_hits_since_generator_start: int | None = None
    recurrent_cache_entries: int | None = None
    recurrent_cache_bytes: int | None = None
    recurrent_cache_evictions_since_generator_start: int | None = None
    recurrent_cache_pruned_since_generator_start: int | None = None
    vision_cache_budget_bytes: int | None = None
    vision_cache_retained_entries: int | None = None
    vision_cache_retained_tensor_bytes: int | None = None
    vision_cache_queries: int | None = None
    vision_cache_hits: int | None = None
    vision_cache_misses: int | None = None
    vision_cache_evictions: int | None = None
    vision_cache_admission_skipped: int | None = None
    vision_cache_over_budget_requests: int | None = None
    vision_cache_incomplete_prefix_retention_requests: int | None = None
    vision_cache_last_request_unique_media_count: int | None = None
    vision_cache_last_request_unique_media_bytes: int | None = None
    vision_cache_last_request_retained_media_bytes: int | None = None
    vision_cache_last_request_protected_prefix_entries: int | None = None
    vision_cache_last_request_protected_prefix_bytes: int | None = None
    vision_cache_last_request_first_unretained_media_ordinal: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, RuntimeEngineState):
            raise TypeError("state must be a RuntimeEngineState")
        for name in (
            "active_jobs",
            "pending_jobs",
            "max_batch_size",
            "kv_pages_total",
            "kv_pages_referenced",
            "kv_pages_unreferenced",
            "kv_pages_evicted_since_generator_start",
            "kv_cached_pages_evicted_since_generator_start",
            "kv_recurrent_checkpoints_stranded_since_generator_start",
            "kv_pages_allocated_since_generator_start",
            "kv_cached_pages_reused_since_generator_start",
            "kv_pages_restored_from_cpu_tier_since_generator_start",
            "kv_cached_kv_only_pages_since_generator_start",
            "cpu_kv_cached_pages",
            "cpu_kv_cache_max_pages",
            "cpu_kv_cache_pushes_since_generator_start",
            "cpu_kv_cache_restores_since_generator_start",
            "cpu_kv_cache_evictions_since_generator_start",
            "cpu_kv_cache_dedup_hits_since_generator_start",
            "recurrent_cache_entries",
            "recurrent_cache_bytes",
            "recurrent_cache_evictions_since_generator_start",
            "recurrent_cache_pruned_since_generator_start",
            "vision_cache_budget_bytes",
            "vision_cache_retained_entries",
            "vision_cache_retained_tensor_bytes",
            "vision_cache_queries",
            "vision_cache_hits",
            "vision_cache_misses",
            "vision_cache_evictions",
            "vision_cache_admission_skipped",
            "vision_cache_over_budget_requests",
            "vision_cache_incomplete_prefix_retention_requests",
            "vision_cache_last_request_unique_media_count",
            "vision_cache_last_request_unique_media_bytes",
            "vision_cache_last_request_retained_media_bytes",
            "vision_cache_last_request_protected_prefix_entries",
            "vision_cache_last_request_protected_prefix_bytes",
            "vision_cache_last_request_first_unretained_media_ordinal",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer or None")
            if value < 0:
                raise ValueError(f"{name} must be non-negative or None")


class RuntimeEngineStatsProvider(Protocol):
    @property
    def engine_stats(self) -> RuntimeEngineStats:
        ...
