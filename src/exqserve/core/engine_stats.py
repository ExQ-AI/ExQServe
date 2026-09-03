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
