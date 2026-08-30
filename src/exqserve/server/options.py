"""Internal grouped options derived from the flat server configuration facade."""

from __future__ import annotations

from dataclasses import dataclass

from exqserve.model.contracts import ToolConstraintMode


@dataclass(frozen=True, slots=True)
class ResponseStoreOptions:
    max_records: int
    ttl_seconds: float
    max_total_bytes: int


@dataclass(frozen=True, slots=True)
class ToolServingOptions:
    constraint_mode: ToolConstraintMode
    fanout_limit: int
    constrained_parallel_limit: int
