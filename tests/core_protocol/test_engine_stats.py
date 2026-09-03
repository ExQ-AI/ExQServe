from __future__ import annotations

import pytest

from exqserve.core.engine_stats import RuntimeEngineState, RuntimeEngineStats


def test_runtime_engine_stats_preserves_unknown_vs_measured_zero() -> None:
    stats = RuntimeEngineStats(
        RuntimeEngineState.UNINITIALIZED,
        active_jobs=0,
        pending_jobs=0,
        kv_pages_total=8,
        kv_pages_referenced=0,
        kv_pages_unreferenced=8,
    )

    assert stats.active_jobs == 0
    assert stats.kv_pages_evicted_since_generator_start is None


def test_runtime_engine_stats_rejects_negative_and_boolean_values() -> None:
    with pytest.raises(ValueError, match="active_jobs"):
        RuntimeEngineStats(RuntimeEngineState.READY, active_jobs=-1)
    with pytest.raises(TypeError, match="active_jobs"):
        RuntimeEngineStats(RuntimeEngineState.READY, active_jobs=True)  # type: ignore[arg-type]
