from __future__ import annotations

import math

from exqserve.core.engine_stats import RuntimeEngineState, RuntimeEngineStats
from exqserve.observability.metrics import MetricsRegistry


def _sample(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(f"{name} "):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_engine_metrics_replace_ready_values_with_nan_when_unavailable() -> None:
    current = [
        RuntimeEngineStats(
            RuntimeEngineState.READY,
            active_jobs=2,
            pending_jobs=1,
            kv_pages_total=8,
            kv_pages_referenced=3,
            kv_pages_unreferenced=5,
            kv_pages_allocated_since_generator_start=9,
            cpu_kv_cached_pages=4,
            cpu_kv_cache_evictions_since_generator_start=2,
            recurrent_cache_bytes=65536,
            vision_cache_budget_bytes=1024 * 1024,
            vision_cache_retained_entries=3,
            vision_cache_retained_tensor_bytes=4096,
            vision_cache_hits=7,
            vision_cache_admission_skipped=2,
            vision_cache_over_budget_requests=1,
            vision_cache_incomplete_prefix_retention_requests=1,
            vision_cache_last_request_unique_media_count=4,
            vision_cache_last_request_protected_prefix_entries=3,
            vision_cache_last_request_first_unretained_media_ordinal=4,
        )
    ]
    metrics = MetricsRegistry()
    metrics.bind_engine_stats_provider(lambda: current[0])

    ready = metrics.render_text()
    assert _sample(ready, "exqserve_engine_active_jobs") == 2.0
    assert _sample(ready, "exqserve_engine_cpu_kv_cached_pages") == 4.0
    assert _sample(ready, "exqserve_engine_cpu_kv_cache_evictions_since_generator_start") == 2.0
    assert _sample(ready, "exqserve_engine_recurrent_cache_bytes") == 65536.0
    assert _sample(ready, "exqserve_engine_vision_cache_budget_bytes") == 1024.0 * 1024
    assert _sample(ready, "exqserve_engine_vision_cache_retained_entries") == 3.0
    assert _sample(ready, "exqserve_engine_vision_cache_hits") == 7.0
    assert _sample(ready, "exqserve_engine_vision_cache_admission_skipped") == 2.0
    assert _sample(ready, "exqserve_engine_vision_cache_over_budget_requests") == 1.0
    assert _sample(ready, "exqserve_engine_vision_cache_last_request_first_unretained_media_ordinal") == 4.0
    assert 'exqserve_engine_state{state="ready"} 1.0' in ready

    current[0] = RuntimeEngineStats(RuntimeEngineState.UNAVAILABLE)
    unavailable = metrics.render_text()
    active = _sample(unavailable, "exqserve_engine_active_jobs")
    allocated = _sample(unavailable, "exqserve_engine_kv_pages_allocated_since_generator_start")
    vision_budget = _sample(unavailable, "exqserve_engine_vision_cache_budget_bytes")
    assert active is not None and math.isnan(active)
    assert allocated is not None and math.isnan(allocated)
    assert vision_budget is not None and math.isnan(vision_budget)
    assert 'exqserve_engine_state{state="unavailable"} 1.0' in unavailable
    assert 'exqserve_engine_state{state="ready"} 0.0' in unavailable


def test_engine_stats_provider_failure_does_not_break_metrics_render() -> None:
    metrics = MetricsRegistry()

    def fail() -> RuntimeEngineStats:
        raise RuntimeError("stats unavailable")

    metrics.bind_engine_stats_provider(fail)
    text = metrics.render_text()
    active = _sample(text, "exqserve_engine_active_jobs")
    assert active is not None and math.isnan(active)
    assert 'exqserve_engine_state{state="unavailable"} 1.0' in text
