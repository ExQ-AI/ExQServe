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
        )
    ]
    metrics = MetricsRegistry()
    metrics.bind_engine_stats_provider(lambda: current[0])

    ready = metrics.render_text()
    assert _sample(ready, "exqserve_engine_active_jobs") == 2.0
    assert 'exqserve_engine_state{state="ready"} 1.0' in ready

    current[0] = RuntimeEngineStats(RuntimeEngineState.UNAVAILABLE)
    unavailable = metrics.render_text()
    active = _sample(unavailable, "exqserve_engine_active_jobs")
    allocated = _sample(unavailable, "exqserve_engine_kv_pages_allocated_since_generator_start")
    assert active is not None and math.isnan(active)
    assert allocated is not None and math.isnan(allocated)
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
