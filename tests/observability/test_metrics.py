from __future__ import annotations

from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage
from exqserve.observability.metrics import MetricsRegistry


def _sample(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(f"{name} "):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_metrics_registry_uses_private_registry_and_stable_names() -> None:
    first = MetricsRegistry()
    second = MetricsRegistry()
    first.request_started()
    second.request_started()

    assert _sample(first.render_text(), "exqserve_active_requests") == 1.0
    assert _sample(second.render_text(), "exqserve_active_requests") == 1.0


def test_truthful_usage_and_timing_only_record_measured_values_and_rates() -> None:
    metrics = MetricsRegistry()
    metrics.observe_backend(
        GenerationTiming(queue_seconds=0.1, prefill_seconds=0.5, generation_seconds=0.25),
        TokenUsage(input_tokens=100, cached_input_tokens=60, output_tokens=20),
    )
    text = metrics.render_text()

    assert _sample(text, "exqserve_input_tokens_total") == 100.0
    assert _sample(text, "exqserve_cached_input_tokens_total") == 60.0
    assert _sample(text, "exqserve_output_tokens_total") == 20.0
    assert _sample(text, "exqserve_cached_input_ratio_sum") == 0.6
    assert _sample(text, "exqserve_prefill_tokens_per_second_sum") == 80.0
    assert _sample(text, "exqserve_decode_tokens_per_second_sum") == 80.0
    assert _sample(text, "exqserve_backend_queue_seconds_sum") == 0.1
    assert _sample(text, "exqserve_backend_prefill_seconds_sum") == 0.5
    assert _sample(text, "exqserve_backend_generation_seconds_sum") == 0.25


def test_unknown_measurements_do_not_turn_into_zero_or_derived_observations() -> None:
    metrics = MetricsRegistry()
    metrics.observe_backend(GenerationTiming(), TokenUsage(input_tokens=10))
    text = metrics.render_text()

    assert _sample(text, "exqserve_input_tokens_total") == 10.0
    assert _sample(text, "exqserve_cached_input_tokens_total") == 0.0
    assert _sample(text, "exqserve_output_tokens_total") == 0.0
    assert _sample(text, "exqserve_cached_input_ratio_count") == 0.0
    assert _sample(text, "exqserve_prefill_tokens_per_second_count") == 0.0
    assert _sample(text, "exqserve_decode_tokens_per_second_count") == 0.0
    assert _sample(text, "exqserve_backend_prefill_seconds_count") == 0.0
