from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.core.timing import GenerationTiming


def test_generation_timing_preserves_measured_optional_durations() -> None:
    timing = GenerationTiming(queue_seconds=0.1, prefill_seconds=0.2, generation_seconds=0.3)
    assert timing.queue_seconds == 0.1
    assert timing.prefill_seconds == 0.2
    assert timing.generation_seconds == 0.3


def test_generation_timing_allows_truthful_unknowns() -> None:
    assert GenerationTiming() == GenerationTiming(None, None, None)


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_generation_timing_rejects_negative_or_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError):
        GenerationTiming(prefill_seconds=value)


def test_generation_timing_is_immutable() -> None:
    timing = GenerationTiming(prefill_seconds=1.0)
    with pytest.raises(FrozenInstanceError):
        timing.prefill_seconds = 2.0  # type: ignore[misc]
