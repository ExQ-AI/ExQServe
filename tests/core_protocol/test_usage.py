from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.core.usage import TokenUsage


def test_usage_preserves_unmeasured_values_and_measured_zero() -> None:
    unknown = TokenUsage()
    measured_zero = TokenUsage(input_tokens=0, cached_input_tokens=0, output_tokens=0)

    assert unknown.input_tokens is None
    assert unknown.cached_input_tokens is None
    assert unknown.output_tokens is None
    assert unknown.reasoning_tokens is None
    assert measured_zero.input_tokens == 0
    assert measured_zero.cached_input_tokens == 0
    assert measured_zero.output_tokens == 0


@pytest.mark.parametrize(
    "field_name",
    ["input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"],
)
def test_usage_rejects_negative_measured_counts(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        TokenUsage(**{field_name: -1})


def test_usage_rejects_cached_input_greater_than_input() -> None:
    with pytest.raises(ValueError, match="cached_input_tokens"):
        TokenUsage(input_tokens=10, cached_input_tokens=11)


def test_usage_allows_cached_input_when_total_input_is_unmeasured() -> None:
    usage = TokenUsage(cached_input_tokens=7)

    assert usage.cached_input_tokens == 7
    assert usage.input_tokens is None


def test_usage_is_immutable() -> None:
    usage = TokenUsage(input_tokens=1)

    with pytest.raises(FrozenInstanceError):
        usage.input_tokens = 2  # type: ignore[misc]
