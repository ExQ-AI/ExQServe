from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.control.request import (
    RequestControlConfig,
    RequestRejected,
    RequestTerminalReason,
)
from exqserve.core.errors import CanonicalError, ErrorCategory


def test_request_control_config_is_immutable_and_has_no_queue_size() -> None:
    config = RequestControlConfig(
        max_in_flight=4,
        max_prompt_tokens=100,
        max_output_tokens=20,
        max_total_tokens=110,
        timeout_seconds=5.0,
    )

    assert config.max_in_flight == 4
    assert "queue" not in {field for field in config.__dataclass_fields__}
    with pytest.raises(FrozenInstanceError):
        config.max_in_flight = 5  # type: ignore[misc]


@pytest.mark.parametrize("name", ["max_in_flight", "max_prompt_tokens", "max_output_tokens", "max_total_tokens"])
def test_positive_integer_limits_are_enforced(name: str) -> None:
    kwargs: dict[str, object] = {"max_in_flight": 1, name: 0}
    with pytest.raises(ValueError, match=name):
        RequestControlConfig(**kwargs)  # type: ignore[arg-type]


def test_timeout_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        RequestControlConfig(1, timeout_seconds=0)
    with pytest.raises(ValueError, match="finite"):
        RequestControlConfig(1, timeout_seconds=float("inf"))


def test_auto_output_limit_uses_remaining_context_and_operator_cap() -> None:
    context_limited = RequestControlConfig(max_in_flight=1, max_total_tokens=100)
    capped = RequestControlConfig(max_in_flight=1, max_output_tokens=20, max_total_tokens=100)

    assert context_limited.resolve_output_limit(prompt_tokens=30, requested=None) == 69
    assert capped.resolve_output_limit(prompt_tokens=30, requested=None) == 20
    assert capped.resolve_output_limit(prompt_tokens=30, requested=7) == 7


def test_auto_output_limit_rejects_exhausted_or_unbounded_context() -> None:
    with pytest.raises(RequestRejected, match="no room"):
        RequestControlConfig(max_in_flight=1, max_total_tokens=10).resolve_output_limit(
            prompt_tokens=9,
            requested=None,
        )
    with pytest.raises(ValueError, match="requires max_total_tokens or max_output_tokens"):
        RequestControlConfig(max_in_flight=1).resolve_output_limit(
            prompt_tokens=1,
            requested=None,
        )


def test_terminal_reason_is_closed_protocol_neutral_vocabulary() -> None:
    assert {reason.value for reason in RequestTerminalReason} == {
        "completed",
        "runtime_failed",
        "runtime_cancelled",
        "client_cancelled",
        "application_cancelled",
        "timeout",
        "server_shutdown",
    }


def test_request_rejected_exposes_only_canonical_error() -> None:
    error = CanonicalError(
        ErrorCategory.OVERLOADED,
        "server_overloaded",
        "Server is at capacity.",
        retryable=True,
    )
    rejected = RequestRejected(error)

    assert rejected.error is error
    assert str(rejected) == error.message
    with pytest.raises(TypeError, match="CanonicalError"):
        RequestRejected(object())  # type: ignore[arg-type]
