from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode, ReasoningPolicy


def test_reasoning_mode_values_are_protocol_neutral() -> None:
    assert {mode.value for mode in ReasoningMode} == {"default", "enabled", "disabled"}


def test_reasoning_effort_values_are_protocol_neutral() -> None:
    assert {effort.value for effort in ReasoningEffort} == {
        "low",
        "medium",
        "high",
        "maximum",
    }


def test_default_reasoning_policy_preserves_model_default() -> None:
    assert ReasoningPolicy() == ReasoningPolicy(mode=ReasoningMode.DEFAULT, effort=None)


@pytest.mark.parametrize("mode", list(ReasoningMode))
def test_reasoning_mode_and_effort_are_independent_protocol_neutral_controls(
    mode: ReasoningMode,
) -> None:
    assert ReasoningPolicy(mode=mode, effort=ReasoningEffort.HIGH).effort is ReasoningEffort.HIGH


def test_enabled_reasoning_accepts_effort_or_no_effort() -> None:
    assert ReasoningPolicy(mode=ReasoningMode.ENABLED).effort is None
    assert (
        ReasoningPolicy(mode=ReasoningMode.ENABLED, effort=ReasoningEffort.MAXIMUM).effort
        is ReasoningEffort.MAXIMUM
    )


def test_reasoning_policy_is_immutable() -> None:
    policy = ReasoningPolicy(mode=ReasoningMode.ENABLED, effort=ReasoningEffort.LOW)

    with pytest.raises(FrozenInstanceError):
        policy.mode = ReasoningMode.DISABLED  # type: ignore[misc]
