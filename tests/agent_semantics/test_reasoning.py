from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.agent.reasoning import (
    ReasoningBudgetMode,
    ReasoningBudgetOverride,
    ReasoningEffort,
    ReasoningMode,
    ReasoningPolicy,
)


def test_reasoning_mode_values_are_protocol_neutral() -> None:
    assert {mode.value for mode in ReasoningMode} == {"default", "enabled", "disabled"}


def test_reasoning_effort_values_are_protocol_neutral() -> None:
    assert {effort.value for effort in ReasoningEffort} == {
        "low",
        "medium",
        "high",
        "xhigh",
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


def test_reasoning_budget_override_preserves_inherit_disable_explicit_states() -> None:
    assert ReasoningBudgetOverride().mode is ReasoningBudgetMode.INHERIT
    assert ReasoningBudgetOverride(ReasoningBudgetMode.DISABLE).mode is ReasoningBudgetMode.DISABLE
    explicit = ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 0, "done ")
    assert explicit.max_tokens == 0
    assert explicit.message == "done "

    with pytest.raises(ValueError):
        ReasoningBudgetOverride(ReasoningBudgetMode.INHERIT, 1)
    with pytest.raises(ValueError):
        ReasoningBudgetOverride(ReasoningBudgetMode.INHERIT, message="unused")
    with pytest.raises(ValueError):
        ReasoningBudgetOverride(ReasoningBudgetMode.DISABLE, message="unused")
    with pytest.raises(ValueError):
        ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, -1)
    with pytest.raises(ValueError):
        ReasoningBudgetOverride(ReasoningBudgetMode.INHERIT, message="unused")
    with pytest.raises(TypeError):
        ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, True)  # type: ignore[arg-type]
