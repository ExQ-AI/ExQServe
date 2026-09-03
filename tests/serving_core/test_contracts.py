from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.serving.contracts import MidSystemPolicy, ServingRejected, ServingRequest


def _input() -> CanonicalRequest:
    return CanonicalRequest(
        "req-1",
        "model",
        (MessageItem(MessageRole.USER, "hello"),),
    )


def _tools() -> ToolPolicy:
    return ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)


def test_serving_request_is_immutable_and_contains_no_wire_stream_field() -> None:
    request = ServingRequest(
        input=_input(),
        reasoning=ReasoningPolicy(),
        tools=_tools(),
        max_output_tokens=32,
    )

    assert {field.name for field in fields(request)} == {
        "input",
        "reasoning",
        "reasoning_budget",
        "tools",
        "structured_output",
        "max_output_tokens",
        "mid_system_policy",
        "seed",
        "sampling",
        "stop_conditions",
    }
    assert request.mid_system_policy is MidSystemPolicy.LEGACY_UNSPECIFIED
    with pytest.raises(FrozenInstanceError):
        request.max_output_tokens = 64  # type: ignore[misc]


def test_serving_request_validates_local_types_and_output_limit() -> None:
    with pytest.raises(ValueError, match="max_output_tokens"):
        ServingRequest(_input(), ReasoningPolicy(), _tools(), max_output_tokens=0)
    with pytest.raises(TypeError, match="input"):
        ServingRequest(object(), ReasoningPolicy(), _tools(), max_output_tokens=1)  # type: ignore[arg-type]


def test_serving_rejected_exposes_only_canonical_error() -> None:
    error = CanonicalError(
        ErrorCategory.INVALID_REQUEST,
        "invalid",
        "Invalid serving request.",
        retryable=False,
    )
    rejected = ServingRejected(error)

    assert rejected.error is error
    assert str(rejected) == error.message
    with pytest.raises(TypeError, match="CanonicalError"):
        ServingRejected(object())  # type: ignore[arg-type]
