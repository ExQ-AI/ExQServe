from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    TimingUpdated,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import ToolCallItem
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage


def test_completion_reasons_are_exact_v1_set() -> None:
    assert {reason.value for reason in CompletionReason} == {"stop", "length", "tool_calls", "filter"}


def test_generation_event_union_is_exact_v1_set() -> None:
    assert set(get_args(GenerationEvent)) == {
        GenerationStarted,
        GenerationCompleted,
        GenerationCancelled,
        GenerationFailed,
        TextStarted,
        TextDelta,
        TextCompleted,
        ReasoningStarted,
        ReasoningDelta,
        ReasoningCompleted,
        ToolCallStarted,
        ToolCallArgumentsDelta,
        ToolCallCompleted,
        TimingUpdated,
        UsageUpdated,
    }


def test_terminal_events_keep_completion_cancellation_and_failure_distinct() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=2)
    error = CanonicalError(
        category=ErrorCategory.RUNTIME_FAILURE,
        code="runtime_failed",
        message="Runtime failure.",
        retryable=True,
    )

    completed = GenerationCompleted("req-1", CompletionReason.STOP, usage)
    cancelled = GenerationCancelled("req-1")
    failed = GenerationFailed("req-1", error)

    assert completed.reason is CompletionReason.STOP
    assert completed.usage == usage
    assert cancelled.request_id == "req-1"
    assert failed.error == error


@pytest.mark.parametrize(
    "event",
    [
        TextDelta("req-1", "a"),
        ReasoningDelta("req-1", "b"),
        ToolCallArgumentsDelta("req-1", "call-1", "{", 0),
    ],
)
def test_delta_events_require_non_empty_delta(event: object) -> None:
    assert event is not None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TextDelta("req-1", ""),
        lambda: ReasoningDelta("req-1", ""),
        lambda: ToolCallArgumentsDelta("req-1", "call-1", "", 0),
    ],
)
def test_delta_events_reject_empty_delta(factory: object) -> None:
    with pytest.raises(ValueError, match="empty"):
        factory()  # type: ignore[operator]


def test_completed_text_and_reasoning_keep_full_values() -> None:
    text = TextCompleted("req-1", "final answer")
    reasoning = ReasoningCompleted("req-1", "full reasoning")

    assert text.text == "final answer"
    assert reasoning.text == "full reasoning"


def test_tool_call_events_preserve_identity_and_complete_call() -> None:
    call = ToolCallItem("call-1", "bash", '{"command":"pwd"}', 0)
    started = ToolCallStarted("req-1", "call-1", "bash", 0)
    delta = ToolCallArgumentsDelta("req-1", "call-1", '{"command":', 0)
    completed = ToolCallCompleted("req-1", call)

    assert (started.call_id, started.name, started.index) == ("call-1", "bash", 0)
    assert (delta.call_id, delta.index) == ("call-1", 0)
    assert completed.call == call


def test_tool_call_event_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="index"):
        ToolCallStarted("req-1", "call-1", "bash", -1)


def test_timing_event_preserves_truthful_measured_snapshot() -> None:
    timing = GenerationTiming(queue_seconds=0.1, prefill_seconds=0.2, generation_seconds=0.3)
    assert TimingUpdated("req-1", timing).timing == timing


def test_usage_event_preserves_truthful_usage_snapshot() -> None:
    usage = TokenUsage(input_tokens=20, cached_input_tokens=10, output_tokens=1)

    assert UsageUpdated("req-1", usage).usage == usage


def test_all_events_reject_empty_request_id() -> None:
    with pytest.raises(ValueError, match="request_id"):
        GenerationStarted("   ")


def test_events_are_immutable() -> None:
    event = TextStarted("req-1")

    with pytest.raises(FrozenInstanceError):
        event.request_id = "changed"  # type: ignore[misc]
