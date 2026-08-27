from __future__ import annotations

from collections.abc import Iterable

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
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import ToolCallItem
from exqserve.core.usage import TokenUsage

_TERMINAL_EVENTS = (GenerationCompleted, GenerationCancelled, GenerationFailed)
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
)


def _validate_stream(events: tuple[GenerationEvent, ...]) -> None:
    assert events
    assert isinstance(events[0], GenerationStarted)
    assert sum(isinstance(event, GenerationStarted) for event in events) == 1

    terminals = [event for event in events if isinstance(event, _TERMINAL_EVENTS)]
    assert len(terminals) == 1
    assert events[-1] is terminals[0]

    request_id = events[0].request_id
    assert all(event.request_id == request_id for event in events)

    started_tools: dict[int, ToolCallStarted] = {}
    args_by_index: dict[int, list[str]] = {}
    text_deltas: list[str] = []
    reasoning_deltas: list[str] = []
    last_usage: TokenUsage | None = None

    for event in events:
        if isinstance(event, TextDelta):
            text_deltas.append(event.text)
        elif isinstance(event, TextCompleted):
            assert "".join(text_deltas) == event.text
        elif isinstance(event, ReasoningDelta):
            reasoning_deltas.append(event.text)
        elif isinstance(event, ReasoningCompleted):
            assert "".join(reasoning_deltas) == event.text
        elif isinstance(event, ToolCallStarted):
            assert event.index not in started_tools
            started_tools[event.index] = event
            args_by_index[event.index] = []
        elif isinstance(event, ToolCallArgumentsDelta):
            started = started_tools[event.index]
            assert event.call_id == started.call_id
            args_by_index[event.index].append(event.delta)
        elif isinstance(event, ToolCallCompleted):
            started = started_tools[event.call.index]
            assert event.call.call_id == started.call_id
            assert event.call.name == started.name
            assert event.call.index == started.index
            assert event.call.arguments_json == "".join(args_by_index[event.call.index])
        elif isinstance(event, UsageUpdated):
            if last_usage is not None:
                _assert_usage_not_decreased(last_usage, event.usage)
            last_usage = event.usage


def _assert_usage_not_decreased(previous: TokenUsage, current: TokenUsage) -> None:
    for field_name in _USAGE_FIELDS:
        before = getattr(previous, field_name)
        after = getattr(current, field_name)
        if before is not None and after is not None:
            assert after >= before


def _tool_stream(index: int, call_id: str, name: str, arguments: str) -> Iterable[GenerationEvent]:
    split = max(1, len(arguments) // 2)
    yield ToolCallStarted("req-tool", call_id, name, index)
    yield ToolCallArgumentsDelta("req-tool", call_id, arguments[:split], index)
    if arguments[split:]:
        yield ToolCallArgumentsDelta("req-tool", call_id, arguments[split:], index)
    yield ToolCallCompleted("req-tool", ToolCallItem(call_id, name, arguments, index))


def test_text_stream_reconstructs_complete_text_and_terminates_once() -> None:
    events: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-text"),
        TextStarted("req-text"),
        TextDelta("req-text", "hello "),
        TextDelta("req-text", "world"),
        TextCompleted("req-text", "hello world"),
        UsageUpdated("req-text", TokenUsage(input_tokens=10, output_tokens=2)),
        GenerationCompleted(
            "req-text",
            CompletionReason.STOP,
            TokenUsage(input_tokens=10, output_tokens=2),
        ),
    )

    _validate_stream(events)


def test_reasoning_and_text_channels_remain_distinct() -> None:
    events: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-reason"),
        ReasoningStarted("req-reason"),
        ReasoningDelta("req-reason", "think"),
        ReasoningCompleted("req-reason", "think"),
        TextStarted("req-reason"),
        TextDelta("req-reason", "answer"),
        TextCompleted("req-reason", "answer"),
        GenerationCompleted("req-reason", CompletionReason.STOP),
    )

    _validate_stream(events)


def test_one_tool_stream_reconstructs_complete_arguments() -> None:
    body = tuple(_tool_stream(0, "call-1", "bash", '{"command":"pwd"}'))
    events: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-tool"),
        *body,
        GenerationCompleted("req-tool", CompletionReason.TOOL_CALLS),
    )

    _validate_stream(events)


def test_two_tool_streams_keep_ids_and_indices_independent() -> None:
    first = tuple(_tool_stream(0, "call-1", "bash", '{"command":"pwd"}'))
    second = tuple(_tool_stream(1, "call-2", "read_file", '{"path":"a.py"}'))
    events: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-tool"),
        *first,
        *second,
        GenerationCompleted("req-tool", CompletionReason.TOOL_CALLS),
    )

    _validate_stream(events)


def test_cancelled_and_failed_streams_are_valid_distinct_terminals() -> None:
    cancelled: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-cancel"),
        GenerationCancelled("req-cancel"),
    )
    failed: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-fail"),
        GenerationFailed(
            "req-fail",
            CanonicalError(
                ErrorCategory.RUNTIME_FAILURE,
                "runtime_failed",
                "Runtime failure.",
                True,
            ),
        ),
    )

    _validate_stream(cancelled)
    _validate_stream(failed)


def test_usage_snapshots_may_gain_measurements_but_not_decrease_known_counts() -> None:
    _assert_usage_not_decreased(
        TokenUsage(input_tokens=100, output_tokens=None),
        TokenUsage(input_tokens=100, output_tokens=3),
    )

    with pytest.raises(AssertionError):
        _assert_usage_not_decreased(
            TokenUsage(input_tokens=100, output_tokens=3),
            TokenUsage(input_tokens=99, output_tokens=3),
        )


def test_conformance_helper_rejects_events_after_terminal() -> None:
    invalid: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-invalid"),
        GenerationCompleted("req-invalid", CompletionReason.STOP),
        TextStarted("req-invalid"),
    )

    with pytest.raises(AssertionError):
        _validate_stream(invalid)


def test_conformance_helper_rejects_tool_identity_drift() -> None:
    invalid: tuple[GenerationEvent, ...] = (
        GenerationStarted("req-tool"),
        ToolCallStarted("req-tool", "call-1", "bash", 0),
        ToolCallArgumentsDelta("req-tool", "different", "{}", 0),
        GenerationCompleted("req-tool", CompletionReason.TOOL_CALLS),
    )

    with pytest.raises(AssertionError):
        _validate_stream(invalid)
