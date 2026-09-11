from __future__ import annotations

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.events import (
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import ToolCallItem
from exqserve.serving.tool_batch import ToolCallBatchGate


def _policy() -> ToolPolicy:
    tool = FunctionTool(
        "lookup",
        None,
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"integer"}},'
            '"required":["id"],"additionalProperties":false}'
        ),
    )
    return ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)


def _atomic_gate(*, fanout: int = 32, constrained: int = 8) -> ToolCallBatchGate:
    return ToolCallBatchGate(
        _policy(),
        tool_call_fanout_limit=fanout,
        atomic_parallel_tools=True,
        constrained_parallel_tool_call_limit=constrained,
    )


def _non_atomic_gate() -> ToolCallBatchGate:
    return ToolCallBatchGate(
        _policy(),
        tool_call_fanout_limit=32,
        atomic_parallel_tools=False,
        constrained_parallel_tool_call_limit=8,
    )


def test_atomic_gate_commit_is_one_shot_and_abort_cannot_reopen_it() -> None:
    gate = _atomic_gate()
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    events = (
        ToolCallStarted("req", "call-1", "lookup", 0),
        ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
        ToolCallCompleted("req", call),
    )

    assert gate.on_started(events[0]).events == ()
    assert gate.on_arguments_delta(events[1]).events == ()
    assert gate.on_completed(events[2]).events == ()
    assert gate.buffered_event_count == 3

    assert gate.commit_events() == events
    assert gate.commit_events() == ()
    gate.abort()
    assert gate.commit_events() == ()
    assert not gate.has_buffered_events


def test_atomic_gate_abort_is_idempotent_and_commit_after_abort_is_empty() -> None:
    gate = _atomic_gate()
    started = ToolCallStarted("req", "call-1", "lookup", 0)

    assert gate.on_started(started).failure is None
    assert gate.has_buffered_events
    gate.abort()
    gate.abort()

    assert not gate.has_buffered_events
    assert gate.commit_events() == ()
    late = gate.on_started(ToolCallStarted("req", "call-2", "lookup", 1))
    assert late.failure is not None
    assert late.failure.code == "tool_call_stream_invalid"


def test_non_atomic_gate_keeps_tool_events_immediate_before_completion_barrier() -> None:
    gate = _non_atomic_gate()
    started = ToolCallStarted("req", "call-1", "lookup", 0)

    decision = gate.on_started(started)

    assert decision.failure is None
    assert decision.events == (started,)
    assert not gate.has_buffered_events
    assert gate.commit_events() == ()


def test_non_atomic_completion_barrier_holds_following_tail_in_original_order() -> None:
    gate = _non_atomic_gate()
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    completed = ToolCallCompleted("req", call)
    tail_text = TextDelta("req", "after")
    second_started = ToolCallStarted("req", "call-2", "lookup", 1)
    second_delta = ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1)

    assert gate.on_started(ToolCallStarted("req", "call-1", "lookup", 0)).events
    assert gate.on_arguments_delta(
        ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0)
    ).events
    assert gate.on_completed(completed).events == ()
    assert gate.on_passthrough(tail_text).events == ()
    assert gate.on_started(second_started).events == ()
    assert gate.on_arguments_delta(second_delta).events == ()

    assert gate.commit_events() == (completed, tail_text, second_started, second_delta)


def test_non_atomic_completion_barrier_discards_following_tail_on_abort() -> None:
    gate = _non_atomic_gate()
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)

    gate.on_started(ToolCallStarted("req", "call-1", "lookup", 0))
    gate.on_arguments_delta(ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0))
    gate.on_completed(ToolCallCompleted("req", call))
    gate.on_passthrough(TextDelta("req", "must-not-publish"))
    gate.abort()

    assert not gate.has_buffered_events
    assert gate.commit_events() == ()


def test_atomic_gate_uses_lower_of_global_and_constrained_limits() -> None:
    gate = _atomic_gate(fanout=1, constrained=8)

    assert gate.on_started(ToolCallStarted("req", "call-1", "lookup", 0)).failure is None
    rejected = gate.on_started(ToolCallStarted("req", "call-2", "lookup", 1))

    assert rejected.failure is not None
    assert rejected.failure.code == "tool_policy_violation"
