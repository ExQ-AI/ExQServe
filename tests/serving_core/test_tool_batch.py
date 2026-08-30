from __future__ import annotations

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.events import ToolCallArgumentsDelta, ToolCallCompleted, ToolCallStarted
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


def test_non_atomic_gate_keeps_tool_events_immediate() -> None:
    gate = ToolCallBatchGate(
        _policy(),
        tool_call_fanout_limit=32,
        atomic_parallel_tools=False,
        constrained_parallel_tool_call_limit=8,
    )
    started = ToolCallStarted("req", "call-1", "lookup", 0)

    decision = gate.on_started(started)

    assert decision.failure is None
    assert decision.events == (started,)
    assert not gate.has_buffered_events
    assert gate.commit_events() == ()


def test_atomic_gate_uses_lower_of_global_and_constrained_limits() -> None:
    gate = _atomic_gate(fanout=1, constrained=8)

    assert gate.on_started(ToolCallStarted("req", "call-1", "lookup", 0)).failure is None
    rejected = gate.on_started(ToolCallStarted("req", "call-2", "lookup", 1))

    assert rejected.failure is not None
    assert rejected.failure.code == "tool_policy_violation"
