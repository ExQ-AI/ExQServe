from __future__ import annotations

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire.controls.qwen import compile_qwen_tool_wire
from exqserve.tool_wire.engine import ProductionToolWireSession, ToolWireEngineStatus


def _policy(*, allow_parallel: bool = True) -> ToolPolicy:
    tool = FunctionTool(
        "write",
        "write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=True,
    )
    return ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=allow_parallel)


def _bundle(*, allow_parallel: bool = True):
    return compile_qwen_tool_wire(
        _policy(allow_parallel=allow_parallel), ToolConstraintMode.SCHEMA, max_parallel_calls=2
    )


def _finite_bundle():
    tool = FunctionTool(
        "write",
        "write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string","enum":["safe"]}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=True,
    )
    return compile_qwen_tool_wire(
        ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False),
        ToolConstraintMode.SCHEMA,
    )


def _wire(value: str) -> str:
    return (
        "<tool_call><function=write><parameter=content>\n"
        + value
        + "\n</parameter>\n</function></tool_call>"
    )


def test_production_session_native_raw_fake_full_close_is_data() -> None:
    bundle = _bundle()
    semantic = "prefix </parameter></function></tool_call> suffix"
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    wire = _wire(semantic)
    for split in range(0, len(wire), 7):
        result = session.feed(wire[split : split + 7])
        assert result.status is ToolWireEngineStatus.IN_PROGRESS
    result = session.finish()
    assert result.is_complete
    assert result.sequence is not None
    assert result.sequence.calls[0].occurrences[0].canonical_value_json == (
        '"prefix </parameter></function></tool_call> suffix"'
    )
    assert result.raw_region == wire
    assert result.remainder == ""


def test_production_session_adjacent_tools_and_exact_remainder() -> None:
    bundle = _bundle(allow_parallel=True)
    first = _wire("one")
    second = _wire("two")
    tail = "  tail text\n"
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed(first + second + tail)
    result = session.finish()
    assert result.is_complete
    assert result.sequence is not None
    assert [call.occurrences[0].canonical_value_json for call in result.sequence.calls] == [
        '"one"',
        '"two"',
    ]
    assert result.raw_region == first + second
    assert result.remainder == tail


def test_production_session_rejects_adjacent_tool_when_plan_disallows_parallel() -> None:
    bundle = _bundle(allow_parallel=False)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed(_wire("one") + _wire("two"))
    result = session.finish()
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert result.sequence is None
    assert {issue.code for issue in result.issues} == {"adjacent_tool_not_allowed"}


def test_production_session_rejects_structurally_valid_unselected_tool() -> None:
    bundle = _bundle(allow_parallel=False)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed("<tool_call><function=other></function></tool_call>")
    result = session.finish()
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert result.sequence is None
    assert {issue.code for issue in result.issues} == {"plan_tool_not_selected"}


def test_production_session_rejects_raw_value_outside_compiled_finite_language() -> None:
    bundle = _finite_bundle()
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed(_wire("other"))
    result = session.finish()
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert result.sequence is None
    assert {issue.code for issue in result.issues} == {"plan_finite_value_mismatch"}


def test_production_feed_parses_each_completed_envelope_exactly_once(monkeypatch) -> None:
    import exqserve.tool_wire.engine as engine_module

    bundle = _bundle()
    wire = _wire("x" * 16_384)
    calls = 0
    original = engine_module._parse_sequence

    def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "_parse_sequence", counted)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    last = None
    for split in range(0, len(wire), 257):
        last = session.feed(wire[split : split + 257])
    assert last is not None
    assert calls == 1
    assert last.status is ToolWireEngineStatus.IN_PROGRESS
    assert last.completed_envelopes == 1
    assert last.awaiting_next_envelope

    result = session.finish()
    assert result.is_complete
    assert result.completed_envelopes == 1
    assert calls == 1


def test_production_session_truncated_second_envelope_discards_whole_batch() -> None:
    bundle = _bundle(allow_parallel=True)
    first = _wire("one")
    second = _wire("two")
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed(first + second[:-9])
    result = session.finish()
    assert result.status is ToolWireEngineStatus.INCOMPLETE
    assert result.sequence is None
    assert result.completed_envelopes == 1


def test_production_session_enforces_the_compiled_generation_call_limit() -> None:
    bundle = _bundle(allow_parallel=True)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed(_wire("one") + _wire("two") + _wire("three"))
    result = session.finish()
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert result.sequence is None
    assert {issue.code for issue in result.issues} == {"tool_sequence_limit_exceeded"}


def test_production_session_holds_partial_adjacent_opener_until_finish() -> None:
    bundle = _bundle(allow_parallel=True)
    first = _wire("one")
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    result = session.feed(first + "  <tool_")

    assert result.status is ToolWireEngineStatus.IN_PROGRESS
    assert result.completed_envelopes == 1
    assert result.awaiting_next_envelope

    finished = session.finish()
    assert finished.status is ToolWireEngineStatus.INCOMPLETE
    assert finished.sequence is None


def test_production_session_returns_tail_without_stripping_or_rescanning() -> None:
    bundle = _bundle(allow_parallel=True)
    first = _wire("one")
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    result = session.feed(first + "  tail text\n")

    assert result.is_complete
    assert result.raw_region == first
    assert result.remainder == "  tail text\n"
    assert result.completed_envelopes == 1

    updated = session.feed("next line\n")
    assert updated.is_complete
    assert updated.raw_region == first
    assert updated.remainder == "  tail text\nnext line\n"


@pytest.mark.parametrize("payload_size", (1024, 4096, 16384))
def test_production_session_scaling_parses_one_large_envelope_once(
    monkeypatch,
    payload_size: int,
) -> None:
    import exqserve.tool_wire.engine as engine_module

    bundle = _bundle()
    wire = _wire("x" * payload_size)
    calls = 0
    original = engine_module._parse_sequence

    def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "_parse_sequence", counted)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    for offset in range(0, len(wire), 113):
        session.feed(wire[offset : offset + 113])
    result = session.finish()

    assert result.is_complete
    assert result.raw_region == wire
    assert calls == 1


def _finish_with_chunk_size(source: str, chunk_size: int):
    bundle = _bundle(allow_parallel=True)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    for offset in range(0, len(source), chunk_size):
        session.feed(source[offset : offset + chunk_size])
    return session.finish()


@pytest.mark.parametrize("chunk_size", (1, 2, 3, 7, 19, 257))
@pytest.mark.parametrize(
    "source",
    (
        _wire("prefix </parameter></function></tool_call> suffix"),
        _wire("one") + _wire("two"),
        _wire("one") + _wire("two") + "  tail text\n",
    ),
)
def test_production_session_chunking_is_semantically_invariant(
    source: str,
    chunk_size: int,
) -> None:
    whole = _finish_with_chunk_size(source, max(1, len(source)))
    chunked = _finish_with_chunk_size(source, chunk_size)

    assert chunked == whole


@pytest.mark.parametrize(
    ("gap", "expected_status"),
    (
        (" " * 8, ToolWireEngineStatus.COMPLETE),
        (" " * 9, ToolWireEngineStatus.MALFORMED),
        (chr(0xA0), ToolWireEngineStatus.MALFORMED),
    ),
)
def test_production_session_structural_whitespace_matches_constraint_boundary(
    gap: str,
    expected_status: ToolWireEngineStatus,
) -> None:
    bundle = _bundle(allow_parallel=False)
    wire = (
        "<tool_call><function=write><parameter=content>\nvalue\n</parameter>\n</function>"
        + gap
        + "</tool_call>"
    )
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed(wire)
    result = session.finish()

    assert result.status is expected_status
    if expected_status is ToolWireEngineStatus.COMPLETE:
        assert result.sequence is not None
    else:
        assert result.sequence is None


def test_production_session_long_incremental_whitespace_keeps_lookahead_bounded() -> None:
    bundle = _bundle(allow_parallel=True)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed(_wire("one"))

    for _ in range(4096):
        session.feed(" ")

    assert session._between == ""
    assert session._between_ws == ""
    assert session._between_ws_invalid is True
    result = session.finish()
    assert result.status is ToolWireEngineStatus.COMPLETE
    assert result.remainder == " " * 4096


def test_production_session_invalid_adjacent_whitespace_is_chunk_invariant() -> None:
    source = _wire("one") + (" " * 9) + _wire("two")
    whole = _finish_with_chunk_size(source, len(source))
    chunked = _finish_with_chunk_size(source, 1)

    assert chunked == whole
    assert chunked.status is ToolWireEngineStatus.MALFORMED
    assert chunked.sequence is None
    assert chunked.issues[0].code == "structural_whitespace_invalid"
