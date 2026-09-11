from __future__ import annotations

import codecs
import json
from dataclasses import replace

import pytest

from tests.tool_wire._deepseek_fixture import deepseek_v4_structured_dsml_spec
from tests.tool_wire._legacy_api import (
    ArgumentOrderingMode,
    CloseLanguage,
    DeterministicToolWireEngine,
    LiteralTerminal,
    ToolWireEngineStatus,
    WireToolSequence,
    admit_tool_sequence,
)
from tests.tool_wire._support import policy, schema_plan, tool


def _plan(properties: dict[str, dict[str, object]], *, ordering: ArgumentOrderingMode | None = None):
    spec = deepseek_v4_structured_dsml_spec()
    if ordering is not None:
        spec = replace(spec, ordering=ordering)
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    fn = tool("calc", json.dumps(schema, separators=(",", ":")), strict=True)
    return spec, schema_plan(spec, policy(fn), {"calc": tuple(properties)})


def _wire(spec, plan, calls: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]) -> str:
    parts = [spec.tool_open.text]
    for tool_name, occurrences in calls:
        parts.append("\n")
        parts.append(spec.function_open.render(tool_name))
        for name, value_json in occurrences:
            variant = spec.argument_framings[0]
            try:
                branch = plan.tool(tool_name)
            except KeyError:
                branch = None
            if branch is not None:
                planned = next((arg for arg in branch.arguments if arg.name == name), None)
                if planned is not None and planned.framing_variant_id is not None:
                    variant = spec.framing_variant(planned.framing_variant_id)
            parts.append("\n")
            parts.append(variant.argument_open.render(name))
            parts.append(value_json)
            parts.append(variant.argument_close.canonical.text)
        parts.append("\n")
        parts.append(spec.function_close.canonical.text)
    parts.append("\n")
    parts.append(spec.tool_close.canonical.text)
    return "".join(parts)


def _complete(engine: DeterministicToolWireEngine, chunks: tuple[str, ...]):
    for chunk in chunks:
        engine.feed(chunk)
    return engine.finish()


def test_a1_empty_feed_one_char_chunks_and_repeated_finish() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})
    wire = _wire(spec, plan, (("calc", (("value", "7"),)),))
    engine = DeterministicToolWireEngine(spec, plan)

    before = engine.feed("")
    assert before.status is ToolWireEngineStatus.IN_PROGRESS
    result = _complete(engine, tuple(wire))
    assert result.status is ToolWireEngineStatus.COMPLETE
    assert result.sequence is not None
    assert result.sequence.calls[0].occurrences[0].canonical_value_json == "7"
    assert engine.finish() is result
    assert engine.feed("ignored-after-finish") is result


def test_a1_chunk_partition_invariance_at_every_character_boundary() -> None:
    spec, plan = _plan({"value": {"type": "object"}})
    wire = _wire(
        spec,
        plan,
        (("calc", (("value", '{"nested":[1,{"ok":true}],"s":"x"}'),)),),
    )
    baseline = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert baseline.is_complete

    for split in range(len(wire) + 1):
        result = _complete(
            DeterministicToolWireEngine(spec, plan),
            (wire[:split], wire[split:]),
        )
        assert result == baseline


def test_a1_utf8_byte_partition_invariance_across_structural_markers() -> None:
    spec, plan = _plan({"value": {"type": "string"}})
    wire = _wire(spec, plan, (("calc", (("value", '"跨字节"'),)),))
    baseline = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert baseline.is_complete
    encoded = wire.encode("utf-8")

    for split in range(len(encoded) + 1):
        decoder = codecs.getincrementaldecoder("utf-8")()
        first = decoder.decode(encoded[:split])
        second = decoder.decode(encoded[split:], final=True)
        result = _complete(
            DeterministicToolWireEngine(spec, plan),
            (first, second),
        )
        assert result == baseline


def test_a1_multiple_markers_in_one_chunk_and_multiple_invokes() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})
    wire = _wire(
        spec,
        plan,
        (
            ("calc", (("value", "1"),)),
            ("calc", (("value", "2"),)),
        ),
    )
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.is_complete
    assert result.sequence is not None
    assert [call.index for call in result.sequence.calls] == [0, 1]
    assert [call.occurrences[0].canonical_value_json for call in result.sequence.calls] == ["1", "2"]


@pytest.mark.parametrize(
    ("schema", "wire_value", "canonical"),
    [
        ({"type": "integer"}, "42", "42"),
        ({"type": "boolean"}, "true", "true"),
        ({"type": "null"}, "null", "null"),
        ({"type": "string"}, '"hello"', '"hello"'),
        ({"type": "array"}, '[1,{"a":false},null]', '[1,{"a":false},null]'),
        ({"type": "object"}, '{"b":[2,1],"a":true}', '{"a":true,"b":[2,1]}'),
    ],
)
def test_a1_structured_json_value_matrix(
    schema: dict[str, object],
    wire_value: str,
    canonical: str,
) -> None:
    spec, plan = _plan({"value": schema})
    wire = _wire(spec, plan, (("calc", (("value", wire_value),)),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.is_complete
    assert result.sequence is not None
    assert result.sequence.calls[0].occurrences[0].canonical_value_json == canonical


def test_a1_json_string_protects_close_looking_dsml_and_escapes() -> None:
    spec, plan = _plan({"value": {"type": "string"}})
    close_text = spec.argument_framings[0].argument_close.canonical.text
    value = json.dumps(f'x {close_text} y "quoted" \\ tail', ensure_ascii=False)
    wire = _wire(spec, plan, (("calc", (("value", value),)),))
    result = _complete(DeterministicToolWireEngine(spec, plan), tuple(wire))
    assert result.is_complete
    assert result.sequence is not None
    assert json.loads(result.sequence.calls[0].occurrences[0].canonical_value_json) == json.loads(value)


def test_a1_partial_terminal_and_incomplete_nested_json_are_incomplete_at_eos() -> None:
    spec, plan = _plan({"value": {"type": "object"}})
    wire = _wire(spec, plan, (("calc", (("value", '{"a":[1,2]}'),)),))

    partial_terminal = wire[: -len(spec.tool_close.canonical.text) + 3]
    terminal_result = _complete(DeterministicToolWireEngine(spec, plan), (partial_terminal,))
    assert terminal_result.status is ToolWireEngineStatus.INCOMPLETE
    assert {issue.code for issue in terminal_result.issues} == {"incomplete_wire"}

    incomplete_json = _wire(spec, plan, (("calc", (("value", '{"a":[1,2]'),)),))
    json_result = _complete(DeterministicToolWireEngine(spec, plan), (incomplete_json,))
    assert json_result.status is ToolWireEngineStatus.INCOMPLETE
    assert {issue.code for issue in json_result.issues} == {"incomplete_wire"}


def test_a1_malformed_json_is_classified_without_publication() -> None:
    spec, plan = _plan({"value": {"type": "object"}})
    wire = _wire(spec, plan, (("calc", (("value", '{"a":]'),)),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert result.sequence is None
    assert not hasattr(result, "published_events")


def test_a1_duplicate_argument_is_preserved_then_rejected_by_a0_admission() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})
    wire = _wire(spec, plan, (("calc", (("value", "1"), ("value", "2"))),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.is_complete
    assert result.sequence is not None
    assert [occ.canonical_value_json for occ in result.sequence.calls[0].occurrences] == ["1", "2"]

    admission = admit_tool_sequence(spec, plan, result.sequence)
    assert not admission.is_valid
    assert "duplicate_argument" in {issue.code for issue in admission.issues}


def test_a1_required_omission_is_structural_success_then_a0_rejection() -> None:
    spec, plan = _plan(
        {
            "first": {"type": "integer"},
            "second": {"type": "integer"},
        }
    )
    wire = _wire(spec, plan, (("calc", (("first", "1"),)),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.is_complete
    assert result.sequence is not None

    admission = admit_tool_sequence(spec, plan, result.sequence)
    assert not admission.is_valid
    codes = {issue.code for issue in admission.issues}
    assert "required_argument_missing" in codes


def test_a1_order_is_preserved_and_declaration_order_is_owned_by_admission() -> None:
    spec, plan = _plan(
        {
            "first": {"type": "integer"},
            "second": {"type": "integer"},
        },
        ordering=ArgumentOrderingMode.DECLARATION_ORDER,
    )
    wire = _wire(spec, plan, (("calc", (("second", "2"), ("first", "1"))),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.is_complete
    assert result.sequence is not None
    assert [occ.name for occ in result.sequence.calls[0].occurrences] == ["second", "first"]

    admission = admit_tool_sequence(spec, plan, result.sequence)
    assert not admission.is_valid
    assert "argument_order_invalid" in {issue.code for issue in admission.issues}


def test_a1_schema_semantics_remain_a0_admission_authority() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})
    wire = _wire(spec, plan, (("calc", (("value", '"syntactically-json-but-not-integer"'),)),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.is_complete
    assert result.sequence is not None
    assert result.sequence.calls[0].occurrences[0].canonical_value_json == (
        '"syntactically-json-but-not-integer"'
    )

    admission = admit_tool_sequence(spec, plan, result.sequence)
    assert not admission.is_valid
    assert "structured_value_not_admitted_by_plan" in {
        issue.code for issue in admission.issues
    }


@pytest.mark.parametrize(
    "bad_name",
    ["bad name", "bad\tname", "bad<name", "bad>name", "bad\u3000name"],
)
def test_a1_deepseek_static_function_name_language_rejects_invalid_names(
    bad_name: str,
) -> None:
    spec = deepseek_v4_structured_dsml_spec()
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    bad_tool = tool(bad_name, json.dumps(schema, separators=(",", ":")), strict=True)
    bad_plan = schema_plan(spec, policy(bad_tool), {bad_name: ("value",)})
    assert not bad_plan.constrained_executable
    assert not bad_plan.tool(bad_name).representable

    _, plan = _plan({"value": {"type": "integer"}})
    wire = _wire(spec, plan, ((bad_name, (("value", "1"),)),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert {issue.code for issue in result.issues} == {"function_name_not_representable"}


def test_a1_deepseek_static_function_name_whitespace_matches_runtime_isspace() -> None:
    spec = deepseek_v4_structured_dsml_spec()
    declared = {
        value
        for value in spec.function_name_codec.forbidden_sequences
        if len(value) == 1 and value.isspace()
    }
    runtime = {chr(codepoint) for codepoint in range(0x110000) if chr(codepoint).isspace()}
    assert declared == runtime


def test_a1_undeclared_tool_and_argument_are_not_hard_coded_into_engine() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})

    undeclared_tool_wire = _wire(spec, plan, (("other", (("value", "1"),)),))
    tool_result = _complete(DeterministicToolWireEngine(spec, plan), (undeclared_tool_wire,))
    assert tool_result.is_complete
    assert tool_result.sequence is not None
    tool_admission = admit_tool_sequence(spec, plan, tool_result.sequence)
    assert "undeclared_tool_branch" in {issue.code for issue in tool_admission.issues}

    undeclared_arg_wire = _wire(spec, plan, (("calc", (("other", "1"),)),))
    arg_result = _complete(DeterministicToolWireEngine(spec, plan), (undeclared_arg_wire,))
    assert arg_result.is_complete
    assert arg_result.sequence is not None
    arg_admission = admit_tool_sequence(spec, plan, arg_result.sequence)
    assert "undeclared_argument" in {issue.code for issue in arg_admission.issues}


def test_a1_deepseek_empty_tool_block_is_rejected_by_shared_cardinality() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})
    empty_wire = f"{spec.tool_open.text}{spec.tool_close.canonical.text}"

    result = _complete(DeterministicToolWireEngine(spec, plan), (empty_wire,))
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert {issue.code for issue in result.issues} == {"tool_multiplicity_below_minimum"}

    admission = admit_tool_sequence(spec, plan, WireToolSequence(()))
    assert not admission.is_valid
    assert {issue.code for issue in admission.issues} == {"tool_multiplicity_below_minimum"}


def test_a1_raw_string_true_form_is_out_of_scope_and_not_silently_accepted() -> None:
    spec, plan = _plan({"value": {"type": "string"}})
    variant = spec.argument_framings[0]
    wire = _wire(spec, plan, (("calc", (("value", '"safe"'),)),))
    raw_wire = wire.replace(variant.argument_open.suffix, '" string="true">', 1)
    result = _complete(DeterministicToolWireEngine(spec, plan), (raw_wire,))
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert result.sequence is None


def test_a1_engine_result_is_occurrence_preserving_wire_sequence_only() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})
    wire = _wire(spec, plan, (("calc", (("value", "1"),)),))
    result = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert result.is_complete
    assert isinstance(result.sequence, WireToolSequence)
    assert not hasattr(result, "tool_batch")
    assert not hasattr(result, "publication")


def _plan_for_spec(spec):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    fn = tool("calc", json.dumps(schema, separators=(",", ":")), strict=True)
    return schema_plan(spec, policy(fn), {"calc": ("value",)})


def _wire_with_closes(
    spec,
    *,
    argument_close: str,
    function_close: str,
    tool_close: str,
) -> str:
    variant = spec.argument_framings[0]
    return "".join(
        (
            spec.tool_open.text,
            "\n",
            spec.function_open.render("calc"),
            "\n",
            variant.argument_open.render("value"),
            "1",
            argument_close,
            "\n",
            function_close,
            "\n",
            tool_close,
        )
    )


@pytest.mark.parametrize("target", ["tool", "function", "argument"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_a1_overlapping_close_forms_are_order_independent_and_chunk_safe(
    target: str,
    reverse_order: bool,
) -> None:
    spec = deepseek_v4_structured_dsml_spec()
    short = f"</{target}>"
    long = f"</{target}>x"
    ordered = (long, short) if reverse_order else (short, long)
    close_language = CloseLanguage(tuple(LiteralTerminal(value) for value in ordered))

    argument_close = spec.argument_framings[0].argument_close.canonical.text
    function_close = spec.function_close.canonical.text
    tool_close = spec.tool_close.canonical.text
    if target == "tool":
        spec = replace(spec, tool_close=close_language)
        tool_close = long
    elif target == "function":
        spec = replace(spec, function_close=close_language)
        function_close = long
    else:
        variant = replace(spec.argument_framings[0], argument_close=close_language)
        spec = replace(spec, argument_framings=(variant,))
        argument_close = long

    plan = _plan_for_spec(spec)
    wire = _wire_with_closes(
        spec,
        argument_close=argument_close,
        function_close=function_close,
        tool_close=tool_close,
    )
    complete = _complete(DeterministicToolWireEngine(spec, plan), (wire,))
    assert complete.is_complete

    long_at = wire.index(long)
    boundary = long_at + len(short)
    engine = DeterministicToolWireEngine(spec, plan)
    provisional = engine.feed(wire[:boundary])
    assert provisional.status is ToolWireEngineStatus.IN_PROGRESS
    engine.feed(wire[boundary:])
    assert engine.finish() == complete

    short_wire = wire[:long_at] + short + wire[long_at + len(long) :]
    short_engine = DeterministicToolWireEngine(spec, plan)
    short_engine.feed(short_wire)
    assert short_engine.finish().is_complete
