from __future__ import annotations

import json

import pytest

from exqserve.agent._json import canonical_json_dumps, parse_json_strict
from exqserve.core.events import ToolCallCompleted
from exqserve.model.deepseek_v4 import DeepSeekV4IncrementalParser, DeepSeekV4ParserContext
from exqserve.tool_wire import (
    ToolWireEngineStatus,
    WireArgumentOccurrence,
    WireToolCall,
    WireToolSequence,
    certify_engine_shadow,
)
from exqserve.tool_wire.controls.deepseek_v4 import deepseek_v4_structured_dsml_spec
from tests.tool_wire._support import policy, schema_plan, tool


def _plan(properties: dict[str, dict[str, object]]):
    spec = deepseek_v4_structured_dsml_spec()
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    fn = tool("calc", json.dumps(schema, separators=(",", ":")), strict=True)
    return spec, schema_plan(spec, policy(fn), {"calc": tuple(properties)})


def _wire(spec, plan, calls: tuple[tuple[tuple[str, str], ...], ...]) -> str:
    parts = [spec.tool_open.text]
    for occurrences in calls:
        parts.extend(("\n", spec.function_open.render("calc")))
        for name, value_json in occurrences:
            branch = plan.tool("calc")
            argument = next(value for value in branch.arguments if value.name == name)
            assert argument.framing_variant_id is not None
            variant = spec.framing_variant(argument.framing_variant_id)
            parts.extend(
                (
                    "\n",
                    variant.argument_open.render(name),
                    value_json,
                    variant.argument_close.canonical.text,
                )
            )
        parts.extend(("\n", spec.function_close.canonical.text))
    parts.extend(("\n", spec.tool_close.canonical.text))
    return "".join(parts)


def _deepseek_reference_decoder(
    spec,
    property_names: tuple[str, ...],
    occurrence_orders: tuple[tuple[str, ...], ...],
):
    variant_id = spec.argument_framings[0].variant_id

    def decode(wire: str) -> WireToolSequence:
        parser = DeepSeekV4IncrementalParser(
            "a1-shadow",
            start_in_reasoning=False,
            parser_context=DeepSeekV4ParserContext(
                True,
                {"calc": frozenset(property_names)},
            ),
        )
        events = list(parser.feed(wire))
        events.extend(parser.finish().events)
        calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
        assert len(calls) == len(occurrence_orders)
        wire_calls: list[WireToolCall] = []
        for call, occurrence_order in zip(calls, occurrence_orders, strict=True):
            arguments = parse_json_strict(call.arguments_json)
            assert isinstance(arguments, dict)
            occurrences = tuple(
                WireArgumentOccurrence(
                    name,
                    canonical_json_dumps(arguments[name]),
                    variant_id,
                )
                for name in occurrence_order
            )
            wire_calls.append(WireToolCall(call.name, call.index, occurrences))
        return WireToolSequence(tuple(wire_calls))

    return decode


@pytest.mark.parametrize(
    ("schema", "wire_value", "admission_expected"),
    [
        ({"type": "integer"}, "42", True),
        ({"type": "boolean"}, "false", True),
        ({"type": "null"}, "null", True),
        ({"type": "string"}, '"literal </｜DSML｜parameter> text"', True),
        ({"type": "string"}, '"quote: \\" and slash: \\\\"', True),
        ({"type": "array"}, '[1,{"nested":[true,null,"x"]}]', True),
        ({"type": "object"}, '{"a":{"b":[1,2]},"s":"</｜DSML｜invoke>"}', True),
    ],
)
def test_a1_deepseek_structured_shadow_semantic_parity(
    schema: dict[str, object],
    wire_value: str,
    admission_expected: bool,
) -> None:
    spec, plan = _plan({"value": schema})
    wire = _wire(spec, plan, ((("value", wire_value),),))
    reference = _deepseek_reference_decoder(spec, ("value",), (("value",),))

    shadow = certify_engine_shadow(
        spec,
        plan,
        tuple(wire),
        reference_decoder=reference,
    )
    assert shadow.engine_result.status is ToolWireEngineStatus.COMPLETE
    assert shadow.admission is not None
    assert shadow.admission.is_valid is admission_expected
    assert shadow.semantic_match
    assert shadow.is_certified is admission_expected
    assert shadow.reference_sequence == shadow.engine_result.sequence


def test_a1_deepseek_multiple_arguments_and_invokes_shadow_parity() -> None:
    spec, plan = _plan(
        {
            "first": {"type": "integer"},
            "second": {"type": "object"},
        }
    )
    calls = (
        (("second", '{"z":2,"a":1}'), ("first", "7")),
        (("first", "8"), ("second", '{"nested":[1,2,3]}')),
    )
    wire = _wire(spec, plan, calls)
    orders = tuple(tuple(name for name, _ in call) for call in calls)
    reference = _deepseek_reference_decoder(spec, ("first", "second"), orders)

    shadow = certify_engine_shadow(
        spec,
        plan,
        (wire[:17], wire[17:53], wire[53:]),
        reference_decoder=reference,
    )
    assert shadow.engine_result.is_complete
    assert shadow.admission is not None and shadow.admission.is_valid
    assert shadow.semantic_match
    assert shadow.is_certified
    assert shadow.engine_result.sequence is not None
    assert [tuple(occ.name for occ in call.occurrences) for call in shadow.engine_result.sequence.calls] == list(orders)


def test_a1_shadow_malformed_or_incomplete_never_reaches_a0_admission() -> None:
    spec, plan = _plan({"value": {"type": "object"}})
    complete = _wire(spec, plan, ((("value", '{"a":[1,2]}'),),))
    incomplete = complete[: complete.index('{"a":[1,2]}') + len('{"a":[1,2')]

    shadow = certify_engine_shadow(spec, plan, (incomplete,))
    assert shadow.engine_result.status is ToolWireEngineStatus.INCOMPLETE
    assert shadow.admission is None
    assert shadow.reference_sequence is None
    assert not shadow.semantic_match


def test_a1_shadow_without_reference_never_certifies() -> None:
    spec, plan = _plan({"value": {"type": "integer"}})
    wire = _wire(spec, plan, ((("value", "7"),),))

    shadow = certify_engine_shadow(spec, plan, (wire,))
    assert shadow.engine_result.is_complete
    assert shadow.admission is not None and shadow.admission.is_valid
    assert shadow.reference_sequence is None
    assert shadow.semantic_match is None
    assert not shadow.is_certified


def test_a1_deepseek_control_contains_no_raw_string_true_variant() -> None:
    spec = deepseek_v4_structured_dsml_spec()
    assert len(spec.argument_framings) == 1
    variant = spec.argument_framings[0]
    assert 'string="false"' in variant.argument_open.suffix_remainder
    assert 'string="true"' not in variant.argument_open.suffix_remainder
