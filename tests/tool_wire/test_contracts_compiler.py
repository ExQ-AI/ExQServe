from __future__ import annotations

from dataclasses import fields, replace

import pytest

from exqserve.agent._json import parse_json_strict
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire import (
    ArgumentFramingVariant,
    ArgumentOccurrenceCapabilities,
    ArgumentOrderingMode,
    CloseLanguage,
    LiteralTerminal,
    NamedTerminal,
    RepresentabilityStatus,
    ToolMultiplicity,
    ToolWireCompileError,
    ToolWireSpec,
    ValueCodecKind,
    ValueFraming,
    ValueFramingKind,
    compile_tool_wire_plan,
    encode_lossless_raw_string,
)
from tests.tool_wire._support import (
    budget,
    policy,
    raw_compiler_capabilities,
    raw_spec,
    schema_plan,
    structured_spec,
    tool,
)


def test_tool_multiplicity_rejects_unsatisfiable_non_adjacent_minimums() -> None:
    with pytest.raises(ValueError, match="must not exceed one"):
        ToolMultiplicity(None, False, min_calls_per_sequence=2)
    with pytest.raises(ValueError, match="must not exceed one"):
        ToolMultiplicity(8, False, min_calls_per_sequence=7)


def test_tool_multiplicity_accepts_satisfiable_non_adjacent_and_adjacent_states() -> None:
    assert ToolMultiplicity(None, False, min_calls_per_sequence=0).min_calls_per_sequence == 0
    assert ToolMultiplicity(None, False, min_calls_per_sequence=1).min_calls_per_sequence == 1
    assert ToolMultiplicity(1, False, min_calls_per_sequence=1).max_calls_per_sequence == 1
    assert ToolMultiplicity(4, True, min_calls_per_sequence=2).min_calls_per_sequence == 2

    with pytest.raises(ValueError, match="must not exceed max_calls_per_sequence"):
        ToolMultiplicity(2, True, min_calls_per_sequence=3)


def test_static_spec_is_request_neutral_and_fingerprint_is_deterministic() -> None:
    first = raw_spec()
    second = raw_spec()

    assert first == second
    assert first.fingerprint == second.fingerprint
    static_fields = {field.name for field in fields(ToolWireSpec)}
    assert "tools" not in static_fields
    assert "tool_policy" not in static_fields
    assert "prompt" not in static_fields
    assert "request_schema" not in static_fields
    assert "schema_capabilities" not in static_fields
    assert "compiler_capabilities" not in static_fields
    raw_close = first.argument_framings[0].argument_close
    assert raw_close.forms[0].native_token_ids == (500,)
    assert raw_close.forms[1].native_token_ids == (501,)


def test_raw_codec_normalization_and_json_string_probe_are_explicit_not_generic() -> None:
    exact = ValueCodecKind.RAW_STRING
    native = ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT
    close = CloseLanguage((LiteralTerminal("</parameter>"),))
    framing = ValueFraming(ValueFramingKind.RAW_UNTIL, native, close)

    assert exact.decode_raw_payload("\nvalue\n") == "\nvalue\n"
    assert native.decode_raw_payload("\nvalue\n") == "value"
    assert native.decode_raw_payload('"value"') == "value"
    assert native.decode_raw_payload('"a\\nb"') == "a\nb"
    assert native.decode_raw_payload("true") == "true"
    assert native.decode_raw_payload("line1\nline2") == "line1\nline2"
    assert exact.is_losslessly_representable_raw_value(" leading ")
    assert not native.is_losslessly_representable_raw_value(" leading ")
    assert encode_lossless_raw_string(" leading ", framing) == '" leading "'
    safe_close = encode_lossless_raw_string("x</parameter>y", framing)
    assert safe_close == '"x\\u003c/parameter>y"'
    assert native.decode_raw_payload(safe_close) == "x</parameter>y"


def test_raw_until_requires_forbidden_close_language_to_exactly_cover_accepted_closes() -> None:
    accepted = CloseLanguage(
        (LiteralTerminal("</arg>", (10,)), LiteralTerminal("</arg >", (11,)))
    )
    incomplete = CloseLanguage((LiteralTerminal("</arg>", (10,)),))

    with pytest.raises(ValueError, match="exactly cover"):
        ArgumentFramingVariant(
            "bad-raw",
            NamedTerminal("<arg=", ">"),
            accepted,
            ValueFraming(
                ValueFramingKind.RAW_UNTIL,
                ValueCodecKind.RAW_STRING,
                incomplete,
            ),
        )


def test_compiler_requires_explicit_presentation_order_not_canonical_schema_order() -> None:
    write = tool(
        "write",
        '{"type":"object","properties":{'
        '"file_path":{"type":"string"},"content":{"type":"string"}},'
        '"required":["file_path","content"],"additionalProperties":false}',
    )
    parsed = parse_json_strict(write.parameters.canonical_json)
    assert isinstance(parsed, dict)
    properties = parsed["properties"]
    assert isinstance(properties, dict)
    assert tuple(properties) == ("content", "file_path")

    plan = schema_plan(raw_spec(), policy(write), {"write": ("file_path", "content")})
    assert tuple(argument.name for argument in plan.tool("write").arguments) == (
        "file_path",
        "content",
    )
    assert plan.presentation_orders == (("write", ("file_path", "content")),)

    with pytest.raises(ToolWireCompileError, match="presentation order evidence"):
        schema_plan(raw_spec(), policy(write), {})


def test_raw_enum_intersects_every_close_alias_and_keeps_schema_guarantee() -> None:
    choose = tool(
        "choose",
        '{"type":"object","properties":{"value":{"type":"string","enum":['
        '"safe","bad</parameter>","also bad</parameter >"]}},'
        '"required":["value"],"additionalProperties":false}',
    )

    branch = schema_plan(raw_spec(), policy(choose), {"choose": ("value",)}).tool("choose")
    argument = branch.arguments[0]
    assert argument.representability is RepresentabilityStatus.REPRESENTABLE
    assert argument.generated is True
    assert argument.admitted_values_json == ('"safe"',)
    assert argument.proof.guarantee is GenerationGuarantee.SCHEMA
    assert branch.guarantee is GenerationGuarantee.SCHEMA


def test_raw_const_collision_makes_required_strict_branch_unrepresentable() -> None:
    write = tool(
        "write",
        '{"type":"object","properties":{"content":{'
        '"type":"string","const":"x</parameter >y"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )

    branch = schema_plan(raw_spec(), policy(write), {"write": ("content",)}).tool("write")
    assert branch.arguments[0].representability is RepresentabilityStatus.EMPTY
    assert branch.arguments[0].generated is False
    assert branch.representable is False
    assert branch.guarantee is GenerationGuarantee.NONE


def test_schema_mode_may_narrow_unsupported_optional_but_not_required_branch() -> None:
    optional = tool(
        "search",
        '{"type":"object","properties":{'
        '"query":{"type":"string"},"hint":{"type":"string","pattern":"^x"}},'
        '"required":["query"],"additionalProperties":false}',
    )
    optional_branch = schema_plan(
        raw_spec(ordering=ArgumentOrderingMode.DECLARATION_ORDER),
        policy(optional),
        {"search": ("query", "hint")},
    ).tool("search")

    assert optional_branch.guarantee is GenerationGuarantee.SCHEMA
    assert optional_branch.representable is True
    assert optional_branch.arguments[1].generated is False
    assert optional_branch.order_plan.orders == (("query",),)

    required = tool(
        "search",
        '{"type":"object","properties":{'
        '"query":{"type":"string"},"hint":{"type":"string","pattern":"^x"}},'
        '"required":["query","hint"],"additionalProperties":false}',
        strict=True,
    )
    required_branch = schema_plan(
        raw_spec(ordering=ArgumentOrderingMode.DECLARATION_ORDER),
        policy(required),
        {"search": ("query", "hint")},
    ).tool("search")
    assert required_branch.arguments[1].generated is False
    assert required_branch.guarantee is GenerationGuarantee.NONE
    assert required_branch.representable is False


def test_wire_required_optional_cannot_be_silently_narrowed_away() -> None:
    fn = tool(
        "search",
        '{"type":"object","properties":{'
        '"query":{"type":"string"},"hint":{"type":"string","pattern":"^x"}},'
        '"required":["query"],"additionalProperties":false}',
    )
    spec = replace(
        raw_spec(ordering=ArgumentOrderingMode.DECLARATION_ORDER),
        occurrence=ArgumentOccurrenceCapabilities(1, False, False),
    )
    branch = schema_plan(spec, policy(fn), {"search": ("query", "hint")}).tool("search")

    assert branch.arguments[0].wire_required is True
    assert branch.arguments[1].wire_required is True
    assert branch.arguments[1].generated is False
    assert branch.representable is False
    assert branch.guarantee is GenerationGuarantee.NONE


def test_object_schema_capability_caps_tool_guarantee_and_plan_identity() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    full = raw_compiler_capabilities()
    missing_object_keyword = replace(
        full,
        capability_id="synthetic-raw-no-additional-properties",
        supported_object_keywords=("type", "properties", "required"),
    )

    full_plan = schema_plan(raw_spec(), policy(fn), {"write": ("content",)})
    weak_plan = schema_plan(
        raw_spec(),
        policy(fn),
        {"write": ("content",)},
        compiler_capabilities=missing_object_keyword,
    )
    weak_branch = weak_plan.tool("write")

    assert full_plan.fingerprint != weak_plan.fingerprint
    assert weak_branch.object_schema_safe is False
    assert "additionalProperties" in weak_branch.object_schema_detail
    assert weak_branch.guarantee is GenerationGuarantee.FORMAT
    assert weak_branch.representable is False


def test_unsupported_object_semantics_cannot_be_misreported_as_schema() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false,"minProperties":2}',
        strict=True,
    )

    branch = schema_plan(raw_spec(), policy(fn), {"write": ("content",)}).tool("write")

    assert branch.object_schema_safe is False
    assert "minProperties" in branch.object_schema_detail
    assert branch.guarantee is GenerationGuarantee.FORMAT
    assert branch.representable is False


def test_raw_const_and_enum_are_intersected_before_schema_guarantee() -> None:
    fn = tool(
        "choose",
        '{"type":"object","properties":{"value":{'
        '"type":"string","const":"a","enum":["b","c"]}},'
        '"required":["value"],"additionalProperties":false}',
        strict=True,
    )

    branch = schema_plan(raw_spec(), policy(fn), {"choose": ("value",)}).tool("choose")
    argument = branch.arguments[0]

    assert argument.representability is RepresentabilityStatus.EMPTY
    assert argument.admitted_values_json == ()
    assert argument.generated is False
    assert branch.guarantee is GenerationGuarantee.NONE
    assert branch.representable is False


def test_structured_schema_capability_is_recursive_not_top_level_only() -> None:
    fn = tool(
        "store",
        '{"type":"object","properties":{"payload":{'
        '"type":"object","properties":{"name":{"type":"string","pattern":"^a"}},'
        '"required":["name"]}},"required":["payload"],"additionalProperties":false}',
        strict=True,
    )

    branch = schema_plan(
        structured_spec(),
        policy(fn),
        {"store": ("payload",)},
    ).tool("store")
    argument = branch.arguments[0]

    assert argument.proof.schema_safe is False
    assert "pattern" in argument.proof.detail
    assert argument.generated is False
    assert branch.guarantee is GenerationGuarantee.NONE
    assert branch.representable is False


@pytest.mark.parametrize(
    ("count", "limit", "expected_count", "narrowed"),
    (
        (0, 1000, 1, False),
        (1, 1000, 1, False),
        (6, 1000, 720, False),
        (7, 1000, None, True),
        (20, 1000, None, True),
    ),
)
def test_permutation_budget_is_bounded_and_truthful(
    count: int,
    limit: int,
    expected_count: int | None,
    narrowed: bool,
) -> None:
    properties = ",".join(f'"p{i}":{{"type":"string"}}' for i in range(count))
    required = ",".join(f'"p{i}"' for i in range(count))
    schema = (
        '{"type":"object","properties":{'
        + properties
        + '},"required":['
        + required
        + '],"additionalProperties":false}'
    )
    fn = tool("f", schema)
    order = tuple(f"p{i}" for i in range(count))
    plan = schema_plan(
        raw_spec(ordering=ArgumentOrderingMode.PERMUTABLE),
        policy(fn),
        {"f": order},
        compile_budget=budget(limit),
    )
    order_plan = plan.tool("f").order_plan

    assert order_plan.full_order_count == expected_count
    assert order_plan.narrowed is narrowed
    assert len(order_plan.orders) == (1 if narrowed else expected_count)
    assert plan.budget_result.narrowed_permutations is narrowed
    if narrowed:
        assert order_plan.orders == (order,)


def test_many_optional_parameters_narrow_to_stable_required_subset_without_enumeration() -> None:
    optional_count = 30
    property_parts = ['"required":{"type":"string"}']
    property_parts.extend(f'"o{i}":{{"type":"string"}}' for i in range(optional_count))
    schema = (
        '{"type":"object","properties":{'
        + ",".join(property_parts)
        + '},"required":["required"],"additionalProperties":false}'
    )
    fn = tool("many", schema)
    order = ("required", *(f"o{i}" for i in range(optional_count)))
    branch = schema_plan(
        raw_spec(ordering=ArgumentOrderingMode.PERMUTABLE),
        policy(fn),
        {"many": order},
        compile_budget=budget(1000),
    ).tool("many")

    assert branch.order_plan.narrowed is True
    assert branch.order_plan.full_order_count is None
    assert branch.order_plan.orders == (("required",),)
    assert branch.guarantee is GenerationGuarantee.SCHEMA


def test_same_contract_compiler_handles_structured_json_value_framing() -> None:
    fn = tool(
        "measure",
        '{"type":"object","properties":{'
        '"count":{"type":"integer","minimum":0},'
        '"enabled":{"type":"boolean"}},'
        '"required":["count","enabled"],"additionalProperties":false}',
    )
    branch = schema_plan(
        structured_spec(),
        policy(fn),
        {"measure": ("count", "enabled")},
    ).tool("measure")

    assert branch.representable is True
    assert branch.guarantee is GenerationGuarantee.SCHEMA
    assert all(argument.generated for argument in branch.arguments)


def test_same_inputs_have_stable_plan_fingerprint_and_off_mode_cannot_claim_activation() -> None:
    fn = tool(
        "read",
        '{"type":"object","properties":{"path":{"type":"string"}},'
        '"required":["path"],"additionalProperties":false}',
    )
    spec = raw_spec()
    first = schema_plan(spec, policy(fn), {"read": ("path",)})
    second = schema_plan(spec, policy(fn), {"read": ("path",)})
    assert first == second
    assert first.fingerprint == second.fingerprint

    with pytest.raises(ToolWireCompileError, match="OFF Tool-wire plans"):
        compile_tool_wire_plan(
            spec,
            policy(fn),
            ToolConstraintMode.OFF,
            compiler_capabilities=raw_compiler_capabilities(),
            presentation_orders={"read": ("path",)},
            budget=budget(),
            parser_branch_id="off",
            constraint_fingerprint="must-not-be-here",
        )
