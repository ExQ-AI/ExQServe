from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode
from tests.tool_wire._legacy_api import (
    ArgumentFramingSelector,
    ArgumentFramingSelectorRule,
    ArgumentFramingVariant,
    ConstraintCompilerCapabilities,
    NameCodec,
    NamedTerminal,
    PlanCompileDisposition,
    SchemaSemanticAuthority,
    WireArgumentOccurrence,
    WireToolCall,
    WireToolSequence,
    admit_tool_sequence,
    compile_tool_wire_plan,
)
from tests.tool_wire._support import (
    budget,
    policy,
    raw_compiler_capabilities,
    raw_spec,
    schema_plan,
    structured_compiler_capabilities,
    structured_spec,
    tool,
)


def _semantic_capabilities(*extra_property_keywords: str) -> ConstraintCompilerCapabilities:
    base = structured_compiler_capabilities()
    return ConstraintCompilerCapabilities(
        "v13-exact-schema-authority",
        base.supported_object_keywords,
        (*base.supported_property_keywords, *extra_property_keywords),
        SchemaSemanticAuthority.DRAFT_2020_12,
    )


def _schema_plan_for_property(property_schema_json: str, *, strict: bool = True):
    fn = tool(
        "check",
        '{"type":"object","properties":{"value":'
        + property_schema_json
        + '},"required":["value"],"additionalProperties":false}',
        strict=strict,
    )
    spec = structured_spec()
    tool_policy = policy(fn)
    plan = schema_plan(spec, tool_policy, {"check": ("value",)})
    return spec, tool_policy, plan


def test_exact_nonemptiness_rejects_object_const_required_contradiction() -> None:
    _, _, plan = _schema_plan_for_property(
        '{"type":"object","const":{},"properties":{"x":{"type":"integer"}},"required":["x"]}'
    )
    argument = plan.tool("check").arguments[0]
    assert argument.generated is False
    assert argument.guarantee is GenerationGuarantee.NONE
    assert plan.disposition is PlanCompileDisposition.REJECTED
    assert plan.activation is None


def test_exact_nonemptiness_rejects_array_const_items_contradiction() -> None:
    _, _, plan = _schema_plan_for_property(
        '{"type":"array","const":[1],"items":{"type":"string"}}'
    )
    argument = plan.tool("check").arguments[0]
    assert argument.generated is False
    assert argument.guarantee is GenerationGuarantee.NONE
    assert plan.disposition is PlanCompileDisposition.REJECTED


def test_nonfinite_contradictory_bounds_never_fabricate_nonempty_authority() -> None:
    controls = (
        (
            '{"type":"string","minLength":3,"maxLength":1}',
            ("minLength", "maxLength"),
        ),
        (
            '{"type":"array","minItems":2,"maxItems":1,"items":{"type":"integer"}}',
            ("minItems", "maxItems"),
        ),
        (
            '{"type":"number","exclusiveMinimum":2,"exclusiveMaximum":1}',
            ("exclusiveMinimum", "exclusiveMaximum"),
        ),
    )
    for schema_json, extra_keywords in controls:
        fn = tool(
            "check",
            '{"type":"object","properties":{"value":'
            + schema_json
            + '},"required":["value"],"additionalProperties":false}',
            strict=True,
        )
        spec = structured_spec()
        plan = compile_tool_wire_plan(
            spec,
            policy(fn),
            ToolConstraintMode.SCHEMA,
            compiler_capabilities=_semantic_capabilities(*extra_keywords),
            presentation_orders={"check": ("value",)},
            budget=budget(),
            constraint_fingerprint="constraint-v13",
            activation_trigger_ids=("tool-open",),
        )
        argument = plan.tool("check").arguments[0]
        assert argument.generated is False
        assert argument.guarantee is GenerationGuarantee.NONE
        assert plan.disposition is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_exact_finite_witness_and_satisfiable_control() -> None:
    _, _, plan = _schema_plan_for_property('{"type":"integer","const":2,"minimum":0}')
    argument = plan.tool("check").arguments[0]
    assert argument.generated is True
    assert argument.guarantee is GenerationGuarantee.SCHEMA
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_root_object_schema_also_requires_exact_nonempty_authority() -> None:
    fn = tool(
        "root",
        '{"type":"object","const":{},"properties":{"value":{"type":"integer"}},'
        '"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = structured_spec()
    base = structured_compiler_capabilities()
    capabilities = replace(
        base,
        supported_object_keywords=(*base.supported_object_keywords, "const"),
    )
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=capabilities,
        presentation_orders={"root": ("value",)},
        budget=budget(),
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    assert plan.tool("root").representable is False
    assert plan.tool("root").guarantee is not GenerationGuarantee.SCHEMA
    assert plan.disposition is PlanCompileDisposition.REJECTED
    assert plan.activation is None


def test_schema_capability_without_exact_semantic_authority_cannot_claim_schema() -> None:
    fn = tool(
        "check",
        '{"type":"object","properties":{"value":{"type":"integer","const":1}},'
        '"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = structured_spec()
    base = structured_compiler_capabilities()
    no_authority = replace(base, schema_semantic_authority=SchemaSemanticAuthority.NONE)
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=no_authority,
        presentation_orders={"check": ("value",)},
        budget=budget(),
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    argument = plan.tool("check").arguments[0]
    assert argument.generated is False
    assert argument.guarantee is GenerationGuarantee.NONE
    assert plan.disposition is PlanCompileDisposition.REJECTED


def test_raw_finite_values_are_validated_against_additional_string_semantics() -> None:
    fn = tool(
        "choose",
        '{"type":"object","properties":{"value":{"type":"string","enum":["a","safe"],'
        '"minLength":2}},"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = raw_spec()
    base = raw_compiler_capabilities()
    capabilities = replace(
        base,
        supported_property_keywords=(*base.supported_property_keywords, "minLength"),
    )
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=capabilities,
        presentation_orders={"choose": ("value",)},
        budget=budget(),
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    argument = plan.tool("choose").arguments[0]
    assert argument.admitted_wire_payloads == ("safe",)
    assert argument.guarantee is GenerationGuarantee.SCHEMA
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE

    empty_fn = tool(
        "choose",
        '{"type":"object","properties":{"value":{"type":"string","enum":["a"],'
        '"minLength":2}},"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    empty_plan = compile_tool_wire_plan(
        spec,
        policy(empty_fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=capabilities,
        presentation_orders={"choose": ("value",)},
        budget=budget(),
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    empty_argument = empty_plan.tool("choose").arguments[0]
    assert empty_argument.generated is False
    assert empty_argument.guarantee is GenerationGuarantee.NONE


def test_structured_plan_admission_uses_exact_schema_membership() -> None:
    spec, _, plan = _schema_plan_for_property('{"type":"integer","minimum":0}')
    invalid = WireToolSequence(
        (WireToolCall("check", 0, (WireArgumentOccurrence("value", "-1"),)),)
    )
    valid = WireToolSequence(
        (WireToolCall("check", 0, (WireArgumentOccurrence("value", "0"),)),)
    )

    invalid_admission = admit_tool_sequence(spec, plan, invalid)
    assert "structured_value_not_admitted_by_plan" in {
        issue.code for issue in invalid_admission.issues
    }
    assert admit_tool_sequence(spec, plan, valid).is_valid


def test_nested_structured_plan_membership_rejects_invalid_child() -> None:
    fn = tool(
        "check",
        '{"type":"object","properties":{"value":{"type":"object","properties":{'
        '"count":{"type":"integer","minimum":0}},"required":["count"],'
        '"additionalProperties":false}},"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = structured_spec()
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=_semantic_capabilities("additionalProperties"),
        presentation_orders={"check": ("value",)},
        budget=budget(),
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    invalid = WireToolSequence(
        (
            WireToolCall(
                "check",
                0,
                (WireArgumentOccurrence("value", '{"count":-1}'),),
            ),
        )
    )
    valid = WireToolSequence(
        (
            WireToolCall(
                "check",
                0,
                (WireArgumentOccurrence("value", '{"count":0}'),),
            ),
        )
    )
    assert "structured_value_not_admitted_by_plan" in {
        issue.code for issue in admit_tool_sequence(spec, plan, invalid).issues
    }
    assert admit_tool_sequence(spec, plan, valid).is_valid


def test_name_codec_is_bound_to_actual_named_terminal_boundaries() -> None:
    base = raw_spec()
    with pytest.raises(ValueError, match="function_name_codec"):
        replace(base, function_name_codec=NameCodec("permissive"))
    with pytest.raises(ValueError, match="argument_name_codec"):
        replace(base, argument_name_codec=NameCodec("permissive"))

    first = base.argument_framings[0]
    second = ArgumentFramingVariant(
        "raw-square",
        NamedTerminal("<parameter=", "]"),
        first.argument_close,
        first.value_framing,
    )
    selector = ArgumentFramingSelector(
        "two-boundaries",
        (
            ArgumentFramingSelectorRule(first.variant_id, ("string",)),
            ArgumentFramingSelectorRule(second.variant_id, ("integer",)),
        ),
    )
    with pytest.raises(ValueError, match="unprotected variant: raw-square"):
        replace(base, argument_framings=(first, second), framing_selector=selector)

    assert base.function_name_codec.decode(base.function_name_codec.encode("valid")) == "valid"
    assert base.argument_name_codec.decode(base.argument_name_codec.encode("valid")) == "valid"


def test_qwen_c1f1e3c_counterexamples_are_frozen_future_a1_a2_fixtures() -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "qwen_c1f1e3c_a1a2_regressions.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "c1f1e3cc9988291a21e8e04904b780801fc567af"
    assert payload["production_patch_authorized"] is False
    assert {case["id"] for case in payload["cases"]} == {
        "unique_valid_late_tool_hidden_by_lexical_pruning",
        "early_tool_commit_before_later_competing_valid_interpretation_resolves",
    }
    assert all(case["required_phases"] == ["A1", "A2"] for case in payload["cases"])
