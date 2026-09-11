from __future__ import annotations

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.model import tool_constraints
from exqserve.model.contracts import (
    ToolConstraintGuarantee,
    ToolConstraintMode,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
)
from exqserve.model.gemma4 import gemma4_tool_constraint
from exqserve.model.tool_constraints import (
    qwen_parameter_schema,
    qwen_property_schema,
    qwen_property_schema_type,
)
from exqserve.tool_wire.controls.qwen import (
    qwen_production_tool_constraint as qwen_tool_constraint,
)


def _tool(name: str, schema: str, *, strict: bool = False) -> FunctionTool:
    return FunctionTool(name, None, JsonSchema(schema), strict)


def _policy(
    *tools: FunctionTool,
    mode: ToolChoiceMode = ToolChoiceMode.AUTO,
    name: str | None = None,
    parallel: bool = False,
) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(mode, name), parallel)


def _schema() -> str:
    return (
        '{"type":"object","properties":{'
        '"count":{"type":"integer","minimum":1},'
        '"mode":{"type":"string","enum":["fast","safe"],"pattern":"^[a-z]+$"},'
        '"tags":{"type":"array","items":{"type":"string"}}'
        '},"required":["count","mode"],"additionalProperties":false}'
    )


def _qwen_schema() -> str:
    return (
        '{"type":"object","properties":{'
        '"command":{"type":"string"},'
        '"count":{"type":"integer","minimum":1},'
        '"enabled":{"type":"boolean"},'
        '"mode":{"type":"string","enum":["fast","safe"]}'
        '},"required":["command","count","mode"],"additionalProperties":false}'
    )


def _qwen_sound_schema() -> str:
    return (
        '{"type":"object","properties":{'
        '"command":{"type":"string","enum":["run","check"]},'
        '"count":{"type":"integer","minimum":1},'
        '"enabled":{"type":"boolean"},'
        '"fixed":{"type":"string","const":"stable"},'
        '"mode":{"type":"string","enum":["fast","safe"]}'
        '},"required":["command","count","fixed","mode"],"additionalProperties":false}'
    )


def test_constraint_modes_are_stable() -> None:
    assert [mode.value for mode in ToolConstraintMode] == ["off", "format", "schema"]
    assert [guarantee.value for guarantee in ToolConstraintGuarantee] == [
        "none",
        "format",
        "schema",
        "unknown",
    ]


def test_qwen_schema_constraint_uses_native_parameter_envelope_when_sound() -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("save", _qwen_sound_schema()), parallel=False),
        ToolConstraintMode.SCHEMA,
    )

    assert constraint is not None
    assert constraint.trigger == "<tool_call>"
    assert constraint.eos_after_completed is True
    assert '"<function=save>"' in constraint.lark_grammar
    assert '"<parameter=command>\\n"' in constraint.lark_grammar
    assert '"<parameter=count>\\n"' in constraint.lark_grammar
    assert '"<parameter=mode>\\n"' in constraint.lark_grammar
    assert '"minimum":1' in constraint.lark_grammar
    assert 'qwen_raw_string[suffix="</parameter>"]' not in constraint.lark_grammar
    assert '"check\\n</parameter>\\n"' in constraint.lark_grammar
    assert '"run\\n</parameter>\\n"' in constraint.lark_grammar
    assert '"fast\\n</parameter>\\n"' in constraint.lark_grammar
    assert '"safe\\n</parameter>\\n"' in constraint.lark_grammar
    assert '"stable\\n</parameter>\\n"' in constraint.lark_grammar
    assert "%json" in constraint.lark_grammar
    assert "function_0_argument_" in constraint.lark_grammar
    assert 'start: WS? function WS? "</tool_call>"' in constraint.lark_grammar
    assert "<tool_call> WS? function" not in constraint.lark_grammar
    assert constraint.guarantee_for_tool("save") is ToolConstraintGuarantee.SCHEMA


def test_qwen_constraint_bounds_structural_whitespace() -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("save", _qwen_sound_schema()), parallel=False),
        ToolConstraintMode.SCHEMA,
    )

    assert constraint is not None
    assert "WS: /[ \\t\\r\\n]{1,8}/" in constraint.lark_grammar
    assert "WS: /[ \\t\\r\\n]+/" not in constraint.lark_grammar


def test_qwen_schema_constraint_supports_prevention_first_raw_strings() -> None:
    write_schema = (
        '{"type":"object","properties":{'
        '"content":{"type":"string"},"file_path":{"type":"string"}'
        '},"required":["content","file_path"],"additionalProperties":false}'
    )

    constraint = qwen_tool_constraint(
        _policy(_tool("write", write_schema), parallel=False),
        ToolConstraintMode.SCHEMA,
    )
    assert constraint is not None
    assert constraint.guarantee_for_tool("write") is ToolConstraintGuarantee.SCHEMA
    assert '"<parameter=content>\\n"' in constraint.lark_grammar
    assert '"<parameter=file_path>\\n"' in constraint.lark_grammar


def test_qwen_strict_raw_string_schema_uses_tool_wire_schema_guarantee() -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("write", _qwen_schema(), strict=True), parallel=False),
        ToolConstraintMode.SCHEMA,
    )
    assert constraint is not None
    assert constraint.guarantee_for_tool("write") is ToolConstraintGuarantee.SCHEMA


def test_qwen_mixed_sound_and_raw_string_tools_share_tool_wire_schema_authority() -> None:
    safe_schema = (
        '{"type":"object","properties":{"count":{"type":"integer"}},'
        '"required":["count"],"additionalProperties":false}'
    )
    constraint = qwen_tool_constraint(
        _policy(
            _tool("safe", safe_schema),
            _tool("write", _qwen_schema()),
            parallel=True,
        ),
        ToolConstraintMode.SCHEMA,
    )
    assert constraint is not None
    assert constraint.guarantee_for_tool("safe") is ToolConstraintGuarantee.SCHEMA
    assert constraint.guarantee_for_tool("write") is ToolConstraintGuarantee.SCHEMA


def test_qwen_schema_mode_validation_only_fallback_for_unsupported_non_strict_branch() -> None:
    schema = (
        '{"type":"object","properties":{'
        '"command":{"type":"string","pattern":"^git .+$"}'
        '},"required":["command"]}'
    )

    constraint = qwen_tool_constraint(_policy(_tool("run", schema)), ToolConstraintMode.SCHEMA)
    assert constraint is not None
    assert constraint.guarantee_for_tool("run") is ToolConstraintGuarantee.FORMAT
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(
            _policy(_tool("run", schema, strict=True)),
            ToolConstraintMode.SCHEMA,
        )


def test_qwen_format_constraint_limits_tool_name_but_not_parameter_schema() -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("lookup", _schema())),
        ToolConstraintMode.FORMAT,
    )

    assert constraint is not None
    assert constraint.eos_after_completed is True
    assert '"<function=lookup>"' in constraint.lark_grammar
    assert '"<parameter=count>\\n"' in constraint.lark_grammar
    assert '"<parameter=mode>\\n"' in constraint.lark_grammar
    assert '"<parameter=tags>\\n"' in constraint.lark_grammar
    assert 'start: WS? function WS? "</tool_call>"' in constraint.lark_grammar
    assert constraint.guarantee_for_tool("lookup") is ToolConstraintGuarantee.FORMAT


@pytest.mark.parametrize("mode", (ToolConstraintMode.FORMAT, ToolConstraintMode.SCHEMA))
def test_qwen_constrained_parallel_restores_native_one_to_many_grammar(
    mode: ToolConstraintMode,
) -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("save", _qwen_sound_schema()), parallel=True),
        mode,
    )

    assert constraint is not None
    assert (
        'start: WS? function WS? "</tool_call>" '
        '(WS? "<tool_call>" WS? function WS? "</tool_call>"){0,7}'
    ) in constraint.lark_grammar

    strict_constraint = qwen_tool_constraint(
        _policy(_tool("save", _qwen_sound_schema(), strict=True), parallel=True),
        ToolConstraintMode.OFF,
    )
    assert strict_constraint is not None
    assert '"<parameter=count>\\n"' in strict_constraint.lark_grammar

    assert (
        qwen_tool_constraint(
            _policy(_tool("save", _qwen_schema()), parallel=True),
            ToolConstraintMode.OFF,
        )
        is None
    )


def test_qwen_strict_tool_escalates_off_baseline_to_schema() -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("save", _qwen_sound_schema(), strict=True), parallel=False),
        ToolConstraintMode.OFF,
    )

    assert constraint is not None
    assert '"<parameter=count>\\n"' in constraint.lark_grammar
    assert '"minimum":1' in constraint.lark_grammar
    assert 'parameter: "<parameter=" NAME ">" value "</parameter>" WS?' not in constraint.lark_grammar


def test_qwen_mixed_strict_and_non_strict_tools_keep_distinct_branches() -> None:
    strict_schema = (
        '{"type":"object","properties":{"strict_value":{"type":"integer"}},'
        '"required":["strict_value"],"additionalProperties":false}'
    )
    loose_schema = (
        '{"type":"object","properties":{"loose_value":{"type":"integer"}},'
        '"required":["loose_value"],"additionalProperties":false}'
    )
    constraint = qwen_tool_constraint(
        _policy(
            _tool("strict_tool", strict_schema, strict=True),
            _tool("loose_tool", loose_schema),
            parallel=False,
        ),
        ToolConstraintMode.OFF,
    )

    assert constraint is not None
    assert '"<function=strict_tool>"' in constraint.lark_grammar
    assert '"<parameter=strict_value>\\n"' in constraint.lark_grammar
    assert '"<function=loose_tool>"' in constraint.lark_grammar
    assert '"<parameter=loose_value>\\n"' in constraint.lark_grammar
    assert constraint.guarantee_for_tool("strict_tool") is ToolConstraintGuarantee.SCHEMA
    assert constraint.guarantee_for_tool("loose_tool") is ToolConstraintGuarantee.FORMAT


def test_legacy_tool_generation_constraint_defaults_to_unknown_branch_metadata() -> None:
    constraint = ToolGenerationConstraint("<tool>", 'start: "ok"', True)

    assert constraint.branch_guarantees is None
    assert constraint.guarantee_for_tool("lookup") is ToolConstraintGuarantee.UNKNOWN


def test_qwen_named_non_strict_tool_ignores_hidden_strict_tools_when_baseline_off() -> None:
    strict_tool = _tool("strict_tool", _qwen_schema(), strict=True)
    loose_tool = _tool("loose_tool", _qwen_schema())
    policy = _policy(
        strict_tool,
        loose_tool,
        mode=ToolChoiceMode.NAMED,
        name="loose_tool",
        parallel=False,
    )

    assert qwen_tool_constraint(policy, ToolConstraintMode.OFF) is None


def test_gemma_strict_tool_escalates_off_baseline_but_rejects_parallel() -> None:
    tool = _tool("save", _schema(), strict=True)
    constraint = gemma4_tool_constraint(
        _policy(tool, parallel=False),
        ToolConstraintMode.OFF,
    )

    assert constraint is not None
    assert '"count"' in constraint.lark_grammar
    assert constraint.eos_after_completed is True
    assert constraint.guarantee_for_tool("save") is ToolConstraintGuarantee.SCHEMA

    with pytest.raises(ToolConstraintUnsupported, match="parallel"):
        gemma4_tool_constraint(_policy(tool, parallel=True), ToolConstraintMode.OFF)


def test_gemma_schema_constraint_uses_complete_standard_json_schema() -> None:
    constraint = gemma4_tool_constraint(
        _policy(_tool("save", _schema()), parallel=False),
        ToolConstraintMode.SCHEMA,
    )

    assert constraint is not None
    assert constraint.trigger == "<|tool_call>"
    assert constraint.eos_after_completed is True
    assert '"call:save"' in constraint.lark_grammar
    assert "%json" in constraint.lark_grammar
    assert "<tool_call|>" in constraint.lark_grammar
    assert '"<tool_call|>"' not in constraint.lark_grammar
    assert '"enum":["fast","safe"]' in constraint.lark_grammar
    assert '"minimum":1' in constraint.lark_grammar
    assert '"pattern":"^[a-z]+$"' in constraint.lark_grammar


def test_gemma_format_constraint_requires_json_object_only() -> None:
    constraint = gemma4_tool_constraint(
        _policy(_tool("save", _schema())),
        ToolConstraintMode.FORMAT,
    )

    assert constraint is not None
    assert '%json {"type":"object"}' in constraint.lark_grammar
    assert '"count"' not in constraint.lark_grammar


def test_gemma_constraint_has_no_unbounded_whitespace_path_before_close() -> None:
    constraint = gemma4_tool_constraint(
        _policy(_tool("save", _schema()), parallel=False),
        ToolConstraintMode.SCHEMA,
    )

    assert constraint is not None
    assert "start: tool" in constraint.lark_grammar
    assert "WS" not in constraint.lark_grammar
    assert '%json ' in constraint.lark_grammar
    assert " <tool_call|>" in constraint.lark_grammar


def test_named_choice_exposes_only_selected_tool() -> None:
    first = _tool("first", _schema())
    second = _tool("second", _schema())
    policy = _policy(first, second, mode=ToolChoiceMode.NAMED, name="second")

    qwen = qwen_tool_constraint(policy, ToolConstraintMode.FORMAT)
    gemma = gemma4_tool_constraint(policy, ToolConstraintMode.FORMAT)

    assert qwen is not None and gemma is not None
    assert "first" not in qwen.lark_grammar
    assert "first" not in gemma.lark_grammar
    assert "second" in qwen.lark_grammar
    assert "second" in gemma.lark_grammar


def test_none_choice_and_off_mode_do_not_create_filter() -> None:
    tool = _tool("save", _schema())
    none_policy = _policy(tool, mode=ToolChoiceMode.NONE)
    auto_policy = _policy(tool)

    assert qwen_tool_constraint(none_policy, ToolConstraintMode.SCHEMA) is None
    assert gemma4_tool_constraint(none_policy, ToolConstraintMode.SCHEMA) is None
    assert qwen_tool_constraint(auto_policy, ToolConstraintMode.OFF) is None
    assert gemma4_tool_constraint(auto_policy, ToolConstraintMode.OFF) is None


def test_qwen_schema_mode_supports_simple_root_defs_for_property_refs() -> None:
    schema = (
        '{"$defs":{"item":{"type":"string"}},'
        '"type":"object","properties":{"value":{"$ref":"#/$defs/item"}},'
        '"required":["value"]}'
    )
    constraint = qwen_tool_constraint(
        _policy(_tool("save", schema)),
        ToolConstraintMode.SCHEMA,
    )

    assert constraint is not None
    assert '"<parameter=value>\\n"' in constraint.lark_grammar
    assert constraint.guarantee_for_tool("save") is ToolConstraintGuarantee.SCHEMA


def test_qwen_property_schema_does_not_copy_unrelated_root_defs() -> None:
    property_schema = {"type": "integer"}
    root_schema = {
        "$defs": {f"d{index}": {"type": "integer"} for index in range(256)},
        "type": "object",
        "properties": {"value": property_schema},
    }

    resolved = qwen_property_schema(root_schema, property_schema)

    assert resolved is property_schema
    assert resolved == {"type": "integer"}
    assert "$defs" not in resolved


def test_qwen_property_schema_resolves_chained_root_defs_without_copying_scope() -> None:
    root_schema = {
        "$defs": {
            "base": {"type": "integer", "minimum": 5},
            "alias": {"$ref": "#/$defs/base"},
        }
    }

    resolved = qwen_property_schema(root_schema, {"$ref": "#/$defs/alias"})

    assert resolved == {"type": "integer", "minimum": 5}
    assert "$defs" not in resolved
    assert "$ref" not in resolved


@pytest.mark.parametrize("definitions_key", ("$defs", "definitions"))
@pytest.mark.parametrize(
    ("terminal_index", "expected_type"),
    ((63, "string"), (64, "string"), (65, None)),
)
def test_qwen_property_schema_type_matches_generation_ref_boundary(
    definitions_key: str,
    terminal_index: int,
    expected_type: str | None,
) -> None:
    definitions: dict[str, object] = {
        f"d{index}": {"$ref": f"#/{definitions_key}/d{index + 1}"}
        for index in range(terminal_index)
    }
    definitions[f"d{terminal_index}"] = {"type": "string"}
    root_schema = {definitions_key: definitions}
    property_schema = {"$ref": f"#/{definitions_key}/d0"}

    assert qwen_property_schema_type(root_schema, property_schema) == expected_type


def test_qwen_property_schema_stops_detached_expansion_at_remaining_node_limit() -> None:
    class TrackingList(list[None]):
        reads = 0

        def __iter__(self):
            for item in super().__iter__():
                self.reads += 1
                yield item

    examples = TrackingList([None] * 100)
    root_schema = {
        "$defs": {
            "large": {
                "type": "integer",
                "examples": examples,
            }
        }
    }
    context_handle = tool_constraints._QWEN_PROPERTY_RESOLUTION_NODE_LIMIT.set(12)
    try:
        with pytest.raises(tool_constraints._QwenPropertyResolutionNodeLimit):
            qwen_property_schema(root_schema, {"$ref": "#/$defs/large"})
    finally:
        tool_constraints._QWEN_PROPERTY_RESOLUTION_NODE_LIMIT.reset(context_handle)

    assert examples.reads < len(examples)
    assert examples.reads <= 12


def test_qwen_schema_mode_falls_back_for_unsupported_root_defs_target() -> None:
    schema = (
        '{"$defs":{"item":{"type":"string","pattern":"^[A-Z]+$"}},'
        '"type":"object","properties":{"value":{"$ref":"#/$defs/item"}},'
        '"required":["value"]}'
    )

    constraint = qwen_tool_constraint(_policy(_tool("save", schema)), ToolConstraintMode.SCHEMA)
    assert constraint is not None
    assert constraint.guarantee_for_tool("save") is ToolConstraintGuarantee.FORMAT


def test_qwen_schema_mode_rejects_cross_property_top_level_assertions() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{"name":{"type":"string"}},"minProperties":1}'
    )

    with pytest.raises(ToolConstraintUnsupported, match="minProperties"):
        qwen_parameter_schema(schema)


def test_qwen_schema_mode_can_omit_unframed_optional_nonlocal_property_ref() -> None:
    schema = (
        '{"type":"object","properties":{"value":{"anyOf":['
        '{"type":"string"},{"$ref":"#"}]}}}'
    )

    constraint = qwen_tool_constraint(_policy(_tool("save", schema)), ToolConstraintMode.SCHEMA)
    assert constraint is not None
    assert constraint.guarantee_for_tool("save") is ToolConstraintGuarantee.SCHEMA


def test_qwen_schema_mode_formats_loose_dynamic_ref_but_strict_fails_closed() -> None:
    schema = (
        '{"type":"object","properties":{"value":'
        '{"type":"string","$dynamicRef":"#node"}},"required":["value"]}'
    )

    loose = qwen_tool_constraint(_policy(_tool("save", schema)), ToolConstraintMode.SCHEMA)
    assert loose is not None
    assert loose.guarantee_for_tool("save") is ToolConstraintGuarantee.FORMAT
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(_policy(_tool("save", schema, strict=True)), ToolConstraintMode.SCHEMA)


def test_qwen_schema_mode_rejects_required_property_without_declared_schema() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{},"required":["missing"],"additionalProperties":true}'
    )

    with pytest.raises(ToolConstraintUnsupported, match="missing"):
        qwen_parameter_schema(schema)


def test_qwen_constraint_falls_back_or_narrows_for_unrepresentable_names() -> None:
    invalid_tool = _tool("bad name", _schema())
    optional_invalid_parameter = _tool(
        "save",
        '{"type":"object","properties":{"bad name":{"type":"string"}}}',
    )
    required_invalid_parameter = _tool(
        "save",
        '{"type":"object","properties":{"bad name":{"type":"string"}},'
        '"required":["bad name"]}',
    )

    assert qwen_tool_constraint(_policy(invalid_tool), ToolConstraintMode.FORMAT) is None
    narrowed = qwen_tool_constraint(_policy(optional_invalid_parameter), ToolConstraintMode.SCHEMA)
    assert narrowed is not None
    assert narrowed.guarantee_for_tool("save") is ToolConstraintGuarantee.SCHEMA
    assert "<parameter=bad name>" not in narrowed.lark_grammar
    assert qwen_tool_constraint(_policy(required_invalid_parameter), ToolConstraintMode.SCHEMA) is None
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(
            _policy(_tool("bad name", _schema(), strict=True)),
            ToolConstraintMode.OFF,
        )
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(
            _policy(
                _tool(
                    "save",
                    '{"type":"object","properties":{"bad name":{"type":"string"}},'
                    '"required":["bad name"]}',
                    strict=True,
                )
            ),
            ToolConstraintMode.OFF,
        )


@pytest.mark.parametrize("name", ("bad name", "bad{name", "bad}name", "bad<name", "bad>name"))
def test_gemma_constraint_rejects_every_name_the_native_parser_rejects(name: str) -> None:
    with pytest.raises(ToolConstraintUnsupported, match="tool name"):
        gemma4_tool_constraint(
            _policy(_tool(name, _schema())),
            ToolConstraintMode.FORMAT,
        )


def test_constraint_lark_grammars_compile_when_llguidance_is_installed() -> None:
    llguidance = pytest.importorskip("llguidance")
    policy = _policy(_tool("save", _schema()))
    qwen_policy = _policy(_tool("qwen_save", _qwen_sound_schema()))
    ref_policy = _policy(
        _tool(
            "ref_save",
            '{"$defs":{"item":{"type":"string"}},"type":"object",'
            '"properties":{"value":{"$ref":"#/$defs/item"}},"required":["value"]}',
        )
    )

    for constraint in (
        qwen_tool_constraint(policy, ToolConstraintMode.FORMAT),
        qwen_tool_constraint(qwen_policy, ToolConstraintMode.SCHEMA),
        qwen_tool_constraint(ref_policy, ToolConstraintMode.SCHEMA),
        gemma4_tool_constraint(policy, ToolConstraintMode.FORMAT),
        gemma4_tool_constraint(policy, ToolConstraintMode.SCHEMA),
    ):
        assert constraint is not None
        llguidance.LLMatcher.grammar_from_lark(constraint.lark_grammar)
