from __future__ import annotations

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.model.contracts import ToolConstraintMode, ToolConstraintUnsupported
from exqserve.model.gemma4 import gemma4_tool_constraint
from exqserve.model.qwen import qwen_tool_constraint
from exqserve.model.tool_constraints import qwen_parameter_schema


def _tool(name: str, schema: str) -> FunctionTool:
    return FunctionTool(name, None, JsonSchema(schema))


def _policy(
    *tools: FunctionTool,
    mode: ToolChoiceMode = ToolChoiceMode.AUTO,
    name: str | None = None,
    parallel: bool = True,
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


def test_constraint_modes_are_stable() -> None:
    assert [mode.value for mode in ToolConstraintMode] == ["off", "format", "schema"]


def test_qwen_schema_constraint_uses_native_parameter_envelope() -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("save", _schema()), parallel=False),
        ToolConstraintMode.SCHEMA,
    )

    assert constraint is not None
    assert constraint.trigger == "<tool_call>"
    assert constraint.eos_after_completed is True
    assert '"<function=save>"' in constraint.lark_grammar
    assert '"<parameter=count>"' in constraint.lark_grammar
    assert '"<parameter=mode>"' in constraint.lark_grammar
    assert '"<parameter=tags>"' in constraint.lark_grammar
    assert '"minimum":1' in constraint.lark_grammar
    assert '"pattern":"^[a-z]+$"' in constraint.lark_grammar
    assert "%json" in constraint.lark_grammar
    assert "function_0_parameter_2?" in constraint.lark_grammar
    assert '"</tool_call>"' in constraint.lark_grammar


def test_qwen_format_constraint_limits_tool_name_but_not_parameter_schema() -> None:
    constraint = qwen_tool_constraint(
        _policy(_tool("lookup", _schema())),
        ToolConstraintMode.FORMAT,
    )

    assert constraint is not None
    assert constraint.eos_after_completed is False
    assert '"<function=lookup>"' in constraint.lark_grammar
    assert 'parameter: "<parameter=" NAME ">" value "</parameter>" WS?' in constraint.lark_grammar
    assert "%json" not in constraint.lark_grammar


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
    assert '"<tool_call|>"' in constraint.lark_grammar
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


def test_qwen_schema_mode_supports_root_defs_for_property_refs() -> None:
    schema = (
        '{"$defs":{"item":{"type":"string","pattern":"^[A-Z]+$"}},'
        '"type":"object","properties":{"value":{"$ref":"#/$defs/item"}},'
        '"required":["value"]}'
    )
    constraint = qwen_tool_constraint(
        _policy(_tool("save", schema)),
        ToolConstraintMode.SCHEMA,
    )

    assert constraint is not None
    assert '"$ref":"#/$defs/item"' in constraint.lark_grammar
    assert '"$defs":{"item":{"pattern":"^[A-Z]+$","type":"string"}}' in constraint.lark_grammar


def test_qwen_schema_mode_rejects_cross_property_top_level_assertions() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{"name":{"type":"string"}},"minProperties":1}'
    )

    with pytest.raises(ToolConstraintUnsupported, match="minProperties"):
        qwen_parameter_schema(schema)


def test_qwen_schema_mode_rejects_required_property_without_declared_schema() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{},"required":["missing"],"additionalProperties":true}'
    )

    with pytest.raises(ToolConstraintUnsupported, match="missing"):
        qwen_parameter_schema(schema)


def test_qwen_constraint_rejects_names_that_native_tag_parser_cannot_accept() -> None:
    invalid_tool = _tool("bad name", _schema())
    invalid_parameter = _tool(
        "save",
        '{"type":"object","properties":{"bad name":{"type":"string"}}}',
    )

    with pytest.raises(ToolConstraintUnsupported, match="tool name"):
        qwen_tool_constraint(_policy(invalid_tool), ToolConstraintMode.FORMAT)
    with pytest.raises(ToolConstraintUnsupported, match="parameter name"):
        qwen_tool_constraint(_policy(invalid_parameter), ToolConstraintMode.SCHEMA)


def test_gemma_constraint_rejects_name_that_conflicts_with_json_body_delimiter() -> None:
    with pytest.raises(ToolConstraintUnsupported, match="tool name"):
        gemma4_tool_constraint(
            _policy(_tool("bad{name", _schema())),
            ToolConstraintMode.FORMAT,
        )


def test_constraint_lark_grammars_compile_when_llguidance_is_installed() -> None:
    llguidance = pytest.importorskip("llguidance")
    policy = _policy(_tool("save", _schema()))
    ref_policy = _policy(
        _tool(
            "ref_save",
            '{"$defs":{"item":{"type":"string"}},"type":"object",'
            '"properties":{"value":{"$ref":"#/$defs/item"}},"required":["value"]}',
        )
    )

    for constraint in (
        qwen_tool_constraint(policy, ToolConstraintMode.FORMAT),
        qwen_tool_constraint(policy, ToolConstraintMode.SCHEMA),
        qwen_tool_constraint(ref_policy, ToolConstraintMode.SCHEMA),
        gemma4_tool_constraint(policy, ToolConstraintMode.FORMAT),
        gemma4_tool_constraint(policy, ToolConstraintMode.SCHEMA),
    ):
        assert constraint is not None
        llguidance.LLMatcher.grammar_from_lark(constraint.lark_grammar)
