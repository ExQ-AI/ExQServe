from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.agent.validation import (
    ValidationCode,
    ValidationIssue,
    ValidationResult,
    validate_tool_calls,
    validate_tool_calls_with_canonical_arguments,
)
from exqserve.core.items import ToolCallItem


def _tool(name: str, schema: str = '{"type":"object"}') -> FunctionTool:
    return FunctionTool(name, None, JsonSchema(schema))


def _policy(
    *tools: FunctionTool,
    mode: ToolChoiceMode = ToolChoiceMode.AUTO,
    name: str | None = None,
    allow_parallel: bool = True,
) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(mode, name), allow_parallel)


def _call(index: int, name: str = "bash", call_id: str | None = None, args: str = "{}") -> ToolCallItem:
    return ToolCallItem(call_id or f"call-{index}", name, args, index)


def test_validation_result_contract_is_immutable_and_derived() -> None:
    valid = ValidationResult(())
    issue = ValidationIssue(ValidationCode.UNDECLARED_TOOL, "missing", ("calls", 0, "name"))
    invalid = ValidationResult((issue,))

    assert valid.is_valid is True
    assert invalid.is_valid is False
    assert invalid.issues == (issue,)

    with pytest.raises(FrozenInstanceError):
        issue.message = "changed"  # type: ignore[misc]


def test_detailed_tool_validation_returns_canonical_arguments_from_same_parse() -> None:
    tool = _tool(
        "lookup",
        '{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},'
        '"required":["a","b"],"additionalProperties":false}',
    )
    detailed = validate_tool_calls_with_canonical_arguments(
        (_call(0, "lookup", args='{ "b" : 2, "a" : 1 }'),),
        _policy(tool),
    )

    assert detailed.result.is_valid
    assert detailed.canonical_arguments == ('{"a":1,"b":2}',)


def test_tool_validation_reports_numeric_overflow_as_invalid_json() -> None:
    tool = _tool(
        "lookup",
        '{"type":"object","properties":{"x":{"type":"number"}},"required":["x"]}',
    )
    detailed = validate_tool_calls_with_canonical_arguments(
        (_call(0, "lookup", args='{"x":1e999}'),),
        _policy(tool),
    )

    assert detailed.result.is_valid is False
    assert [issue.code for issue in detailed.result.issues] == [ValidationCode.INVALID_JSON]
    assert "non-finite JSON number" in detailed.result.issues[0].message
    assert detailed.canonical_arguments == (None,)


def test_tool_validation_reports_host_integer_limit_as_invalid_json() -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Python integer digit safety limit is disabled")

    huge_integer = "9" * (limit + 1)
    tool = _tool(
        "lookup",
        '{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}',
    )
    detailed = validate_tool_calls_with_canonical_arguments(
        (_call(0, "lookup", args=f'{{"x":{huge_integer}}}'),),
        _policy(tool),
    )

    assert detailed.result.is_valid is False
    assert [issue.code for issue in detailed.result.issues] == [ValidationCode.INVALID_JSON]
    assert "supported digit limit" in detailed.result.issues[0].message
    assert detailed.canonical_arguments == (None,)


def test_tool_validation_reports_decoder_depth_as_invalid_json() -> None:
    nested = "[" * 10_000 + "0" + "]" * 10_000
    detailed = validate_tool_calls_with_canonical_arguments(
        (_call(0, "lookup", args=f'{{"x":{nested}}}'),),
        _policy(_tool("lookup")),
    )

    assert detailed.result.is_valid is False
    assert [issue.code for issue in detailed.result.issues] == [ValidationCode.INVALID_JSON]
    assert "decoder depth" in detailed.result.issues[0].message
    assert detailed.canonical_arguments == (None,)


def test_validation_codes_cover_v1_agent_semantics() -> None:
    assert {code.value for code in ValidationCode} == {
        "invalid_json",
        "duplicate_json_key",
        "json_value_not_object",
        "schema_validation_failed",
        "undeclared_tool",
        "tool_call_forbidden",
        "tool_call_required",
        "wrong_named_tool",
        "parallel_tool_calls_forbidden",
        "duplicate_tool_call_id",
        "invalid_tool_call_order",
        "unknown_tool_result",
        "duplicate_tool_result",
    }


def test_none_policy_rejects_any_call() -> None:
    result = validate_tool_calls((_call(0),), _policy(_tool("bash"), mode=ToolChoiceMode.NONE))

    assert [issue.code for issue in result.issues] == [ValidationCode.TOOL_CALL_FORBIDDEN]
    assert result.issues[0].path == ("calls",)


def test_required_and_named_policy_require_a_call() -> None:
    required = validate_tool_calls((), _policy(_tool("bash"), mode=ToolChoiceMode.REQUIRED))
    named = validate_tool_calls(
        (), _policy(_tool("bash"), mode=ToolChoiceMode.NAMED, name="bash")
    )

    assert [issue.code for issue in required.issues] == [ValidationCode.TOOL_CALL_REQUIRED]
    assert [issue.code for issue in named.issues] == [ValidationCode.TOOL_CALL_REQUIRED]


def test_named_policy_accepts_named_tool_and_rejects_other_declared_tool() -> None:
    policy = _policy(
        _tool("bash"),
        _tool("read_file"),
        mode=ToolChoiceMode.NAMED,
        name="bash",
    )

    assert validate_tool_calls((_call(0, "bash"),), policy).is_valid
    wrong = validate_tool_calls((_call(0, "read_file"),), policy)
    assert [issue.code for issue in wrong.issues] == [ValidationCode.WRONG_NAMED_TOOL]
    assert wrong.issues[0].path == ("calls", 0, "name")


def test_declaration_check_rejects_unknown_tool() -> None:
    result = validate_tool_calls((_call(0, "unknown"),), _policy(_tool("bash")))

    assert [issue.code for issue in result.issues] == [ValidationCode.UNDECLARED_TOOL]


def test_parallel_policy_is_turn_level() -> None:
    calls = (_call(0), _call(1))

    assert validate_tool_calls(calls, _policy(_tool("bash"), allow_parallel=True)).is_valid
    forbidden = validate_tool_calls(calls, _policy(_tool("bash"), allow_parallel=False))
    assert [issue.code for issue in forbidden.issues] == [
        ValidationCode.PARALLEL_TOOL_CALLS_FORBIDDEN
    ]


def test_duplicate_call_ids_are_reported_at_later_position() -> None:
    calls = (_call(0, call_id="same"), _call(1, call_id="same"))

    result = validate_tool_calls(calls, _policy(_tool("bash")))

    assert [issue.code for issue in result.issues] == [ValidationCode.DUPLICATE_TOOL_CALL_ID]
    assert result.issues[0].path == ("calls", 1, "call_id")


def test_indices_must_match_exact_zero_based_tuple_order() -> None:
    calls = (
        ToolCallItem("call-a", "bash", "{}", 1),
        ToolCallItem("call-b", "bash", "{}", 0),
    )

    result = validate_tool_calls(calls, _policy(_tool("bash")))

    assert [issue.code for issue in result.issues] == [
        ValidationCode.INVALID_TOOL_CALL_ORDER,
        ValidationCode.INVALID_TOOL_CALL_ORDER,
    ]
    assert [issue.path for issue in result.issues] == [
        ("calls", 0, "index"),
        ("calls", 1, "index"),
    ]


def test_policy_wide_issues_precede_per_call_issues_deterministically() -> None:
    calls = (
        ToolCallItem("same", "unknown", "{}", 1),
        ToolCallItem("same", "unknown", "{}", 0),
    )
    policy = _policy(_tool("bash"), allow_parallel=False)

    first = validate_tool_calls(calls, policy)
    second = validate_tool_calls(calls, policy)

    assert first == second
    assert [issue.code for issue in first.issues] == [
        ValidationCode.PARALLEL_TOOL_CALLS_FORBIDDEN,
        ValidationCode.INVALID_TOOL_CALL_ORDER,
        ValidationCode.UNDECLARED_TOOL,
        ValidationCode.DUPLICATE_TOOL_CALL_ID,
        ValidationCode.INVALID_TOOL_CALL_ORDER,
        ValidationCode.UNDECLARED_TOOL,
    ]


def test_malformed_arguments_json_is_reported_without_exception_leakage() -> None:
    call = _call(0, args='{"x":')

    result = validate_tool_calls((call,), _policy(_tool("bash")))

    assert [issue.code for issue in result.issues] == [ValidationCode.INVALID_JSON]
    assert result.issues[0].path == ("calls", 0, "arguments")
    assert call.arguments_json == '{"x":'


def test_duplicate_argument_keys_have_distinct_validation_code() -> None:
    result = validate_tool_calls(
        (_call(0, args='{"x":1,"x":2}'),),
        _policy(_tool("bash")),
    )

    assert [issue.code for issue in result.issues] == [ValidationCode.DUPLICATE_JSON_KEY]


def test_tool_arguments_must_be_json_object() -> None:
    for args in ('[1,2]', '"text"', '42', 'null'):
        result = validate_tool_calls((_call(0, args=args),), _policy(_tool("bash")))
        assert [issue.code for issue in result.issues] == [ValidationCode.JSON_VALUE_NOT_OBJECT]


def test_nested_schema_validation_reports_semantic_argument_path() -> None:
    schema = """{
      "type":"object",
      "properties":{
        "user":{
          "type":"object",
          "properties":{"age":{"type":"integer","minimum":18}},
          "required":["age"]
        }
      },
      "required":["user"]
    }"""
    policy = _policy(_tool("create_user", schema))
    call = _call(0, "create_user", args='{"user":{"age":12}}')

    result = validate_tool_calls((call,), policy)

    assert [issue.code for issue in result.issues] == [ValidationCode.SCHEMA_VALIDATION_FAILED]
    assert result.issues[0].path == ("calls", 0, "arguments", "user", "age")


def test_unicode_arguments_validate_without_mutation() -> None:
    schema = '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}'
    call = _call(0, "save", args='{"name":"中文-β"}')

    result = validate_tool_calls((call,), _policy(_tool("save", schema)))

    assert result.is_valid
    assert call.arguments_json == '{"name":"中文-β"}'


def test_undeclared_tool_does_not_get_cascading_json_or_schema_errors() -> None:
    result = validate_tool_calls(
        (_call(0, "unknown", args='{"broken":'),),
        _policy(_tool("bash", '{"type":"object","required":["x"]}')),
    )

    assert [issue.code for issue in result.issues] == [ValidationCode.UNDECLARED_TOOL]


def test_schema_issue_order_is_deterministic_across_repeated_validation() -> None:
    schema = """{
      "type":"object",
      "properties":{
        "a":{"type":"integer"},
        "b":{"type":"string"}
      },
      "required":["a","b"]
    }"""
    call = _call(0, "check", args='{"a":"wrong","b":3}')
    policy = _policy(_tool("check", schema))

    first = validate_tool_calls((call,), policy)
    second = validate_tool_calls((call,), policy)

    assert first == second
    assert [issue.code for issue in first.issues] == [
        ValidationCode.SCHEMA_VALIDATION_FAILED,
        ValidationCode.SCHEMA_VALIDATION_FAILED,
    ]
    assert [issue.path for issue in first.issues] == [
        ("calls", 0, "arguments", "a"),
        ("calls", 0, "arguments", "b"),
    ]
