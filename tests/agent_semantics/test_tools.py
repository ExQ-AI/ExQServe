from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy


def _tool(name: str = "bash") -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"Tool {name}",
        parameters=JsonSchema('{"type":"object"}'),
    )


def test_function_tool_has_only_v1_function_fields() -> None:
    tool = _tool()

    assert [field.name for field in fields(tool)] == ["name", "description", "parameters"]
    assert tool.name == "bash"


def test_function_tool_rejects_empty_name_and_invalid_fields() -> None:
    with pytest.raises(ValueError, match="name"):
        FunctionTool("   ", None, JsonSchema('{}'))

    with pytest.raises(TypeError, match="description"):
        FunctionTool("bash", 123, JsonSchema('{}'))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="parameters"):
        FunctionTool("bash", None, object())  # type: ignore[arg-type]


def test_tool_choice_modes_are_exact_v1_set() -> None:
    assert {mode.value for mode in ToolChoiceMode} == {"none", "auto", "required", "named"}


def test_named_choice_requires_non_empty_name() -> None:
    assert ToolChoice(ToolChoiceMode.NAMED, "bash").name == "bash"

    with pytest.raises(ValueError, match="name"):
        ToolChoice(ToolChoiceMode.NAMED)

    with pytest.raises(ValueError, match="name"):
        ToolChoice(ToolChoiceMode.NAMED, "   ")


def test_non_named_choices_reject_name() -> None:
    for mode in (ToolChoiceMode.NONE, ToolChoiceMode.AUTO, ToolChoiceMode.REQUIRED):
        with pytest.raises(ValueError, match="name"):
            ToolChoice(mode, "bash")


def test_tool_policy_enforces_unique_names_and_required_tools() -> None:
    bash = _tool("bash")

    with pytest.raises(ValueError, match="unique"):
        ToolPolicy(
            tools=(bash, _tool("bash")),
            choice=ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=True,
        )

    with pytest.raises(ValueError, match="requires at least one"):
        ToolPolicy(
            tools=(),
            choice=ToolChoice(ToolChoiceMode.REQUIRED),
            allow_parallel=False,
        )


def test_named_choice_must_reference_declared_tool() -> None:
    with pytest.raises(ValueError, match="declared"):
        ToolPolicy(
            tools=(_tool("bash"),),
            choice=ToolChoice(ToolChoiceMode.NAMED, "read_file"),
            allow_parallel=False,
        )


def test_auto_with_zero_tools_is_valid_and_parallel_is_explicit() -> None:
    policy = ToolPolicy(
        tools=(),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=False,
    )

    assert policy.tools == ()
    assert policy.allow_parallel is False

    with pytest.raises(TypeError, match="allow_parallel"):
        ToolPolicy(
            tools=(),
            choice=ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=1,  # type: ignore[arg-type]
        )


def test_tool_policy_requires_immutable_tuple_tools() -> None:
    with pytest.raises(TypeError, match="tools"):
        ToolPolicy(
            tools=[_tool()],  # type: ignore[arg-type]
            choice=ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=True,
        )


def test_tool_policy_values_are_immutable() -> None:
    policy = ToolPolicy(
        tools=(_tool(),),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=True,
    )

    with pytest.raises(FrozenInstanceError):
        policy.allow_parallel = False  # type: ignore[misc]
