from __future__ import annotations

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire.controls import qwen as qwen_control


def _policy(schema: str, *, parallel: bool = True) -> ToolPolicy:
    return ToolPolicy(
        (
            FunctionTool("lookup", "lookup data", JsonSchema(schema), False),
        ),
        ToolChoice(ToolChoiceMode.AUTO),
        parallel,
    )


def test_qwen_compile_cache_reuses_only_identical_immutable_inputs() -> None:
    qwen_control._clear_qwen_tool_wire_compile_cache()
    policy = _policy(
        '{"type":"object","properties":{"x":{"type":"integer"}},'
        '"required":["x"],"additionalProperties":false}'
    )

    first = qwen_control.compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
    second = qwen_control.compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert second is first
    assert qwen_control.qwen_tool_wire_compile_cache_stats() == (1, 1, 128, 1)

    different_mode = qwen_control.compile_qwen_tool_wire(policy, ToolConstraintMode.FORMAT)
    different_parallel_limit = qwen_control.compile_qwen_tool_wire(
        policy,
        ToolConstraintMode.SCHEMA,
        max_parallel_calls=2,
    )
    different_schema = qwen_control.compile_qwen_tool_wire(
        _policy(
            '{"type":"object","properties":{"x":{"type":"string"}},'
            '"required":["x"],"additionalProperties":false}'
        ),
        ToolConstraintMode.SCHEMA,
    )

    assert different_mode is not first
    assert different_parallel_limit is not first
    assert different_schema is not first
    hits, misses, maxsize, currsize = qwen_control.qwen_tool_wire_compile_cache_stats()
    assert hits == 1
    assert misses == 4
    assert maxsize == 128
    assert currsize == 4

    qwen_control._clear_qwen_tool_wire_compile_cache()


def test_qwen_compile_cache_bypasses_large_schema_sources() -> None:
    qwen_control._clear_qwen_tool_wire_compile_cache()
    description = "x" * (qwen_control._QWEN_COMPILE_CACHE_MAX_SOURCE_CHARS + 1024)
    schema = (
        '{"type":"object","description":' + '"' + description + '"'
        + ',"properties":{"x":{"type":"integer"}},"additionalProperties":false}'
    )
    policy = _policy(schema)

    first = qwen_control.compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
    second = qwen_control.compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert second is not first
    hits, misses, maxsize, currsize = qwen_control.qwen_tool_wire_compile_cache_stats()
    assert hits == 0
    assert misses == 0
    assert maxsize == 128
    assert currsize == 0

    qwen_control._clear_qwen_tool_wire_compile_cache()

def test_qwen_compile_cache_evicts_to_global_weight_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_policy = _policy(
        '{"type":"object","properties":{"a":{"type":"integer"}},'
        '"required":["a"],"additionalProperties":false}'
    )
    second_policy = _policy(
        '{"type":"object","properties":{"b":{"type":"string"}},'
        '"required":["b"],"additionalProperties":false}'
    )

    qwen_control._clear_qwen_tool_wire_compile_cache()
    sample_first = qwen_control.compile_qwen_tool_wire(first_policy, ToolConstraintMode.SCHEMA)
    sample_second = qwen_control.compile_qwen_tool_wire(second_policy, ToolConstraintMode.SCHEMA)
    first_weight = qwen_control._qwen_compile_cache_weight(first_policy, sample_first)
    second_weight = qwen_control._qwen_compile_cache_weight(second_policy, sample_second)
    qwen_control._clear_qwen_tool_wire_compile_cache()

    cap = max(first_weight, second_weight) + min(first_weight, second_weight) // 2
    monkeypatch.setattr(qwen_control, "_QWEN_COMPILE_CACHE_MAX_WEIGHT_BYTES", cap)

    first = qwen_control.compile_qwen_tool_wire(first_policy, ToolConstraintMode.SCHEMA)
    qwen_control.compile_qwen_tool_wire(second_policy, ToolConstraintMode.SCHEMA)

    _, misses, _, currsize = qwen_control.qwen_tool_wire_compile_cache_stats()
    max_weight, current_weight = qwen_control._qwen_tool_wire_compile_cache_weight_stats()
    assert misses == 2
    assert currsize == 1
    assert max_weight == cap
    assert 0 < current_weight <= cap

    recompiled_first = qwen_control.compile_qwen_tool_wire(first_policy, ToolConstraintMode.SCHEMA)
    assert recompiled_first is not first
    assert qwen_control.qwen_tool_wire_compile_cache_stats()[1] == 3
