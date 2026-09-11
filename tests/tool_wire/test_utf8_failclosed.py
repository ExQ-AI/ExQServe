from __future__ import annotations

import json

import pytest

from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire import CompileBudget, PlanCompileDisposition, compile_tool_wire_plan
from exqserve.tool_wire.controls.qwen import compile_qwen_a2a_shadow
from tests.tool_wire._support import policy, raw_compiler_capabilities, raw_spec, tool

_ROOMY = CompileBudget(1000, 1_000_000, 100_000_000, 10_000_000)


def _compile_qwen(fn, orders: dict[str, tuple[str, ...]]):
    return compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        orders,
        budget=_ROOMY,
    )


def _compile_generic(fn, orders: dict[str, tuple[str, ...]]):
    return compile_tool_wire_plan(
        raw_spec(max_calls=1),
        policy(fn, allow_parallel=False),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=raw_compiler_capabilities(),
        presentation_orders=orders,
        budget=_ROOMY,
        parser_branch_id="utf8-failclosed",
        constraint_fingerprint="static",
        activation_trigger_ids=("tool-open",),
    )


@pytest.mark.parametrize(
    ("schema", "order"),
    (
        (
            '{"type":"object","properties":{"x":{"type":"string","enum":["\\ud800"]}},"required":["x"],"additionalProperties":false}',
            ("x",),
        ),
        (
            '{"type":"object","properties":{"x":{"type":"string","const":"\\udc00"}},"required":["x"],"additionalProperties":false}',
            ("x",),
        ),
        (
            '{"type":"object","properties":{"r":{"type":"integer"},"o":{"type":"string","const":"\\ud800"}},"required":["r"],"additionalProperties":false}',
            ("r", "o"),
        ),
        (
            '{"type":"object","properties":{"x":{"type":"string","description":"\\ud800"}},"required":["x"],"additionalProperties":false}',
            ("x",),
        ),
    ),
)
def test_non_utf8_schema_strings_fail_closed_in_qwen_and_generic(
    schema: str,
    order: tuple[str, ...],
) -> None:
    fn = tool("f", schema, strict=False)

    qwen = _compile_qwen(fn, {"f": order})
    generic = _compile_generic(fn, {"f": order})

    assert qwen.plan.disposition is PlanCompileDisposition.REJECTED
    assert qwen.constraint is None
    assert generic.disposition is PlanCompileDisposition.REJECTED


def test_non_utf8_tool_name_fails_closed_in_qwen_and_generic() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    surrogate_name = chr(0xD800)
    fn = tool(surrogate_name, schema, strict=False)

    qwen = _compile_qwen(fn, {surrogate_name: ("x",)})
    generic = _compile_generic(fn, {surrogate_name: ("x",)})

    assert qwen.plan.disposition is PlanCompileDisposition.REJECTED
    assert qwen.constraint is None
    assert generic.disposition is PlanCompileDisposition.REJECTED


def test_non_utf8_presentation_name_fails_closed_in_qwen_and_generic() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    fn = tool("f", schema, strict=False)
    surrogate_name = chr(0xD800)

    qwen = _compile_qwen(fn, {"f": (surrogate_name,)})
    generic = _compile_generic(fn, {"f": (surrogate_name,)})

    assert qwen.plan.disposition is PlanCompileDisposition.REJECTED
    assert qwen.constraint is None
    assert generic.disposition is PlanCompileDisposition.REJECTED


def test_valid_non_bmp_scalar_remains_accepted_in_qwen_and_generic() -> None:
    schema = '{"type":"object","properties":{"x":{"type":"string","const":"\\ud83d\\ude42"}},"required":["x"],"additionalProperties":false}'
    fn = tool("f", schema, strict=False)

    qwen = _compile_qwen(fn, {"f": ("x",)})
    generic = _compile_generic(fn, {"f": ("x",)})

    assert qwen.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert qwen.constraint is not None
    assert generic.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
