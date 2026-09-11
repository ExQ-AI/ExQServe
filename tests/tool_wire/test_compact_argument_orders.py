from __future__ import annotations

import json
from dataclasses import replace

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire.contracts import ArgumentOrderingMode
from exqserve.tool_wire.controls.qwen import compile_qwen_tool_wire
from exqserve.tool_wire.engine import ProductionToolWireSession
from tests.tool_wire.test_a2b_qwen_production import _constraint_accepts


def _bundle(count: int, *, all_required: bool = False):
    names = [f"p{i:02}" for i in range(count)]
    schema = {
        "type": "object",
        "properties": {name: {"type": "integer"} for name in names},
        "required": names if all_required else ["p00"],
        "additionalProperties": False,
    }
    tool = FunctionTool("write", None, JsonSchema(json.dumps(schema)), strict=False)
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
    return compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)


def _suffix(indices: tuple[int, ...]) -> str:
    parameters = "".join(f"<parameter=p{i:02}>\n{i}\n</parameter>\n" for i in indices)
    return f"<function=write>{parameters}</function></tool_call>"


@pytest.mark.parametrize("count", (6, 7, 12))
def test_supported_optional_arguments_remain_generation_choices(count: int) -> None:
    bundle = _bundle(count)
    assert bundle.constrained
    generated = {argument.name for argument in bundle.plan.tools[0].arguments if argument.generated}
    assert generated == {f"p{i:02}" for i in range(count)}


@pytest.mark.parametrize("count", (6, 7, 12))
def test_declared_order_retains_independent_optional_arguments_and_values(count: int) -> None:
    bundle = _bundle(count)
    assert bundle.constraint is not None
    combinations = [(0,), tuple(range(count))]
    combinations.extend((0, i) for i in range(1, count))
    for indices in combinations:
        suffix = _suffix(indices)
        assert _constraint_accepts(bundle.constraint, suffix), indices
        session = ProductionToolWireSession(bundle.spec, bundle.plan)
        session.feed("<tool_call>" + suffix)
        result = session.finish()
        assert result.is_complete
        assert result.sequence is not None
        assert {
            item.name: item.canonical_value_json for item in result.sequence.calls[0].occurrences
        } == {f"p{i:02}": str(i) for i in indices}


def test_required_parameters_use_compact_grammar_without_losing_values() -> None:
    bundle = _bundle(6, all_required=True)
    assert bundle.constraint is not None
    assert len(bundle.constraint.lark_grammar.encode("utf-8")) < 10_000
    assert _constraint_accepts(bundle.constraint, _suffix(tuple(range(6))))
    assert not _constraint_accepts(bundle.constraint, _suffix((0,)))


def test_generation_uses_declaration_order_instead_of_permutations() -> None:
    bundle = _bundle(3, all_required=True)
    assert bundle.constraint is not None
    assert _constraint_accepts(bundle.constraint, _suffix((0, 1, 2)))
    assert not _constraint_accepts(bundle.constraint, _suffix((2, 1, 0)))


@pytest.mark.parametrize("indices", ((2, 1, 0), (6, 0)))
def test_decoder_preserves_semantically_equivalent_parameter_order(indices: tuple[int, ...]) -> None:
    bundle = _bundle(7)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    for char in "<tool_call>" + _suffix(indices):
        session.feed(char)
    result = session.finish()
    assert result.is_complete
    assert result.sequence is not None
    assert {item.name: item.canonical_value_json for item in result.sequence.calls[0].occurrences} == {
        f"p{i:02}": str(i) for i in indices
    }


@pytest.mark.parametrize("indices", ((0, 0), (1,), (0, 7)))
def test_order_tolerance_does_not_allow_duplicate_missing_or_unknown_parameters(indices: tuple[int, ...]) -> None:
    bundle = _bundle(7)
    session = ProductionToolWireSession(bundle.spec, bundle.plan)
    session.feed("<tool_call>" + _suffix(indices))
    assert not session.finish().is_complete


def test_declaration_only_wire_contract_still_requires_declared_order() -> None:
    bundle = _bundle(3, all_required=True)
    spec = replace(bundle.spec, ordering=ArgumentOrderingMode.DECLARATION_ORDER)
    plan = replace(bundle.plan, spec_fingerprint=spec.fingerprint)
    session = ProductionToolWireSession(spec, plan)
    session.feed("<tool_call>" + _suffix((2, 1, 0)))
    assert not session.finish().is_complete
