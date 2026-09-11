from __future__ import annotations

import json

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire.compiler import _raw_argument_proof
from exqserve.tool_wire.contracts import SchemaSemanticAuthority
from exqserve.tool_wire.semantic_authority import exact_finite_non_emptiness
from tests.tool_wire._support import raw_compiler_capabilities, raw_spec


def test_finite_batch_prepares_one_schema_without_changing_membership(monkeypatch) -> None:
    preparations = 0
    original = JsonSchema.__init__

    def counted(self, schema_json):
        nonlocal preparations
        preparations += 1
        original(self, schema_json)

    monkeypatch.setattr(JsonSchema, "__init__", counted)
    schema = json.dumps({"type": "integer", "minimum": 2, "maximum": 31})
    authoritative, admitted = exact_finite_non_emptiness(
        SchemaSemanticAuthority.DRAFT_2020_12,
        schema,
        (None, True, "3", *range(64)),
    )

    assert authoritative is True
    assert admitted == tuple(range(2, 32))
    # Resource contract: preparing the same schema must not scale with candidate count.
    assert preparations == 1


@pytest.mark.parametrize(
    ("schema", "candidates", "expected"),
    [
        ('{"type":"null"}', (None, False, ""), (None,)),
        ('{"type":"string","enum":["a","b"],"const":"b"}', ("a", "b", "b"), ("b", "b")),
        ('{"$ref":"#/$defs/missing"}', (0, "x"), ()),
        ('{"type":"array","items":{"type":"integer"}}', ([1], [True], []), ([1], [])),
    ],
)
def test_finite_batch_preserves_exact_values_order_and_reference_behavior(
    schema, candidates, expected
) -> None:
    assert exact_finite_non_emptiness(
        SchemaSemanticAuthority.DRAFT_2020_12, schema, candidates
    ) == (True, expected)


def test_unavailable_authority_does_not_validate_or_claim_membership() -> None:
    assert exact_finite_non_emptiness(
        SchemaSemanticAuthority.NONE, "not a schema", (1,)
    ) == (False, ())


def test_raw_enum_compilation_prepares_schema_once(monkeypatch) -> None:
    preparations = 0
    original = JsonSchema.__init__

    def counted(self, schema_json):
        nonlocal preparations
        preparations += 1
        original(self, schema_json)

    monkeypatch.setattr(JsonSchema, "__init__", counted)
    values = [f"item-{index}" for index in range(64)]
    result = _raw_argument_proof(
        raw_spec().argument_framings[0],
        raw_compiler_capabilities(),
        {"type": "string", "enum": values},
        mode=ToolConstraintMode.SCHEMA,
    )
    assert result[2] == tuple(values)
    assert result[3] is GenerationGuarantee.SCHEMA
    assert preparations == 1
