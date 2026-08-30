from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.agent.schema import JsonSchema, _schema_violations, validate_strict_function_schema

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def test_schema_defaults_to_draft_2020_12_and_canonicalizes() -> None:
    left = JsonSchema('{"properties":{"β":{"type":"string"}},"type":"object"}')
    right = JsonSchema('{"type":"object","properties":{"β":{"type":"string"}}}')

    assert left == right
    assert left.canonical_json == '{"properties":{"β":{"type":"string"}},"type":"object"}'


def test_explicit_draft_2020_12_is_accepted() -> None:
    schema = JsonSchema(f'{{"$schema":"{_DRAFT_2020_12}","type":"integer"}}')

    assert _DRAFT_2020_12 in schema.canonical_json


def test_explicit_other_draft_is_rejected() -> None:
    with pytest.raises(ValueError, match="Draft 2020-12"):
        JsonSchema('{"$schema":"http://json-schema.org/draft-07/schema#","type":"object"}')


def test_schema_document_must_be_an_object() -> None:
    with pytest.raises(TypeError, match="schema document must be a JSON object"):
        JsonSchema('[{"type":"string"}]')


def test_invalid_draft_2020_12_schema_is_rejected_without_third_party_exception() -> None:
    with pytest.raises(ValueError, match="invalid Draft 2020-12 schema") as exc_info:
        JsonSchema('{"type":42}')

    assert "SchemaError" not in type(exc_info.value).__name__


@pytest.mark.parametrize(
    "keyword",
    ["$ref", "$dynamicRef"],
)
def test_external_schema_references_are_rejected(keyword: str) -> None:
    text = f'{{"{keyword}":"https://example.com/schema.json"}}'

    with pytest.raises(ValueError, match="external schema references"):
        JsonSchema(text)


def test_local_fragment_reference_is_supported() -> None:
    schema = JsonSchema(
        """{
          "$defs": {"name": {"type": "string", "minLength": 2}},
          "type": "object",
          "properties": {"name": {"$ref": "#/$defs/name"}},
          "required": ["name"]
        }"""
    )

    assert _schema_violations(schema, {"name": "ok"}) == ()
    violations = _schema_violations(schema, {"name": "x"})
    assert len(violations) == 1
    assert violations[0].path == ("name",)


def test_format_is_annotation_only_in_v1() -> None:
    schema = JsonSchema('{"type":"string","format":"email"}')

    assert _schema_violations(schema, "not-an-email") == ()


def test_schema_rejects_duplicate_keys_and_non_finite_json() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        JsonSchema('{"type":"object","type":"string"}')

    with pytest.raises(ValueError, match="non-finite"):
        JsonSchema('{"const":NaN}')


def test_strict_function_schema_accepts_required_nullable_nested_objects() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{'
        '"name":{"type":["string","null"]},'
        '"options":{"type":"object","properties":{"enabled":{"type":"boolean"}},'
        '"required":["enabled"],"additionalProperties":false}'
        '},"required":["name","options"],"additionalProperties":false}'
    )

    validate_strict_function_schema(schema)


def test_strict_function_schema_requires_closed_objects_and_all_properties_required() -> None:
    with pytest.raises(ValueError, match="additionalProperties"):
        validate_strict_function_schema(
            JsonSchema('{"type":"object","properties":{},"required":[]}')
        )

    with pytest.raises(ValueError, match="missing 'name'"):
        validate_strict_function_schema(
            JsonSchema(
                '{"type":"object","properties":{"name":{"type":"string"}},'
                '"required":[],"additionalProperties":false}'
            )
        )

    with pytest.raises(ValueError, match="additionalProperties"):
        validate_strict_function_schema(
            JsonSchema(
                '{"type":"object","properties":{"nested":{"type":"object",'
                '"properties":{},"required":[]}},"required":["nested"],'
                '"additionalProperties":false}'
            )
        )


def test_strict_function_schema_requires_object_root() -> None:
    with pytest.raises(ValueError, match="root type 'object'"):
        validate_strict_function_schema(JsonSchema('{"type":"string"}'))


def test_json_schema_is_immutable() -> None:
    schema = JsonSchema('{"type":"object"}')

    with pytest.raises(FrozenInstanceError):
        schema.canonical_json = "{}"  # type: ignore[misc]
