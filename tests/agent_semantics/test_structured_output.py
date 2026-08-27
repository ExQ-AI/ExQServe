from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec, validate_structured_output
from exqserve.agent.validation import ValidationCode


def _spec(schema: str) -> StructuredOutputSpec:
    return StructuredOutputSpec(JsonSchema(schema))


def test_structured_output_spec_contains_only_schema_and_is_immutable() -> None:
    spec = _spec('{"type":"object"}')

    assert [field.name for field in fields(spec)] == ["schema"]

    with pytest.raises(FrozenInstanceError):
        spec.schema = JsonSchema('{}')  # type: ignore[misc]


def test_structured_output_spec_requires_json_schema() -> None:
    with pytest.raises(TypeError, match="schema"):
        StructuredOutputSpec(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "schema"),
    [
        ('{"x":1}', '{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}'),
        ('[1,2,3]', '{"type":"array","items":{"type":"integer"}}'),
        ('"hello"', '{"type":"string","minLength":2}'),
        ('42', '{"type":"integer","minimum":0}'),
        ('true', '{"type":"boolean"}'),
        ('null', '{"type":"null"}'),
    ],
)
def test_schema_controls_allowed_top_level_json_type(text: str, schema: str) -> None:
    assert validate_structured_output(text, _spec(schema)).is_valid


def test_malformed_json_returns_validation_issue_without_exception() -> None:
    result = validate_structured_output('{"x":', _spec('{"type":"object"}'))

    assert [issue.code for issue in result.issues] == [ValidationCode.INVALID_JSON]
    assert result.issues[0].path == ()


def test_duplicate_json_key_is_distinct_issue() -> None:
    result = validate_structured_output(
        '{"x":1,"x":2}',
        _spec('{"type":"object"}'),
    )

    assert [issue.code for issue in result.issues] == [ValidationCode.DUPLICATE_JSON_KEY]


def test_nested_schema_failure_has_protocol_neutral_schema_path() -> None:
    spec = _spec(
        """{
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
    )

    result = validate_structured_output('{"user":{"age":12}}', spec)

    assert [issue.code for issue in result.issues] == [ValidationCode.SCHEMA_VALIDATION_FAILED]
    assert result.issues[0].path == ("user", "age")


def test_multiple_schema_failures_are_deterministic() -> None:
    spec = _spec(
        '{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}}}'
    )
    text = '{"a":"wrong","b":3}'

    first = validate_structured_output(text, spec)
    second = validate_structured_output(text, spec)

    assert first == second
    assert [issue.path for issue in first.issues] == [("a",), ("b",)]


def test_markdown_fence_is_not_repaired_or_extracted() -> None:
    result = validate_structured_output(
        '```json\n{"x":1}\n```',
        _spec('{"type":"object"}'),
    )

    assert [issue.code for issue in result.issues] == [ValidationCode.INVALID_JSON]


def test_format_remains_annotation_only_for_structured_output() -> None:
    result = validate_structured_output(
        '"not-an-email"',
        _spec('{"type":"string","format":"email"}'),
    )

    assert result.is_valid
