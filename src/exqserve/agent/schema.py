"""Immutable Draft 2020-12 JSON Schema boundary for Agent semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from exqserve.agent._json import JsonValue, canonical_json_dumps, parse_json_strict

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_REFERENCE_KEYWORDS = frozenset({"$ref", "$dynamicRef"})


@dataclass(frozen=True, slots=True)
class _SchemaViolation:
    path: tuple[str | int, ...]
    validator: str
    message: str


@dataclass(frozen=True, slots=True, init=False)
class JsonSchema:
    canonical_json: str

    def __init__(self, schema_json: str) -> None:
        value = parse_json_strict(schema_json)
        if not isinstance(value, dict):
            raise TypeError("schema document must be a JSON object")

        declared_draft = value.get("$schema")
        if declared_draft is not None and declared_draft != _DRAFT_2020_12:
            raise ValueError("V1 schemas must use JSON Schema Draft 2020-12")

        _reject_external_references(value)

        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as exc:
            raise ValueError(f"invalid Draft 2020-12 schema: {exc.message}") from exc

        object.__setattr__(self, "canonical_json", canonical_json_dumps(value))


def _reject_external_references(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _REFERENCE_KEYWORDS and isinstance(child, str) and not child.startswith("#"):
                raise ValueError("external schema references are not supported in V1")
            _reject_external_references(child)
    elif isinstance(value, list):
        for child in value:
            _reject_external_references(child)


def _schema_violations(schema: JsonSchema, value: JsonValue) -> tuple[_SchemaViolation, ...]:
    schema_value = parse_json_strict(schema.canonical_json)
    assert isinstance(schema_value, dict)  # constructor invariant

    validator = Draft202012Validator(cast(dict[str, object], schema_value))
    violations = tuple(
        _SchemaViolation(
            path=tuple(cast(str | int, part) for part in error.path),
            validator=str(error.validator),
            message=error.message,
        )
        for error in validator.iter_errors(value)
    )
    return tuple(sorted(violations, key=_violation_sort_key))


def validate_strict_function_schema(schema: JsonSchema) -> None:
    """Validate strict function declaration structure before generation."""

    if not isinstance(schema, JsonSchema):
        raise TypeError("schema must be a JsonSchema")
    value = parse_json_strict(schema.canonical_json)
    assert isinstance(value, dict)
    if value.get("type") != "object":
        raise ValueError("strict function parameters must declare root type 'object'")
    _validate_strict_schema_node(value, ())


def _validate_strict_schema_node(value: JsonValue, path: tuple[str | int, ...]) -> None:
    if not isinstance(value, dict):
        return

    type_value = value.get("type")
    object_schema = (
        type_value == "object"
        or (isinstance(type_value, list) and "object" in type_value)
        or "properties" in value
    )
    if object_schema:
        location = _format_schema_path(path)
        if value.get("additionalProperties") is not False:
            raise ValueError(f"object schema at {location} must set additionalProperties to false")
        properties = value.get("properties", {})
        assert isinstance(properties, dict)
        required = value.get("required", [])
        assert isinstance(required, list)
        required_names = {item for item in required if isinstance(item, str)}
        missing = sorted(set(properties) - required_names)
        if missing:
            raise ValueError(
                f"object schema at {location} must require every property; missing {missing[0]!r}"
            )

    for child_path, child in _strict_schema_children(value, path):
        _validate_strict_schema_node(child, child_path)


def _strict_schema_children(
    value: dict[str, JsonValue],
    path: tuple[str | int, ...],
) -> tuple[tuple[tuple[str | int, ...], JsonValue], ...]:
    children: list[tuple[tuple[str | int, ...], JsonValue]] = []

    for keyword in ("properties", "patternProperties", "$defs", "dependentSchemas"):
        mapping = value.get(keyword)
        if isinstance(mapping, dict):
            for name, child in mapping.items():
                if isinstance(child, dict):
                    children.append(((*path, keyword, name), child))

    for keyword in (
        "items",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "propertyNames",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contentSchema",
    ):
        child = value.get(keyword)
        if isinstance(child, dict):
            children.append(((*path, keyword), child))

    for keyword in ("prefixItems", "allOf", "anyOf", "oneOf"):
        sequence = value.get(keyword)
        if isinstance(sequence, list):
            for index, child in enumerate(sequence):
                if isinstance(child, dict):
                    children.append(((*path, keyword, index), child))

    return tuple(children)


def _format_schema_path(path: tuple[str | int, ...]) -> str:
    if not path:
        return "$"
    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


def _violation_sort_key(violation: _SchemaViolation) -> tuple[object, ...]:
    path_key = tuple(
        (0, f"{part:020d}") if isinstance(part, int) else (1, part)
        for part in violation.path
    )
    return (path_key, violation.validator, violation.message)
