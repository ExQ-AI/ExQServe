"""Exact schema semantic authority for production Tool-Wire compilation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite

from jsonschema import Draft202012Validator
from referencing.exceptions import Unresolvable

from exqserve.agent._json import JsonValue, canonical_json_dumps, parse_json_strict
from exqserve.agent.schema import JsonSchema
from exqserve.tool_wire.contracts import SchemaSemanticAuthority


@dataclass(frozen=True, slots=True)
class ExactWitness:
    """Explicit witness presence so JSON null is not confused with no witness."""

    value: JsonValue


def schema_value_is_valid(
    authority: SchemaSemanticAuthority,
    schema_json: str,
    value: JsonValue,
) -> bool:
    """Return exact branch membership under the declared semantic authority."""

    if not isinstance(authority, SchemaSemanticAuthority):
        raise TypeError("authority must be a SchemaSemanticAuthority")
    if authority is SchemaSemanticAuthority.NONE:
        return False
    if authority is not SchemaSemanticAuthority.DRAFT_2020_12:
        raise ValueError(f"unsupported schema semantic authority: {authority.value}")
    return _value_is_valid(_prepare_validator(schema_json), value)


def _prepare_validator(schema_json: str) -> Draft202012Validator:
    schema = JsonSchema(schema_json)
    value = parse_json_strict(schema.canonical_json)
    assert isinstance(value, dict)
    return Draft202012Validator(value)


def _value_is_valid(validator: Draft202012Validator, value: JsonValue) -> bool:
    try:
        return validator.is_valid(value)
    except Unresolvable:
        # Detached/local compilation may intentionally lose a reference scope that exists only in
        # the original root schema. Exact membership is UNKNOWN there and must fail closed.
        return False


def exact_finite_non_emptiness(
    authority: SchemaSemanticAuthority,
    schema_json: str,
    candidates: tuple[JsonValue, ...],
) -> tuple[bool, tuple[JsonValue, ...]]:
    """Validate an exhaustive finite candidate domain against the exact schema authority."""

    if authority is SchemaSemanticAuthority.NONE:
        return False, ()
    if not candidates:
        return True, ()
    if not isinstance(authority, SchemaSemanticAuthority):
        raise TypeError("authority must be a SchemaSemanticAuthority")
    if authority is not SchemaSemanticAuthority.DRAFT_2020_12:
        raise ValueError(f"unsupported schema semantic authority: {authority.value}")
    validator = _prepare_validator(schema_json)
    admitted = tuple(
        candidate
        for candidate in candidates
        if _value_is_valid(validator, candidate)
    )
    return True, admitted


def find_exact_witness(
    authority: SchemaSemanticAuthority,
    schema_json: str,
    *,
    parsed_schema: dict[str, JsonValue] | None = None,
) -> ExactWitness | None:
    """Validate one deterministic witness from the bounded generation schema.

    Candidate construction is deliberately incomplete. The compiler only needs a sound witness
    for constrained shapes it can actually emit; inability to construct one means UNKNOWN rather
    than invoking a second general JSON-Schema search engine.
    """

    if authority is SchemaSemanticAuthority.NONE:
        return None
    schema_value: JsonValue = (
        parsed_schema if parsed_schema is not None else parse_json_strict(schema_json)
    )
    if not isinstance(schema_value, dict):
        return None
    for candidate in _minimal_witness_candidates(schema_value, authority=authority, depth=0):
        if schema_value_is_valid(authority, schema_json, candidate.value):
            return candidate
    return None


def _is_finite_numeric(value: JsonValue) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and isfinite(value)


def _minimal_witness_candidates(
    schema: dict[str, JsonValue],
    *,
    authority: SchemaSemanticAuthority,
    depth: int,
) -> tuple[ExactWitness, ...]:
    """Construct only minimal values for the compiler's supported generation-schema vocabulary."""

    if depth > 64:
        return ()
    if "const" in schema:
        return (ExactWitness(schema["const"]),)
    enum_value = schema.get("enum")
    if isinstance(enum_value, list):
        return tuple(ExactWitness(value) for value in enum_value)

    schema_type = schema.get("type")
    if schema_type == "string":
        return (ExactWitness(""),)
    if schema_type == "boolean":
        return (ExactWitness(False),)
    if schema_type == "null":
        return (ExactWitness(None),)
    if schema_type == "array":
        # The production compiler does not support minItems, so [] is the actual minimal emitted
        # array shape whenever an array generation schema is admitted.
        return (ExactWitness([]),)
    if schema_type == "integer":
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if _is_finite_numeric(minimum):
            assert isinstance(minimum, (int, float)) and not isinstance(minimum, bool)
            candidate: JsonValue = ceil(minimum)
        elif _is_finite_numeric(maximum):
            assert isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
            candidate = floor(maximum)
        else:
            candidate = 0
        return (ExactWitness(candidate),)
    if schema_type == "number":
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if _is_finite_numeric(minimum):
            assert isinstance(minimum, (int, float)) and not isinstance(minimum, bool)
            candidate = minimum
        elif _is_finite_numeric(maximum):
            assert isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
            candidate = maximum
        else:
            candidate = 0
        return (ExactWitness(candidate),)
    if schema_type != "object":
        return ()

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return ()
    candidate_object: dict[str, JsonValue] = {}
    for name in required:
        if not isinstance(name, str):
            return ()
        child_schema = properties.get(name)
        if not isinstance(child_schema, dict):
            return ()
        children = _minimal_witness_candidates(
            child_schema,
            authority=authority,
            depth=depth + 1,
        )
        if not children:
            return ()
        child_schema_json = canonical_json_dumps(child_schema)
        child = next(
            (
                candidate
                for candidate in children
                if schema_value_is_valid(authority, child_schema_json, candidate.value)
            ),
            None,
        )
        if child is None:
            return ()
        candidate_object[name] = child.value
    return (ExactWitness(candidate_object),)
