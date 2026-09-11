"""Exact schema semantic authority used by Tool-wire A0 proof/certification."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from exqserve.agent._json import JsonValue, canonical_json_dumps, parse_json_strict
from exqserve.agent.schema import JsonSchema, _schema_violations
from exqserve.tool_wire.contracts import SchemaSemanticAuthority


def _charge_json_traversal(
    value: JsonValue,
    charge_work: Callable[[int], None] | None,
) -> None:
    """Authorize one request-scaled canonicalization/materialization before it executes."""

    del value
    if charge_work is not None:
        charge_work(1)


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
    schema = JsonSchema(schema_json)
    return not _schema_violations(schema, value)


def schema_values_equal(
    authority: SchemaSemanticAuthority,
    left: JsonValue,
    right: JsonValue,
) -> bool:
    """Compare JSON values using the declared schema authority's ``const`` semantics."""

    if not isinstance(authority, SchemaSemanticAuthority):
        raise TypeError("authority must be a SchemaSemanticAuthority")
    if authority is SchemaSemanticAuthority.NONE:
        return False
    if authority is not SchemaSemanticAuthority.DRAFT_2020_12:
        raise ValueError(f"unsupported schema semantic authority: {authority.value}")
    return schema_value_is_valid(authority, canonical_json_dumps({"const": left}), right)


def exact_finite_non_emptiness(
    authority: SchemaSemanticAuthority,
    schema_json: str,
    candidates: tuple[JsonValue, ...],
) -> tuple[bool, tuple[JsonValue, ...]]:
    """Validate an exhaustive finite candidate domain against the exact schema authority."""

    if authority is SchemaSemanticAuthority.NONE:
        return False, ()
    admitted = tuple(
        candidate
        for candidate in candidates
        if schema_value_is_valid(authority, schema_json, candidate)
    )
    return True, admitted


def find_exact_witness(
    authority: SchemaSemanticAuthority,
    schema_json: str,
) -> ExactWitness | None:
    """Find one bounded candidate witness; failure means UNKNOWN, never EMPTY.

    Candidate construction is intentionally incomplete. Soundness comes exclusively from exact
    Draft-2020-12 validation of a candidate, not from this bounded constructor.
    """

    return _find_exact_witness(authority, schema_json, charge_work=None)


def find_exact_witness_metered(
    authority: SchemaSemanticAuthority,
    schema_json: str,
    charge_work: Callable[[int], None],
    *,
    parsed_schema: dict[str, JsonValue] | None = None,
) -> ExactWitness | None:
    """Find an exact witness while charging candidate construction and validation work.

    ``parsed_schema`` lets metered compiler callers reuse schema state they already own instead of
    reparsing an arbitrarily large JSON document before the first affordable work step.
    """

    if not callable(charge_work):
        raise TypeError("charge_work must be callable")
    return _find_exact_witness(
        authority,
        schema_json,
        charge_work=charge_work,
        parsed_schema=parsed_schema,
    )


def _find_exact_witness(
    authority: SchemaSemanticAuthority,
    schema_json: str,
    *,
    charge_work: Callable[[int], None] | None,
    parsed_schema: dict[str, JsonValue] | None = None,
) -> ExactWitness | None:
    if authority is SchemaSemanticAuthority.NONE:
        return None
    schema_value: JsonValue = (
        parsed_schema if parsed_schema is not None else parse_json_strict(schema_json)
    )
    if not isinstance(schema_value, dict):
        return None
    return _find_exact_witness_in_schema(
        authority,
        schema_value,
        schema_json,
        depth=0,
        charge_work=charge_work,
    )


def _find_exact_witness_in_schema(
    authority: SchemaSemanticAuthority,
    schema: dict[str, JsonValue],
    schema_json: str,
    *,
    depth: int,
    charge_work: Callable[[int], None] | None,
) -> ExactWitness | None:
    for candidate in _iter_candidate_witnesses(
        schema,
        authority=authority,
        depth=depth,
        charge_work=charge_work,
    ):
        if charge_work is not None:
            charge_work(1)
        if schema_value_is_valid(authority, schema_json, candidate):
            return ExactWitness(candidate)
    return None


def _iter_candidate_witnesses(
    schema: dict[str, JsonValue],
    *,
    authority: SchemaSemanticAuthority,
    depth: int,
    charge_work: Callable[[int], None] | None,
) -> Iterator[JsonValue]:
    if depth > 4:
        return
    if charge_work is not None:
        charge_work(1)

    seen: set[str] = set()

    def emit(candidate: JsonValue) -> tuple[JsonValue, ...]:
        _charge_json_traversal(candidate, charge_work)
        identity = canonical_json_dumps(candidate)
        if identity in seen:
            return ()
        seen.add(identity)
        return (candidate,)

    if "const" in schema:
        yield from emit(schema["const"])

    enum_value = schema.get("enum")
    if isinstance(enum_value, list):
        for value in enum_value:
            yield from emit(value)

    schema_type = schema.get("type")
    defaults: tuple[JsonValue, ...] = ()
    if schema_type == "string":
        defaults = ("", "x")
    elif schema_type == "integer":
        defaults = (0, 1, -1)
    elif schema_type == "number":
        defaults = (0, 1, -1, 0.5)
    elif schema_type == "boolean":
        defaults = (False, True)
    elif schema_type == "null":
        defaults = (None,)
    for value in defaults:
        yield from emit(value)

    if schema_type == "array":
        yield from emit([])
        items = schema.get("items")
        if isinstance(items, dict):
            for emitted_children, child in enumerate(
                _iter_candidate_witnesses(
                    items,
                    authority=authority,
                    depth=depth + 1,
                    charge_work=charge_work,
                ),
                start=1,
            ):
                for value in ([child], [child, child]):
                    yield from emit(value)
                if emitted_children >= 4:
                    break
        else:
            fallback_arrays: tuple[JsonValue, ...] = ([None], [0], [""])
            for value in fallback_arrays:
                yield from emit(value)
    elif schema_type == "object":
        yield from emit({})
        properties = schema.get("properties")
        required = schema.get("required")
        if isinstance(properties, dict) and isinstance(required, list):
            required_object: dict[str, JsonValue] = {}
            complete = True
            for name in required:
                if charge_work is not None:
                    charge_work(1)
                if not isinstance(name, str):
                    complete = False
                    break
                child_schema = properties.get(name)
                if not isinstance(child_schema, dict):
                    complete = False
                    break
                _charge_json_traversal(child_schema, charge_work)
                child_schema_json = canonical_json_dumps(child_schema)
                child_witness = _find_exact_witness_in_schema(
                    authority,
                    child_schema,
                    child_schema_json,
                    depth=depth + 1,
                    charge_work=charge_work,
                )
                if child_witness is None:
                    complete = False
                    break
                required_object[name] = child_witness.value
            if complete:
                yield from emit(required_object)


def _candidate_witnesses(
    schema: dict[str, JsonValue],
    *,
    authority: SchemaSemanticAuthority,
    depth: int,
) -> tuple[JsonValue, ...]:
    """Legacy eager candidate helper retained for compatibility with existing callers/tests."""

    return tuple(
        _iter_candidate_witnesses(
            schema,
            authority=authority,
            depth=depth,
            charge_work=None,
        )
    )
