"""Strict JSON helpers shared by Agent semantic validation paths."""

from __future__ import annotations

import json
from typing import cast

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]


class InvalidJsonError(ValueError):
    """Raised when input is not strict RFC-style JSON accepted by ExQServe."""


class DuplicateJsonKeyError(InvalidJsonError):
    """Raised when a JSON object contains a duplicate member name."""


def _object_from_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> JsonValue:
    raise InvalidJsonError(f"non-finite JSON constant is not allowed: {value}")


def parse_json_strict(text: str) -> JsonValue:
    """Parse JSON without duplicate keys, non-finite constants, or repair behavior."""

    if not isinstance(text, str):
        raise TypeError("JSON input must be a string")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise InvalidJsonError(f"invalid JSON: {exc.msg}") from exc

    return cast(JsonValue, value)


def canonical_json_dumps(value: JsonValue) -> str:
    """Serialize an accepted JSON value deterministically without ASCII escaping."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
