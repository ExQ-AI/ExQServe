from __future__ import annotations

import math

import pytest

from exqserve.agent._json import (
    DuplicateJsonKeyError,
    InvalidJsonError,
    canonical_json_dumps,
    parse_json_strict,
)


def test_strict_json_parses_standard_values() -> None:
    assert parse_json_strict('{"a":[1,true,null,"x"]}') == {"a": [1, True, None, "x"]}
    assert parse_json_strict('[1,2,3]') == [1, 2, 3]
    assert parse_json_strict('"text"') == "text"
    assert parse_json_strict("3.5") == 3.5


@pytest.mark.parametrize(
    "text",
    [
        '{"a":1,"a":2}',
        '{"outer":{"x":1,"x":2}}',
        '[{"x":1,"x":2}]',
    ],
)
def test_duplicate_keys_are_rejected_at_any_nesting_level(text: str) -> None:
    with pytest.raises(DuplicateJsonKeyError, match="duplicate JSON object key"):
        parse_json_strict(text)


def test_malformed_json_is_distinct_from_duplicate_keys() -> None:
    with pytest.raises(InvalidJsonError, match="invalid JSON"):
        parse_json_strict('{"a":')


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", '{"x":NaN}'])
def test_non_finite_constants_are_rejected(text: str) -> None:
    with pytest.raises(InvalidJsonError, match="non-finite"):
        parse_json_strict(text)


def test_canonical_json_is_deterministic_compact_and_unicode_preserving() -> None:
    left = {"z": 1, "a": {"β": "值", "x": 2}}
    right = {"a": {"x": 2, "β": "值"}, "z": 1}

    assert canonical_json_dumps(left) == canonical_json_dumps(right)
    assert canonical_json_dumps(left) == '{"a":{"x":2,"β":"值"},"z":1}'


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json_dumps({"x": math.nan})


def test_strict_json_does_not_repair_or_extract_markdown() -> None:
    with pytest.raises(InvalidJsonError):
        parse_json_strict('```json\n{"x":1}\n```')
