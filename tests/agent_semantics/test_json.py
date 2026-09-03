from __future__ import annotations

import math
import sys

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


@pytest.mark.parametrize(
    "text",
    [
        "1e999",
        "-1e999",
        "1e309",
        '{"x":1e999}',
        '[0,{"x":-1e999}]',
    ],
)
def test_numeric_overflow_to_non_finite_float_is_rejected(text: str) -> None:
    with pytest.raises(InvalidJsonError, match="non-finite JSON number"):
        parse_json_strict(text)


def test_large_finite_float_remains_valid_json() -> None:
    assert parse_json_strict("1e308") == 1e308
    assert parse_json_strict('{"x":-1e308}') == {"x": -1e308}


def test_integer_beyond_python_digit_limit_is_rejected_as_invalid_json() -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Python integer digit safety limit is disabled")

    accepted = "9" * limit
    assert parse_json_strict(accepted) == int(accepted)
    with pytest.raises(InvalidJsonError, match="supported digit limit"):
        parse_json_strict("9" * (limit + 1))


def test_decoder_recursion_limit_is_rejected_as_invalid_json() -> None:
    text = "[" * 10_000 + "0" + "]" * 10_000
    with pytest.raises(InvalidJsonError, match="decoder depth"):
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
