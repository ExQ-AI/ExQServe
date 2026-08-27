from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from exqserve.core.errors import CanonicalError, ErrorCategory


def test_error_categories_are_exact_v1_set() -> None:
    assert {category.value for category in ErrorCategory} == {
        "invalid_request",
        "unsupported_capability",
        "context_length",
        "overloaded",
        "model_failure",
        "runtime_failure",
        "internal",
    }


def test_canonical_error_has_only_protocol_neutral_fields() -> None:
    error = CanonicalError(
        category=ErrorCategory.INVALID_REQUEST,
        code="invalid_role",
        message="The requested role is not valid here.",
        retryable=False,
    )

    assert [field.name for field in fields(error)] == [
        "category",
        "code",
        "message",
        "retryable",
    ]


def test_canonical_error_rejects_empty_machine_code() -> None:
    with pytest.raises(ValueError, match="code"):
        CanonicalError(
            category=ErrorCategory.INTERNAL,
            code="   ",
            message="Unexpected failure.",
            retryable=False,
        )


def test_canonical_error_is_immutable() -> None:
    error = CanonicalError(
        category=ErrorCategory.OVERLOADED,
        code="queue_full",
        message="The serving runtime is overloaded.",
        retryable=True,
    )

    with pytest.raises(FrozenInstanceError):
        error.retryable = False  # type: ignore[misc]
