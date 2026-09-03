from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from exqserve.core.errors import (
    CanonicalError,
    ErrorCategory,
    FailureCause,
    SemanticCommitClass,
    commit_aware_error,
    public_error_code,
)


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
        "cause",
    ]
    assert error.cause is None


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


def test_failure_causes_are_small_stable_fact_vocabulary() -> None:
    assert {cause.value for cause in FailureCause} == {
        "output_eos",
        "output_length",
        "parser_ambiguity_limit",
        "runtime_recovering",
        "restart_required",
        "constraint_failure",
        "model_tool_output_invalid",
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_incomplete",
                "Incomplete tool call.",
                False,
                FailureCause.OUTPUT_EOS,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_incomplete",
                "Incomplete tool call.",
                False,
                FailureCause.OUTPUT_LENGTH,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "protocol_ambiguity",
                "Ambiguous model output.",
                False,
                FailureCause.OUTPUT_EOS,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "protocol_ambiguity",
                "Ambiguous model output.",
                False,
                FailureCause.OUTPUT_LENGTH,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "protocol_ambiguity",
                "Ambiguous model output.",
                False,
                FailureCause.PARSER_AMBIGUITY_LIMIT,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_invalid",
                "Model produced an invalid tool call.",
                False,
                FailureCause.MODEL_TOOL_OUTPUT_INVALID,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.RUNTIME_FAILURE,
                "generation_failed",
                "Runtime failed.",
                False,
                FailureCause.RUNTIME_RECOVERING,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.RUNTIME_FAILURE,
                "generation_failed",
                "Runtime failed.",
                False,
                FailureCause.RESTART_REQUIRED,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.CONTEXT_LENGTH,
                "prompt_limit_exceeded",
                "Too long.",
                False,
            ),
            "context_length_exceeded",
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "bad_output",
                "Bad output.",
                False,
                FailureCause.OUTPUT_EOS,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "other_model_failure",
                "Bad output.",
                False,
                FailureCause.MODEL_TOOL_OUTPUT_INVALID,
            ),
            None,
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "constraint_failed",
                "Constraint failed.",
                False,
                FailureCause.CONSTRAINT_FAILURE,
            ),
            None,
        ),
    ],
)
def test_public_error_code_only_normalizes_standard_compatibility_identity(
    error: CanonicalError,
    expected: str | None,
) -> None:
    assert public_error_code(error) == expected


def test_commit_aware_error_preserves_precommit_replay_and_clears_postcommit() -> None:
    source = CanonicalError(
        ErrorCategory.RUNTIME_FAILURE,
        "transient",
        "Transient backend failure.",
        True,
        FailureCause.RUNTIME_RECOVERING,
    )

    assert commit_aware_error(source, SemanticCommitClass.NO_SEMANTIC_COMMIT) is source
    for commit_class in (
        SemanticCommitClass.CONTENT_COMMITTED,
        SemanticCommitClass.PARTIAL_TOOL_COMMITTED,
        SemanticCommitClass.TOOL_COMPLETED,
    ):
        committed = commit_aware_error(source, commit_class)
        assert committed.retryable is False
        assert committed.cause is FailureCause.RUNTIME_RECOVERING
        assert committed.code == source.code
