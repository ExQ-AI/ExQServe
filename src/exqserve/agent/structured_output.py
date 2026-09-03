"""Structured JSON output validation independent of tool calling and decoding constraints."""

from __future__ import annotations

from dataclasses import dataclass

from exqserve.agent._json import DuplicateJsonKeyError, InvalidJsonError, parse_json_strict
from exqserve.agent.schema import JsonSchema, _schema_violations
from exqserve.agent.validation import ValidationCode, ValidationIssue, ValidationResult
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee


@dataclass(frozen=True, slots=True)
class StructuredOutputSpec:
    schema: JsonSchema
    requested_guarantee: GenerationGuarantee = GenerationGuarantee.NONE
    fallback_policy: ConstraintFallbackPolicy = ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY

    def __post_init__(self) -> None:
        if not isinstance(self.schema, JsonSchema):
            raise TypeError("schema must be a JsonSchema")
        if not isinstance(self.requested_guarantee, GenerationGuarantee):
            raise TypeError("requested_guarantee must be a GenerationGuarantee")
        if self.requested_guarantee is GenerationGuarantee.UNKNOWN:
            raise ValueError("requested_guarantee cannot be UNKNOWN")
        if not isinstance(self.fallback_policy, ConstraintFallbackPolicy):
            raise TypeError("fallback_policy must be a ConstraintFallbackPolicy")
        if self.requested_guarantee is GenerationGuarantee.NONE:
            if self.fallback_policy is not ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY:
                raise ValueError("NONE requested guarantee requires validation-only fallback")
        elif self.fallback_policy is not ConstraintFallbackPolicy.FAIL_CLOSED:
            raise ValueError("FORMAT/SCHEMA requested guarantees require fail-closed fallback")


def violates_structured_constraint_guarantee(
    result: ValidationResult,
    guarantee: GenerationGuarantee,
) -> bool:
    """Return whether validation contradicts the effective structured generation guarantee."""

    if not isinstance(result, ValidationResult):
        raise TypeError("result must be a ValidationResult")
    if not isinstance(guarantee, GenerationGuarantee):
        raise TypeError("guarantee must be a GenerationGuarantee")
    if result.is_valid:
        return False
    issue_codes = {issue.code for issue in result.issues}
    if guarantee is GenerationGuarantee.SCHEMA:
        return bool(issue_codes)
    if guarantee is GenerationGuarantee.FORMAT:
        return bool(
            issue_codes
            & {ValidationCode.INVALID_JSON, ValidationCode.DUPLICATE_JSON_KEY}
        )
    return False


def validate_structured_output(text: str, spec: StructuredOutputSpec) -> ValidationResult:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(spec, StructuredOutputSpec):
        raise TypeError("spec must be a StructuredOutputSpec")

    try:
        value = parse_json_strict(text)
    except DuplicateJsonKeyError as exc:
        return ValidationResult(
            (
                ValidationIssue(
                    ValidationCode.DUPLICATE_JSON_KEY,
                    str(exc),
                ),
            )
        )
    except InvalidJsonError as exc:
        return ValidationResult(
            (
                ValidationIssue(
                    ValidationCode.INVALID_JSON,
                    str(exc),
                ),
            )
        )

    issues = tuple(
        ValidationIssue(
            ValidationCode.SCHEMA_VALIDATION_FAILED,
            violation.message,
            violation.path,
        )
        for violation in _schema_violations(spec.schema, value)
    )
    return ValidationResult(issues)
