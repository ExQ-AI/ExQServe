"""Structured JSON output validation independent of tool calling and decoding constraints."""

from __future__ import annotations

from dataclasses import dataclass

from exqserve.agent._json import DuplicateJsonKeyError, InvalidJsonError, parse_json_strict
from exqserve.agent.schema import JsonSchema, _schema_violations
from exqserve.agent.validation import ValidationCode, ValidationIssue, ValidationResult


@dataclass(frozen=True, slots=True)
class StructuredOutputSpec:
    schema: JsonSchema

    def __post_init__(self) -> None:
        if not isinstance(self.schema, JsonSchema):
            raise TypeError("schema must be a JsonSchema")


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
