"""Protocol-neutral validation results and Agent tool semantic checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from exqserve.agent._json import (
    DuplicateJsonKeyError,
    InvalidJsonError,
    canonical_json_dumps,
    parse_json_strict,
)
from exqserve.agent.schema import _schema_violations
from exqserve.agent.tools import ToolChoiceMode, ToolPolicy
from exqserve.core.items import (
    CanonicalItem,
    MultimodalToolResultItem,
    ToolCallItem,
    ToolResultItem,
)


class ValidationCode(str, Enum):
    INVALID_JSON = "invalid_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    JSON_VALUE_NOT_OBJECT = "json_value_not_object"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    UNDECLARED_TOOL = "undeclared_tool"
    TOOL_CALL_FORBIDDEN = "tool_call_forbidden"
    TOOL_CALL_REQUIRED = "tool_call_required"
    WRONG_NAMED_TOOL = "wrong_named_tool"
    PARALLEL_TOOL_CALLS_FORBIDDEN = "parallel_tool_calls_forbidden"
    DUPLICATE_TOOL_CALL_ID = "duplicate_tool_call_id"
    INVALID_TOOL_CALL_ORDER = "invalid_tool_call_order"
    UNKNOWN_TOOL_RESULT = "unknown_tool_result"
    DUPLICATE_TOOL_RESULT = "duplicate_tool_result"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, ValidationCode):
            raise TypeError("code must be a ValidationCode")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        if not isinstance(self.path, tuple):
            raise TypeError("path must be a tuple")
        if not all(isinstance(part, str | int) and not isinstance(part, bool) for part in self.path):
            raise TypeError("path entries must be strings or integers")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.issues, tuple):
            raise TypeError("issues must be a tuple")
        if not all(isinstance(issue, ValidationIssue) for issue in self.issues):
            raise TypeError("issues must contain only ValidationIssue values")

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class ToolCallsValidation:
    result: ValidationResult
    canonical_arguments: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result, ValidationResult):
            raise TypeError("result must be a ValidationResult")
        if not isinstance(self.canonical_arguments, tuple):
            raise TypeError("canonical_arguments must be a tuple")
        if not all(value is None or isinstance(value, str) for value in self.canonical_arguments):
            raise TypeError("canonical_arguments entries must be strings or None")


def validate_tool_calls(
    calls: tuple[ToolCallItem, ...],
    policy: ToolPolicy,
) -> ValidationResult:
    return validate_tool_calls_with_canonical_arguments(calls, policy).result


def validate_tool_calls_with_canonical_arguments(
    calls: tuple[ToolCallItem, ...],
    policy: ToolPolicy,
) -> ToolCallsValidation:
    if not isinstance(calls, tuple):
        raise TypeError("calls must be a tuple")
    if not all(isinstance(call, ToolCallItem) for call in calls):
        raise TypeError("calls must contain only ToolCallItem values")
    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")

    issues: list[ValidationIssue] = []

    if policy.choice.mode is ToolChoiceMode.NONE and calls:
        return ToolCallsValidation(
            ValidationResult(
                (
                    ValidationIssue(
                        ValidationCode.TOOL_CALL_FORBIDDEN,
                        "tool calls are forbidden by the current policy",
                        ("calls",),
                    ),
                )
            ),
            tuple(None for _ in calls),
        )

    if policy.choice.mode in {ToolChoiceMode.REQUIRED, ToolChoiceMode.NAMED} and not calls:
        issues.append(
            ValidationIssue(
                ValidationCode.TOOL_CALL_REQUIRED,
                "at least one tool call is required by the current policy",
                ("calls",),
            )
        )

    if not policy.allow_parallel and len(calls) > 1:
        issues.append(
            ValidationIssue(
                ValidationCode.PARALLEL_TOOL_CALLS_FORBIDDEN,
                "multiple tool calls are forbidden by the current policy",
                ("calls",),
            )
        )

    declared_tools = {tool.name: tool for tool in policy.tools}
    seen_call_ids: set[str] = set()
    canonical_arguments: list[str | None] = []

    for position, call in enumerate(calls):
        base_path = ("calls", position)

        if call.call_id in seen_call_ids:
            issues.append(
                ValidationIssue(
                    ValidationCode.DUPLICATE_TOOL_CALL_ID,
                    "tool call id must be unique within the assistant turn",
                    (*base_path, "call_id"),
                )
            )
        else:
            seen_call_ids.add(call.call_id)

        if call.index != position:
            issues.append(
                ValidationIssue(
                    ValidationCode.INVALID_TOOL_CALL_ORDER,
                    "tool call index must match zero-based tuple order",
                    (*base_path, "index"),
                )
            )

        tool = declared_tools.get(call.name)
        if tool is None:
            issues.append(
                ValidationIssue(
                    ValidationCode.UNDECLARED_TOOL,
                    f"tool {call.name!r} is not declared",
                    (*base_path, "name"),
                )
            )
            canonical_arguments.append(None)
            continue

        if policy.choice.mode is ToolChoiceMode.NAMED and call.name != policy.choice.name:
            issues.append(
                ValidationIssue(
                    ValidationCode.WRONG_NAMED_TOOL,
                    f"tool call must use named tool {policy.choice.name!r}",
                    (*base_path, "name"),
                )
            )

        try:
            arguments = parse_json_strict(call.arguments_json)
        except DuplicateJsonKeyError as exc:
            issues.append(
                ValidationIssue(
                    ValidationCode.DUPLICATE_JSON_KEY,
                    str(exc),
                    (*base_path, "arguments"),
                )
            )
            canonical_arguments.append(None)
            continue
        except InvalidJsonError as exc:
            issues.append(
                ValidationIssue(
                    ValidationCode.INVALID_JSON,
                    str(exc),
                    (*base_path, "arguments"),
                )
            )
            canonical_arguments.append(None)
            continue

        if not isinstance(arguments, dict):
            issues.append(
                ValidationIssue(
                    ValidationCode.JSON_VALUE_NOT_OBJECT,
                    "tool arguments must be a JSON object",
                    (*base_path, "arguments"),
                )
            )
            canonical_arguments.append(None)
            continue

        violations = _schema_violations(tool.parameters, arguments)
        for violation in violations:
            issues.append(
                ValidationIssue(
                    ValidationCode.SCHEMA_VALIDATION_FAILED,
                    violation.message,
                    (*base_path, "arguments", *violation.path),
                )
            )
        canonical_arguments.append(None if violations else canonical_json_dumps(arguments))

    return ToolCallsValidation(ValidationResult(tuple(issues)), tuple(canonical_arguments))


def validate_tool_history(items: tuple[CanonicalItem, ...]) -> ValidationResult:
    if not isinstance(items, tuple):
        raise TypeError("items must be a tuple")
    if not all(isinstance(item, CanonicalItem) for item in items):
        raise TypeError("items must contain only CanonicalItem values")

    issues: list[ValidationIssue] = []
    seen_calls: set[str] = set()
    resolved_calls: set[str] = set()

    for item_index, item in enumerate(items):
        if isinstance(item, ToolCallItem):
            if item.call_id in seen_calls:
                issues.append(
                    ValidationIssue(
                        ValidationCode.DUPLICATE_TOOL_CALL_ID,
                        "tool call id must be unique across canonical history",
                        ("items", item_index, "call_id"),
                    )
                )
            else:
                seen_calls.add(item.call_id)
            continue

        if not isinstance(item, ToolResultItem | MultimodalToolResultItem):
            continue

        if item.call_id not in seen_calls:
            issues.append(
                ValidationIssue(
                    ValidationCode.UNKNOWN_TOOL_RESULT,
                    "tool result must reference a preceding tool call",
                    ("items", item_index, "call_id"),
                )
            )
            continue

        if item.call_id in resolved_calls:
            issues.append(
                ValidationIssue(
                    ValidationCode.DUPLICATE_TOOL_RESULT,
                    "tool call may have at most one result",
                    ("items", item_index, "call_id"),
                )
            )
            continue

        resolved_calls.add(item.call_id)

    return ValidationResult(tuple(issues))
