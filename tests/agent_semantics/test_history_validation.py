from __future__ import annotations

from exqserve.agent.validation import ValidationCode, validate_tool_history
from exqserve.core.items import (
    MessageItem,
    MessageRole,
    ReasoningItem,
    ToolCallItem,
    ToolResultItem,
)


def _call(call_id: str, index: int = 0) -> ToolCallItem:
    return ToolCallItem(call_id, "bash", "{}", index)


def test_matching_call_and_result_is_valid() -> None:
    result = validate_tool_history((_call("call-1"), ToolResultItem("call-1", "ok")))

    assert result.is_valid


def test_unknown_result_must_reference_preceding_call() -> None:
    result = validate_tool_history((ToolResultItem("missing", "result"),))

    assert [issue.code for issue in result.issues] == [ValidationCode.UNKNOWN_TOOL_RESULT]
    assert result.issues[0].path == ("items", 0, "call_id")


def test_later_call_does_not_retroactively_validate_earlier_unknown_result() -> None:
    result = validate_tool_history((ToolResultItem("call-1", "early"), _call("call-1")))

    assert [issue.code for issue in result.issues] == [ValidationCode.UNKNOWN_TOOL_RESULT]
    assert result.issues[0].path == ("items", 0, "call_id")


def test_duplicate_result_is_rejected_at_later_result() -> None:
    items = (
        _call("call-1"),
        ToolResultItem("call-1", "first"),
        ToolResultItem("call-1", "second"),
    )

    result = validate_tool_history(items)

    assert [issue.code for issue in result.issues] == [ValidationCode.DUPLICATE_TOOL_RESULT]
    assert result.issues[0].path == ("items", 2, "call_id")


def test_duplicate_call_id_is_rejected_across_history() -> None:
    result = validate_tool_history((_call("same", 0), _call("same", 1)))

    assert [issue.code for issue in result.issues] == [ValidationCode.DUPLICATE_TOOL_CALL_ID]
    assert result.issues[0].path == ("items", 1, "call_id")


def test_unresolved_call_at_end_of_history_is_valid() -> None:
    assert validate_tool_history((_call("call-1"),)).is_valid


def test_parallel_results_may_arrive_in_different_order() -> None:
    items = (
        _call("call-1", 0),
        _call("call-2", 1),
        ToolResultItem("call-2", "second first"),
        ToolResultItem("call-1", "first second"),
    )

    assert validate_tool_history(items).is_valid


def test_error_tool_result_still_resolves_call() -> None:
    items = (_call("call-1"), ToolResultItem("call-1", "failed", is_error=True))

    assert validate_tool_history(items).is_valid


def test_message_and_reasoning_items_are_transparent() -> None:
    items = (
        MessageItem(MessageRole.USER, "do it"),
        ReasoningItem("thinking"),
        _call("call-1"),
        MessageItem(MessageRole.ASSISTANT, "tool requested"),
        ToolResultItem("call-1", "ok"),
    )

    assert validate_tool_history(items).is_valid


def test_history_issues_follow_canonical_item_order_deterministically() -> None:
    items = (
        ToolResultItem("missing", "unknown"),
        _call("same", 0),
        _call("same", 1),
        ToolResultItem("same", "first"),
        ToolResultItem("same", "duplicate"),
    )

    first = validate_tool_history(items)
    second = validate_tool_history(items)

    assert first == second
    assert [issue.code for issue in first.issues] == [
        ValidationCode.UNKNOWN_TOOL_RESULT,
        ValidationCode.DUPLICATE_TOOL_CALL_ID,
        ValidationCode.DUPLICATE_TOOL_RESULT,
    ]
    assert [issue.path for issue in first.issues] == [
        ("items", 0, "call_id"),
        ("items", 2, "call_id"),
        ("items", 4, "call_id"),
    ]
