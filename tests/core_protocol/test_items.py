from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from exqserve.core.items import (
    CanonicalItem,
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)


def test_message_roles_keep_developer_distinct() -> None:
    assert {role.value for role in MessageRole} == {
        "system",
        "developer",
        "user",
        "assistant",
    }
    assert MessageRole.DEVELOPER is not MessageRole.SYSTEM


def test_canonical_item_union_is_exact_v1_set() -> None:
    assert set(get_args(CanonicalItem)) == {
        MessageItem,
        MultimodalMessageItem,
        MultimodalToolResultItem,
        ReasoningItem,
        ToolCallItem,
        ToolResultItem,
    }


def test_message_and_reasoning_items_preserve_text() -> None:
    message = MessageItem(role=MessageRole.USER, text="hello")
    reasoning = ReasoningItem(text="considering options")

    assert message.role is MessageRole.USER
    assert message.text == "hello"
    assert reasoning.text == "considering options"


def test_multimodal_user_item_preserves_order_and_image_detail() -> None:
    item = MultimodalMessageItem(
        MessageRole.USER,
        (
            TextContentPart("before"),
            ImageContentPart("data:image/png;base64,AA==", "high"),
            TextContentPart("after"),
        ),
    )
    assert item.parts[0] == TextContentPart("before")
    assert item.parts[1] == ImageContentPart("data:image/png;base64,AA==", "high")
    with pytest.raises(ValueError, match="user role"):
        MultimodalMessageItem(MessageRole.ASSISTANT, item.parts)


def test_tool_call_preserves_json_text_without_parsing_it() -> None:
    call = ToolCallItem(
        call_id="call-1",
        name="bash",
        arguments_json='{"command":"echo hi"}',
        index=0,
    )

    assert call.arguments_json == '{"command":"echo hi"}'


@pytest.mark.parametrize(
    ("call_id", "name", "index", "match"),
    [
        ("", "bash", 0, "call_id"),
        ("call-1", "   ", 0, "name"),
        ("call-1", "bash", -1, "index"),
    ],
)
def test_tool_call_rejects_invalid_local_fields(
    call_id: str,
    name: str,
    index: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ToolCallItem(
            call_id=call_id,
            name=name,
            arguments_json="{}",
            index=index,
        )


def test_tool_result_requires_call_id_but_preserves_tool_error_semantics() -> None:
    result = ToolResultItem(call_id="call-1", text="command failed", is_error=True)

    assert result.is_error is True
    assert result.text == "command failed"

    with pytest.raises(ValueError, match="call_id"):
        ToolResultItem(call_id="   ", text="anything")


def test_items_are_immutable() -> None:
    message = MessageItem(role=MessageRole.SYSTEM, text="rules")

    with pytest.raises(FrozenInstanceError):
        message.text = "changed"  # type: ignore[misc]
