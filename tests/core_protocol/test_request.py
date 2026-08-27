from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from exqserve.core.items import MessageItem, MessageRole, ReasoningItem
from exqserve.core.request import CanonicalRequest


def test_canonical_request_has_only_accepted_core_fields() -> None:
    request = CanonicalRequest(request_id="req-1", model="qwen", items=())

    assert [field.name for field in fields(request)] == ["request_id", "model", "items"]


@pytest.mark.parametrize(("request_id", "model", "match"), [("", "qwen", "request_id"), ("req-1", "   ", "model")])
def test_canonical_request_rejects_empty_identity_fields(
    request_id: str,
    model: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        CanonicalRequest(request_id=request_id, model=model, items=())


def test_canonical_request_preserves_item_order_exactly() -> None:
    first = MessageItem(role=MessageRole.SYSTEM, text="rules")
    second = MessageItem(role=MessageRole.USER, text="question")
    third = ReasoningItem(text="thinking")

    request = CanonicalRequest(
        request_id="req-1",
        model="qwen",
        items=(first, second, third),
    )

    assert request.items == (first, second, third)


def test_canonical_request_requires_immutable_tuple_items() -> None:
    with pytest.raises(TypeError, match="items"):
        CanonicalRequest(
            request_id="req-1",
            model="qwen",
            items=[MessageItem(role=MessageRole.USER, text="hello")],  # type: ignore[arg-type]
        )


def test_canonical_request_is_immutable() -> None:
    request = CanonicalRequest(request_id="req-1", model="qwen", items=())

    with pytest.raises(FrozenInstanceError):
        request.model = "other"  # type: ignore[misc]
