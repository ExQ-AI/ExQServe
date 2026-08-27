from __future__ import annotations

import pytest

from exqserve.core.items import MessageItem, MessageRole
from exqserve.protocol.openai.common import OpenAIProtocolError
from exqserve.protocol.openai.responses import ResponsesRequestAdapter


def test_responses_parser_separates_transient_instructions_from_state_input() -> None:
    parsed = ResponsesRequestAdapter().parse(
        {
            "model": "m",
            "instructions": "NEW RULES",
            "input": "current",
            "max_output_tokens": 8,
            "previous_response_id": "resp_old",
            "store": True,
        },
        request_id="req",
    )

    assert parsed.previous_response_id == "resp_old"
    assert parsed.store is True
    assert parsed.instruction_items == (MessageItem(MessageRole.DEVELOPER, "NEW RULES"),)
    assert parsed.state_input_items == (MessageItem(MessageRole.USER, "current"),)

    previous = (
        MessageItem(MessageRole.USER, "old"),
        MessageItem(MessageRole.ASSISTANT, "answer"),
    )
    rebuilt = parsed.serving_with_context(previous)
    assert rebuilt.input.items == (
        MessageItem(MessageRole.DEVELOPER, "NEW RULES"),
        *previous,
        MessageItem(MessageRole.USER, "current"),
    )


def test_responses_store_defaults_true_and_previous_id_is_optional() -> None:
    parsed = ResponsesRequestAdapter(default_max_output_tokens=8).parse(
        {"model": "m", "input": "hello"},
        request_id="r",
    )
    assert parsed.store is True
    assert parsed.previous_response_id is None


def test_responses_store_false_is_accepted_but_conversation_remains_unsupported() -> None:
    parsed = ResponsesRequestAdapter(default_max_output_tokens=8).parse(
        {"model": "m", "input": "hello", "store": False},
        request_id="r",
    )
    assert parsed.store is False

    with pytest.raises(OpenAIProtocolError) as exc_info:
        ResponsesRequestAdapter(default_max_output_tokens=8).parse(
            {"model": "m", "input": "hello", "conversation": "conv"},
            request_id="r2",
        )
    assert exc_info.value.code == "unsupported_conversation"


def test_responses_invalid_previous_or_store_fail_explicitly() -> None:
    adapter = ResponsesRequestAdapter(default_max_output_tokens=8)
    with pytest.raises(OpenAIProtocolError) as exc_info:
        adapter.parse({"model": "m", "input": "x", "previous_response_id": ""}, request_id="r")
    assert exc_info.value.code == "invalid_previous_response_id"

    with pytest.raises(OpenAIProtocolError) as exc_info:
        adapter.parse({"model": "m", "input": "x", "store": "yes"}, request_id="r")
    assert exc_info.value.code == "invalid_store"
