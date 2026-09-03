from __future__ import annotations

import json

import pytest

from exqserve.core.errors import CanonicalError, ErrorCategory, FailureCause
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationFailed,
    GenerationStarted,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import ToolCallItem
from exqserve.core.usage import TokenUsage
from exqserve.protocol.anthropic.common import AnthropicProtocolError, map_canonical_error
from exqserve.protocol.anthropic.serialization import (
    AnthropicMessageAccumulator,
    AnthropicMessageStreamSerializer,
    anthropic_sse,
)


def _events():  # type: ignore[no-untyped-def]
    usage = TokenUsage(input_tokens=10, cached_input_tokens=4, output_tokens=7)
    call = ToolCallItem("toolu_1", "lookup", '{"id":1}', 0)
    return [
        GenerationStarted("req_1"),
        ReasoningStarted("req_1"),
        ReasoningDelta("req_1", "Need lookup"),
        ReasoningCompleted("req_1", "Need lookup"),
        TextStarted("req_1"),
        TextDelta("req_1", "Checking."),
        TextCompleted("req_1", "Checking."),
        ToolCallStarted("req_1", "toolu_1", "lookup", 0),
        ToolCallArgumentsDelta("req_1", "toolu_1", '{"id":1}', 0),
        ToolCallCompleted("req_1", call),
        UsageUpdated("req_1", usage),
        GenerationCompleted("req_1", CompletionReason.TOOL_CALLS, usage),
    ]


def test_nonstream_accumulator_builds_anthropic_content_and_usage() -> None:
    accumulator = AnthropicMessageAccumulator("local-qwen", message_id="msg_test")
    for event in _events():
        accumulator.consume(event)

    result = accumulator.result()
    assert result["id"] == "msg_test"
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["model"] == "local-qwen"
    assert result["stop_reason"] == "tool_use"
    assert result["stop_sequence"] is None
    content = result["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "thinking"
    assert content[0]["thinking"] == "Need lookup"
    assert content[0]["signature"].startswith("exqserve_")
    assert content[1] == {"type": "text", "text": "Checking."}
    assert content[2] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "lookup",
        "input": {"id": 1},
    }
    assert result["usage"] == {
        "input_tokens": 6,
        "output_tokens": 7,
        "cache_read_input_tokens": 4,
        "cache_creation_input_tokens": 0,
    }


def test_nonstream_accumulator_preserves_repeated_text_and_reasoning_blocks() -> None:
    call = ToolCallItem("toolu_1", "lookup", '{"id":1}', 0)
    events = (
        TextStarted("req_1"),
        TextDelta("req_1", "before"),
        TextCompleted("req_1", "before"),
        ToolCallStarted("req_1", "toolu_1", "lookup", 0),
        ToolCallArgumentsDelta("req_1", "toolu_1", '{"id":1}', 0),
        ToolCallCompleted("req_1", call),
        TextStarted("req_1"),
        TextDelta("req_1", "after"),
        TextCompleted("req_1", "after"),
        ReasoningStarted("req_1"),
        ReasoningDelta("req_1", "think1"),
        ReasoningCompleted("req_1", "think1"),
        TextStarted("req_1"),
        TextDelta("req_1", "middle"),
        TextCompleted("req_1", "middle"),
        ReasoningStarted("req_1"),
        ReasoningDelta("req_1", "think2"),
        ReasoningCompleted("req_1", "think2"),
        GenerationCompleted("req_1", CompletionReason.STOP),
    )
    accumulator = AnthropicMessageAccumulator("local-qwen", message_id="msg_segments")
    for event in events:
        accumulator.consume(event)

    content = accumulator.result()["content"]
    assert content[0] == {"type": "text", "text": "before"}
    assert content[1] == {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"id": 1}}
    assert content[2] == {"type": "text", "text": "after"}
    assert content[3]["type"] == "thinking"
    assert content[3]["thinking"] == "think1"
    assert content[4] == {"type": "text", "text": "middle"}
    assert content[5]["type"] == "thinking"
    assert content[5]["thinking"] == "think2"

    serializer = AnthropicMessageStreamSerializer("local-qwen", message_id="msg_segments_stream")
    payloads = [payload for event in events for payload in serializer.feed(event)]
    starts = [payload for name, payload in payloads if name == "content_block_start"]
    assert [payload["content_block"]["type"] for payload in starts] == [  # type: ignore[index]
        "text",
        "tool_use",
        "text",
        "thinking",
        "text",
        "thinking",
    ]


def test_stream_serializer_emits_anthropic_event_flow_and_tool_json_delta() -> None:
    serializer = AnthropicMessageStreamSerializer(
        "local-qwen", message_id="msg_stream", input_token_count=9
    )
    payloads: list[tuple[str, dict[str, object]]] = []
    for event in _events():
        payloads.extend(serializer.feed(event))

    names = [name for name, _ in payloads]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert payloads[0][1]["message"]["id"] == "msg_stream"  # type: ignore[index]
    assert payloads[0][1]["message"]["usage"] == {  # type: ignore[index]
        "input_tokens": 9,
        "output_tokens": 0,
    }
    assert payloads[1][1] == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    }
    assert payloads[2][1]["delta"] == {"type": "thinking_delta", "thinking": "Need lookup"}
    signature = payloads[3][1]["delta"]
    assert signature["type"] == "signature_delta"  # type: ignore[index]
    assert signature["signature"].startswith("exqserve_")  # type: ignore[index]
    assert payloads[8][1] == {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
    }
    assert payloads[9][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"id":1}',
    }
    assert payloads[-2][1] == {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {
            "input_tokens": 6,
            "output_tokens": 7,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 0,
        },
    }
    assert payloads[-1][1] == {"type": "message_stop"}


def test_anthropic_stream_defers_post_tool_text_until_tool_commit() -> None:
    serializer = AnthropicMessageStreamSerializer("local-qwen", message_id="msg_order_single")
    call = ToolCallItem("toolu_1", "lookup", '{"id":1}', 0)

    before_commit_events = (
        GenerationStarted("req_1"),
        ToolCallStarted("req_1", "toolu_1", "lookup", 0),
        ToolCallArgumentsDelta("req_1", "toolu_1", '{"id":1}', 0),
        TextStarted("req_1"),
        TextDelta("req_1", "after tool"),
        TextCompleted("req_1", "after tool"),
    )
    before_commit = [
        payload
        for event in before_commit_events
        for payload in serializer.feed(event)
    ]

    assert [name for name, _ in before_commit] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
    ]

    committed = serializer.feed(ToolCallCompleted("req_1", call))
    assert [name for name, _ in committed] == [
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    assert committed[0][1]["index"] == 0
    assert committed[1][1]["index"] == 1
    assert committed[1][1]["content_block"] == {"type": "text", "text": ""}  # type: ignore[index]
    assert committed[2][1]["delta"] == {"type": "text_delta", "text": "after tool"}  # type: ignore[index]


def test_anthropic_stream_serializes_multiple_tools_before_post_tool_content() -> None:
    serializer = AnthropicMessageStreamSerializer("local-qwen", message_id="msg_order_multi")
    first = ToolCallItem("toolu_1", "lookup", '{"id":1}', 0)
    second = ToolCallItem("toolu_2", "lookup", '{"id":2}', 1)

    before_commit_events = (
        GenerationStarted("req_1"),
        ToolCallStarted("req_1", "toolu_1", "lookup", 0),
        ToolCallArgumentsDelta("req_1", "toolu_1", '{"id":1}', 0),
        ToolCallStarted("req_1", "toolu_2", "lookup", 1),
        ToolCallArgumentsDelta("req_1", "toolu_2", '{"id":2}', 1),
        TextStarted("req_1"),
        TextDelta("req_1", "after tools"),
        TextCompleted("req_1", "after tools"),
        ReasoningStarted("req_1"),
        ReasoningDelta("req_1", "tail thought"),
        ReasoningCompleted("req_1", "tail thought"),
    )
    before_commit = [
        payload
        for event in before_commit_events
        for payload in serializer.feed(event)
    ]
    assert [name for name, _ in before_commit] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
    ]

    first_commit = serializer.feed(ToolCallCompleted("req_1", first))
    assert [name for name, _ in first_commit] == [
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
    ]
    assert [payload["index"] for _, payload in first_commit] == [0, 1, 1]

    second_commit = serializer.feed(ToolCallCompleted("req_1", second))
    names = [name for name, _ in second_commit]
    assert names == [
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
    ]
    starts = [payload for name, payload in second_commit if name == "content_block_start"]
    assert [payload["content_block"]["type"] for payload in starts] == ["text", "thinking"]  # type: ignore[index]
    stops = [payload["index"] for name, payload in (*first_commit, *second_commit) if name == "content_block_stop"]
    assert stops == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "terminal",
    (
        GenerationFailed(
            "req_1",
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_policy_violation",
                "Model output violated the requested tool policy.",
                False,
            ),
        ),
        GenerationCancelled("req_1"),
    ),
)
def test_anthropic_stream_abort_discards_queued_post_tool_content(
    terminal: GenerationFailed | GenerationCancelled,
) -> None:
    serializer = AnthropicMessageStreamSerializer("local-qwen", message_id="msg_order_abort")

    payloads = []
    for event in (
        GenerationStarted("req_1"),
        ToolCallStarted("req_1", "toolu_1", "lookup", 0),
        ToolCallArgumentsDelta("req_1", "toolu_1", '{"id":1}', 0),
        TextStarted("req_1"),
        TextDelta("req_1", "must not publish"),
        TextCompleted("req_1", "must not publish"),
        terminal,
    ):
        payloads.extend(serializer.feed(event))

    names = [name for name, _ in payloads]
    assert names[:3] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
    ]
    assert "content_block_stop" not in names
    assert sum(name == "content_block_start" for name in names) == 1
    assert names[-1] == "error"


@pytest.mark.parametrize(
    ("error", "status", "error_type", "fact_code"),
    [
        (
            CanonicalError(
                ErrorCategory.CONTEXT_LENGTH,
                "prompt_limit_exceeded",
                "Context is too long.",
                False,
            ),
            400,
            "invalid_request_error",
            "context_length_exceeded",
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_invalid",
                "Model produced an invalid tool call.",
                False,
                FailureCause.MODEL_TOOL_OUTPUT_INVALID,
            ),
            500,
            "api_error",
            "tool_call_invalid",
        ),
        (
            CanonicalError(
                ErrorCategory.OVERLOADED,
                "runtime_recovering",
                "Runtime is rebuilding.",
                True,
                FailureCause.RUNTIME_RECOVERING,
            ),
            529,
            "overloaded_error",
            "runtime_recovering",
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_incomplete",
                "Model output ended with an incomplete tool call.",
                False,
                FailureCause.OUTPUT_EOS,
            ),
            500,
            "api_error",
            "tool_call_incomplete",
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_incomplete",
                "Model output ended with an incomplete tool call.",
                False,
                FailureCause.OUTPUT_LENGTH,
            ),
            500,
            "api_error",
            "tool_call_incomplete",
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "protocol_ambiguity",
                "Model output ended at an ambiguous protocol boundary.",
                False,
                FailureCause.OUTPUT_EOS,
            ),
            500,
            "api_error",
            "protocol_ambiguity",
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "protocol_ambiguity",
                "Model output ended at an ambiguous protocol boundary.",
                False,
                FailureCause.OUTPUT_LENGTH,
            ),
            500,
            "api_error",
            "protocol_ambiguity",
        ),
        (
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "protocol_ambiguity",
                "Model output ended at an ambiguous protocol boundary.",
                False,
                FailureCause.PARSER_AMBIGUITY_LIMIT,
            ),
            500,
            "api_error",
            "protocol_ambiguity",
        ),
        (
            CanonicalError(
                ErrorCategory.RUNTIME_FAILURE,
                "restart_required",
                "Runtime restart is required.",
                False,
                FailureCause.RESTART_REQUIRED,
            ),
            500,
            "api_error",
            "restart_required",
        ),
    ],
)
def test_anthropic_projection_keeps_standard_identity_and_optional_diagnostic_code(
    error: CanonicalError,
    status: int,
    error_type: str,
    fact_code: str,
) -> None:
    mapped = map_canonical_error(error)
    assert mapped.status_code == status
    assert mapped.type == error_type
    assert mapped.exqserve_code == fact_code
    body = mapped.to_body("req_map")
    assert body["error"]["type"] == error_type  # type: ignore[index]
    assert body["error"]["message"] == error.message  # type: ignore[index]
    assert body["error"]["exqserve_code"] == fact_code  # type: ignore[index]

    serializer = AnthropicMessageStreamSerializer("local-qwen", message_id="msg_fact")
    stream_error = serializer.feed(GenerationFailed("req_stream", error))
    assert stream_error == (("error", mapped.to_body("req_stream")),)


def test_anthropic_model_failure_causes_do_not_become_hidden_retry_statuses() -> None:
    cases = (
        ("tool_call_incomplete", FailureCause.OUTPUT_EOS),
        ("tool_call_incomplete", FailureCause.OUTPUT_LENGTH),
        ("protocol_ambiguity", FailureCause.OUTPUT_EOS),
        ("protocol_ambiguity", FailureCause.OUTPUT_LENGTH),
        ("protocol_ambiguity", FailureCause.PARSER_AMBIGUITY_LIMIT),
        ("tool_call_invalid", FailureCause.MODEL_TOOL_OUTPUT_INVALID),
    )
    for code, cause in cases:
        source = CanonicalError(ErrorCategory.MODEL_FAILURE, code, "Model output failed.", False, cause)
        mapped = map_canonical_error(source)
        assert mapped.status_code == 500
        assert mapped.type == "api_error"
        assert mapped.message == source.message
        assert mapped.exqserve_code == code


def test_anthropic_protocol_error_omits_extension_without_machine_fact() -> None:
    error = AnthropicProtocolError(500, "api_error", "Ordinary failure.")
    assert error.exqserve_code is None
    assert error.to_body("req_plain") == {
        "type": "error",
        "error": {"type": "api_error", "message": "Ordinary failure."},
        "request_id": "req_plain",
    }


def test_anthropic_invalid_tool_candidate_leaves_tentative_block_open_then_errors() -> None:
    serializer = AnthropicMessageStreamSerializer("local-qwen", message_id="msg_invalid")
    events = (
        GenerationStarted("req_1"),
        ToolCallStarted("req_1", "toolu_1", "lookup", 0),
        ToolCallArgumentsDelta("req_1", "toolu_1", '{"id":"bad"}', 0),
        GenerationFailed(
            "req_1",
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_invalid",
                "Model produced an invalid tool call.",
                False,
            ),
        ),
    )
    payloads = [payload for event in events for payload in serializer.feed(event)]
    names = [name for name, _ in payloads]

    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert "content_block_stop" not in names
    assert "message_stop" not in names
    assert names[-1] == "error"
    assert payloads[-1][1]["type"] == "error"
    assert payloads[-1][1]["error"]["type"] == "api_error"  # type: ignore[index]


def test_anthropic_invalid_tool_candidate_nonstream_raises_instead_of_returning_tool_use() -> None:
    accumulator = AnthropicMessageAccumulator("local-qwen", message_id="msg_invalid")
    accumulator.consume(ToolCallStarted("req_1", "toolu_1", "lookup", 0))
    accumulator.consume(ToolCallArgumentsDelta("req_1", "toolu_1", '{"id":"bad"}', 0))
    accumulator.consume(
        GenerationFailed(
            "req_1",
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_invalid",
                "Model produced an invalid tool call.",
                False,
            ),
        )
    )
    with pytest.raises(AnthropicProtocolError) as exc_info:
        accumulator.result()
    assert exc_info.value.type == "api_error"
    assert exc_info.value.message == "Model produced an invalid tool call."


def test_omitted_thinking_hides_reasoning_text_but_preserves_signature() -> None:
    accumulator = AnthropicMessageAccumulator("local-qwen", message_id="msg_omit", omit_thinking=True)
    for event in (
        ReasoningStarted("req_1"),
        ReasoningDelta("req_1", "hidden"),
        ReasoningCompleted("req_1", "hidden"),
        GenerationCompleted("req_1", CompletionReason.STOP),
    ):
        accumulator.consume(event)

    result = accumulator.result()
    content = result["content"]
    assert isinstance(content, list)
    assert content[0]["thinking"] == ""
    assert content[0]["signature"].startswith("exqserve_")

    serializer = AnthropicMessageStreamSerializer(
        "local-qwen",
        message_id="msg_omit_stream",
        omit_thinking=True,
    )
    payloads: list[tuple[str, dict[str, object]]] = []
    for event in (
        GenerationStarted("req_1"),
        ReasoningStarted("req_1"),
        ReasoningDelta("req_1", "hidden"),
        ReasoningCompleted("req_1", "hidden"),
        GenerationCompleted("req_1", CompletionReason.STOP),
    ):
        payloads.extend(serializer.feed(event))
    assert not any(
        payload.get("delta", {}).get("type") == "thinking_delta"  # type: ignore[union-attr]
        for _, payload in payloads
    )
    assert any(
        payload.get("delta", {}).get("type") == "signature_delta"  # type: ignore[union-attr]
        for _, payload in payloads
    )


def test_stop_sequence_is_reported_in_nonstream_and_stream_results() -> None:
    terminal = GenerationCompleted("req_1", CompletionReason.STOP, stop_sequence="END")

    accumulator = AnthropicMessageAccumulator("local-qwen", message_id="msg_stop")
    accumulator.consume(terminal)
    result = accumulator.result()
    assert result["stop_reason"] == "stop_sequence"
    assert result["stop_sequence"] == "END"

    serializer = AnthropicMessageStreamSerializer("local-qwen", message_id="msg_stop_stream")
    payloads = serializer.feed(terminal)
    assert payloads[0][0] == "message_delta"
    assert payloads[0][1]["delta"] == {
        "stop_reason": "stop_sequence",
        "stop_sequence": "END",
    }


def test_anthropic_sse_uses_named_event_and_matching_json_data() -> None:
    rendered = anthropic_sse("message_stop", {"type": "message_stop"})
    assert rendered.startswith("event: message_stop\n")
    data_line = next(line for line in rendered.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: ")) == {"type": "message_stop"}
