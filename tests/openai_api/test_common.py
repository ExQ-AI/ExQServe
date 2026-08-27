from __future__ import annotations

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.common import (
    OpenAIProtocol,
    OpenAIProtocolError,
    ParsedOpenAIRequest,
    chat_usage,
    map_canonical_error,
    parse_sampling,
    responses_usage,
)
from exqserve.serving.contracts import ServingRequest


def _parsed() -> ParsedOpenAIRequest:
    serving = ServingRequest(
        CanonicalRequest("req", "model", (MessageItem(MessageRole.USER, "hello"),)),
        ReasoningPolicy(),
        ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True),
        max_output_tokens=8,
    )
    return ParsedOpenAIRequest(serving, "model", False, OpenAIProtocol.CHAT, False)


def test_parsed_openai_request_is_protocol_neutral_serving_wrapper() -> None:
    parsed = _parsed()
    assert parsed.serving.input.request_id == "req"
    assert parsed.protocol is OpenAIProtocol.CHAT
    assert parsed.stream is False


def test_protocol_error_serializes_openai_error_envelope() -> None:
    error = OpenAIProtocolError(400, "invalid_request_error", "bad_field", "Bad field.", "foo")
    assert error.to_body() == {
        "error": {
            "message": "Bad field.",
            "type": "invalid_request_error",
            "param": "foo",
            "code": "bad_field",
        }
    }
    assert str(error) == "Bad field."


def test_canonical_error_mapping_preserves_safe_message_and_category_status() -> None:
    overloaded = CanonicalError(
        ErrorCategory.OVERLOADED,
        "server_overloaded",
        "Server is at capacity.",
        retryable=True,
    )
    mapped = map_canonical_error(overloaded)
    assert mapped.status_code == 429
    assert mapped.type == "rate_limit_error"
    assert mapped.code == "server_overloaded"
    assert mapped.message == "Server is at capacity."

    failure = map_canonical_error(
        CanonicalError(ErrorCategory.RUNTIME_FAILURE, "runtime_failed", "Runtime failed.", False)
    )
    assert failure.status_code == 500
    assert failure.type == "server_error"


def test_sampling_accepts_local_sampler_extensions() -> None:
    sampling = parse_sampling(
        {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.08,
            "repetition_penalty": 1.0,
            "penalty_range": -1,
            "repetition_decay": 64,
            "temperature_last": True,
            "adaptive_target": 0.5,
            "adaptive_decay": 0.8,
            "logit_bias": {"10": 5.0, "20": -4.0},
        }
    )
    assert sampling is not None
    assert sampling.temperature == 0.6
    assert sampling.top_p == 0.95
    assert sampling.top_k == 20
    assert sampling.min_p == 0.08
    assert sampling.repetition_penalty == 1.0
    assert sampling.repetition_penalty_range == 100_000_000
    assert sampling.repetition_decay == 64
    assert sampling.adaptive_target == 0.5
    assert sampling.adaptive_decay == 0.8
    assert sampling.logit_bias == ((10, 5.0), (20, -4.0))
    assert sampling.temperature_last is True

    for invalid in (
        {"temperature_last": 1},
        {"penalty_range": -2},
        {"repetition_decay": -1},
        {"adaptive_target": 1.1},
        {"adaptive_decay": 1.0},
        {"logit_bias": {"not-a-token": 1.0}},
        {"logit_bias": {"10": 101.0}},
    ):
        with pytest.raises(OpenAIProtocolError) as exc_info:
            parse_sampling(invalid)
        assert exc_info.value.code == "invalid_sampling"


def test_usage_mapping_includes_measured_cached_tokens_without_invention() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=4, cached_input_tokens=7)
    assert chat_usage(usage) == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 7},
    }
    assert responses_usage(usage) == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "input_tokens_details": {"cached_tokens": 7},
    }

    unknown = TokenUsage(input_tokens=10)
    assert "prompt_tokens_details" not in chat_usage(unknown)
    assert "input_tokens_details" not in responses_usage(unknown)
