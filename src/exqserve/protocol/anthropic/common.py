"""Common Anthropic Messages wire contracts and error mapping."""

from __future__ import annotations

from dataclasses import dataclass

from exqserve.core.errors import CanonicalError, ErrorCategory, public_error_code
from exqserve.core.usage import TokenUsage
from exqserve.serving.contracts import ServingRequest


@dataclass(frozen=True, slots=True)
class ParsedAnthropicRequest:
    serving: ServingRequest
    model: str
    stream: bool
    omit_thinking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.serving, ServingRequest):
            raise TypeError("serving must be a ServingRequest")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.stream, bool):
            raise TypeError("stream must be a bool")
        if not isinstance(self.omit_thinking, bool):
            raise TypeError("omit_thinking must be a bool")


class AnthropicProtocolError(Exception):
    def __init__(
        self,
        status_code: int,
        type: str,
        message: str,
        exqserve_code: str | None = None,
    ) -> None:
        if not isinstance(status_code, int) or isinstance(status_code, bool):
            raise TypeError("status_code must be an integer")
        if not 400 <= status_code <= 599:
            raise ValueError("status_code must be an HTTP error status")
        if not isinstance(type, str) or not type.strip():
            raise ValueError("type must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if exqserve_code is not None and (
            not isinstance(exqserve_code, str) or not exqserve_code.strip()
        ):
            raise ValueError("exqserve_code must be a non-empty string or None")
        self.status_code = status_code
        self.type = type
        self.message = message
        self.exqserve_code = exqserve_code
        super().__init__(message)

    def to_body(self, request_id: str) -> dict[str, object]:
        error: dict[str, object] = {"type": self.type, "message": self.message}
        if self.exqserve_code is not None:
            error["exqserve_code"] = self.exqserve_code
        return {
            "type": "error",
            "error": error,
            "request_id": request_id,
        }


def invalid_request(message: str) -> AnthropicProtocolError:
    return AnthropicProtocolError(400, "invalid_request_error", message)


def map_canonical_error(error: CanonicalError) -> AnthropicProtocolError:
    if not isinstance(error, CanonicalError):
        raise TypeError("error must be a CanonicalError")
    compatibility_code = public_error_code(error)
    diagnostic_code = compatibility_code or (error.code if error.cause is not None else None)
    if error.category is ErrorCategory.OVERLOADED:
        return AnthropicProtocolError(529, "overloaded_error", error.message, diagnostic_code)
    if error.category in {
        ErrorCategory.INVALID_REQUEST,
        ErrorCategory.UNSUPPORTED_CAPABILITY,
        ErrorCategory.CONTEXT_LENGTH,
    }:
        return AnthropicProtocolError(400, "invalid_request_error", error.message, diagnostic_code)
    return AnthropicProtocolError(500, "api_error", error.message, diagnostic_code)


def anthropic_usage(usage: TokenUsage | None) -> dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    input_total = usage.input_tokens or 0
    cached = usage.cached_input_tokens or 0
    result = {
        "input_tokens": max(0, input_total - cached),
        "output_tokens": usage.output_tokens or 0,
    }
    if usage.cached_input_tokens is not None:
        result["cache_read_input_tokens"] = cached
        result["cache_creation_input_tokens"] = 0
    return result
