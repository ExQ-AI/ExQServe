"""Common OpenAI wire contracts and protocol-neutral value mappings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from exqserve.agent.reasoning import (
    ReasoningBudgetMode,
    ReasoningBudgetOverride,
    ReasoningEffort,
    ReasoningMode,
    ReasoningPolicy,
)
from exqserve.core.errors import CanonicalError, ErrorCategory, public_error_code
from exqserve.core.sampling import SamplingOverridePolicy
from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import RuntimeSamplingConfig
from exqserve.serving.contracts import ServingRequest


class OpenAIProtocol(str, Enum):
    CHAT = "chat"
    RESPONSES = "responses"


@dataclass(frozen=True, slots=True)
class ParsedOpenAIRequest:
    serving: ServingRequest
    model: str
    stream: bool
    protocol: OpenAIProtocol
    include_usage: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.serving, ServingRequest):
            raise TypeError("serving must be a ServingRequest")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.stream, bool):
            raise TypeError("stream must be a bool")
        if not isinstance(self.protocol, OpenAIProtocol):
            raise TypeError("protocol must be an OpenAIProtocol")
        if not isinstance(self.include_usage, bool):
            raise TypeError("include_usage must be a bool")


class OpenAIProtocolError(Exception):
    def __init__(
        self,
        status_code: int,
        type: str,
        code: str,
        message: str,
        param: str | None = None,
        exqserve_cause: str | None = None,
    ) -> None:
        if not isinstance(status_code, int) or isinstance(status_code, bool):
            raise TypeError("status_code must be an integer")
        if status_code < 400 or status_code > 599:
            raise ValueError("status_code must be an HTTP error status")
        for name, value in (("type", type), ("code", code), ("message", message)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if param is not None and not isinstance(param, str):
            raise TypeError("param must be a string or None")
        if exqserve_cause is not None and (
            not isinstance(exqserve_cause, str) or not exqserve_cause.strip()
        ):
            raise ValueError("exqserve_cause must be a non-empty string or None")
        self.status_code = status_code
        self.type = type
        self.code = code
        self.message = message
        self.param = param
        self.exqserve_cause = exqserve_cause
        super().__init__(message)

    def to_error_object(self, *, include_param: bool = True) -> dict[str, object]:
        error: dict[str, object] = {
            "message": self.message,
            "type": self.type,
            "code": self.code,
        }
        if include_param:
            error["param"] = self.param
        if self.exqserve_cause is not None:
            error["exqserve_cause"] = self.exqserve_cause
        return error

    def to_body(self) -> dict[str, object]:
        return {"error": self.to_error_object()}


def map_canonical_error(error: CanonicalError) -> OpenAIProtocolError:
    if not isinstance(error, CanonicalError):
        raise TypeError("error must be a CanonicalError")
    cause = None if error.cause is None else error.cause.value
    fact_code = public_error_code(error)
    wire_code = fact_code or error.code
    if error.category is ErrorCategory.OVERLOADED:
        status = 503 if error.code == "runtime_recovering" else 429
        return OpenAIProtocolError(status, "rate_limit_error", wire_code, error.message, None, cause)
    if error.category is ErrorCategory.CONTEXT_LENGTH:
        return OpenAIProtocolError(
            400,
            "invalid_request_error",
            wire_code,
            "Request exceeds the model context window.",
            None,
            cause,
        )
    if error.category in {
        ErrorCategory.INVALID_REQUEST,
        ErrorCategory.UNSUPPORTED_CAPABILITY,
    }:
        return OpenAIProtocolError(400, "invalid_request_error", wire_code, error.message, None, cause)
    return OpenAIProtocolError(500, "server_error", wire_code, error.message, None, cause)


def map_stream_canonical_error(error: CanonicalError) -> OpenAIProtocolError:
    """Map an HTTP-200 stream failure without client-specific message encoding."""

    return map_canonical_error(error)


def _usage_counts(usage: TokenUsage) -> tuple[int, int, int]:
    if not isinstance(usage, TokenUsage):
        raise TypeError("usage must be TokenUsage")
    input_count = usage.input_tokens if usage.input_tokens is not None else 0
    output_count = usage.output_tokens if usage.output_tokens is not None else 0
    return input_count, output_count, input_count + output_count


def chat_usage(usage: TokenUsage) -> dict[str, object]:
    input_count, output_count, total_count = _usage_counts(usage)
    cached_count = usage.cached_input_tokens
    result: dict[str, object] = {
        "prompt_tokens": input_count,
        "completion_tokens": output_count,
        "total_tokens": total_count,
    }
    if cached_count is not None:
        result["prompt_tokens_details"] = {"cached_tokens": cached_count}
    return result


def responses_usage(usage: TokenUsage) -> dict[str, object]:
    input_count, output_count, total_count = _usage_counts(usage)
    cached_count = usage.cached_input_tokens
    result: dict[str, object] = {
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": total_count,
    }
    if cached_count is not None:
        result["input_tokens_details"] = {"cached_tokens": cached_count}
    return result


def invalid_request(code: str, message: str, param: str | None = None) -> OpenAIProtocolError:
    return OpenAIProtocolError(400, "invalid_request_error", code, message, param)


def parse_reasoning_budget(body: dict[str, object]) -> ReasoningBudgetOverride:
    aliases = ("reasoning_budget_tokens", "thinking_token_budget")
    supplied = [(name, body[name]) for name in aliases if name in body]
    message_present = "reasoning_budget_message" in body
    raw_message = body.get("reasoning_budget_message")
    if message_present and not isinstance(raw_message, str):
        raise invalid_request(
            "invalid_reasoning_budget_message",
            "reasoning_budget_message must be a string.",
            "reasoning_budget_message",
        )
    message = raw_message if isinstance(raw_message, str) else None
    if not supplied:
        if message_present:
            raise invalid_request(
                "reasoning_budget_message_requires_budget",
                "reasoning_budget_message requires an explicit reasoning budget.",
                "reasoning_budget_message",
            )
        return ReasoningBudgetOverride()

    parsed: list[tuple[str, int]] = []
    for name, value in supplied:
        if not isinstance(value, int) or isinstance(value, bool):
            raise invalid_request(
                "invalid_reasoning_budget",
                f"{name} must be an integer.",
                name,
            )
        if value < -1:
            raise invalid_request(
                "invalid_reasoning_budget",
                f"{name} must be -1 or a non-negative integer.",
                name,
            )
        parsed.append((name, value))
    values = {value for _, value in parsed}
    if len(values) != 1:
        raise invalid_request(
            "conflicting_reasoning_budget",
            "Reasoning budget aliases must not contain conflicting values.",
            parsed[-1][0],
        )
    value = parsed[0][1]
    if value == -1:
        if message_present:
            raise invalid_request(
                "reasoning_budget_message_with_disabled_budget",
                "reasoning_budget_message cannot be used when the reasoning budget is disabled.",
                "reasoning_budget_message",
            )
        return ReasoningBudgetOverride(ReasoningBudgetMode.DISABLE)
    return ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, value, message)


def parse_reasoning_effort(value: object, *, param: str) -> ReasoningPolicy:
    if value is None:
        return ReasoningPolicy()
    if not isinstance(value, str):
        raise invalid_request("invalid_reasoning_effort", "Reasoning effort must be a string.", param)
    normalized = value.lower()
    if normalized in {"none", "disabled"}:
        return ReasoningPolicy(ReasoningMode.DISABLED)
    effort_map = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
        "xhigh": ReasoningEffort.XHIGH,
        "max": ReasoningEffort.MAXIMUM,
        "maximum": ReasoningEffort.MAXIMUM,
    }
    effort = effort_map.get(normalized)
    if effort is None:
        raise invalid_request("invalid_reasoning_effort", "Unsupported reasoning effort.", param)
    return ReasoningPolicy(ReasoningMode.ENABLED, effort)


_FULL_CONTEXT_PENALTY_RANGE = 100_000_000


def _parse_logit_bias(value: object) -> tuple[tuple[int, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise TypeError("logit_bias must be an object")

    parsed: list[tuple[int, float]] = []
    seen_token_ids: set[int] = set()
    for raw_token_id, raw_bias in value.items():
        if isinstance(raw_token_id, bool):
            raise TypeError("logit_bias token ids must be integers")
        if isinstance(raw_token_id, int):
            token_id = raw_token_id
        elif isinstance(raw_token_id, str):
            try:
                token_id = int(raw_token_id)
            except ValueError as exc:
                raise ValueError("logit_bias token ids must be decimal integers") from exc
        else:
            raise TypeError("logit_bias token ids must be integers")
        if token_id < 0 or token_id in seen_token_ids:
            raise ValueError("logit_bias token ids must be unique and non-negative")
        if not isinstance(raw_bias, int | float) or isinstance(raw_bias, bool):
            raise TypeError("logit_bias values must be numbers")
        bias = float(raw_bias)
        if not math.isfinite(bias):
            raise ValueError("logit_bias values must be finite")
        if not -100 <= bias <= 100:
            raise ValueError("logit_bias values must be between -100 and 100")
        seen_token_ids.add(token_id)
        parsed.append((token_id, bias))
    return tuple(parsed)


def _parse_penalty_range(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("penalty_range must be an integer")
    if value < -1:
        raise ValueError("penalty_range must be -1 or non-negative")
    return _FULL_CONTEXT_PENALTY_RANGE if value == -1 else value


def parse_stop(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value:
            raise invalid_request("invalid_stop", "stop strings must not be empty.", "stop")
        return (value,)
    if isinstance(value, list):
        if not 1 <= len(value) <= 4 or not all(isinstance(item, str) and item for item in value):
            raise invalid_request(
                "invalid_stop",
                "stop must contain between one and four non-empty strings.",
                "stop",
            )
        return tuple(value)
    raise invalid_request(
        "invalid_stop",
        "stop must be a string or a list of up to four strings.",
        "stop",
    )


_OVERRIDE_BODY_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty_range": "penalty_range",
    "repetition_decay": "repetition_decay",
    "temperature_last": "temperature_last",
    "adaptive_target": "adaptive_target",
    "adaptive_decay": "adaptive_decay",
    "logit_bias": "logit_bias",
}


def _apply_sampling_overrides(
    body: dict[str, object],
    policy: SamplingOverridePolicy | None,
) -> dict[str, object]:
    if policy is None or not policy.enabled:
        return body
    effective = dict(body)
    for override in policy.overrides:
        body_field = _OVERRIDE_BODY_FIELDS[override.field]
        if override.force or body_field not in body:
            value: object = override.value
            if override.field == "logit_bias":
                assert isinstance(value, tuple)
                value = dict(value)
            effective[body_field] = value
    return effective


def parse_sampling(
    body: dict[str, object],
    policy: SamplingOverridePolicy | None = None,
) -> RuntimeSamplingConfig | None:
    body = _apply_sampling_overrides(body, policy)
    names = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "penalty_range",
        "repetition_decay",
        "temperature_last",
        "adaptive_target",
        "adaptive_decay",
        "logit_bias",
    )
    if not any(name in body for name in names):
        return None
    try:
        return RuntimeSamplingConfig(
            temperature=body.get("temperature", 1.0),  # type: ignore[arg-type]
            min_p=body.get("min_p", 0.0),  # type: ignore[arg-type]
            top_k=body.get("top_k", 0),  # type: ignore[arg-type]
            top_p=body.get("top_p", 1.0),  # type: ignore[arg-type]
            repetition_penalty=body.get("repetition_penalty", 1.0),  # type: ignore[arg-type]
            frequency_penalty=body.get("frequency_penalty", 0.0),  # type: ignore[arg-type]
            presence_penalty=body.get("presence_penalty", 0.0),  # type: ignore[arg-type]
            temperature_last=body.get("temperature_last", False),  # type: ignore[arg-type]
            repetition_penalty_range=_parse_penalty_range(body.get("penalty_range", -1)),
            repetition_decay=body.get("repetition_decay", 0),  # type: ignore[arg-type]
            adaptive_target=body.get("adaptive_target", 1.0),  # type: ignore[arg-type]
            adaptive_decay=body.get("adaptive_decay", 0.9),  # type: ignore[arg-type]
            logit_bias=_parse_logit_bias(body.get("logit_bias")),
        )
    except (TypeError, ValueError) as exc:
        raise invalid_request("invalid_sampling", "Sampling parameters are invalid.") from exc
