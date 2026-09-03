"""Legacy OpenAI Completions adapter over protocol-neutral raw prompts."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from exqserve.core.events import (
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    TextDelta,
    UsageUpdated,
)
from exqserve.core.items import RawPromptItem
from exqserve.core.request import RawPromptRequest
from exqserve.core.sampling import SamplingOverridePolicy
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.common import (
    OpenAIProtocolError,
    chat_usage,
    invalid_request,
    map_canonical_error,
    parse_sampling,
    parse_stop,
)
from exqserve.serving.contracts import RawServingRequest


def _parse_prompt(value: object) -> RawPromptItem:
    if value is None:
        return RawPromptItem(text="")
    if isinstance(value, str):
        return RawPromptItem(text=value)
    if isinstance(value, list) and value and all(
        isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
        for token_id in value
    ):
        return RawPromptItem(token_ids=tuple(value))
    raise invalid_request(
        "unsupported_prompt_form",
        "Only one string prompt or one flat token-id array is supported.",
        "prompt",
    )


def _positive_max_tokens(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise invalid_request("invalid_max_tokens", "max_tokens must be a positive integer.", "max_tokens")
    return value


def _parse_bool(body: dict[str, object], name: str, default: bool = False) -> bool:
    value = body.get(name, default)
    if not isinstance(value, bool):
        raise invalid_request(f"invalid_{name}", f"{name} must be boolean.", name)
    return value


def _reject_unsupported(body: dict[str, object]) -> None:
    n = body.get("n", 1)
    if isinstance(n, bool) or n != 1:
        raise invalid_request("unsupported_n", "Only n=1 is supported.", "n")
    best_of = body.get("best_of", 1)
    if isinstance(best_of, bool) or best_of not in {None, 1}:
        raise invalid_request("unsupported_best_of", "best_of greater than 1 is not supported.", "best_of")
    if body.get("logprobs") is not None:
        raise invalid_request("unsupported_logprobs", "logprobs are not supported.", "logprobs")
    if body.get("suffix") is not None:
        raise invalid_request("unsupported_suffix", "suffix is not supported.", "suffix")


@dataclass(frozen=True, slots=True)
class ParsedCompletionsRequest:
    raw: RawServingRequest
    model: str
    stream: bool
    echo: bool
    include_usage: bool


class CompletionsRequestAdapter:
    def __init__(self, sampling_overrides: SamplingOverridePolicy | None = None) -> None:
        if sampling_overrides is not None and not isinstance(sampling_overrides, SamplingOverridePolicy):
            raise TypeError("sampling_overrides must be SamplingOverridePolicy or None")
        self._sampling_overrides = sampling_overrides

    def parse(self, body: dict[str, object], *, request_id: str) -> ParsedCompletionsRequest:
        if not isinstance(body, dict):
            raise TypeError("body must be a dictionary")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise invalid_request("invalid_model", "model must be a non-empty string.", "model")

        _reject_unsupported(body)
        prompt = _parse_prompt(body.get("prompt"))
        stream = _parse_bool(body, "stream")
        echo = _parse_bool(body, "echo")
        if echo and prompt.token_ids is not None:
            raise invalid_request(
                "unsupported_echo_token_prompt",
                "echo=true is supported only for text prompts.",
                "echo",
            )

        include_usage = False
        stream_options = body.get("stream_options")
        if stream_options is not None:
            if not isinstance(stream_options, dict):
                raise invalid_request("invalid_stream_options", "stream_options must be an object.", "stream_options")
            include_value = stream_options.get("include_usage", False)
            if not isinstance(include_value, bool):
                raise invalid_request(
                    "invalid_stream_options",
                    "stream_options.include_usage must be boolean.",
                    "stream_options.include_usage",
                )
            include_usage = include_value

        seed = body.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise invalid_request("invalid_seed", "seed must be an integer.", "seed")

        raw = RawServingRequest(
            RawPromptRequest(request_id, model, (prompt,)),
            _positive_max_tokens(body.get("max_tokens")),
            parse_stop(body.get("stop")),
            seed,
            parse_sampling(body, self._sampling_overrides),
        )
        return ParsedCompletionsRequest(raw, model, stream, echo, include_usage)


def _completion_id(value: str | None) -> str:
    if value is None:
        return f"cmpl-{uuid.uuid4().hex}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("response_id must be a non-empty string or None")
    return value


def _created(value: int | None) -> int:
    if value is None:
        return int(time.time())
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("created must be a non-negative integer or None")
    return value


def _cancel_error() -> OpenAIProtocolError:
    return OpenAIProtocolError(
        500,
        "server_error",
        "generation_cancelled",
        "Generation was cancelled.",
    )


class CompletionsStreamSerializer:
    def __init__(
        self,
        model: str,
        *,
        response_id: str | None = None,
        created: int | None = None,
        echo_text: str = "",
        include_usage: bool = False,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(echo_text, str):
            raise TypeError("echo_text must be a string")
        if not isinstance(include_usage, bool):
            raise TypeError("include_usage must be a bool")
        self._model = model
        self._id = _completion_id(response_id)
        self._created = _created(created)
        self._echo_text = echo_text
        self._include_usage = include_usage
        self._usage: TokenUsage | None = None
        self._terminal = False

    def _chunk(self, text: str, finish_reason: str | None) -> dict[str, object]:
        return {
            "id": self._id,
            "object": "text_completion",
            "created": self._created,
            "model": self._model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def feed(self, event: GenerationEvent) -> tuple[dict[str, object], ...]:
        if self._terminal:
            return ()
        if isinstance(event, GenerationStarted):
            return (self._chunk(self._echo_text, None),) if self._echo_text else ()
        if isinstance(event, TextDelta):
            return (self._chunk(event.text, None),)
        if isinstance(event, UsageUpdated):
            self._usage = event.usage
            return ()
        if isinstance(event, GenerationCompleted):
            self._terminal = True
            usage = event.usage or self._usage
            chunks: list[dict[str, object]] = [self._chunk("", event.reason.value)]
            if self._include_usage and usage is not None:
                chunks.append(
                    {
                        "id": self._id,
                        "object": "text_completion",
                        "created": self._created,
                        "model": self._model,
                        "choices": [],
                        "usage": chat_usage(usage),
                    }
                )
            return tuple(chunks)
        if isinstance(event, GenerationFailed):
            self._terminal = True
            return (map_canonical_error(event.error).to_body(),)
        if isinstance(event, GenerationCancelled):
            self._terminal = True
            return (_cancel_error().to_body(),)
        return ()


class CompletionsAccumulator:
    def __init__(
        self,
        model: str,
        *,
        response_id: str | None = None,
        created: int | None = None,
        echo_text: str = "",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(echo_text, str):
            raise TypeError("echo_text must be a string")
        self._model = model
        self._id = _completion_id(response_id)
        self._created = _created(created)
        self._echo_text = echo_text
        self._parts: list[str] = []
        self._usage: TokenUsage | None = None
        self._reason: str | None = None
        self._error: OpenAIProtocolError | None = None

    def consume(self, event: GenerationEvent) -> None:
        if isinstance(event, TextDelta):
            self._parts.append(event.text)
        elif isinstance(event, UsageUpdated):
            self._usage = event.usage
        elif isinstance(event, GenerationCompleted):
            self._usage = event.usage or self._usage
            self._reason = event.reason.value
        elif isinstance(event, GenerationFailed):
            self._error = map_canonical_error(event.error)
        elif isinstance(event, GenerationCancelled):
            self._error = _cancel_error()

    def result(self) -> dict[str, object]:
        if self._error is not None:
            raise self._error
        if self._reason is None:
            raise RuntimeError("Completions accumulation is not terminal")
        result: dict[str, object] = {
            "id": self._id,
            "object": "text_completion",
            "created": self._created,
            "model": self._model,
            "choices": [
                {
                    "text": self._echo_text + "".join(self._parts),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": self._reason,
                }
            ],
        }
        if self._usage is not None:
            result["usage"] = chat_usage(self._usage)
        return result
