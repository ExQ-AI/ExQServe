"""OpenAI Chat Completions request codec over serving-core semantics."""

from __future__ import annotations

import time
import uuid

from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    ReasoningDelta,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import (
    ToolCallItem,
)
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.common import (
    OpenAIProtocolError,
    chat_usage,
    map_canonical_error,
    map_stream_canonical_error,
)


def _chat_id(value: str | None) -> str:
    if value is None:
        return f"chatcmpl-{uuid.uuid4().hex}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("response_id must be a non-empty string or None")
    return value


def _created(value: int | None) -> int:
    if value is None:
        return int(time.time())
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("created must be a non-negative integer or None")
    return value


def _finish_reason(reason: CompletionReason) -> str:
    if reason is CompletionReason.FILTER:
        return CompletionReason.STOP.value
    return str(reason.value)


def _cancel_error() -> OpenAIProtocolError:
    return OpenAIProtocolError(
        500,
        "server_error",
        "generation_cancelled",
        "Generation was cancelled.",
    )


class ChatStreamSerializer:
    def __init__(
        self,
        model: str,
        *,
        response_id: str | None = None,
        created: int | None = None,
        include_usage: bool = False,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(include_usage, bool):
            raise TypeError("include_usage must be a bool")
        self._model = model
        self._id = _chat_id(response_id)
        self._created = _created(created)
        self._include_usage = include_usage
        self._usage: TokenUsage | None = None
        self._terminal = False

    def _chunk(self, delta: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
        return {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def feed(self, event: GenerationEvent) -> tuple[dict[str, object], ...]:
        if self._terminal:
            return ()
        if isinstance(event, GenerationStarted):
            return (self._chunk({"role": "assistant", "content": ""}),)
        if isinstance(event, ReasoningDelta):
            return (self._chunk({"reasoning_content": event.text}),)
        if isinstance(event, TextDelta):
            return (self._chunk({"content": event.text}),)
        if isinstance(event, ToolCallStarted):
            return (
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": event.index,
                                "id": event.call_id,
                                "type": "function",
                                "function": {"name": event.name, "arguments": ""},
                            }
                        ]
                    }
                ),
            )
        if isinstance(event, ToolCallArgumentsDelta):
            return (
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": event.index,
                                "function": {"arguments": event.delta},
                            }
                        ]
                    }
                ),
            )
        if isinstance(event, UsageUpdated):
            self._usage = event.usage
            return ()
        if isinstance(event, GenerationCompleted):
            self._terminal = True
            usage = event.usage or self._usage
            chunks: list[dict[str, object]] = [self._chunk({}, _finish_reason(event.reason))]
            if self._include_usage and usage is not None:
                chunks.append(
                    {
                        "id": self._id,
                        "object": "chat.completion.chunk",
                        "created": self._created,
                        "model": self._model,
                        "choices": [],
                        "usage": chat_usage(usage),
                    }
                )
            return tuple(chunks)
        if isinstance(event, GenerationFailed):
            self._terminal = True
            return (map_stream_canonical_error(event.error).to_body(),)
        if isinstance(event, GenerationCancelled):
            self._terminal = True
            return (_cancel_error().to_body(),)
        return ()


class ChatAccumulator:
    def __init__(
        self,
        model: str,
        *,
        response_id: str | None = None,
        created: int | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self._model = model
        self._id = _chat_id(response_id)
        self._created = _created(created)
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, ToolCallItem] = {}
        self._usage: TokenUsage | None = None
        self._reason: str | None = None
        self._error: OpenAIProtocolError | None = None

    def consume(self, event: GenerationEvent) -> None:
        if isinstance(event, ReasoningDelta):
            self._reasoning.append(event.text)
        elif isinstance(event, TextDelta):
            self._text.append(event.text)
        elif isinstance(event, ToolCallCompleted):
            self._calls[event.call.index] = event.call
        elif isinstance(event, UsageUpdated):
            self._usage = event.usage
        elif isinstance(event, GenerationCompleted):
            self._usage = event.usage or self._usage
            self._reason = _finish_reason(event.reason)
        elif isinstance(event, GenerationFailed):
            self._error = map_canonical_error(event.error)
        elif isinstance(event, GenerationCancelled):
            self._error = _cancel_error()

    def result(self) -> dict[str, object]:
        if self._error is not None:
            raise self._error
        if self._reason is None:
            raise RuntimeError("Chat accumulation is not terminal")

        text = "".join(self._text)
        reasoning = "".join(self._reasoning)
        calls = [self._calls[index] for index in sorted(self._calls)]
        message: dict[str, object] = {
            "role": "assistant",
            "content": text if text else (None if calls else ""),
        }
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in calls
            ]

        result: dict[str, object] = {
            "id": self._id,
            "object": "chat.completion",
            "created": self._created,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self._reason,
                }
            ],
        }
        if self._usage is not None:
            result["usage"] = chat_usage(self._usage)
        return result
