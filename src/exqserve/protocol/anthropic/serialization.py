"""Anthropic Messages response accumulation and SSE serialization."""

from __future__ import annotations

import json
import uuid

from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
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
from exqserve.core.usage import TokenUsage
from exqserve.protocol.anthropic.common import (
    AnthropicProtocolError,
    anthropic_usage,
    map_canonical_error,
)


def _message_id(value: str | None) -> str:
    if value is None:
        return f"msg_{uuid.uuid4().hex}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("message_id must be a non-empty string or None")
    return value


def _local_signature(message_id: str) -> str:
    return f"exqserve_{message_id}_{uuid.uuid4().hex}"


def _stop_reason(reason: CompletionReason, stop_sequence: str | None) -> str:
    if stop_sequence is not None:
        return "stop_sequence"
    mapping = {
        CompletionReason.STOP: "end_turn",
        CompletionReason.LENGTH: "max_tokens",
        CompletionReason.TOOL_CALLS: "tool_use",
    }
    return mapping[reason]


def _cancel_error() -> AnthropicProtocolError:
    return AnthropicProtocolError(500, "api_error", "Generation was cancelled.")


def anthropic_sse(event: str, payload: dict[str, object]) -> str:
    if not isinstance(event, str) or not event.strip():
        raise ValueError("event must be a non-empty string")
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


class AnthropicMessageAccumulator:
    def __init__(
        self,
        model: str,
        *,
        message_id: str | None = None,
        omit_thinking: bool = False,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(omit_thinking, bool):
            raise TypeError("omit_thinking must be a bool")
        self._model = model
        self._id = _message_id(message_id)
        self._signature = _local_signature(self._id)
        self._omit_thinking = omit_thinking
        self._content: list[dict[str, object]] = []
        self._reasoning_index: int | None = None
        self._text_index: int | None = None
        self._tool_blocks: dict[int, int] = {}
        self._usage: TokenUsage | None = None
        self._reason: CompletionReason | None = None
        self._stop_sequence: str | None = None
        self._error: AnthropicProtocolError | None = None

    def _ensure_reasoning(self) -> int:
        if self._reasoning_index is None:
            self._reasoning_index = len(self._content)
            self._content.append(
                {"type": "thinking", "thinking": "", "signature": self._signature}
            )
        return self._reasoning_index

    def _ensure_text(self) -> int:
        if self._text_index is None:
            self._text_index = len(self._content)
            self._content.append({"type": "text", "text": ""})
        return self._text_index

    def consume(self, event: GenerationEvent) -> None:
        if isinstance(event, ReasoningStarted):
            self._ensure_reasoning()
        elif isinstance(event, ReasoningDelta):
            index = self._ensure_reasoning()
            if not self._omit_thinking:
                self._content[index]["thinking"] = str(self._content[index]["thinking"]) + event.text
        elif isinstance(event, ReasoningCompleted):
            index = self._ensure_reasoning()
            if not self._omit_thinking:
                self._content[index]["thinking"] = event.text
            self._reasoning_index = None
        elif isinstance(event, TextStarted):
            self._ensure_text()
        elif isinstance(event, TextDelta):
            index = self._ensure_text()
            self._content[index]["text"] = str(self._content[index]["text"]) + event.text
        elif isinstance(event, TextCompleted):
            index = self._ensure_text()
            self._content[index]["text"] = event.text
            self._text_index = None
        elif isinstance(event, ToolCallStarted):
            new_block_index = len(self._content)
            self._tool_blocks[event.index] = new_block_index
            self._content.append(
                {"type": "tool_use", "id": event.call_id, "name": event.name, "input": {}}
            )
        elif isinstance(event, ToolCallCompleted):
            block_index = self._tool_blocks.get(event.call.index)
            try:
                arguments = json.loads(event.call.arguments_json)
            except json.JSONDecodeError as exc:  # pragma: no cover - canonical parser invariant
                raise RuntimeError("canonical tool call arguments are not valid JSON") from exc
            if not isinstance(arguments, dict):  # pragma: no cover - tool validation invariant
                raise TypeError("canonical tool call arguments must be an object")
            if block_index is None:
                self._tool_blocks[event.call.index] = len(self._content)
                self._content.append(
                    {
                        "type": "tool_use",
                        "id": event.call.call_id,
                        "name": event.call.name,
                        "input": arguments,
                    }
                )
            else:
                self._content[block_index]["input"] = arguments
        elif isinstance(event, UsageUpdated):
            self._usage = event.usage
        elif isinstance(event, GenerationCompleted):
            self._usage = event.usage or self._usage
            self._reason = event.reason
            self._stop_sequence = event.stop_sequence
        elif isinstance(event, GenerationFailed):
            self._error = map_canonical_error(event.error)
        elif isinstance(event, GenerationCancelled):
            self._error = _cancel_error()

    def result(self) -> dict[str, object]:
        if self._error is not None:
            raise self._error
        if self._reason is None:
            raise RuntimeError("Anthropic Message accumulation is not terminal")
        return {
            "id": self._id,
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "content": self._content,
            "stop_reason": _stop_reason(self._reason, self._stop_sequence),
            "stop_sequence": self._stop_sequence,
            "usage": anthropic_usage(self._usage),
        }


class AnthropicMessageStreamSerializer:
    def __init__(
        self,
        model: str,
        *,
        message_id: str | None = None,
        omit_thinking: bool = False,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(omit_thinking, bool):
            raise TypeError("omit_thinking must be a bool")
        self._model = model
        self._id = _message_id(message_id)
        self._signature = _local_signature(self._id)
        self._omit_thinking = omit_thinking
        self._next_block_index = 0
        self._reasoning_block: int | None = None
        self._text_block: int | None = None
        self._tool_blocks: dict[int, int] = {}
        self._usage: TokenUsage | None = None
        self._terminal = False

    def _start_block(self, content_block: dict[str, object]) -> tuple[str, dict[str, object]]:
        index = self._next_block_index
        self._next_block_index += 1
        return (
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": content_block},
        )

    def feed(self, event: GenerationEvent) -> tuple[tuple[str, dict[str, object]], ...]:
        if self._terminal:
            return ()
        if isinstance(event, GenerationStarted):
            return (
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": self._id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self._model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": anthropic_usage(None),
                        },
                    },
                ),
            )
        if isinstance(event, ReasoningStarted):
            self._reasoning_block = self._next_block_index
            return (self._start_block({"type": "thinking", "thinking": "", "signature": ""}),)
        if isinstance(event, ReasoningDelta):
            if self._reasoning_block is None:
                self._reasoning_block = self._next_block_index
                start = self._start_block({"type": "thinking", "thinking": "", "signature": ""})
                if self._omit_thinking:
                    return (start,)
                delta = (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._reasoning_block,
                        "delta": {"type": "thinking_delta", "thinking": event.text},
                    },
                )
                return (start, delta)
            if self._omit_thinking:
                return ()
            return (
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._reasoning_block,
                        "delta": {"type": "thinking_delta", "thinking": event.text},
                    },
                ),
            )
        if isinstance(event, ReasoningCompleted):
            if self._reasoning_block is None:
                self._reasoning_block = self._next_block_index
                start = self._start_block({"type": "thinking", "thinking": "", "signature": ""})
                prefix: tuple[tuple[str, dict[str, object]], ...] = (start,)
            else:
                prefix = ()
            index = self._reasoning_block
            return (
                *prefix,
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "signature_delta", "signature": self._signature},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": index}),
            )
        if isinstance(event, TextStarted):
            self._text_block = self._next_block_index
            return (self._start_block({"type": "text", "text": ""}),)
        if isinstance(event, TextDelta):
            if self._text_block is None:
                self._text_block = self._next_block_index
                start = self._start_block({"type": "text", "text": ""})
                delta = (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._text_block,
                        "delta": {"type": "text_delta", "text": event.text},
                    },
                )
                return (start, delta)
            return (
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._text_block,
                        "delta": {"type": "text_delta", "text": event.text},
                    },
                ),
            )
        if isinstance(event, TextCompleted):
            if self._text_block is None:
                self._text_block = self._next_block_index
                start = self._start_block({"type": "text", "text": ""})
                prefix = (start,)
            else:
                prefix = ()
            return (
                *prefix,
                (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._text_block},
                ),
            )
        if isinstance(event, ToolCallStarted):
            new_tool_block_index = self._next_block_index
            self._tool_blocks[event.index] = new_tool_block_index
            return (
                self._start_block(
                    {"type": "tool_use", "id": event.call_id, "name": event.name, "input": {}}
                ),
            )
        if isinstance(event, ToolCallArgumentsDelta):
            argument_block_index = self._tool_blocks.get(event.index)
            if argument_block_index is None:  # pragma: no cover - canonical parser emits start first
                return ()
            return (
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": argument_block_index,
                        "delta": {"type": "input_json_delta", "partial_json": event.delta},
                    },
                ),
            )
        if isinstance(event, ToolCallCompleted):
            completed_block_index = self._tool_blocks.get(event.call.index)
            if completed_block_index is None:  # pragma: no cover - canonical parser emits start first
                return ()
            return (
                (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": completed_block_index},
                ),
            )
        if isinstance(event, UsageUpdated):
            self._usage = event.usage
            return ()
        if isinstance(event, GenerationCompleted):
            self._terminal = True
            usage = event.usage or self._usage
            return (
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": _stop_reason(event.reason, event.stop_sequence),
                            "stop_sequence": event.stop_sequence,
                        },
                        "usage": anthropic_usage(usage),
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            )
        if isinstance(event, GenerationFailed):
            self._terminal = True
            error = map_canonical_error(event.error)
            return (("error", error.to_body(event.request_id)),)
        if isinstance(event, GenerationCancelled):
            self._terminal = True
            return (("error", _cancel_error().to_body(event.request_id)),)
        return ()
