"""Opt-in versioned canonical capture and replay."""

from __future__ import annotations

import asyncio
import json
import math
from enum import Enum
from pathlib import Path
from typing import Protocol, cast

from exqserve.core.errors import CanonicalError, ErrorCategory
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
    TimingUpdated,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import (
    CanonicalItem,
    ImageContentPart,
    MessageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    RawPromptItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest, RawPromptRequest
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage

_SCHEMA_VERSION = 1


class CaptureMode(str, Enum):
    OFF = "off"
    METADATA = "metadata"
    FULL = "full"


class CaptureSink(Protocol):
    async def write(self, record: dict[str, object]) -> None:
        ...


class MemoryCaptureSink:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    async def write(self, record: dict[str, object]) -> None:
        self.records.append(record)


class JsonlCaptureSink:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    async def write(self, record: dict[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class CaptureManager:
    def __init__(self, mode: CaptureMode = CaptureMode.OFF, sink: CaptureSink | None = None) -> None:
        if not isinstance(mode, CaptureMode):
            raise TypeError("mode must be CaptureMode")
        if mode is not CaptureMode.OFF and sink is None:
            raise ValueError("metadata/full capture requires a sink")
        self.mode = mode
        self._sink = sink

    @property
    def enabled(self) -> bool:
        return self.mode is not CaptureMode.OFF

    async def record_terminal(
        self,
        *,
        request: CanonicalRequest | RawPromptRequest,
        prompt_hash: str | None,
        status: str,
        elapsed_seconds: float,
        usage: TokenUsage,
        timing: GenerationTiming,
        error: CanonicalError | None,
        events: tuple[GenerationEvent, ...],
        runtime_trace: tuple[dict[str, object], ...] = (),
    ) -> None:
        if self.mode is CaptureMode.OFF:
            return
        if status not in {"completed", "failed", "cancelled", "rejected"}:
            raise ValueError("unsupported capture terminal status")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        assert self._sink is not None

        record: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "mode": self.mode.value,
            "request_id": request.request_id,
            "model": request.model,
            "prompt_hash": prompt_hash,
            "status": status,
            "elapsed_seconds": elapsed_seconds,
            "usage": _encode_usage(usage),
            "timing": _encode_timing(timing),
            "error": _encode_error_metadata(error),
        }
        if self.mode is CaptureMode.FULL:
            record["request"] = _encode_request(request)
            record["events"] = [_encode_event(event) for event in events]
            if runtime_trace:
                record["runtime_trace"] = [dict(entry) for entry in runtime_trace]
        await self._sink.write(record)


def read_capture_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"capture line {line_number} must be a JSON object")
            record = cast(dict[str, object], value)
            if record.get("schema_version") != _SCHEMA_VERSION:
                raise ValueError(f"unsupported capture schema_version on line {line_number}")
            records.append(record)
    return records


def replay_request(record: dict[str, object]) -> CanonicalRequest | RawPromptRequest:
    _require_full(record)
    value = record.get("request")
    if not isinstance(value, dict):
        raise TypeError("full capture is missing canonical request")
    request_id = value.get("request_id")
    model = value.get("model")
    items = value.get("items")
    if not isinstance(request_id, str) or not isinstance(model, str) or not isinstance(items, list):
        raise TypeError("captured canonical request is malformed")
    decoded = tuple(_decode_item(item) for item in items)
    if any(isinstance(item, RawPromptItem) for item in decoded):
        if len(decoded) != 1 or not isinstance(decoded[0], RawPromptItem):
            raise TypeError("captured raw prompt request is malformed")
        return RawPromptRequest(request_id, model, (decoded[0],))
    return CanonicalRequest(request_id, model, cast(tuple[CanonicalItem, ...], decoded))


def replay_events(record: dict[str, object]) -> tuple[GenerationEvent, ...]:
    _require_full(record)
    values = record.get("events")
    if not isinstance(values, list):
        raise TypeError("full capture is missing canonical events")
    return tuple(_decode_event(value) for value in values)


def _require_full(record: dict[str, object]) -> None:
    if record.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported capture schema_version")
    if record.get("mode") != CaptureMode.FULL.value:
        raise ValueError("replay requires a full capture")


def _encode_timing(timing: GenerationTiming) -> dict[str, object]:
    return {
        "queue_seconds": timing.queue_seconds,
        "prefill_seconds": timing.prefill_seconds,
        "generation_seconds": timing.generation_seconds,
    }


def _decode_timing(value: object) -> GenerationTiming:
    if not isinstance(value, dict):
        raise TypeError("captured timing is malformed")
    return GenerationTiming(
        cast(float | None, value.get("queue_seconds")),
        cast(float | None, value.get("prefill_seconds")),
        cast(float | None, value.get("generation_seconds")),
    )


def _encode_error_metadata(error: CanonicalError | None) -> dict[str, object] | None:
    if error is None:
        return None
    return {
        "category": error.category.value,
        "code": error.code,
        "retryable": error.retryable,
    }


def _encode_error_full(error: CanonicalError) -> dict[str, object]:
    result = _encode_error_metadata(error)
    assert result is not None
    result["message"] = error.message
    return result


def _decode_error(value: object) -> CanonicalError:
    if not isinstance(value, dict):
        raise TypeError("captured error is malformed")
    category = value.get("category")
    code = value.get("code")
    message = value.get("message")
    retryable = value.get("retryable")
    if not isinstance(category, str) or not isinstance(code, str) or not isinstance(message, str):
        raise TypeError("captured error is malformed")
    if not isinstance(retryable, bool):
        raise TypeError("captured error retryable flag is malformed")
    return CanonicalError(ErrorCategory(category), code, message, retryable)


def _encode_usage(usage: TokenUsage) -> dict[str, object]:
    result: dict[str, object] = {}
    result["input_" + "tokens"] = usage.input_tokens
    result["cached_input_" + "tokens"] = usage.cached_input_tokens
    result["output_" + "tokens"] = usage.output_tokens
    result["reasoning_" + "tokens"] = usage.reasoning_tokens
    return result


def _decode_usage(value: object) -> TokenUsage:
    if not isinstance(value, dict):
        raise TypeError("captured usage is malformed")
    input_count = cast(int | None, value.get("input_" + "tokens"))
    cached_count = cast(int | None, value.get("cached_input_" + "tokens"))
    output_count = cast(int | None, value.get("output_" + "tokens"))
    reasoning_count = cast(int | None, value.get("reasoning_" + "tokens"))
    return TokenUsage(input_count, cached_count, output_count, reasoning_count)


def _encode_content_parts(parts: tuple[MessageContentPart, ...]) -> list[dict[str, object]]:
    encoded: list[dict[str, object]] = []
    for part in parts:
        if isinstance(part, TextContentPart):
            encoded.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContentPart):
            row: dict[str, object] = {"type": "image", "source": part.source}
            if part.detail is not None:
                row["detail"] = part.detail
            encoded.append(row)
        else:  # pragma: no cover - canonical validation prevents this
            raise TypeError("unsupported canonical content part")
    return encoded


def _decode_content_parts(value: object) -> tuple[MessageContentPart, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError("captured multimodal content is malformed")
    parts: list[MessageContentPart] = []
    for raw_part in value:
        if not isinstance(raw_part, dict):
            raise TypeError("captured multimodal content is malformed")
        kind = raw_part.get("type")
        if kind == "text":
            text = raw_part.get("text")
            if not isinstance(text, str):
                raise TypeError("captured multimodal text part is malformed")
            parts.append(TextContentPart(text))
            continue
        if kind == "image":
            source = raw_part.get("source")
            detail = raw_part.get("detail")
            if not isinstance(source, str) or (detail is not None and not isinstance(detail, str)):
                raise TypeError("captured multimodal image part is malformed")
            parts.append(ImageContentPart(source, detail))
            continue
        raise ValueError(f"unsupported captured content part type: {kind!r}")
    return tuple(parts)


def _encode_item(item: CanonicalItem | RawPromptItem) -> dict[str, object]:
    if isinstance(item, MessageItem):
        return {"type": "message", "role": item.role.value, "text": item.text}
    if isinstance(item, MultimodalMessageItem):
        return {
            "type": "multimodal_message",
            "role": item.role.value,
            "parts": _encode_content_parts(item.parts),
        }
    if isinstance(item, RawPromptItem):
        return {
            "type": "raw_prompt",
            "text": item.text,
            "token_ids": list(item.token_ids) if item.token_ids is not None else None,
        }
    if isinstance(item, ReasoningItem):
        return {
            "type": "reasoning",
            "text": item.text,
            "starts_new_assistant_segment": item.starts_new_assistant_segment,
        }
    if isinstance(item, ToolCallItem):
        return {
            "type": "tool_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments_json": item.arguments_json,
            "index": item.index,
        }
    if isinstance(item, ToolResultItem):
        return {
            "type": "tool_result",
            "call_id": item.call_id,
            "text": item.text,
            "is_error": item.is_error,
        }
    if isinstance(item, MultimodalToolResultItem):
        return {
            "type": "multimodal_tool_result",
            "call_id": item.call_id,
            "parts": _encode_content_parts(item.parts),
            "is_error": item.is_error,
        }
    raise TypeError("unsupported canonical item")


def _decode_item(value: object) -> CanonicalItem | RawPromptItem:
    if not isinstance(value, dict):
        raise TypeError("captured item is malformed")
    kind = value.get("type")
    if kind == "message":
        return MessageItem(MessageRole(cast(str, value.get("role"))), cast(str, value.get("text")))
    if kind == "multimodal_message":
        return MultimodalMessageItem(
            MessageRole(cast(str, value.get("role"))),
            _decode_content_parts(value.get("parts")),
        )
    if kind == "raw_prompt":
        text = value.get("text")
        token_ids = value.get("token_ids")
        if text is not None:
            if not isinstance(text, str) or token_ids is not None:
                raise TypeError("captured raw text prompt is malformed")
            return RawPromptItem(text=text)
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in token_ids
        ):
            raise TypeError("captured raw token prompt is malformed")
        return RawPromptItem(token_ids=tuple(token_ids))
    if kind == "reasoning":
        return ReasoningItem(
            cast(str, value.get("text")),
            cast(bool, value.get("starts_new_assistant_segment", False)),
        )
    if kind == "tool_call":
        return ToolCallItem(
            cast(str, value.get("call_id")),
            cast(str, value.get("name")),
            cast(str, value.get("arguments_json")),
            cast(int, value.get("index")),
        )
    if kind == "tool_result":
        return ToolResultItem(
            cast(str, value.get("call_id")),
            cast(str, value.get("text")),
            cast(bool, value.get("is_error", False)),
        )
    if kind == "multimodal_tool_result":
        return MultimodalToolResultItem(
            cast(str, value.get("call_id")),
            _decode_content_parts(value.get("parts")),
            cast(bool, value.get("is_error", False)),
        )
    raise ValueError(f"unsupported captured item type: {kind!r}")


def _encode_request(request: CanonicalRequest | RawPromptRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "model": request.model,
        "items": [_encode_item(item) for item in request.items],
    }


def _event_base(kind: str, request_id: str) -> dict[str, object]:
    return {"type": kind, "request_id": request_id}


def _encode_event(event: GenerationEvent) -> dict[str, object]:
    if isinstance(event, GenerationStarted):
        return _event_base("generation_started", event.request_id)
    if isinstance(event, GenerationCompleted):
        result = _event_base("generation_completed", event.request_id)
        result["reason"] = event.reason.value
        result["usage"] = _encode_usage(event.usage) if event.usage is not None else None
        result["stop_sequence"] = event.stop_sequence
        return result
    if isinstance(event, GenerationCancelled):
        return _event_base("generation_cancelled", event.request_id)
    if isinstance(event, GenerationFailed):
        result = _event_base("generation_failed", event.request_id)
        result["error"] = _encode_error_full(event.error)
        return result
    if isinstance(event, TextStarted):
        return _event_base("text_started", event.request_id)
    if isinstance(event, TextDelta):
        return {**_event_base("text_delta", event.request_id), "text": event.text}
    if isinstance(event, TextCompleted):
        return {**_event_base("text_completed", event.request_id), "text": event.text}
    if isinstance(event, ReasoningStarted):
        return _event_base("reasoning_started", event.request_id)
    if isinstance(event, ReasoningDelta):
        return {**_event_base("reasoning_delta", event.request_id), "text": event.text}
    if isinstance(event, ReasoningCompleted):
        return {**_event_base("reasoning_completed", event.request_id), "text": event.text}
    if isinstance(event, ToolCallStarted):
        return {
            **_event_base("tool_call_started", event.request_id),
            "call_id": event.call_id,
            "name": event.name,
            "index": event.index,
        }
    if isinstance(event, ToolCallArgumentsDelta):
        return {
            **_event_base("tool_call_arguments_delta", event.request_id),
            "call_id": event.call_id,
            "delta": event.delta,
            "index": event.index,
        }
    if isinstance(event, ToolCallCompleted):
        return {
            **_event_base("tool_call_completed", event.request_id),
            "call": _encode_item(event.call),
        }
    if isinstance(event, TimingUpdated):
        return {
            **_event_base("timing_updated", event.request_id),
            "timing": _encode_timing(event.timing),
        }
    if isinstance(event, UsageUpdated):
        return {
            **_event_base("usage_updated", event.request_id),
            "usage": _encode_usage(event.usage),
        }
    raise TypeError("unsupported canonical event")


def _decode_event(value: object) -> GenerationEvent:
    if not isinstance(value, dict):
        raise TypeError("captured event is malformed")
    kind = value.get("type")
    request_id = value.get("request_id")
    if not isinstance(request_id, str):
        raise TypeError("captured event request_id is malformed")
    if kind == "generation_started":
        return GenerationStarted(request_id)
    if kind == "generation_completed":
        raw_usage = value.get("usage")
        usage = None if raw_usage is None else _decode_usage(raw_usage)
        return GenerationCompleted(
            request_id,
            CompletionReason(cast(str, value.get("reason"))),
            usage,
            cast(str | None, value.get("stop_sequence")),
        )
    if kind == "generation_cancelled":
        return GenerationCancelled(request_id)
    if kind == "generation_failed":
        return GenerationFailed(request_id, _decode_error(value.get("error")))
    if kind == "text_started":
        return TextStarted(request_id)
    if kind == "text_delta":
        return TextDelta(request_id, cast(str, value.get("text")))
    if kind == "text_completed":
        return TextCompleted(request_id, cast(str, value.get("text")))
    if kind == "reasoning_started":
        return ReasoningStarted(request_id)
    if kind == "reasoning_delta":
        return ReasoningDelta(request_id, cast(str, value.get("text")))
    if kind == "reasoning_completed":
        return ReasoningCompleted(request_id, cast(str, value.get("text")))
    if kind == "tool_call_started":
        return ToolCallStarted(
            request_id,
            cast(str, value.get("call_id")),
            cast(str, value.get("name")),
            cast(int, value.get("index")),
        )
    if kind == "tool_call_arguments_delta":
        return ToolCallArgumentsDelta(
            request_id,
            cast(str, value.get("call_id")),
            cast(str, value.get("delta")),
            cast(int, value.get("index")),
        )
    if kind == "tool_call_completed":
        call = _decode_item(value.get("call"))
        if not isinstance(call, ToolCallItem):
            raise ValueError("captured completed tool call is malformed")
        return ToolCallCompleted(request_id, call)
    if kind == "timing_updated":
        return TimingUpdated(request_id, _decode_timing(value.get("timing")))
    if kind == "usage_updated":
        return UsageUpdated(request_id, _decode_usage(value.get("usage")))
    raise ValueError(f"unsupported captured event type: {kind!r}")
