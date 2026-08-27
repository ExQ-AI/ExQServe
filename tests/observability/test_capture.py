from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
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
    MessageItem,
    MessageRole,
    RawPromptItem,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest, RawPromptRequest
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage
from exqserve.observability.capture import (
    CaptureManager,
    CaptureMode,
    JsonlCaptureSink,
    MemoryCaptureSink,
    read_capture_records,
    replay_events,
    replay_request,
)


def _request() -> CanonicalRequest:
    return CanonicalRequest(
        "r",
        "m",
        (
            MessageItem(MessageRole.USER, "SECRET USER TEXT"),
            ToolCallItem("c", "lookup", '{"secret":"ARG"}', 0),
            ToolResultItem("c", "SECRET RESULT"),
        ),
    )


def _events():  # type: ignore[no-untyped-def]
    usage = TokenUsage(input_tokens=10, cached_input_tokens=6, output_tokens=2)
    timing = GenerationTiming(queue_seconds=0.1, prefill_seconds=0.2, generation_seconds=0.3)
    call = ToolCallItem("c", "lookup", '{"secret":"ARG"}', 0)
    return (
        GenerationStarted("r"),
        ReasoningStarted("r"),
        ReasoningDelta("r", "SECRET REASONING"),
        ReasoningCompleted("r", "SECRET REASONING"),
        TextStarted("r"),
        TextDelta("r", "SECRET ANSWER"),
        TextCompleted("r", "SECRET ANSWER"),
        ToolCallStarted("r", "c", "lookup", 0),
        ToolCallArgumentsDelta("r", "c", '{"secret":"ARG"}', 0),
        ToolCallCompleted("r", call),
        TimingUpdated("r", timing),
        UsageUpdated("r", usage),
        GenerationCompleted("r", CompletionReason.TOOL_CALLS, usage),
    )


def test_capture_off_performs_no_sink_write() -> None:
    async def scenario() -> None:
        sink = MemoryCaptureSink()
        manager = CaptureManager(CaptureMode.OFF, sink)
        await manager.record_terminal(
            request=_request(),
            prompt_hash="a" * 64,
            status="completed",
            elapsed_seconds=1.0,
            usage=TokenUsage(input_tokens=1),
            timing=GenerationTiming(),
            error=None,
            events=_events(),
        )
        assert sink.records == []

    asyncio.run(scenario())


def test_metadata_capture_contains_no_user_reasoning_tool_payloads() -> None:
    async def scenario() -> None:
        sink = MemoryCaptureSink()
        manager = CaptureManager(CaptureMode.METADATA, sink)
        await manager.record_terminal(
            request=_request(),
            prompt_hash="b" * 64,
            status="failed",
            elapsed_seconds=1.25,
            usage=TokenUsage(input_tokens=10, cached_input_tokens=5, output_tokens=2),
            timing=GenerationTiming(prefill_seconds=0.2),
            error=CanonicalError(ErrorCategory.MODEL_FAILURE, "bad_output", "Safe failure.", False),
            events=_events(),
        )

        assert len(sink.records) == 1
        record = sink.records[0]
        text = json.dumps(record, sort_keys=True)
        assert record["mode"] == "metadata"
        assert record["request_id"] == "r"
        assert record["model"] == "m"
        assert record["prompt_hash"] == "b" * 64
        assert record["status"] == "failed"
        assert "request" not in record
        assert "events" not in record
        assert "SECRET" not in text
        assert "ARG" not in text

    asyncio.run(scenario())


def test_full_capture_round_trips_canonical_request_and_event_order() -> None:
    async def scenario() -> None:
        sink = MemoryCaptureSink()
        manager = CaptureManager(CaptureMode.FULL, sink)
        events = _events()
        request = _request()
        await manager.record_terminal(
            request=request,
            prompt_hash="c" * 64,
            status="completed",
            elapsed_seconds=0.5,
            usage=TokenUsage(input_tokens=10, cached_input_tokens=6, output_tokens=2),
            timing=GenerationTiming(0.1, 0.2, 0.3),
            error=None,
            events=events,
        )

        record = sink.records[0]
        assert replay_request(record) == request
        assert replay_events(record) == events

    asyncio.run(scenario())


def test_full_capture_round_trips_raw_prompt_items() -> None:
    async def scenario() -> None:
        sink = MemoryCaptureSink()
        manager = CaptureManager(CaptureMode.FULL, sink)
        for request in (
            RawPromptRequest("raw-text", "m", (RawPromptItem(text="RAW"),)),
            RawPromptRequest("raw-tokens", "m", (RawPromptItem(token_ids=(1, 2, 3)),)),
        ):
            await manager.record_terminal(
                request=request,
                prompt_hash="e" * 64,
                status="completed",
                elapsed_seconds=0.1,
                usage=TokenUsage(input_tokens=3, output_tokens=1),
                timing=GenerationTiming(),
                error=None,
                events=(),
            )
        assert replay_request(sink.records[0]) == RawPromptRequest(
            "raw-text", "m", (RawPromptItem(text="RAW"),)
        )
        assert replay_request(sink.records[1]) == RawPromptRequest(
            "raw-tokens", "m", (RawPromptItem(token_ids=(1, 2, 3)),)
        )

    asyncio.run(scenario())


def test_jsonl_sink_and_reader_are_versioned_and_reject_unknown_schema(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "capture.jsonl"
        sink = JsonlCaptureSink(path)
        manager = CaptureManager(CaptureMode.FULL, sink)
        await manager.record_terminal(
            request=_request(),
            prompt_hash="d" * 64,
            status="completed",
            elapsed_seconds=0.5,
            usage=TokenUsage(input_tokens=1),
            timing=GenerationTiming(),
            error=None,
            events=(GenerationStarted("r"), GenerationCompleted("r", CompletionReason.STOP)),
        )
        records = read_capture_records(path)
        assert len(records) == 1
        assert records[0]["schema_version"] == 1

        path.write_text('{"schema_version":2,"mode":"full"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            read_capture_records(path)

    asyncio.run(scenario())


def test_full_replay_preserves_safe_failure_error() -> None:
    async def scenario() -> None:
        sink = MemoryCaptureSink()
        manager = CaptureManager(CaptureMode.FULL, sink)
        error = CanonicalError(ErrorCategory.RUNTIME_FAILURE, "backend_failed", "Runtime failed.", True)
        events = (GenerationStarted("r"), GenerationFailed("r", error))
        await manager.record_terminal(
            request=_request(),
            prompt_hash=None,
            status="failed",
            elapsed_seconds=0.1,
            usage=TokenUsage(),
            timing=GenerationTiming(),
            error=error,
            events=events,
        )
        assert replay_events(sink.records[0]) == events

    asyncio.run(scenario())
