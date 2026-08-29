from __future__ import annotations

from collections.abc import Iterable

from exqserve.core.events import (
    GenerationEvent,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.model.gemma4 import Gemma4IncrementalParser


def _parse(
    chunks: Iterable[str],
    *,
    start_in_reasoning: bool = False,
) -> tuple[list[GenerationEvent], bool]:
    parser = Gemma4IncrementalParser("req-gemma", start_in_reasoning=start_in_reasoning)
    events: list[GenerationEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    finished = parser.finish()
    events.extend(finished.events)
    return events, finished.incomplete_tool_call


def _reasoning(events: Iterable[GenerationEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, ReasoningDelta))


def _text(events: Iterable[GenerationEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


def _completed(events: Iterable[GenerationEvent]) -> list[ToolCallCompleted]:
    return [event for event in events if isinstance(event, ToolCallCompleted)]


def _argument_stream(events: Iterable[GenerationEvent], index: int) -> str:
    return "".join(
        event.delta
        for event in events
        if isinstance(event, ToolCallArgumentsDelta) and event.index == index
    )


def test_preopened_gemma4_thought_channel_switches_to_text() -> None:
    events, incomplete = _parse(
        ["reasoning here<channel|>final answer"],
        start_in_reasoning=True,
    )

    assert incomplete is False
    assert _reasoning(events) == "reasoning here"
    assert _text(events) == "final answer"
    assert [type(event) for event in events] == [
        ReasoningStarted,
        ReasoningDelta,
        ReasoningCompleted,
        TextStarted,
        TextDelta,
        TextCompleted,
    ]


def test_explicit_gemma4_thought_marker_is_stripped_and_chunk_safe() -> None:
    source = "<|channel>thought\nαβ<channel|>answer 中文"
    baseline, _ = _parse([source])

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert _reasoning(events) == _reasoning(baseline) == "αβ"
        assert _text(events) == _text(baseline) == "answer 中文"
        assert "<|channel>" not in _reasoning(events) + _text(events)
        assert "<channel|>" not in _reasoning(events) + _text(events)


def test_gemma4_tool_call_normalizes_native_argument_syntax_to_json() -> None:
    source = (
        '<|tool_call>call:save{path:<|"|>/tmp/a b<|"|>,'
        'count:3,enabled:true,tags:[<|"|>a<|"|>,<|"|>β<|"|>],'
        'meta:{mode:<|"|>fast<|"|>,ratio:0.5}}<tool_call|>'
    )
    events, incomplete = _parse([source])
    completed = _completed(events)

    assert incomplete is False
    assert len(completed) == 1
    call = completed[0].call
    assert call.name == "save"
    assert call.index == 0
    assert call.arguments_json == (
        '{"count":3,"enabled":true,"meta":{"mode":"fast","ratio":0.5},'
        '"path":"/tmp/a b","tags":["a","β"]}'
    )
    assert _argument_stream(events, 0) == call.arguments_json
    starts = [event for event in events if isinstance(event, ToolCallStarted)]
    assert len(starts) == 1
    assert starts[0].call_id == call.call_id


def test_gemma4_tool_call_accepts_standard_json_as_robust_fallback() -> None:
    source = '<|tool_call>  call:lookup{"query":"hello","limit":5}<tool_call|>'
    events, incomplete = _parse([source])
    calls = _completed(events)

    assert incomplete is False
    assert len(calls) == 1
    assert calls[0].call.name == "lookup"
    assert calls[0].call.arguments_json == '{"limit":5,"query":"hello"}'


def test_gemma4_parallel_tool_calls_are_repeated_and_indexed() -> None:
    source = (
        '<|tool_call>call:first{x:1}<tool_call|>'
        '<|tool_call>call:second{label:<|"|>two<|"|>}<tool_call|>'
    )
    events, incomplete = _parse(list(source))
    calls = [event.call for event in _completed(events)]

    assert incomplete is False
    assert [(call.name, call.index, call.arguments_json) for call in calls] == [
        ("first", 0, '{"x":1}'),
        ("second", 1, '{"label":"two"}'),
    ]
    assert calls[0].call_id != calls[1].call_id


def test_gemma4_tool_call_can_split_at_every_character_boundary() -> None:
    source = '<|tool_call>call:lookup{query:<|"|>hello 中文<|"|>,limit:5}<tool_call|>'
    baseline, _ = _parse([source])
    expected = _completed(baseline)[0].call

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed(events)
        assert incomplete is False
        assert len(calls) == 1
        assert calls[0].call == expected
        assert _argument_stream(events, 0) == expected.arguments_json


def test_gemma4_text_tool_text_channels_close_cleanly() -> None:
    source = (
        'before'
        '<|tool_call>call:ping{}<tool_call|>'
        'after'
    )
    events, incomplete = _parse([source])

    assert incomplete is False
    assert _text(events) == "beforeafter"
    assert len(_completed(events)) == 1
    assert sum(isinstance(event, TextStarted) for event in events) == 2
    assert sum(isinstance(event, TextCompleted) for event in events) == 2


def test_incomplete_or_malformed_gemma4_tool_call_is_flagged_without_fabrication() -> None:
    incomplete_events, incomplete = _parse(['<|tool_call>call:save{x:<|"|>unfinished'])
    assert incomplete is True
    assert _completed(incomplete_events) == []

    malformed_events, malformed = _parse(['<|tool_call>not-a-call{}<tool_call|>after'])
    assert malformed is True
    assert _completed(malformed_events) == []
    assert _text(malformed_events) == "after"


def test_finish_is_idempotent() -> None:
    parser = Gemma4IncrementalParser("req-gemma")
    parser.feed("hello")
    first = parser.finish()
    second = parser.finish()

    assert first.incomplete_tool_call is False
    assert any(isinstance(event, TextCompleted) for event in first.events)
    assert second.events == ()
    assert second.incomplete_tool_call is False
