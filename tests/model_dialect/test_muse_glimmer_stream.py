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
from exqserve.model.muse_glimmer import MuseGlimmerIncrementalParser


def _parse(chunks: Iterable[str]) -> tuple[list[GenerationEvent], bool]:
    parser = MuseGlimmerIncrementalParser("req-muse")
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


def _calls(events: Iterable[GenerationEvent]) -> list[ToolCallCompleted]:
    return [event for event in events if isinstance(event, ToolCallCompleted)]


def test_muse_reasoning_then_user_channel_is_separated() -> None:
    source = (
        " to=self<|message|>Need to think.<|eom|>"
        "<|start|>assistant to=user<|message|>Final answer<|eot|>"
    )
    events, incomplete = _parse([source])

    assert incomplete is False
    assert _reasoning(events) == "Need to think."
    assert _text(events) == "Final answer"
    assert [type(event) for event in events] == [
        ReasoningStarted,
        ReasoningDelta,
        ReasoningCompleted,
        TextStarted,
        TextDelta,
        TextCompleted,
    ]


def test_muse_recipient_channels_are_chunk_boundary_safe() -> None:
    source = (
        " to=self<|message|>分析<|eom|>"
        "<|start|>assistant to=user<|message|>答案<|eot|>"
    )
    baseline, _ = _parse([source])

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert _reasoning(events) == _reasoning(baseline) == "分析"
        assert _text(events) == _text(baseline) == "答案"


def test_muse_atem_tool_call_normalizes_string_and_json_values() -> None:
    source = (
        " to=files.read<|message|>"
        '<atem:function_calls>\n<atem:invoke name="files.read">\n'
        '<atem:parameter name="path">/tmp/a b</atem:parameter>\n'
        '<atem:parameter name="limit">5</atem:parameter>\n'
        '<atem:parameter name="options">{"hidden":true}</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    events, incomplete = _parse([source])
    calls = _calls(events)

    assert incomplete is False
    assert len(calls) == 1
    call = calls[0].call
    assert call.name == "files.read"
    assert call.arguments_json == '{"limit":5,"options":{"hidden":true},"path":"/tmp/a b"}'
    assert [type(event) for event in events] == [
        ToolCallStarted,
        ToolCallArgumentsDelta,
        ToolCallCompleted,
    ]


def test_muse_atem_parallel_invokes_are_repeated_and_indexed() -> None:
    source = (
        " to=tools.batch<|message|>"
        "<atem:function_calls>\n"
        '<atem:invoke name="weather">\n<atem:parameter name="city">Paris</atem:parameter>\n</atem:invoke>\n'
        '<atem:invoke name="time">\n<atem:parameter name="city">Tokyo</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls><|eot|>"
    )
    events, incomplete = _parse(list(source))
    calls = [event.call for event in _calls(events)]

    assert incomplete is False
    assert [(call.name, call.index, call.arguments_json) for call in calls] == [
        ("weather", 0, '{"city":"Paris"}'),
        ("time", 1, '{"city":"Tokyo"}'),
    ]
    assert calls[0].call_id != calls[1].call_id


def test_muse_tool_call_can_finish_without_eot_when_runtime_consumes_stop() -> None:
    source = (
        " to=lookup<|message|>"
        '<atem:function_calls>\n<atem:invoke name="lookup">\n'
        '<atem:parameter name="id">7</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    events, incomplete = _parse([source])
    calls = _calls(events)

    assert incomplete is False
    assert len(calls) == 1
    assert calls[0].call.arguments_json == '{"id":7}'


def test_muse_malformed_or_incomplete_atem_is_flagged_without_fabrication() -> None:
    incomplete_events, incomplete = _parse(
        [
            (
                " to=lookup<|message|><atem:function_calls>\n"
                '<atem:invoke name="lookup"><atem:parameter name="id">7'
            )
        ]
    )
    assert incomplete is True
    assert _calls(incomplete_events) == []

    malformed_events, malformed = _parse(
        [
            (
                " to=lookup<|message|><atem:function_calls>\n"
                '<atem:invoke name="lookup">garbage</atem:invoke>\n'
                "</atem:function_calls><|eot|>"
            )
        ]
    )
    assert malformed is True
    assert _calls(malformed_events) == []


def test_muse_atem_rejects_non_whitespace_outside_function_calls_block() -> None:
    valid_block = (
        '<atem:function_calls><atem:invoke name="lookup">'
        '<atem:parameter name="id">7</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    )

    for body in (f"GARBAGE{valid_block}", f"{valid_block}TRAILING"):
        events, incomplete = _parse([f" to=lookup<|message|>{body}<|eot|>"])
        assert incomplete is True
        assert _calls(events) == []


def test_muse_raw_text_fallback_stays_text_for_protocol_robustness() -> None:
    events, incomplete = _parse(["plain answer"])

    assert incomplete is False
    assert _text(events) == "plain answer"


def test_finish_is_idempotent() -> None:
    parser = MuseGlimmerIncrementalParser("req-muse")
    parser.feed(" to=user<|message|>hello")
    first = parser.finish()
    second = parser.finish()

    assert first.incomplete_tool_call is False
    assert any(isinstance(event, TextCompleted) for event in first.events)
    assert second.events == ()
    assert second.incomplete_tool_call is False
