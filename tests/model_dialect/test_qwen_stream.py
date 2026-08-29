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
from exqserve.model.qwen import QwenIncrementalParser


def _parse(chunks: Iterable[str], request_id: str = "req-1") -> tuple[list[GenerationEvent], bool]:
    parser = QwenIncrementalParser(request_id)
    events: list[GenerationEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    finished = parser.finish()
    events.extend(finished.events)
    return events, finished.incomplete_tool_call


def _reasoning_text(events: Iterable[GenerationEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, ReasoningDelta))


def _text(events: Iterable[GenerationEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


def _tool_argument_stream(events: Iterable[GenerationEvent], index: int) -> str:
    return "".join(
        event.delta
        for event in events
        if isinstance(event, ToolCallArgumentsDelta) and event.index == index
    )


def _completed_calls(events: Iterable[GenerationEvent]) -> list[ToolCallCompleted]:
    return [event for event in events if isinstance(event, ToolCallCompleted)]


def test_preopened_reasoning_from_generation_prompt_is_separated() -> None:
    parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
    events = list(parser.feed("reason from preopened think</think>answer"))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == "reason from preopened think"
    assert _text(events) == "answer"


def test_reasoning_and_text_channels_strip_model_markers() -> None:
    events, incomplete = _parse(["<think>reason</think>answer"])

    assert incomplete is False
    assert _reasoning_text(events) == "reason"
    assert _text(events) == "answer"
    assert [type(event) for event in events] == [
        ReasoningStarted,
        ReasoningDelta,
        ReasoningCompleted,
        TextStarted,
        TextDelta,
        TextCompleted,
    ]
    assert "<think>" not in _reasoning_text(events) + _text(events)
    assert "</think>" not in _reasoning_text(events) + _text(events)


def test_reasoning_markers_can_split_at_every_character_boundary() -> None:
    source = "<think>αβ reasoning</think>final 中文"
    baseline, _ = _parse([source])
    expected_reasoning = _reasoning_text(baseline)
    expected_text = _text(baseline)

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert _reasoning_text(events) == expected_reasoning
        assert _text(events) == expected_text
        assert sum(isinstance(event, ReasoningStarted) for event in events) == 1
        assert sum(isinstance(event, ReasoningCompleted) for event in events) == 1
        assert sum(isinstance(event, TextStarted) for event in events) == 1
        assert sum(isinstance(event, TextCompleted) for event in events) == 1


def test_character_by_character_stream_preserves_reasoning_and_text() -> None:
    source = "<think>one two</think>three four"
    events, incomplete = _parse(list(source))
    assert incomplete is False
    assert _reasoning_text(events) == "one two"
    assert _text(events) == "three four"


def test_finish_closes_open_reasoning_without_fabricating_marker_text() -> None:
    events, incomplete = _parse(["<think>unfinished reasoning"])
    assert incomplete is False
    assert _reasoning_text(events) == "unfinished reasoning"
    completed = [event for event in events if isinstance(event, ReasoningCompleted)]
    assert len(completed) == 1
    assert completed[0].text == "unfinished reasoning"


def test_complete_tool_call_streams_arguments_and_completed_item() -> None:
    source = (
        "<tool_call>\n"
        "<function=list_files>\n"
        "<parameter=path>\n/tmp\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    events, incomplete = _parse([source])

    assert incomplete is False
    starts = [event for event in events if isinstance(event, ToolCallStarted)]
    completed = _completed_calls(events)
    assert len(starts) == 1
    assert len(completed) == 1
    assert starts[0].name == "list_files"
    assert starts[0].index == 0
    assert starts[0].call_id == completed[0].call.call_id
    assert completed[0].call.arguments_json == '{"path":"/tmp"}'
    assert _tool_argument_stream(events, 0) == completed[0].call.arguments_json


def test_zero_parameter_tool_call_emits_safe_empty_object() -> None:
    source = "<tool_call><function=ping></function></tool_call>"
    events, incomplete = _parse([source])
    completed = _completed_calls(events)

    assert incomplete is False
    assert len(completed) == 1
    assert completed[0].call.arguments_json == "{}"
    assert _tool_argument_stream(events, 0) == "{}"


def test_tool_call_can_split_at_every_character_boundary() -> None:
    source = (
        "<tool_call><function=save>"
        "<parameter=count>3</parameter>"
        "<parameter=enabled>true</parameter>"
        "</function></tool_call>"
    )
    baseline, _ = _parse([source])
    baseline_call = _completed_calls(baseline)[0].call

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        completed = _completed_calls(events)
        assert incomplete is False
        assert len(completed) == 1
        assert completed[0].call == baseline_call
        assert _tool_argument_stream(events, 0) == baseline_call.arguments_json
        assert sum(isinstance(event, ToolCallStarted) for event in events) == 1


def test_parameter_values_use_strict_json_when_possible_and_string_otherwise() -> None:
    source = (
        "<tool_call><function=save>"
        "<parameter=count>3</parameter>"
        "<parameter=enabled>true</parameter>"
        "<parameter=meta>{\"x\":1}</parameter>"
        "<parameter=label>hello 中文</parameter>"
        "</function></tool_call>"
    )
    events, incomplete = _parse([source])
    call = _completed_calls(events)[0].call

    assert incomplete is False
    assert call.arguments_json == (
        '{"count":3,"enabled":true,"meta":{"x":1},"label":"hello 中文"}'
    )


def test_duplicate_parameter_names_are_preserved_for_downstream_strict_validation() -> None:
    source = (
        "<tool_call><function=f>"
        "<parameter=x>1</parameter>"
        "<parameter=x>2</parameter>"
        "</function></tool_call>"
    )
    events, incomplete = _parse([source])
    call = _completed_calls(events)[0].call

    assert incomplete is False
    assert call.arguments_json == '{"x":1,"x":2}'


def test_truncated_tool_before_first_parameter_never_becomes_actionable() -> None:
    events, incomplete = _parse(["<tool_call><function=run><parameter=x>partial"])

    assert incomplete is True
    assert not any(isinstance(event, ToolCallStarted) for event in events)
    assert not any(isinstance(event, ToolCallCompleted) for event in events)


def test_truncated_tool_after_safe_parameter_never_fabricates_completion() -> None:
    events, incomplete = _parse(
        ["<tool_call><function=run><parameter=x>1</parameter><parameter=y>partial"]
    )

    assert incomplete is True
    assert sum(isinstance(event, ToolCallStarted) for event in events) == 1
    assert not any(isinstance(event, ToolCallCompleted) for event in events)


def test_multiple_tool_calls_have_deterministic_distinct_ids_and_indices() -> None:
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )
    first, incomplete = _parse([source])
    second, _ = _parse([source])
    calls = [event.call for event in _completed_calls(first)]
    repeated = [event.call for event in _completed_calls(second)]

    assert incomplete is False
    assert [call.index for call in calls] == [0, 1]
    assert calls[0].call_id != calls[1].call_id
    assert calls == repeated


def test_tool_call_inside_reasoning_is_parsed_not_leaked_as_reasoning_text() -> None:
    source = (
        "<think>before "
        "<tool_call><function=lookup><parameter=q>\"x\"</parameter></function></tool_call>"
        " after</think>final"
    )
    events, incomplete = _parse(list(source))

    assert incomplete is False
    assert _reasoning_text(events) == "before  after"
    assert _text(events) == "final"
    assert len(_completed_calls(events)) == 1
    assert "tool_call" not in _reasoning_text(events)


def test_unconfirmed_tool_marker_in_reasoning_remains_literal() -> None:
    source = "The parser treats <tool_call> as a candidate marker in prose."
    parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
    events: list[GenerationEvent] = []
    for character in source:
        events.extend(parser.feed(character))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert _reasoning_text(events) == source
    assert not _completed_calls(events)


def test_partial_confirming_function_prefix_remains_incomplete_candidate() -> None:
    events, incomplete = _parse(["<tool_call>\n<fun"])

    assert incomplete is True
    assert not any(isinstance(event, ToolCallStarted) for event in events)
    assert not _completed_calls(events)


def test_real_markers_after_reasoning_inline_code_still_parse() -> None:
    source = (
        'Discuss `<think>`, `</think>`, and `<tool_call>` literally.'
        "</think>"
        "<tool_call><function=lookup><parameter=q>1</parameter></function></tool_call>"
    )
    parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
    events: list[GenerationEvent] = []
    for character in source:
        events.extend(parser.feed(character))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert _reasoning_text(events) == 'Discuss `<think>`, `</think>`, and `<tool_call>` literally.'
    assert len(_completed_calls(events)) == 1
    assert _completed_calls(events)[0].call.name == "lookup"


def test_quoted_source_markers_remain_literal_across_character_boundaries() -> None:
    source = 'The source says _PLAIN_MARKERS = ("<think>", "</think>", "<tool_call>").'
    parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
    events: list[GenerationEvent] = []
    for character in source:
        events.extend(parser.feed(character))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert _reasoning_text(events) == source
    assert not _completed_calls(events)


def test_single_quoted_end_think_marker_remains_reasoning_text() -> None:
    source = "The Qwen tool-call format is '</think>', followed by more analysis."
    parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
    events: list[GenerationEvent] = []
    for split in range(len(source) + 1):
        parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
        events = list(parser.feed(source[:split]))
        events.extend(parser.feed(source[split:]))
        finished = parser.finish()
        events.extend(finished.events)
        assert finished.incomplete_tool_call is False
        assert _reasoning_text(events) == source


def test_backtick_code_protocol_markers_remain_literal_at_every_split() -> None:
    samples = (
        "```text\n<think>\n</think>\n<tool_call>\n```\nstill reasoning",
        "Use ``<tool_call>`` and then continue reasoning.",
    )
    for source in samples:
        for split in range(len(source) + 1):
            parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
            events = list(parser.feed(source[:split]))
            events.extend(parser.feed(source[split:]))
            finished = parser.finish()
            events.extend(finished.events)
            assert finished.incomplete_tool_call is False
            assert _reasoning_text(events) == source
            assert not _completed_calls(events)


def test_real_end_think_after_closed_code_span_still_switches_channel() -> None:
    source = "Discuss `</think>` literally.</think>final"
    for split in range(len(source) + 1):
        parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
        events = list(parser.feed(source[:split]))
        events.extend(parser.feed(source[split:]))
        finished = parser.finish()
        events.extend(finished.events)
        assert finished.incomplete_tool_call is False
        assert _reasoning_text(events) == "Discuss `</think>` literally."
        assert _text(events) == "final"


def test_split_fenced_literal_then_real_close_streams_text_before_eos() -> None:
    opening_splits = (("`", "``"), ("``", "`"), ("`", "`", "`"))
    suffixes = ("FINAL text", "FINAL `x`", "FINAL ```code```")

    for opening_chunks in opening_splits:
        for suffix in suffixes:
            parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
            events: list[GenerationEvent] = []
            for chunk in opening_chunks:
                events.extend(parser.feed(chunk))
            events.extend(parser.feed("xml\n</think>\n"))
            streamed = list(parser.feed(f"```\nactual reasoning</think>{suffix}"))
            events.extend(streamed)

            assert _text(streamed) == suffix
            assert any(isinstance(event, ReasoningCompleted) for event in streamed)
            assert _reasoning_text(events) == "```xml\n</think>\n```\nactual reasoning"
            assert _text(events) == suffix

            finished = parser.finish()
            assert finished.incomplete_tool_call is False
            assert _text(finished.events) == ""


def test_parameter_close_literal_inside_json_string_is_not_treated_as_envelope_close() -> None:
    source = (
        '<tool_call><function=run><parameter=cmd>"echo </parameter> here"'
        '</parameter></function></tool_call>'
    )
    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False
        assert len(calls) == 1
        assert calls[0].call.name == "run"
        assert calls[0].call.arguments_json == '{"cmd":"echo </parameter> here"}'
