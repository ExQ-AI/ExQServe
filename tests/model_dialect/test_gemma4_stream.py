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

    literal = '<|tool_call>not-a-call{}<tool_call|>after'
    literal_events, literal_incomplete = _parse([literal])
    assert literal_incomplete is False
    assert _completed(literal_events) == []
    assert _text(literal_events) == literal

    malformed_events, malformed = _parse(['<|tool_call>call:save{broken}<tool_call|>after'])
    assert malformed is True
    assert _completed(malformed_events) == []
    assert _text(malformed_events) == "after"


def test_literal_gemma_tool_opener_replays_as_text_at_every_split() -> None:
    source = "Review literal `<|tool_call>` marker in parser source."
    baseline, baseline_incomplete = _parse([source])
    assert baseline_incomplete is False
    assert _text(baseline) == source
    assert _completed(baseline) == []

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert _text(events) == source
        assert _completed(events) == []


def test_literal_gemma_tool_candidate_replays_through_reasoning_channel() -> None:
    source = "Review `<|tool_call>` literally.<channel|>final"
    events, incomplete = _parse([source], start_in_reasoning=True)

    assert incomplete is False
    assert _reasoning(events) == "Review `<|tool_call>` literally."
    assert _text(events) == "final"
    assert _completed(events) == []


def test_disqualified_candidate_does_not_swallow_later_real_tool_call() -> None:
    source = (
        "literal <|tool_call> prose "
        "<|tool_call>call:ping{}<tool_call|> done"
    )
    baseline, baseline_incomplete = _parse([source])
    assert baseline_incomplete is False
    assert _text(baseline) == "literal <|tool_call> prose  done"
    assert [event.call.name for event in _completed(baseline)] == ["ping"]

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert _text(events) == _text(baseline)
        assert [event.call for event in _completed(events)] == [
            event.call for event in _completed(baseline)
        ]


def test_gemma_candidate_preserves_whitespace_accepted_by_body_parser() -> None:
    source = "<|tool_call> \n call:   ping  {}<tool_call|>"
    events, incomplete = _parse([source])

    assert incomplete is False
    calls = _completed(events)
    assert len(calls) == 1
    assert calls[0].call.name == "ping"
    assert calls[0].call.arguments_json == "{}"


def test_gemma_valid_candidate_prefix_eof_remains_incomplete() -> None:
    candidates = (
        "<|tool_call>",
        "<|tool_call> ",
        "<|tool_call> call",
        "<|tool_call> call:",
        "<|tool_call> call: ",
        "<|tool_call> call: ping",
        "<|tool_call> call: ping ",
        "<|tool_call> call: ping {",
    )
    for source in candidates:
        events, incomplete = _parse([source])
        assert incomplete is True
        assert _completed(events) == []


def test_gemma_close_inside_native_string_is_not_envelope_close_and_is_chunk_invariant() -> None:
    source = (
        '<|tool_call>call:save{value:<|"|>before<tool_call|>after<|"|>}'
        '<tool_call|>'
    )
    baseline, baseline_incomplete = _parse([source])
    assert baseline_incomplete is False
    calls = _completed(baseline)
    assert len(calls) == 1
    assert calls[0].call.arguments_json == '{"value":"before<tool_call|>after"}'

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert events == baseline


def test_gemma_close_inside_json_string_and_escape_parity_are_chunk_invariant() -> None:
    sources = (
        '<|tool_call>call:save{"value":"before<tool_call|>after"}<tool_call|>',
        '<|tool_call>call:save{"value":"a\\\\\\"b<tool_call|>c"}<tool_call|>',
        '<|tool_call>call:save{"value":"a\\\\\\\\","next":"<tool_call|>"}<tool_call|>',
    )
    for source in sources:
        baseline, baseline_incomplete = _parse([source])
        assert baseline_incomplete is False
        assert len(_completed(baseline)) == 1
        for split in range(len(source) + 1):
            events, incomplete = _parse([source[:split], source[split:]])
            assert incomplete is False
            assert events == baseline


def test_gemma_close_scanner_preserves_quote_in_accepted_tool_name() -> None:
    source = '<|tool_call>call:foo"bar{}<tool_call|>'
    baseline, baseline_incomplete = _parse([source])
    calls = _completed(baseline)

    assert baseline_incomplete is False
    assert len(calls) == 1
    assert calls[0].call.name == 'foo"bar'
    assert calls[0].call.arguments_json == "{}"

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert events == baseline


def test_gemma_close_scanner_preserves_quote_in_fallback_unquoted_key() -> None:
    source = '<|tool_call>call:save{foo"bar:1}<tool_call|>'
    baseline, baseline_incomplete = _parse([source])
    calls = _completed(baseline)

    assert baseline_incomplete is False
    assert len(calls) == 1
    assert calls[0].call.name == "save"
    assert calls[0].call.arguments_json == '{"foo\\"bar":1}'

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert events == baseline


def test_gemma_close_scanner_rejects_marker_in_fallback_unquoted_key() -> None:
    source = '<|tool_call>call:save{foo<tool_call|>bar:1}<tool_call|>'
    baseline, baseline_incomplete = _parse([source])

    assert baseline_incomplete is True
    assert _completed(baseline) == []

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is True
        assert _completed(events) == []
        assert _text(events) == _text(baseline)


def test_gemma_close_scanner_allows_marker_in_quoted_keys() -> None:
    sources = (
        '<|tool_call>call:save{"foo<tool_call|>bar":1}<tool_call|>',
        '<|tool_call>call:save{<|"|>foo<tool_call|>bar<|"|>:1}<tool_call|>',
    )

    for source in sources:
        baseline, baseline_incomplete = _parse([source])
        calls = _completed(baseline)

        assert baseline_incomplete is False
        assert len(calls) == 1
        assert calls[0].call.arguments_json == '{"foo<tool_call|>bar":1}'

        for split in range(len(source) + 1):
            events, incomplete = _parse([source[:split], source[split:]])
            assert incomplete is False
            assert events == baseline


def test_gemma_inline_backtick_valid_tool_wire_is_literal_but_unquoted_executes() -> None:
    wire = '<|tool_call>call:read{path:<|"|>/tmp/a<|"|>}<tool_call|>'
    quoted = f'`{wire}`'

    events, incomplete = _parse([quoted])
    assert incomplete is False
    assert _completed(events) == []
    assert _text(events) == quoted

    events, incomplete = _parse([wire])
    calls = _completed(events)
    assert incomplete is False
    assert len(calls) == 1
    assert calls[0].call.name == "read"
    assert calls[0].call.arguments_json == '{"path":"/tmp/a"}'


def test_gemma_inline_presentation_is_chunk_invariant_and_recovers_same_chunk() -> None:
    literal_wire = '<|tool_call>call:read{path:<|"|>/quoted<|"|>}<tool_call|>'
    real_wire = '<|tool_call>call:read{path:<|"|>/real<|"|>}<tool_call|>'
    source = f'before `{literal_wire}` after {real_wire} done'
    baseline, baseline_incomplete = _parse([source])

    assert baseline_incomplete is False
    assert [event.call.arguments_json for event in _completed(baseline)] == ['{"path":"/real"}']
    assert _text(baseline) == f'before `{literal_wire}` after  done'

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert _text(events) == _text(baseline)
        assert _reasoning(events) == _reasoning(baseline)
        assert [event.call for event in _completed(events)] == [
            event.call for event in _completed(baseline)
        ]


def test_gemma_inline_run_width_and_newline_reset_are_frozen() -> None:
    wire = '<|tool_call>call:read{}<tool_call|>'
    source = f'``code ` still code {wire}``\n{wire}'
    events, incomplete = _parse(list(source))

    assert incomplete is False
    assert [event.call.name for event in _completed(events)] == ["read"]
    assert _text(events) == f'``code ` still code {wire}``\n'

    unmatched = f'prefix `unclosed\n{wire}'
    events, incomplete = _parse([unmatched])
    assert incomplete is False
    assert [event.call.name for event in _completed(events)] == ["read"]
    assert _text(events) == "prefix `unclosed\n"


def test_gemma_fenced_backtick_valid_wire_is_literal_and_recovers_after_close() -> None:
    literal_wire = '<|tool_call>call:read{path:<|"|>/quoted<|"|>}<tool_call|>'
    real_wire = '<|tool_call>call:read{path:<|"|>/real<|"|>}<tool_call|>'
    source = f'```python\n{literal_wire}\n```\n{real_wire}'
    baseline, baseline_incomplete = _parse([source])

    assert baseline_incomplete is False
    calls = _completed(baseline)
    assert len(calls) == 1
    assert calls[0].call.arguments_json == '{"path":"/real"}'
    assert _text(baseline) == f'```python\n{literal_wire}\n```\n'

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False
        assert _text(events) == _text(baseline)
        assert _reasoning(events) == _reasoning(baseline)
        assert [event.call for event in _completed(events)] == [
            event.call for event in _completed(baseline)
        ]


def test_gemma_fence_close_requires_whitespace_only_tail() -> None:
    suppressed = '<|tool_call>call:read{path:<|"|>/suppressed<|"|>}<tool_call|>'
    real_wire = '<|tool_call>call:read{path:<|"|>/real<|"|>}<tool_call|>'
    source = f'```python\n```not-a-close\n{suppressed}\n```   \n{real_wire}'
    events, incomplete = _parse([source])

    assert incomplete is False
    calls = _completed(events)
    assert len(calls) == 1
    assert calls[0].call.arguments_json == '{"path":"/real"}'
    assert suppressed in _text(events)


def test_gemma_unclosed_fence_never_publishes_tool_call() -> None:
    wire = '<|tool_call>call:read{path:<|"|>/quoted<|"|>}<tool_call|>'
    source = f'   ```python\n{wire}\n'
    events, incomplete = _parse(list(source))

    assert incomplete is False
    assert _completed(events) == []
    assert _text(events) == source


def test_gemma_inline_thought_close_cannot_escape_presentation() -> None:
    wire = '<|tool_call>call:read{}<tool_call|>'
    source = f'`literal <channel|> {wire}`'
    baseline, baseline_incomplete = _parse([source], start_in_reasoning=True)

    assert baseline_incomplete is False
    assert _completed(baseline) == []
    assert _reasoning(baseline) == source
    assert _text(baseline) == ""

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], start_in_reasoning=True)
        assert incomplete is False
        assert _reasoning(events) == _reasoning(baseline)
        assert _text(events) == _text(baseline)
        assert _completed(events) == []


def test_gemma_fenced_thought_markers_cannot_escape_presentation() -> None:
    wire = '<|tool_call>call:read{}<tool_call|>'
    sources = (
        f'```python\n<channel|>\n{wire}\n```\n',
        f'```python\n<|channel>thought\n{wire}\n```\n',
    )

    for source in sources:
        start_in_reasoning = "<channel|>" in source
        baseline, baseline_incomplete = _parse(
            [source], start_in_reasoning=start_in_reasoning
        )
        assert baseline_incomplete is False
        assert _completed(baseline) == []

        for split in range(len(source) + 1):
            events, incomplete = _parse(
                [source[:split], source[split:]], start_in_reasoning=start_in_reasoning
            )
            assert incomplete is False
            assert _reasoning(events) == _reasoning(baseline)
            assert _text(events) == _text(baseline)
            assert _completed(events) == []


def test_gemma_structural_markers_recover_after_presentation_closes() -> None:
    literal_wire = '<|tool_call>call:read{path:<|"|>/literal<|"|>}<tool_call|>'
    real_wire = '<|tool_call>call:read{path:<|"|>/real<|"|>}<tool_call|>'
    source = f'```python\n<channel|>\n{literal_wire}\n```\n<channel|>{real_wire}'
    baseline, baseline_incomplete = _parse([source], start_in_reasoning=True)

    assert baseline_incomplete is False
    calls = _completed(baseline)
    assert len(calls) == 1
    assert calls[0].call.arguments_json == '{"path":"/real"}'
    assert literal_wire in _reasoning(baseline)

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], start_in_reasoning=True)
        assert incomplete is False
        assert _reasoning(events) == _reasoning(baseline)
        assert _text(events) == _text(baseline)
        assert [event.call for event in _completed(events)] == [
            event.call for event in calls
        ]


def test_gemma_quoted_wire_stays_in_reasoning_source_channel() -> None:
    wire = '<|tool_call>call:read{}<tool_call|>'
    source = f'review `{wire}` literally<channel|>final'
    baseline, baseline_incomplete = _parse([source], start_in_reasoning=True)

    assert baseline_incomplete is False
    assert _completed(baseline) == []
    assert _reasoning(baseline) == f'review `{wire}` literally'
    assert _text(baseline) == "final"

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], start_in_reasoning=True)
        assert incomplete is False
        assert _text(events) == _text(baseline)
        assert _reasoning(events) == _reasoning(baseline)
        assert [event.call for event in _completed(events)] == [
            event.call for event in _completed(baseline)
        ]

def test_gemma_presentation_partial_markers_at_eof_stay_literal() -> None:
    cases = (
        ("`literal <channel|", True, "reasoning"),
        ("```python\n<channel|", True, "reasoning"),
        ("`literal <|channel", False, "text"),
        ("```python\n<|channel", False, "text"),
        ("`literal <|tool_call", True, "reasoning"),
        ("```python\n<|tool_call", False, "text"),
    )

    for source, start_in_reasoning, channel in cases:
        baseline, baseline_incomplete = _parse([source], start_in_reasoning=start_in_reasoning)
        assert baseline_incomplete is False
        assert _completed(baseline) == []
        if channel == "reasoning":
            assert _reasoning(baseline) == source
            assert _text(baseline) == ""
        else:
            assert _text(baseline) == source
            assert _reasoning(baseline) == ""

        for split in range(len(source) + 1):
            events, incomplete = _parse(
                [source[:split], source[split:]], start_in_reasoning=start_in_reasoning
            )
            assert incomplete is False
            assert _reasoning(events) == _reasoning(baseline)
            assert _text(events) == _text(baseline)
            assert _completed(events) == []


def test_gemma_partial_marker_eof_semantics_remain_structural_outside_presentation() -> None:
    tool_events, tool_incomplete = _parse(["plain <|tool_call"])
    assert tool_incomplete is True
    assert _completed(tool_events) == []
    assert _text(tool_events) == "plain "

    thought_close_events, thought_close_incomplete = _parse(
        ["reason <channel|"], start_in_reasoning=True
    )
    assert thought_close_incomplete is False
    assert _reasoning(thought_close_events) == "reason "

    thought_open_events, thought_open_incomplete = _parse(["plain <|channel"])
    assert thought_open_incomplete is False
    assert _text(thought_open_events) == "plain "


def test_finish_is_idempotent() -> None:
    parser = Gemma4IncrementalParser("req-gemma")
    parser.feed("hello")
    first = parser.finish()
    second = parser.finish()

    assert first.incomplete_tool_call is False
    assert any(isinstance(event, TextCompleted) for event in first.events)
    assert second.events == ()
    assert second.incomplete_tool_call is False
