from __future__ import annotations

import sys
from collections.abc import Iterable

import pytest

from exqserve.agent._json import canonical_json_dumps
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
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
from exqserve.core.tokens import NativeTokenSpan
from exqserve.model.contracts import (
    NativeTokenProvenanceError,
    ParserAmbiguityDetail,
    ParserTerminalIssueKind,
)
from exqserve.model.qwen import QwenIncrementalParser


def _parse(
    chunks: Iterable[str],
    request_id: str = "req-1",
    *,
    tool_policy: ToolPolicy | None = None,
) -> tuple[list[GenerationEvent], bool]:
    parser = QwenIncrementalParser(request_id, tool_policy=tool_policy)
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


def test_parser_validates_tool_policy_at_public_boundary() -> None:
    with pytest.raises(TypeError, match="tool_policy"):
        QwenIncrementalParser("req-invalid", tool_policy="bad")  # type: ignore[arg-type]


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


def test_numeric_parameter_overflow_degrades_without_parser_exception() -> None:
    tool = FunctionTool(
        "f",
        None,
        JsonSchema(
            '{"type":"object","properties":{"x":{"type":"number"}},'
            '"required":["x"]}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    source = "<tool_call><function=f><parameter=x>1e999</parameter></function></tool_call>"

    for chunks in ([source], list(source)):
        events, incomplete = _parse(chunks, tool_policy=policy)
        completed = _completed_calls(events)

        assert incomplete is False
        assert len(completed) == 1
        assert completed[0].call.arguments_json == '{"x":"1e999"}'


def test_huge_integer_parameter_degrades_without_parser_exception() -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Python integer digit safety limit is disabled")

    huge_integer = "9" * (limit + 1)
    tool = FunctionTool(
        "f",
        None,
        JsonSchema(
            '{"type":"object","properties":{"x":{"type":"integer"}},'
            '"required":["x"]}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    source = (
        "<tool_call><function=f><parameter=x>"
        + huge_integer
        + "</parameter></function></tool_call>"
    )

    for chunks in ([source], list(source)):
        events, incomplete = _parse(chunks, tool_policy=policy)
        completed = _completed_calls(events)

        assert incomplete is False
        assert len(completed) == 1
        assert completed[0].call.arguments_json == canonical_json_dumps({"x": huge_integer})


def test_deeply_nested_parameter_degrades_without_parser_exception() -> None:
    nested = "[" * 10_000 + "0" + "]" * 10_000
    source = (
        "<tool_call><function=f><parameter=x>"
        + nested
        + "</parameter></function></tool_call>"
    )

    events, incomplete = _parse([source])
    completed = _completed_calls(events)

    assert incomplete is False
    assert len(completed) == 1
    assert completed[0].call.arguments_json == canonical_json_dumps({"x": nested})


def test_string_parameter_schema_preserves_raw_json_looking_text_as_string() -> None:
    tool = FunctionTool(
        "save",
        None,
        JsonSchema(
            '{"type":"object","properties":{'
            '"label":{"type":"string"},'
            '"enabled":{"type":"boolean"},'
            '"count":{"type":"integer"}'
            '},"required":["label","enabled","count"]}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    parser = QwenIncrementalParser("req-string", tool_policy=policy)
    source = (
        "<tool_call><function=save>"
        "<parameter=label>true</parameter>"
        "<parameter=enabled>true</parameter>"
        "<parameter=count>3</parameter>"
        "</function></tool_call>"
    )
    events = list(parser.feed(source))
    events.extend(parser.finish().events)
    call = _completed_calls(events)[0].call

    assert call.arguments_json == '{"label":"true","enabled":true,"count":3}'


def test_string_parameter_schema_still_accepts_json_quoted_string_surface() -> None:
    tool = FunctionTool(
        "save",
        None,
        JsonSchema(
            '{"type":"object","properties":{"label":{"type":"string"}},'
            '"required":["label"]}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    parser = QwenIncrementalParser("req-quoted", tool_policy=policy)
    events = list(
        parser.feed(
            '<tool_call><function=save><parameter=label>"hello"</parameter>'
            "</function></tool_call>"
        )
    )
    events.extend(parser.finish().events)

    assert _completed_calls(events)[0].call.arguments_json == '{"label":"hello"}'


def test_duplicate_parameter_names_fail_closed_in_qwen_envelope_parser() -> None:
    source = (
        "<tool_call><function=f>"
        "<parameter=x>1</parameter>"
        "<parameter=x>2</parameter>"
        "</function></tool_call>"
    )
    events, incomplete = _parse([source])

    assert incomplete is True
    assert _completed_calls(events) == []


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


def test_malformed_inline_backticks_do_not_suppress_later_real_tool_call() -> None:
    source = (
        "analysis ``foo` bar\n"
        "</think>\n"
        "<tool_call><function=read><parameter=file_path>/x</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
        events = list(parser.feed(source[:split]))
        events.extend(parser.feed(source[split:]))
        finished = parser.finish()
        events.extend(finished.events)

        calls = _completed_calls(events)
        assert finished.incomplete_tool_call is False
        assert _reasoning_text(events) == "analysis ``foo` bar\n"
        assert len(calls) == 1
        assert calls[0].call.name == "read"
        assert calls[0].call.arguments_json == '{"file_path":"/x"}'


def test_fenced_literal_tool_marker_then_real_tool_call_remains_distinct() -> None:
    for indent in ("", "   "):
        fenced = f"{indent}```text\n<tool_call>\n{indent}```\n"
        source = (
            fenced
            + "actual reasoning</think>\n"
            + "<tool_call><function=read><parameter=file_path>/x</parameter></function></tool_call>"
        )

        for split in range(len(source) + 1):
            parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
            events = list(parser.feed(source[:split]))
            events.extend(parser.feed(source[split:]))
            finished = parser.finish()
            events.extend(finished.events)

            calls = _completed_calls(events)
            assert finished.incomplete_tool_call is False
            assert _reasoning_text(events) == fenced + "actual reasoning"
            assert len(calls) == 1
            assert calls[0].call.name == "read"
            assert calls[0].call.arguments_json == '{"file_path":"/x"}'


def test_unclosed_fence_recovers_complete_tool_call_deterministically_at_eos() -> None:
    source = (
        "```text\nunclosed code sample\n"
        "</think>\n"
        "<tool_call><function=read><parameter=file_path>/x</parameter></function></tool_call>"
    )
    parser = QwenIncrementalParser("req-1", start_in_reasoning=True)
    streamed = list(parser.feed(source))
    finished = parser.finish()
    events = [*streamed, *finished.events]

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert _reasoning_text(events) == "```text\nunclosed code sample\n"
    assert len(calls) == 1
    assert calls[0].call.name == "read"
    assert calls[0].call.arguments_json == '{"file_path":"/x"}'


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


def _raw_parameter_tool_call(value: str, *, trailing_parameter: bool = False) -> str:
    tail = ""
    if trailing_parameter:
        tail = "<parameter=description>trace parser</parameter>"
    return (
        "<tool_call><function=bash><parameter=command>"
        + value
        + "</parameter>"
        + tail
        + "</function></tool_call>"
    )


def test_parameter_close_inside_single_quoted_source_is_raw_data() -> None:
    value = "text = '<parameter=x>1</parameter>\\n</function>'\nprint(text)"
    events, incomplete = _parse([_raw_parameter_tool_call(value)])
    calls = _completed_calls(events)

    assert incomplete is False
    assert len(calls) == 1
    assert calls[0].call.arguments_json == canonical_json_dumps({"command": value})


def test_exact_dsh_heredoc_parameter_collision_is_preserved() -> None:
    command = (
        "PYTHONPATH=src:/tmp/exqshim python3 - <<'EOF'\n"
        "from exqserve.model.qwen import _QwenToolCallParser, _QwenToolState\n\n"
        'p = _QwenToolCallParser("req-1", 0, {})\n'
        "text = '\\n<function=a>\\n<parameter=x>1</parameter>\\n</function>\\n<\\\\tool_call>\\n"
        "<tool_call>\\n<function=b>\\n<parameter=y>2</parameter>\\n</function>\\n<\\\\tool_call>'\n"
        "r = p.feed(text)\n"
        'print("closed:", r.closed, "completed:", r.completed_call, "incomplete:", r.incomplete)\n'
        'print("state:", p._state)\n'
        'print("buffer:", repr(p._buffer))\n'
        'print("events:", [type(e).__name__ for e in r.events])\n'
        "EOF"
    )
    source = _raw_parameter_tool_call(command, trailing_parameter=True)

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert len(calls) == 1, split
        assert calls[0].call.name == "bash", split
        assert calls[0].call.arguments_json == canonical_json_dumps(
            {"command": command, "description": "trace parser"}
        ), split


def test_multiple_fake_parameter_closes_are_scanned_until_valid_envelope_close() -> None:
    value = (
        "first </parameter> ordinary source\n"
        "second </parameter> </function> not-a-tool-close\n"
        "third </parameter> <parameter=bad name> still source"
    )
    events, incomplete = _parse([_raw_parameter_tool_call(value)])
    calls = _completed_calls(events)

    assert incomplete is False
    assert len(calls) == 1
    assert calls[0].call.arguments_json == canonical_json_dumps({"command": value})


def test_ambiguous_parameter_close_waits_for_partial_envelope_then_rejects_it_as_data() -> None:
    value = "literal </parameter> </fun"
    suffix = "ction> not-structural\nmore source"
    source = _raw_parameter_tool_call(value + suffix)
    split = source.index("</fun") + len("</fun")

    parser = QwenIncrementalParser("req-1")
    first_events = list(parser.feed(source[:split]))
    assert _completed_calls(first_events) == []

    events = [*first_events, *parser.feed(source[split:])]
    finished = parser.finish()
    events.extend(finished.events)
    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert len(calls) == 1
    assert calls[0].call.arguments_json == canonical_json_dumps({"command": value + suffix})


def test_genuine_parameter_close_continuations_and_malformed_suffix_are_distinguished() -> None:
    next_parameter = _raw_parameter_tool_call("echo ok", trailing_parameter=True)
    events, incomplete = _parse([next_parameter])
    assert incomplete is False
    assert len(_completed_calls(events)) == 1

    final_parameter = _raw_parameter_tool_call("echo ok")
    events, incomplete = _parse([final_parameter])
    assert incomplete is False
    assert len(_completed_calls(events)) == 1

    malformed = "<tool_call><function=bash><parameter=command>echo ok</parameter></function>"
    events, incomplete = _parse([malformed])
    assert incomplete is True
    assert _completed_calls(events) == []


def test_complete_tool_close_literal_with_later_real_close_fails_closed_across_splits() -> None:
    source = _raw_parameter_tool_call("before </parameter></function></tool_call> after")

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is True, split
        assert _completed_calls(events) == [], split
        assert not any(isinstance(event, ToolCallStarted) for event in events), split


def test_single_full_close_candidate_completes_and_replays_text_remainder_across_splits() -> None:
    source = _raw_parameter_tool_call("echo ok") + " ordinary remainder"

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert len(calls) == 1, split
        assert calls[0].call.arguments_json == '{"command":"echo ok"}', split
        assert _text(events) == " ordinary remainder", split


def test_back_to_back_tool_calls_keep_order_across_splits() -> None:
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert [(call.call.name, call.call.arguments_json) for call in calls] == [
            ("a", '{"x":1}'),
            ("b", '{"y":2}'),
        ], split


def test_tool_text_tool_keeps_calls_and_replays_text_across_splits() -> None:
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        " ordinary text "
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert [(call.call.name, call.call.arguments_json) for call in calls] == [
            ("a", '{"x":1}'),
            ("b", '{"y":2}'),
        ], split
        assert _text(events) == " ordinary text ", split


def test_tool_text_tool_like_raw_literal_never_commits_side_effects_across_splits() -> None:
    source = _raw_parameter_tool_call(
        "before </parameter></function></tool_call> ordinary "
        "<tool_call><function=fake> literal"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is True, split
        assert _completed_calls(events) == [], split
        assert not any(isinstance(event, ToolCallStarted) for event in events), split


def test_complete_fake_tool_inside_raw_literal_still_fails_closed_across_splits() -> None:
    source = _raw_parameter_tool_call(
        "before </parameter></function></tool_call> ordinary "
        "<tool_call><function=fake><parameter=x>1</parameter></function></tool_call> literal"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is True, split
        assert _completed_calls(events) == [], split
        assert not any(isinstance(event, ToolCallStarted) for event in events), split


def test_immediate_complete_fake_tool_inside_raw_literal_still_fails_closed_across_splits() -> None:
    source = _raw_parameter_tool_call(
        "before </parameter></function></tool_call>"
        "<tool_call><function=fake><parameter=x>1</parameter></function></tool_call> literal"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is True, split
        assert _completed_calls(events) == [], split
        assert not any(isinstance(event, ToolCallStarted) for event in events), split


@pytest.mark.parametrize("depth", (150, 192, 600))
def test_nested_raw_ambiguity_fails_closed_without_recursive_parser_nesting(depth: int) -> None:
    source = "<tool_call><function=leaf><parameter=x>ok</parameter></function></tool_call>"
    for index in range(depth):
        source = (
            f"<tool_call><function=f{index}><parameter=x>"
            "before </parameter></function></tool_call>"
            + source
            + " after</parameter></function></tool_call>"
        )

    events, incomplete = _parse([source])
    assert incomplete is True
    assert _completed_calls(events) == []
    assert not any(isinstance(event, ToolCallStarted) for event in events)


@pytest.mark.parametrize("literal_close", ("</tool_call>", "</function>", "</parameter>"))
def test_tool_text_tool_preserves_top_level_literal_closes_across_splits(literal_close: str) -> None:
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        f" ordinary {literal_close} literal "
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, (literal_close, split)
        assert [(call.call.name, call.call.arguments_json) for call in calls] == [
            ("a", '{"x":1}'),
            ("b", '{"y":2}'),
        ], (literal_close, split)
        assert _text(events) == f" ordinary {literal_close} literal ", (literal_close, split)


def test_tool_text_tool_preserves_backticked_top_level_literal_close_across_splits() -> None:
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        " discuss `</tool_call>` literally "
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert [call.call.name for call in calls] == ["a", "b"], split
        assert _text(events) == " discuss `</tool_call>` literally ", split


def test_tool_text_tool_preserves_backticked_full_close_chain_across_splits() -> None:
    literal = "`</parameter></function></tool_call>`"
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        f" discuss {literal} literally "
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert [call.call.name for call in calls] == ["a", "b"], split
        assert _text(events) == f" discuss {literal} literally ", split


def test_tool_text_tool_preserves_fenced_full_close_chain_across_splits() -> None:
    middle = "\n```text\n</parameter></function></tool_call>\n```\n"
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        + middle
        + "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert [call.call.name for call in calls] == ["a", "b"], split
        assert _text(events) == middle, split


def test_tool_text_tool_preserves_inline_literal_full_close_and_tool_envelope_across_splits() -> None:
    middle = (
        " discuss `</parameter></function></tool_call> "
        "<tool_call><function=read><parameter=file_path>/tmp/literal</parameter></function></tool_call>` "
        "literally "
    )
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        + middle
        + "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert [(call.call.name, call.call.arguments_json) for call in calls] == [
            ("a", '{"x":1}'),
            ("b", '{"y":2}'),
        ], split
        assert _text(events) == middle, split


def test_tool_text_tool_preserves_fenced_literal_full_close_and_tool_envelope_across_splits() -> None:
    middle = (
        "\n```text\n"
        "</parameter></function></tool_call>\n"
        "<tool_call><function=read><parameter=file_path>/tmp/literal</parameter></function></tool_call>\n"
        "```\n"
    )
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        + middle
        + "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert [(call.call.name, call.call.arguments_json) for call in calls] == [
            ("a", '{"x":1}'),
            ("b", '{"y":2}'),
        ], split
        assert _text(events) == middle, split


def test_tool_text_tool_plain_full_close_chain_stays_fail_closed_across_splits() -> None:
    source = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        " ordinary </parameter></function></tool_call> literal "
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is True, split
        assert _completed_calls(events) == [], split
        assert not any(isinstance(event, ToolCallStarted) for event in events), split


def test_declared_unseen_next_parameter_has_structural_precedence() -> None:
    tool = FunctionTool(
        "f",
        None,
        JsonSchema(
            '{"type":"object","properties":{'
            '"x":{"type":"string"},"y":{"type":"string"}'
            '},"required":["x","y"],"additionalProperties":false}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    source = (
        "<tool_call><function=f>"
        "<parameter=x>one</parameter>"
        "<parameter=y>two</parameter>"
        "</function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], tool_policy=policy)
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert len(calls) == 1, split
        assert calls[0].call.arguments_json == '{"x":"one","y":"two"}', split


def test_closed_schema_undeclared_next_parameter_candidate_stays_raw_across_splits() -> None:
    tool = FunctionTool(
        "f",
        None,
        JsonSchema(
            '{"type":"object","properties":{'
            '"content":{"type":"string"},"file_path":{"type":"string"}'
            '},"required":["content","file_path"],"additionalProperties":false}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    value = "before </parameter><parameter=n>fake</parameter> after"
    source = (
        "<tool_call><function=f>"
        f"<parameter=content>{value}</parameter>"
        "<parameter=file_path>/tmp/out</parameter>"
        "</function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], tool_policy=policy)
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert len(calls) == 1, split
        assert calls[0].call.arguments_json == canonical_json_dumps(
            {"content": value, "file_path": "/tmp/out"}
        ), split


def test_pattern_properties_keep_dynamic_next_parameter_structural_across_splits() -> None:
    tool = FunctionTool(
        "f",
        None,
        JsonSchema(
            '{"type":"object","properties":{"x":{"type":"string"}},'
            '"patternProperties":{"^dyn_":{"type":"string"}},'
            '"additionalProperties":false}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    source = (
        "<tool_call><function=f>"
        "<parameter=x>one</parameter>"
        "<parameter=dyn_name>two</parameter>"
        "</function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], tool_policy=policy)
        calls = _completed_calls(events)
        assert incomplete is False, split
        assert len(calls) == 1, split
        assert calls[0].call.arguments_json == '{"x":"one","dyn_name":"two"}', split


def test_dynamic_object_parameters_keep_structural_precedence_across_splits() -> None:
    schemas = (
        '{"type":"object","additionalProperties":true}',
        '{"type":"object","properties":{"known":{"type":"string"}}}',
    )
    source = (
        "<tool_call><function=f>"
        "<parameter=a>1</parameter>"
        "<parameter=b>2</parameter>"
        "</function></tool_call>"
    )

    for schema in schemas:
        tool = FunctionTool("f", None, JsonSchema(schema))
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
        for split in range(len(source) + 1):
            events, incomplete = _parse([source[:split], source[split:]], tool_policy=policy)
            calls = _completed_calls(events)
            assert incomplete is False, (schema, split)
            assert len(calls) == 1, (schema, split)
            assert calls[0].call.arguments_json == '{"a":1,"b":2}', (schema, split)


@pytest.mark.parametrize("raw_value", ("123", "true", "null"))
def test_dynamic_string_schema_preserves_json_literal_text_across_splits(raw_value: str) -> None:
    tool = FunctionTool(
        "f",
        None,
        JsonSchema(
            '{"type":"object","properties":{"count":{"type":"integer"}},'
            '"additionalProperties":{"type":"string"}}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    source = (
        "<tool_call><function=f>"
        "<parameter=count>7</parameter>"
        f"<parameter=dynamic>{raw_value}</parameter>"
        "</function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], tool_policy=policy)
        calls = _completed_calls(events)
        assert incomplete is False, (raw_value, split)
        assert len(calls) == 1, (raw_value, split)
        assert calls[0].call.arguments_json == f'{{"count":7,"dynamic":"{raw_value}"}}', (
            raw_value,
            split,
        )


def test_fake_next_parameter_then_duplicate_real_parameter_fails_closed() -> None:
    tool = FunctionTool(
        "bash",
        None,
        JsonSchema(
            '{"type":"object","properties":{'
            '"command":{"type":"string"},"description":{"type":"string"}'
            '},"required":["command"],"additionalProperties":false}'
        ),
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)
    source = (
        "<tool_call><function=bash><parameter=command>"
        "before </parameter><parameter=description>fake</parameter> after"
        "</parameter><parameter=description>real</parameter>"
        "</function></tool_call>"
    )

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]], tool_policy=policy)
        assert incomplete is True, split
        assert _completed_calls(events) == [], split


def _native_span(text: str, marker: str, token_id: int) -> NativeTokenSpan:
    start = text.index(marker)
    return NativeTokenSpan(start, start + len(marker), token_id, marker)


def _native_tool_spans(text: str) -> tuple[NativeTokenSpan, ...]:
    marker = "<tool_call>"
    spans: list[NativeTokenSpan] = []
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return tuple(spans)
        spans.append(NativeTokenSpan(start, start + len(marker), 248058, marker))
        cursor = start + len(marker)


def test_native_aware_marker_is_candidate_with_backtick_literal_veto() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    first = "The format is `</think>` and still reasoning.\n"
    literal_span = _native_span(first, "</think>", 248069)
    events = list(parser.feed_with_native_tokens(first, (literal_span,)))

    close = "actual close</think>\n"
    close_span = _native_span(close, "</think>", 248069)
    events.extend(parser.feed_with_native_tokens(close, (close_span,)))

    tool = "<tool_call><function=read><parameter=file_path>/x</parameter></function></tool_call>"
    tool_span = _native_span(tool, "<tool_call>", 248058)
    events.extend(parser.feed_with_native_tokens(tool, (tool_span,)))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == "The format is `</think>` and still reasoning.\nactual close"
    calls = _completed_calls(events)
    assert len(calls) == 1
    assert calls[0].call.name == "read"


@pytest.mark.parametrize(
    ("marker", "token_id", "start_in_reasoning"),
    [
        ("<think>", 248068, False),
        ("</think>", 248069, True),
        ("<tool_call>", 248058, True),
    ],
)
def test_native_multiline_inline_marker_is_literal_at_every_valid_split(
    marker: str,
    token_id: int,
    start_in_reasoning: bool,
) -> None:
    source = f"Example: `abc\n{marker} tail` done"
    marker_start = source.index(marker)
    marker_end = marker_start + len(marker)

    for split in range(len(source) + 1):
        if marker_start < split < marker_end:
            continue
        parser = QwenIncrementalParser(
            "req-native-multiline",
            start_in_reasoning=start_in_reasoning,
        )
        events: list[GenerationEvent] = []
        first = source[:split]
        second = source[split:]
        first_spans = (
            (NativeTokenSpan(marker_start, marker_end, token_id, marker),)
            if marker_end <= split
            else ()
        )
        second_spans = (
            (
                NativeTokenSpan(
                    marker_start - split,
                    marker_end - split,
                    token_id,
                    marker,
                ),
            )
            if marker_start >= split
            else ()
        )
        events.extend(parser.feed_with_native_tokens(first, first_spans))
        events.extend(parser.feed_with_native_tokens(second, second_spans))
        finished = parser.finish()
        events.extend(finished.events)

        assert finished.incomplete_tool_call is False, split
        if start_in_reasoning:
            assert _reasoning_text(events) == source, split
            assert _text(events) == "", split
        else:
            assert _reasoning_text(events) == "", split
            assert _text(events) == source, split
        assert not _completed_calls(events), split


def test_native_multiline_double_backtick_marker_is_literal_at_every_valid_split() -> None:
    marker = "</think>"
    token_id = 248069
    source = f"Example: ``abc\n{marker} tail`` done"
    marker_start = source.index(marker)
    marker_end = marker_start + len(marker)

    for split in range(len(source) + 1):
        if marker_start < split < marker_end:
            continue
        parser = QwenIncrementalParser("req-native-multiline-double", start_in_reasoning=True)
        first = source[:split]
        second = source[split:]
        first_spans = (
            (NativeTokenSpan(marker_start, marker_end, token_id, marker),)
            if marker_end <= split
            else ()
        )
        second_spans = (
            (
                NativeTokenSpan(
                    marker_start - split,
                    marker_end - split,
                    token_id,
                    marker,
                ),
            )
            if marker_start >= split
            else ()
        )
        events = list(parser.feed_with_native_tokens(first, first_spans))
        events.extend(parser.feed_with_native_tokens(second, second_spans))
        events.extend(parser.finish().events)

        assert _reasoning_text(events) == source, split
        assert _text(events) == "", split


def test_native_multiline_captured_trace_104_105_keeps_end_think_literal() -> None:
    parser = QwenIncrementalParser("req_fd718d04b137457d93e9596978255ed4", start_in_reasoning=True)
    first = " think the format is:\n- Tool call 1: `<tool_call><"
    first_tool = _native_span(first, "<tool_call>", 248058)
    second = "function=a>...</function>\n</think>\n\n<tool_call>`\n- Tool call 2: `<tool_call"
    end_think = _native_span(second, "</think>", 248069)

    events = list(parser.feed_with_native_tokens(first, (first_tool,)))
    events.extend(parser.feed_with_native_tokens(second, (end_think,)))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert _reasoning_text(events) == first + second
    assert _text(events) == ""
    assert not _completed_calls(events)


def test_native_real_end_think_after_closed_multiline_inline_span_stays_structural() -> None:
    parser = QwenIncrementalParser("req-native-multiline-real-close", start_in_reasoning=True)
    example = "Example: `abc\n</think>` still reasoning."
    example_span = _native_span(example, "</think>", 248069)
    real_close = "</think>final"
    real_close_span = _native_span(real_close, "</think>", 248069)

    events = list(parser.feed_with_native_tokens(example, (example_span,)))
    events.extend(parser.feed_with_native_tokens(real_close, (real_close_span,)))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == example
    assert _text(events) == "final"


def test_native_unmatched_multiline_inline_opener_reports_ambiguity_at_eos() -> None:
    parser = QwenIncrementalParser("req-native-unmatched-inline", start_in_reasoning=True)
    prefix = "analysis `open\n"
    marker_chunk = "</think>after marker"
    marker_span = _native_span(marker_chunk, "</think>", 248069)

    events = list(parser.feed_with_native_tokens(prefix, ()))
    pending = list(parser.feed_with_native_tokens(marker_chunk, (marker_span,)))
    assert not any(isinstance(event, ReasoningCompleted) for event in pending)
    events.extend(pending)

    finished = parser.finish()
    events.extend(finished.events)

    assert _reasoning_text(events) == prefix
    assert _text(events) == ""
    assert finished.incomplete_tool_call is False
    assert finished.terminal_issue is not None
    assert finished.terminal_issue.kind is ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
    assert finished.terminal_issue.ambiguity_detail is ParserAmbiguityDetail.UNRESOLVED_BOUNDARY


@pytest.mark.parametrize(("opener", "longer_run"), [("`", "```"), ("``", "```")])
def test_native_multiline_inline_requires_exact_width_close(
    opener: str,
    longer_run: str,
) -> None:
    parser = QwenIncrementalParser("req-native-exact-inline-close", start_in_reasoning=True)
    prefix = f"analysis {opener}open\n"
    marker_chunk = f"</think>{longer_run} final"
    marker_span = _native_span(marker_chunk, "</think>", 248069)

    events = list(parser.feed_with_native_tokens(prefix, ()))
    events.extend(parser.feed_with_native_tokens(marker_chunk, (marker_span,)))
    finished = parser.finish()
    events.extend(finished.events)

    assert _reasoning_text(events) == prefix
    assert _text(events) == ""
    assert finished.terminal_issue is not None
    assert finished.terminal_issue.kind is ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
    assert finished.terminal_issue.ambiguity_detail is ParserAmbiguityDetail.UNRESOLVED_BOUNDARY


@pytest.mark.parametrize(("opener", "longer_run"), [("`", "```"), ("``", "```")])
def test_plain_multiline_inline_requires_exact_width_close(opener: str, longer_run: str) -> None:
    parser = QwenIncrementalParser("req-plain-exact-inline-close", start_in_reasoning=True)
    source = f"analysis {opener}open\n</think>{longer_run} final"

    events = list(parser.feed(source))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == f"analysis {opener}open\n"
    assert _text(events) == f"{longer_run} final"


def test_native_aware_back_to_back_tool_calls_preserve_verified_second_opener() -> None:
    parser = QwenIncrementalParser("req-native-back-to-back")
    source = (
        "<tool_call>\n<function=read>\n<parameter=file_path>/README.md</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=read>\n<parameter=file_path>/engine.py</parameter>\n"
        "</function>\n</tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [(call.call.index, call.call.name, call.call.arguments_json) for call in calls] == [
        (0, "read", '{"file_path":"/README.md"}'),
        (1, "read", '{"file_path":"/engine.py"}'),
    ]
    assert "<tool_call>" not in _text(events)
    assert "<function=" not in _text(events)


def test_native_aware_tool_text_tool_preserves_verified_second_opener() -> None:
    parser = QwenIncrementalParser("req-native-tool-text-tool")
    source = (
        "<tool_call>\n<function=read>\n<parameter=file_path>/README.md</parameter>\n"
        "</function>\n</tool_call>\n"
        "ordinary text\n"
        "<tool_call>\n<function=read>\n<parameter=file_path>/engine.py</parameter>\n"
        "</function>\n</tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [(call.call.index, call.call.name, call.call.arguments_json) for call in calls] == [
        (0, "read", '{"file_path":"/README.md"}'),
        (1, "read", '{"file_path":"/engine.py"}'),
    ]
    assert _text(events) == "\nordinary text\n"
    assert "<tool_call>" not in _text(events)


def test_native_aware_tool_text_tool_like_raw_literal_never_commits_side_effects() -> None:
    parser = QwenIncrementalParser("req-native-tool-text-tool-literal")
    source = _raw_parameter_tool_call(
        "before </parameter></function></tool_call> ordinary "
        "<tool_call><function=fake> literal"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is True
    assert _completed_calls(events) == []
    assert not any(isinstance(event, ToolCallStarted) for event in events)


@pytest.mark.parametrize("literal_close", ("</tool_call>", "</function>", "</parameter>"))
def test_native_aware_tool_text_tool_preserves_top_level_literal_closes(literal_close: str) -> None:
    parser = QwenIncrementalParser(f"req-native-literal-close-{literal_close}")
    source = (
        "<tool_call><function=read><parameter=file_path>/README.md</parameter></function></tool_call>"
        f" ordinary {literal_close} literal "
        "<tool_call><function=read><parameter=file_path>/engine.py</parameter></function></tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [(call.call.index, call.call.arguments_json) for call in calls] == [
        (0, '{"file_path":"/README.md"}'),
        (1, '{"file_path":"/engine.py"}'),
    ]
    assert _text(events) == f" ordinary {literal_close} literal "


def test_native_aware_tool_text_tool_preserves_backticked_top_level_literal_close() -> None:
    parser = QwenIncrementalParser("req-native-backticked-literal-close")
    source = (
        "<tool_call><function=read><parameter=file_path>/README.md</parameter></function></tool_call>"
        " discuss `</tool_call>` literally "
        "<tool_call><function=read><parameter=file_path>/engine.py</parameter></function></tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert [call.call.index for call in _completed_calls(events)] == [0, 1]
    assert _text(events) == " discuss `</tool_call>` literally "


def test_native_aware_tool_text_tool_preserves_backticked_full_close_chain() -> None:
    parser = QwenIncrementalParser("req-native-backticked-full-close")
    middle = " discuss `</parameter></function></tool_call>` literally "
    source = (
        "<tool_call><function=read><parameter=file_path>/README.md</parameter></function></tool_call>"
        + middle
        + "<tool_call><function=read><parameter=file_path>/engine.py</parameter></function></tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert [call.call.index for call in _completed_calls(events)] == [0, 1]
    assert _text(events) == middle


def test_native_aware_tool_text_tool_preserves_fenced_full_close_chain() -> None:
    parser = QwenIncrementalParser("req-native-fenced-full-close")
    middle = "\n```text\n</parameter></function></tool_call>\n```\n"
    source = (
        "<tool_call><function=read><parameter=file_path>/README.md</parameter></function></tool_call>"
        + middle
        + "<tool_call><function=read><parameter=file_path>/engine.py</parameter></function></tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert [call.call.index for call in _completed_calls(events)] == [0, 1]
    assert _text(events) == middle


def test_native_aware_inline_literal_full_close_and_tool_envelope_stays_text() -> None:
    parser = QwenIncrementalParser("req-native-inline-full-close-tool-literal")
    middle = (
        " discuss `</parameter></function></tool_call> "
        "<tool_call><function=read><parameter=file_path>/tmp/literal</parameter></function></tool_call>` "
        "literally "
    )
    source = (
        "<tool_call><function=read><parameter=file_path>/README.md</parameter></function></tool_call>"
        + middle
        + "<tool_call><function=read><parameter=file_path>/engine.py</parameter></function></tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [call.call.arguments_json for call in calls] == [
        '{"file_path":"/README.md"}',
        '{"file_path":"/engine.py"}',
    ]
    assert _text(events) == middle


def test_native_aware_fenced_literal_full_close_and_tool_envelope_stays_text() -> None:
    parser = QwenIncrementalParser("req-native-fenced-full-close-tool-literal")
    middle = (
        "\n```text\n"
        "</parameter></function></tool_call>\n"
        "<tool_call><function=read><parameter=file_path>/tmp/literal</parameter></function></tool_call>\n"
        "```\n"
    )
    source = (
        "<tool_call><function=read><parameter=file_path>/README.md</parameter></function></tool_call>"
        + middle
        + "<tool_call><function=read><parameter=file_path>/engine.py</parameter></function></tool_call>"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [call.call.arguments_json for call in calls] == [
        '{"file_path":"/README.md"}',
        '{"file_path":"/engine.py"}',
    ]
    assert _text(events) == middle


def test_native_aware_three_back_to_back_tool_calls_replay_recursively() -> None:
    parser = QwenIncrementalParser("req-native-back-to-back-three")
    source = "".join(
        f"<tool_call><function=read><parameter=file_path>/{name}</parameter></function></tool_call>"
        for name in ("one.py", "two.py", "three.py")
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [call.call.index for call in calls] == [0, 1, 2]
    assert [call.call.arguments_json for call in calls] == [
        '{"file_path":"/one.py"}',
        '{"file_path":"/two.py"}',
        '{"file_path":"/three.py"}',
    ]
    assert "<tool_call>" not in _text(events)


def test_native_aware_captured_back_to_back_shape_keeps_second_call_structural() -> None:
    parser = QwenIncrementalParser("req_a859e0e50ca2440ba6798fc726f9d335", start_in_reasoning=True)
    first = (
        "Let me start by reading the README and the main architecture files. I'll read several in parallel.\n"
        "</think>\n\n<tool_call>\n<function=read>\n<parameter=file_path>\n"
        "/root/workspace/.ai-bridge/qwen-tool-leak-ab/target/README.md\n"
        "</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read>\n<parameter=file"
    )
    second = "_path>\n/root/workspace/.ai-bridge/qwen-tool-leak-ab/target/src/exqserve/s"
    third = "erving/engine.py\n</parameter>\n</function>\n</tool_call>"
    first_spans = (
        NativeTokenSpan(99, 107, 248069, "</think>"),
        NativeTokenSpan(109, 120, 248058, "<tool_call>"),
        NativeTokenSpan(246, 258, 248059, "</tool_call>"),
        NativeTokenSpan(259, 270, 248058, "<tool_call>"),
    )
    third_spans = (NativeTokenSpan(42, 54, 248059, "</tool_call>"),)

    events = list(parser.feed_with_native_tokens(first, first_spans))
    events.extend(parser.feed_with_native_tokens(second, ()))
    events.extend(parser.feed_with_native_tokens(third, third_spans))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [call.call.index for call in calls] == [0, 1]
    assert calls[0].call.arguments_json == (
        '{"file_path":"/root/workspace/.ai-bridge/qwen-tool-leak-ab/target/README.md"}'
    )
    assert calls[1].call.arguments_json == (
        '{"file_path":"/root/workspace/.ai-bridge/qwen-tool-leak-ab/target/src/exqserve/serving/engine.py"}'
    )
    visible = _reasoning_text(events) + _text(events)
    assert "<tool_call>" not in visible
    assert "<function=read>" not in visible


@pytest.mark.parametrize(
    "chunks",
    [
        (
            "<tool_call>\n<function=read>\n<parameter=file_path>/README.md</parameter>\n</function>\n</tool_call>",
            "\n<tool_call>",
            "\n<function=read>\n<parameter=file_path>/engine.py</parameter>\n</function>\n</tool_call>",
        ),
        (
            "<tool_call>\n<function=read>\n<parameter=file_path>/README.md</parameter>\n</function>\n</tool_call>\n",
            "<tool_call>\n<function=",
            "read>\n<parameter=file_path>/engine.py</parameter>\n</function>\n</tool_call>",
        ),
        (
            "<tool_call>\n<function=read>\n<parameter=file_path>/README.md</parameter>\n</function>\n</tool_call>\n<tool_call>\n<fun",
            "ction=read>\n<parameter=file_path>/engine.py</parameter>\n",
            "</function>\n</tool_call>",
        ),
    ],
)
def test_native_aware_back_to_back_tool_calls_are_chunk_invariant(chunks: tuple[str, ...]) -> None:
    parser = QwenIncrementalParser("req-native-back-to-back-split")
    events: list[GenerationEvent] = []
    for chunk in chunks:
        events.extend(parser.feed_with_native_tokens(chunk, _native_tool_spans(chunk)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert [call.call.index for call in calls] == [0, 1]
    assert [call.call.arguments_json for call in calls] == [
        '{"file_path":"/README.md"}',
        '{"file_path":"/engine.py"}',
    ]
    assert "<tool_call>" not in _text(events)


def test_native_aware_replayed_tool_opener_still_honors_literal_context() -> None:
    parser = QwenIncrementalParser("req-native-back-to-back-literal")
    source = (
        "<tool_call><function=read><parameter=file_path>/README.md</parameter></function></tool_call>"
        "\n`<tool_call><function=read> literal protocol example`"
    )
    events = list(parser.feed_with_native_tokens(source, _native_tool_spans(source)))
    finished = parser.finish()
    events.extend(finished.events)

    calls = _completed_calls(events)
    assert finished.incomplete_tool_call is False
    assert len(calls) == 1
    assert calls[0].call.arguments_json == '{"file_path":"/README.md"}'
    assert "<tool_call><function=read>" in _text(events)
    assert "literal protocol example" in _text(events)


def test_native_aware_ordinary_same_spelling_marker_is_literal_only() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    source = "literal <tool_call> remains prose"
    events = list(parser.feed_with_native_tokens(source, ()))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == source
    assert not _completed_calls(events)


def test_native_aware_unverified_ambiguous_marker_fails_closed() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)

    with pytest.raises(NativeTokenProvenanceError):
        parser.feed_with_native_tokens("ambiguous </think> outside code", None)


def test_native_aware_unverified_marker_inside_backticks_stays_literal() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    source = "code `</think>` remains reasoning"
    events = list(parser.feed_with_native_tokens(source, None))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == source


def test_native_aware_open_inline_barrier_never_retroactively_promotes_marker() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    first = "`` source <tool_call><function=read><parameter=file_path>/x</parameter></function></tool_call>\n"
    literal_span = _native_span(first, "<tool_call>", 248058)
    events = list(parser.feed_with_native_tokens(first, (literal_span,)))

    second = "</think><tool_call><function=read><parameter=file_path>/real</parameter></function></tool_call>"
    close_span = _native_span(second, "</think>", 248069)
    tool_at = second.index("<tool_call>")
    tool_span = NativeTokenSpan(tool_at, tool_at + len("<tool_call>"), 248058, "<tool_call>")
    events.extend(parser.feed_with_native_tokens(second, (close_span, tool_span)))
    finished = parser.finish()
    events.extend(finished.events)

    assert _reasoning_text(events) == "`` source "
    assert not _completed_calls(events)
    assert finished.terminal_issue is not None
    assert finished.terminal_issue.kind is ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
    assert finished.terminal_issue.ambiguity_detail is ParserAmbiguityDetail.UNRESOLVED_BOUNDARY


def test_native_aware_open_fence_reports_ambiguity_without_literal_tool_side_effect() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    source = "```text\nexample\n</think>\n<tool_call><function=read><parameter=file_path>/x</parameter></function></tool_call>"
    spans = (
        _native_span(source, "</think>", 248069),
        _native_span(source, "<tool_call>", 248058),
    )
    events = list(parser.feed_with_native_tokens(source, spans))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.incomplete_tool_call is False
    assert _reasoning_text(events) == "```text\nexample\n"
    assert not _completed_calls(events)
    assert finished.terminal_issue is not None
    assert finished.terminal_issue.kind is ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
    assert finished.terminal_issue.ambiguity_detail is ParserAmbiguityDetail.UNRESOLVED_BOUNDARY


def test_native_fence_tentative_close_same_line_tool_reports_ambiguity() -> None:
    parser = QwenIncrementalParser("req-native-fence-tail", start_in_reasoning=True)
    tool = "<tool_call><function=read><parameter=file_path>/danger</parameter></function></tool_call>"
    source = "```text\nliteral\n```   " + tool
    tool_span = _native_span(source, "<tool_call>", 248058)

    events = list(parser.feed_with_native_tokens(source, (tool_span,)))
    finished = parser.finish()
    events.extend(finished.events)

    assert _reasoning_text(events) == "```text\nliteral\n```   "
    assert not _completed_calls(events)
    assert finished.terminal_issue is not None
    assert finished.terminal_issue.kind is ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
    assert finished.terminal_issue.ambiguity_detail is ParserAmbiguityDetail.UNRESOLVED_BOUNDARY


def test_native_fence_whitespace_newline_close_resolves_literal_then_real_close() -> None:
    parser = QwenIncrementalParser("req-native-fence-close", start_in_reasoning=True)
    source = "```text\nliteral </think>\n```   \t\nstill reasoning</think>final"
    first_at = source.index("</think>")
    second_at = source.index("</think>", first_at + 1)
    spans = (
        NativeTokenSpan(first_at, first_at + len("</think>"), 248069, "</think>"),
        NativeTokenSpan(second_at, second_at + len("</think>"), 248069, "</think>"),
    )

    events = list(parser.feed_with_native_tokens(source, spans))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.terminal_issue is None
    assert _reasoning_text(events) == "```text\nliteral </think>\n```   \t\nstill reasoning"
    assert _text(events) == "final"


def test_native_fence_whitespace_eos_close_resolves_literal() -> None:
    parser = QwenIncrementalParser("req-native-fence-eos", start_in_reasoning=True)
    source = "```text\nliteral </think>\n```   \t"
    marker_span = _native_span(source, "</think>", 248069)

    events = list(parser.feed_with_native_tokens(source, (marker_span,)))
    finished = parser.finish()
    events.extend(finished.events)

    assert finished.terminal_issue is None
    assert _reasoning_text(events) == source
    assert not _completed_calls(events)


def test_native_semantic_hold_exact_65536_resolves_without_overflow() -> None:
    limit = 64 * 1024
    marker = "</think>"
    close = "\n```\n"
    prefix = "SAFE_PREFIX\n```text\n"
    filler = "x" * (limit - len(marker.encode()) - len(close.encode()))
    source = prefix + marker + filler + close
    marker_span = _native_span(source, marker, 248069)
    parser = QwenIncrementalParser("req-native-hold-exact", start_in_reasoning=True)

    events = list(parser.feed_with_native_tokens(source, (marker_span,)))
    finished = parser.finish()
    events.extend(finished.events)

    assert parser.peak_semantic_hold_bytes == limit
    assert finished.terminal_issue is None
    assert _reasoning_text(events) == source


def test_native_semantic_hold_65537th_byte_fails_before_later_close() -> None:
    limit = 64 * 1024
    marker = "</think>"
    close = "\n```\n"
    prefix = "SAFE_PREFIX\n```text\n"
    filler = "x" * (limit - len(marker.encode()) - len(close.encode()) + 1)
    source = prefix + marker + filler + close
    marker_span = _native_span(source, marker, 248069)
    parser = QwenIncrementalParser("req-native-hold-over", start_in_reasoning=True)

    events = list(parser.feed_with_native_tokens(source, (marker_span,)))

    issue = parser.early_terminal_issue
    assert issue is not None
    assert issue.kind is ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
    assert issue.ambiguity_detail is ParserAmbiguityDetail.HOLD_LIMIT
    assert parser.peak_semantic_hold_bytes == limit
    assert _reasoning_text(events) == prefix
    assert not _completed_calls(events)


def test_native_semantic_hold_multibyte_crossing_never_partially_overflows() -> None:
    limit = 64 * 1024
    marker = "</think>"
    prefix = "SAFE_PREFIX\n```text\n"
    filler = "x" * (limit - len(marker.encode()) - 1)
    source = prefix + marker + filler + "你\n```\n"
    marker_span = _native_span(source, marker, 248069)
    parser = QwenIncrementalParser("req-native-hold-utf8", start_in_reasoning=True)

    events = list(parser.feed_with_native_tokens(source, (marker_span,)))

    issue = parser.early_terminal_issue
    assert issue is not None
    assert issue.ambiguity_detail is ParserAmbiguityDetail.HOLD_LIMIT
    assert parser.peak_semantic_hold_bytes == limit - 1
    assert _reasoning_text(events) == prefix


def test_native_fence_tentative_close_is_chunk_invariant() -> None:
    marker = "<tool_call>"
    tool = marker + "<function=read><parameter=file_path>/danger</parameter></function></tool_call>"
    source = "```text\nliteral\n``` \t " + tool
    marker_at = source.index(marker)
    marker_end = marker_at + len(marker)

    for split in range(1, len(source)):
        if marker_at < split < marker_end:
            continue
        parser = QwenIncrementalParser("req-native-fence-partition", start_in_reasoning=True)
        events: list[GenerationEvent] = []
        first = source[:split]
        second = source[split:]
        first_spans = (
            (NativeTokenSpan(marker_at, marker_end, 248058, marker),)
            if marker_end <= split
            else ()
        )
        second_spans = (
            (NativeTokenSpan(marker_at - split, marker_end - split, 248058, marker),)
            if marker_at >= split
            else ()
        )
        events.extend(parser.feed_with_native_tokens(first, first_spans))
        events.extend(parser.feed_with_native_tokens(second, second_spans))
        finished = parser.finish()
        events.extend(finished.events)

        assert _reasoning_text(events) == "```text\nliteral\n``` \t ", split
        assert not _completed_calls(events), split
        assert finished.terminal_issue is not None, split
        assert finished.terminal_issue.ambiguity_detail is ParserAmbiguityDetail.UNRESOLVED_BOUNDARY, split


def test_native_aware_direct_quote_veto_is_chunk_invariant() -> None:
    marker = "</think>"
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    whole = f"quoted '{marker}' still reasoning"
    span = _native_span(whole, marker, 248069)
    events = list(parser.feed_with_native_tokens(whole, (span,)))
    events.extend(parser.finish().events)
    assert _reasoning_text(events) == whole

    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    first = f"quoted '{marker}"
    span = _native_span(first, marker, 248069)
    events = list(parser.feed_with_native_tokens(first, (span,)))
    events.extend(parser.feed_with_native_tokens("' still reasoning", ()))
    events.extend(parser.finish().events)
    assert _reasoning_text(events) == whole

    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    events = list(parser.feed_with_native_tokens("quoted '", ()))
    marker_span = NativeTokenSpan(0, len(marker), 248069, marker)
    events.extend(parser.feed_with_native_tokens(marker, (marker_span,)))
    events.extend(parser.feed_with_native_tokens("' still reasoning", ()))
    events.extend(parser.finish().events)
    assert _reasoning_text(events) == whole


def test_native_aware_unmatched_direct_quote_at_eof_does_not_veto_verified_marker() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    source = "ends with quote '</think>"
    span = _native_span(source, "</think>", 248069)
    events = list(parser.feed_with_native_tokens(source, (span,)))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == "ends with quote '"


def test_native_aware_unmatched_direct_quote_at_eof_fails_closed_without_provenance() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    source = "ends with quote '</think>"
    parser.feed_with_native_tokens(source, None)

    with pytest.raises(NativeTokenProvenanceError):
        parser.finish()


@pytest.mark.parametrize("marker", ["<think>", "</think>", "<tool_call>"])
def test_native_aware_unverified_marker_prefix_split_fails_closed(marker: str) -> None:
    for split in range(1, len(marker)):
        parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
        parser.feed_with_native_tokens("ambiguous " + marker[:split], None)
        with pytest.raises(NativeTokenProvenanceError):
            parser.feed_with_native_tokens(marker[split:] + " outside", None)


def test_native_aware_unverified_prefix_can_finish_in_verified_literal_context() -> None:
    marker = "</think>"
    split = 4
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    events = list(parser.feed_with_native_tokens("code `" + marker[:split], None))
    events.extend(parser.feed_with_native_tokens(marker[split:] + "` remains reasoning", ()))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == "code `</think>` remains reasoning"


def test_native_aware_verified_ordinary_prefix_does_not_poison_unverified_neighbor() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    events = list(parser.feed_with_native_tokens("literal <tool_", ()))
    events.extend(parser.feed_with_native_tokens("call> remains prose", None))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == "literal <tool_call> remains prose"


def test_native_marker_must_pass_dialect_state_validation() -> None:
    parser = QwenIncrementalParser("req-native", start_in_reasoning=True)
    nested = "nested <think> stays reasoning"
    nested_span = _native_span(nested, "<think>", 248068)
    events = list(parser.feed_with_native_tokens(nested, (nested_span,)))

    close = "</think>final literal </think>"
    first_close = NativeTokenSpan(0, len("</think>"), 248069, "</think>")
    second_at = close.rindex("</think>")
    second_close = NativeTokenSpan(
        second_at,
        second_at + len("</think>"),
        248069,
        "</think>",
    )
    events.extend(parser.feed_with_native_tokens(close, (first_close, second_close)))
    events.extend(parser.finish().events)

    assert _reasoning_text(events) == nested
    assert _text(events) == "final literal </think>"
