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
from exqserve.model.contracts import TemplateTool
from exqserve.model.glm5 import Glm5IncrementalParser


def _tool(
    name: str,
    properties: str,
) -> TemplateTool:
    return TemplateTool(
        name,
        None,
        '{"type":"object","properties":' + properties + "}",
    )


def _parse(
    chunks: Iterable[str],
    *,
    start_in_reasoning: bool = True,
    tools: tuple[TemplateTool, ...] = (),
) -> tuple[list[GenerationEvent], bool]:
    parser = Glm5IncrementalParser(
        "req-glm5",
        start_in_reasoning=start_in_reasoning,
        tools=tools,
    )
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


def test_glm5_generation_prompt_preopened_think_is_parsed_without_open_marker() -> None:
    events, incomplete = _parse(["Need to reason.</think>Final answer"])

    assert incomplete is False
    assert _reasoning(events) == "Need to reason."
    assert _text(events) == "Final answer"
    assert [type(event) for event in events] == [
        ReasoningStarted,
        ReasoningDelta,
        ReasoningCompleted,
        TextStarted,
        TextDelta,
        TextCompleted,
    ]


def test_glm5_explicit_think_marker_is_tolerated_and_chunk_safe() -> None:
    source = "<think>分析</think>答案"
    baseline, _ = _parse([source], start_in_reasoning=False)

    for split in range(len(source) + 1):
        events, incomplete = _parse(
            [source[:split], source[split:]],
            start_in_reasoning=False,
        )
        assert incomplete is False
        assert _reasoning(events) == _reasoning(baseline) == "分析"
        assert _text(events) == _text(baseline) == "答案"


def test_glm5_disabled_reasoning_stays_plain_text() -> None:
    events, incomplete = _parse(["plain answer"], start_in_reasoning=False)

    assert incomplete is False
    assert _reasoning(events) == ""
    assert _text(events) == "plain answer"


def test_glm5_schema_aware_tool_arguments_preserve_optional_string_types() -> None:
    tool = _tool(
        "mix",
        (
            '{"string_id":{"type":"string"},'
            '"optional_id":{"anyOf":[{"type":"string"},{"type":"null"}]},'
            '"count":{"type":"integer"},'
            '"enabled":{"type":"boolean"},'
            '"items":{"type":"array"}}'
        ),
    )
    source = (
        "</think><tool_call>mix"
        "<arg_key>string_id</arg_key><arg_value>123</arg_value>"
        "<arg_key>optional_id</arg_key><arg_value>456</arg_value>"
        "<arg_key>count</arg_key><arg_value>7</arg_value>"
        "<arg_key>enabled</arg_key><arg_value>true</arg_value>"
        "<arg_key>items</arg_key><arg_value>[1,2]</arg_value>"
        "</tool_call>"
    )
    events, incomplete = _parse(list(source), tools=(tool,))
    calls = _calls(events)

    assert incomplete is False
    assert len(calls) == 1
    assert calls[0].call.arguments_json == (
        '{"count":7,"enabled":true,"items":[1,2],'
        '"optional_id":"456","string_id":"123"}'
    )
    assert [type(event) for event in events] == [
        ToolCallStarted,
        ToolCallArgumentsDelta,
        ToolCallCompleted,
    ]


def test_glm5_internal_schema_ref_preserves_numeric_looking_string() -> None:
    tool = TemplateTool(
        "lookup",
        None,
        (
            '{"type":"object","properties":{"id":{"$ref":"#/$defs/StringId"}},'
            '"$defs":{"StringId":{"type":"string"}}}'
        ),
    )
    events, incomplete = _parse(
        ["</think><tool_call>lookup<arg_key>id</arg_key><arg_value>123</arg_value></tool_call>"],
        tools=(tool,),
    )

    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == '{"id":"123"}'



def test_glm5_nullable_string_accepts_actual_null() -> None:
    tool = _tool(
        "lookup",
        '{"id":{"anyOf":[{"type":"string"},{"type":"null"}]}}',
    )
    events, incomplete = _parse(
        ["</think><tool_call>lookup<arg_key>id</arg_key><arg_value>null</arg_value></tool_call>"],
        tools=(tool,),
    )

    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == '{"id":null}'


def test_glm5_zero_argument_tool_call_is_valid() -> None:
    tool = _tool("ping", "{}")
    events, incomplete = _parse(
        ["</think><tool_call>ping</tool_call>"],
        tools=(tool,),
    )

    assert incomplete is False
    calls = _calls(events)
    assert len(calls) == 1
    assert calls[0].call.name == "ping"
    assert calls[0].call.arguments_json == "{}"


def test_glm5_parallel_calls_are_repeated_and_indexed() -> None:
    first = _tool("weather", '{"city":{"type":"string"}}')
    second = _tool("clock", '{"city":{"type":"string"}}')
    source = (
        "</think>"
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        "<tool_call>clock<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>"
    )
    events, incomplete = _parse(list(source), tools=(first, second))
    calls = [event.call for event in _calls(events)]

    assert incomplete is False
    assert [(call.name, call.index, call.arguments_json) for call in calls] == [
        ("weather", 0, '{"city":"Paris"}'),
        ("clock", 1, '{"city":"Tokyo"}'),
    ]
    assert calls[0].call_id != calls[1].call_id


def test_glm5_incomplete_or_malformed_tool_call_never_fabricates_completion() -> None:
    tool = _tool("lookup", '{"id":{"type":"integer"}}')

    events, incomplete = _parse(
        ["</think><tool_call>lookup<arg_key>id</arg_key><arg_value>7"],
        tools=(tool,),
    )
    assert incomplete is True
    assert _calls(events) == []

    events, incomplete = _parse(
        [
            (
                "</think><tool_call>lookup"
                "<arg_key>id</arg_key>GARBAGE<arg_value>7</arg_value>"
                "</tool_call>"
            )
        ],
        tools=(tool,),
    )
    assert incomplete is True
    assert _calls(events) == []


def test_glm5_tool_name_with_whitespace_is_rejected_as_malformed() -> None:
    tool = _tool("lookup", "{}")
    events, incomplete = _parse(
        ["</think><tool_call>lookup extra</tool_call>"],
        tools=(tool,),
    )

    assert incomplete is True
    assert _calls(events) == []



def test_glm5_missing_arg_value_open_tag_is_incomplete_not_silently_empty() -> None:
    tool = _tool("search", '{"query":{"type":"string"}}')
    events, incomplete = _parse(
        [
            (
                "</think><tool_call>search<arg_key>query</arg_key>"
                "how many vacation days left</arg_value></tool_call>"
            )
        ],
        tools=(tool,),
    )

    assert incomplete is True
    assert _calls(events) == []



def test_glm5_tool_marker_is_plain_text_when_no_tools_were_exposed() -> None:
    events, incomplete = _parse(
        ["literal <tool_call>example</tool_call>"],
        start_in_reasoning=False,
        tools=(),
    )

    assert incomplete is False
    assert _calls(events) == []
    assert _text(events) == "literal <tool_call>example</tool_call>"


def test_glm5_finish_is_idempotent() -> None:
    parser = Glm5IncrementalParser("req-glm5", start_in_reasoning=False)
    parser.feed("hello")
    first = parser.finish()
    second = parser.finish()

    assert first.incomplete_tool_call is False
    assert any(isinstance(event, TextCompleted) for event in first.events)
    assert second.events == ()
    assert second.incomplete_tool_call is False
