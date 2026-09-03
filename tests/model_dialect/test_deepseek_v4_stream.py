from __future__ import annotations

import json
from collections.abc import Iterable

from exqserve.core.events import GenerationEvent, ReasoningDelta, TextDelta, ToolCallCompleted
from exqserve.model.deepseek_v4 import DeepSeekV4IncrementalParser, DeepSeekV4ParserContext

_DSML = "｜DSML｜"


_TEST_TOOL_PROPERTIES = {
    "mix": frozenset({"id", "count", "enabled", "items"}),
    "ping": frozenset(),
    "first": frozenset({"x"}),
    "second": frozenset({"n"}),
    "lookup": frozenset({"id", "count", "query", "offset"}),
    "wrapped": frozenset({"city"}),
    "real_arguments": frozenset({"arguments"}),
    "tool_a": frozenset({"city"}),
    "tool_b": frozenset({"tz"}),
}


def _parse(
    chunks: Iterable[str],
    *,
    reasoning: bool = True,
    tools_enabled: bool = True,
    tool_properties: dict[str, frozenset[str]] | None = None,
) -> tuple[list[GenerationEvent], bool]:
    properties = _TEST_TOOL_PROPERTIES if tool_properties is None else tool_properties
    context = DeepSeekV4ParserContext(tools_enabled, properties if tools_enabled else {})
    parser = DeepSeekV4IncrementalParser(
        "req-dsv4",
        start_in_reasoning=reasoning,
        parser_context=context,
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


def _tool_block(invokes: str) -> str:
    return f"<{_DSML}tool_calls>\n{invokes}\n</{_DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    middle = f"\n{params}" if params else ""
    return f'<{_DSML}invoke name="{name}">{middle}\n</{_DSML}invoke>'


def _param(name: str, value: str, *, string: bool) -> str:
    flag = "true" if string else "false"
    return f'<{_DSML}parameter name="{name}" string="{flag}">{value}</{_DSML}parameter>'


def test_deepseek_v4_generation_begins_inside_preopened_think() -> None:
    events, incomplete = _parse(["reasoning</think>answer"])

    assert incomplete is False
    assert _reasoning(events) == "reasoning"
    assert _text(events) == "answer"


def test_deepseek_v4_duplicate_explicit_think_marker_is_tolerated() -> None:
    events, incomplete = _parse(["<think>reasoning</think>answer"])

    assert incomplete is False
    assert _reasoning(events) == "reasoning"
    assert _text(events) == "answer"


def test_deepseek_v4_dsml_string_flag_controls_json_argument_type() -> None:
    source = (
        "reason</think>"
        + _tool_block(
            _invoke(
                "mix",
                "\n".join(
                    (
                        _param("id", "123", string=True),
                        _param("count", "-7", string=False),
                        _param("enabled", "true", string=False),
                        _param("items", "[1, 2]", string=False),
                    )
                ),
            )
        )
    )
    events, incomplete = _parse([source])
    calls = _calls(events)

    assert incomplete is False
    assert len(calls) == 1
    assert calls[0].call.name == "mix"
    assert calls[0].call.arguments_json == (
        '{"count":-7,"enabled":true,"id":"123","items":[1,2]}'
    )


def test_deepseek_v4_zero_argument_tool_call_is_valid() -> None:
    events, incomplete = _parse(["</think>" + _tool_block(_invoke("ping"))])

    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == "{}"


def test_deepseek_v4_parallel_invokes_are_indexed_in_order() -> None:
    source = "</think>" + _tool_block(
        _invoke("first", _param("x", "A", string=True))
        + "\n"
        + _invoke("second", _param("n", "2", string=False))
    )
    events, incomplete = _parse([source])
    calls = [event.call for event in _calls(events)]

    assert incomplete is False
    assert [(call.name, call.index, call.arguments_json) for call in calls] == [
        ("first", 0, '{"x":"A"}'),
        ("second", 1, '{"n":2}'),
    ]
    assert calls[0].call_id != calls[1].call_id


def test_deepseek_v4_accepts_direct_json_invoke_variant_seen_in_serving_frameworks() -> None:
    invoke = f'<{_DSML}invoke name="lookup">\n{{"id":"123","count":2}}\n</{_DSML}invoke>'
    events, incomplete = _parse(["</think>" + _tool_block(invoke)])

    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == '{"count":2,"id":"123"}'


def test_deepseek_v4_direct_json_preserves_dsml_closes_across_every_split() -> None:
    value = (
        f'a </{_DSML}invoke> b </{_DSML}tool_calls> c '
        '"quoted" and \\ escaped'
    )
    payload = json.dumps(
        {"query": value, "nested": {"items": [1, {"text": "ok"}]}},
        ensure_ascii=False,
    )
    invoke = f'<{_DSML}invoke name="lookup">\n{payload}\n</{_DSML}invoke>'
    source = "</think>" + _tool_block(invoke)

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _calls(events)
        assert incomplete is False, split
        assert len(calls) == 1, split
        assert json.loads(calls[0].call.arguments_json) == {
            "query": value,
            "nested": {"items": [1, {"text": "ok"}]},
        }, split


def test_deepseek_v4_parallel_direct_json_scan_state_resets_across_invokes() -> None:
    first_value = f'a </{_DSML}tool_calls> b'
    second_value = f'c </{_DSML}invoke> d'
    first = (
        f'<{_DSML}invoke name="lookup">\n'
        f'{json.dumps({"query": first_value}, ensure_ascii=False)}\n'
        f'</{_DSML}invoke>'
    )
    second = (
        f'<{_DSML}invoke name="first">\n'
        f'{json.dumps({"x": second_value}, ensure_ascii=False)}\n'
        f'</{_DSML}invoke>'
    )
    source = "</think>" + _tool_block(first + "\n" + second)

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        calls = _calls(events)
        assert incomplete is False, split
        assert [json.loads(call.call.arguments_json) for call in calls] == [
            {"query": first_value},
            {"x": second_value},
        ], split


def test_deepseek_v4_unwraps_arguments_wrapper_only_when_schema_proves_it_is_wrapper() -> None:
    wrapped = _invoke(
        "wrapped",
        _param("arguments", '{"city":"Tokyo"}', string=False),
    )
    events, incomplete = _parse(["</think>" + _tool_block(wrapped)])
    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == '{"city":"Tokyo"}'

    real_parameter = _invoke(
        "real_arguments",
        _param("arguments", '{"city":"Tokyo"}', string=False),
    )
    events, incomplete = _parse(["</think>" + _tool_block(real_parameter)])
    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == '{"arguments":{"city":"Tokyo"}}'


def test_deepseek_v4_parallel_wrapper_repair_uses_each_tool_own_schema() -> None:
    shared = _param("arguments", '{"city":"Tokyo"}', string=False)
    source = "</think>" + _tool_block(_invoke("tool_a", shared) + "\n" + _invoke("tool_b", shared))
    events, incomplete = _parse([source])
    calls = [event.call for event in _calls(events)]

    assert incomplete is False
    assert calls[0].arguments_json == '{"city":"Tokyo"}'
    assert calls[1].arguments_json == '{"arguments":{"city":"Tokyo"}}'


def test_deepseek_v4_tool_block_is_safe_across_every_two_chunk_split() -> None:
    source = (
        "reason</think>"
        + _tool_block(
            _invoke(
                "lookup",
                _param("offset", "-7", string=False) + "\n" + _param("query", "abc", string=True),
            )
        )
    )
    baseline, _ = _parse([source])
    baseline_calls = [(event.call.name, event.call.arguments_json) for event in _calls(baseline)]

    for split in range(len(source) + 1):
        events, incomplete = _parse([source[:split], source[split:]])
        assert incomplete is False, split
        assert _reasoning(events) == "reason", split
        assert [(event.call.name, event.call.arguments_json) for event in _calls(events)] == baseline_calls, split


def test_deepseek_v4_json_string_preserves_dsml_close_markers_across_splits() -> None:
    markers = (
        f"</{_DSML}parameter>",
        f"</{_DSML}invoke>",
        f"</{_DSML}tool_calls>",
    )
    for marker in markers:
        value = f'a {marker} b "quoted"'
        json_value = json.dumps(value, ensure_ascii=False)
        source = "</think>" + _tool_block(
            _invoke("lookup", _param("query", json_value, string=False))
        )

        for split in range(len(source) + 1):
            events, incomplete = _parse([source[:split], source[split:]])
            calls = _calls(events)
            assert incomplete is False, (marker, split)
            assert len(calls) == 1, (marker, split)
            assert json.loads(calls[0].call.arguments_json) == {"query": value}, (marker, split)


def test_deepseek_v4_raw_string_dsml_parameter_close_remains_native_wire_limit() -> None:
    source = "</think>" + _tool_block(
        _invoke(
            "lookup",
            _param("query", f"a </{_DSML}parameter> b", string=True),
        )
    )

    events, incomplete = _parse([source])

    assert incomplete is True
    assert _calls(events) == []


def test_deepseek_v4_name_and_arguments_in_same_first_chunk_are_not_lost() -> None:
    source = "</think>" + _tool_block(
        _invoke("lookup", _param("query", "same chunk", string=True))
    )
    split = source.index("same chunk") + len("same chunk")
    events, incomplete = _parse([source[:split], source[split:]])

    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == '{"query":"same chunk"}'


def test_deepseek_v4_bare_invoke_without_tool_calls_start_is_recovered() -> None:
    invoke = _invoke("lookup", _param("query", "recover me", string=True))
    source = f"</think>{invoke}\n</{_DSML}tool_calls>"
    events, incomplete = _parse([source])

    assert incomplete is False
    calls = _calls(events)
    assert len(calls) == 1
    assert calls[0].call.name == "lookup"
    assert calls[0].call.arguments_json == '{"query":"recover me"}'


def test_deepseek_v4_bare_complete_invoke_without_outer_end_is_recovered_on_finish() -> None:
    invoke = _invoke("lookup", _param("query", "finish recovery", string=True))
    events, incomplete = _parse(["</think>" + invoke])

    assert incomplete is False
    assert _calls(events)[0].call.arguments_json == '{"query":"finish recovery"}'


def test_deepseek_v4_known_misspelled_outer_wrapper_is_recovered_without_text_leak() -> None:
    invoke = _invoke("lookup", _param("query", "recover typo", string=True))
    source = f"</think><{_DSML}toolcalls>\n{invoke}\n</{_DSML}tool_calls>"
    events, incomplete = _parse([source])

    assert incomplete is False
    assert _text(events) == ""
    assert _calls(events)[0].call.arguments_json == '{"query":"recover typo"}'


def test_deepseek_v4_recovery_rejects_undeclared_tool_name() -> None:
    invoke = _invoke("not_exposed", _param("query", "nope", string=True))
    events, incomplete = _parse(["</think>" + invoke])

    assert incomplete is True
    assert _calls(events) == []


def test_deepseek_v4_dsml_is_literal_text_when_no_tools_were_exposed() -> None:
    literal = _tool_block(_invoke("lookup", _param("query", "literal", string=True)))
    events, incomplete = _parse([literal], reasoning=False, tools_enabled=False)

    assert incomplete is False
    assert _calls(events) == []
    assert _text(events) == literal


def test_deepseek_v4_malformed_or_incomplete_dsml_never_fabricates_empty_call() -> None:
    malformed = (
        "</think>"
        + f"<{_DSML}tool_calls>\n"
        + f'<{_DSML}invoke name="lookup">\n'
        + f'<{_DSML}parameter name="q" string="true">abc'
        + f"</{_DSML}invoke>\n"
        + f"</{_DSML}tool_calls>"
    )
    events, incomplete = _parse([malformed])

    assert incomplete is True
    assert _calls(events) == []

    partial = f"</think><{_DSML}tool_calls>\n<{_DSML}invoke"
    events, incomplete = _parse([partial])
    assert incomplete is True
    assert _calls(events) == []


def test_deepseek_v4_non_string_parameter_rejects_invalid_json() -> None:
    malformed = "</think>" + _tool_block(
        _invoke("lookup", _param("count", "NOT_JSON", string=False))
    )
    events, incomplete = _parse([malformed])

    assert incomplete is True
    assert _calls(events) == []


def test_deepseek_v4_partial_tool_open_at_end_is_incomplete_not_visible_text() -> None:
    partial = f"</think>answer\n\n<{_DSML}tool_"
    events, incomplete = _parse([partial])

    assert incomplete is True
    assert _calls(events) == []
    assert _text(events) == "answer"


def test_deepseek_v4_disabled_reasoning_stays_plain_text() -> None:
    events, incomplete = _parse(["plain"], reasoning=False)

    assert incomplete is False
    assert _reasoning(events) == ""
    assert _text(events) == "plain"


def test_deepseek_v4_finish_is_idempotent() -> None:
    parser = DeepSeekV4IncrementalParser("req-dsv4", start_in_reasoning=False)
    parser.feed("hello")
    first = parser.finish()
    second = parser.finish()

    assert first.incomplete_tool_call is False
    assert second.events == ()
    assert second.incomplete_tool_call is False
