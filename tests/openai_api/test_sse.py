from __future__ import annotations

from exqserve.protocol.openai.sse import chat_done, chat_sse, compact_json, responses_sse


def test_compact_json_is_utf8_friendly_and_deterministic_for_existing_order() -> None:
    assert compact_json({"type": "δ", "value": [1, 2]}) == '{"type":"δ","value":[1,2]}'


def test_chat_sse_uses_data_only_and_done_sentinel() -> None:
    assert chat_sse({"id": "x", "choices": []}) == 'data: {"id":"x","choices":[]}\n\n'
    assert chat_done() == "data: [DONE]\n\n"


def test_responses_sse_includes_event_name_and_compact_data() -> None:
    event = {"type": "response.output_text.delta", "sequence_number": 2, "delta": "hi"}
    assert responses_sse(event) == (
        "event: response.output_text.delta\n"
        'data: {"type":"response.output_text.delta","sequence_number":2,"delta":"hi"}\n\n'
    )


def test_responses_sse_requires_non_empty_type() -> None:
    try:
        responses_sse({"sequence_number": 1})
    except ValueError as exc:
        assert "type" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("responses_sse must reject a missing event type")
