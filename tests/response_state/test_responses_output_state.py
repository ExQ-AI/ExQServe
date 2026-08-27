from __future__ import annotations

from exqserve.core.events import CompletionReason, GenerationCompleted, GenerationStarted
from exqserve.protocol.openai.responses import ResponsesAccumulator, ResponsesStreamSerializer


def test_responses_stream_uses_exact_state_identity_metadata() -> None:
    serializer = ResponsesStreamSerializer(
        "m",
        response_id="resp_exact",
        created_at=1,
        previous_response_id="resp_parent",
        store=False,
    )
    created = serializer.feed(GenerationStarted("r"))[0]["response"]
    assert created["id"] == "resp_exact"  # type: ignore[index]
    assert created["previous_response_id"] == "resp_parent"  # type: ignore[index]
    assert created["store"] is False  # type: ignore[index]

    terminal = serializer.feed(GenerationCompleted("r", CompletionReason.STOP))[0]["response"]
    assert terminal["id"] == "resp_exact"  # type: ignore[index]
    assert terminal["previous_response_id"] == "resp_parent"  # type: ignore[index]
    assert terminal["store"] is False  # type: ignore[index]


def test_responses_nonstream_uses_exact_state_identity_metadata() -> None:
    accumulator = ResponsesAccumulator(
        "m",
        response_id="resp_exact",
        created_at=1,
        previous_response_id="resp_parent",
        store=True,
    )
    accumulator.consume(GenerationCompleted("r", CompletionReason.STOP))
    response = accumulator.result()
    assert response["id"] == "resp_exact"
    assert response["previous_response_id"] == "resp_parent"
    assert response["store"] is True
