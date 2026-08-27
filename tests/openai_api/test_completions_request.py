from __future__ import annotations

import pytest

from exqserve.core.items import RawPromptItem
from exqserve.protocol.openai.common import OpenAIProtocolError
from exqserve.protocol.openai.completions import CompletionsRequestAdapter
from exqserve.runtime.contracts import RuntimeSamplingConfig


def test_completions_string_prompt_maps_to_raw_serving_without_chat_semantics() -> None:
    parsed = CompletionsRequestAdapter().parse(
        {
            "model": "m",
            "prompt": "Once upon a time",
            "max_tokens": 7,
            "temperature": 0.2,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "logit_bias": {"10": 2.0},
            "seed": 123,
            "stop": ["END", "STOP"],
        },
        request_id="req_raw",
    )

    assert parsed.model == "m"
    assert parsed.stream is False
    assert parsed.echo is False
    assert parsed.include_usage is False
    assert parsed.raw.input.request_id == "req_raw"
    assert parsed.raw.input.model == "m"
    assert parsed.raw.input.items == (RawPromptItem(text="Once upon a time"),)
    assert parsed.raw.max_output_tokens == 7
    assert parsed.raw.stop_conditions == ("END", "STOP")
    assert parsed.raw.seed == 123
    assert parsed.raw.sampling == RuntimeSamplingConfig(
        temperature=0.2,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        logit_bias=((10, 2.0),),
    )


def test_completions_logit_bias_reaches_shared_sampler() -> None:
    parsed = CompletionsRequestAdapter().parse(
        {"model": "m", "prompt": "raw", "logit_bias": {"7": -10.0}},
        request_id="req_bias",
    )
    assert parsed.raw.sampling is not None
    assert parsed.raw.sampling.logit_bias == ((7, -10.0),)


def test_completions_flat_token_prompt_bypasses_text_tokenization_contract() -> None:
    parsed = CompletionsRequestAdapter().parse(
        {"model": "m", "prompt": [11, 22, 33], "max_tokens": 4},
        request_id="req_tokens",
    )
    assert parsed.raw.input.items == (RawPromptItem(token_ids=(11, 22, 33)),)


def test_completions_missing_prompt_means_empty_raw_document_prompt() -> None:
    parsed = CompletionsRequestAdapter().parse(
        {"model": "m", "max_tokens": 3},
        request_id="req_empty",
    )
    assert parsed.raw.input.items == (RawPromptItem(text=""),)


def test_completions_stream_options_and_echo_are_parsed_explicitly() -> None:
    parsed = CompletionsRequestAdapter().parse(
        {
            "model": "m",
            "prompt": "raw",
            "stream": True,
            "stream_options": {"include_usage": True},
            "echo": True,
        },
        request_id="req_stream",
    )
    assert parsed.stream is True
    assert parsed.echo is True
    assert parsed.include_usage is True
    assert parsed.raw.max_output_tokens == 16


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("n", 2, "unsupported_n"),
        ("best_of", 2, "unsupported_best_of"),
        ("logprobs", 1, "unsupported_logprobs"),
        ("suffix", "tail", "unsupported_suffix"),
    ],
)
def test_completions_semantics_changing_legacy_options_reject(field: str, value: object, code: str) -> None:
    with pytest.raises(OpenAIProtocolError) as exc_info:
        CompletionsRequestAdapter().parse(
            {"model": "m", "prompt": "raw", field: value},
            request_id="req_bad",
        )
    assert exc_info.value.code == code
    assert exc_info.value.param == field


@pytest.mark.parametrize(
    "prompt",
    [
        ["a", "b"],
        [[1, 2], [3, 4]],
        ["only-one-list-form"],
        [1, "mixed"],
        [],
    ],
)
def test_completions_multiple_or_ambiguous_prompt_forms_reject(prompt: object) -> None:
    with pytest.raises(OpenAIProtocolError) as exc_info:
        CompletionsRequestAdapter().parse(
            {"model": "m", "prompt": prompt},
            request_id="req_bad_prompt",
        )
    assert exc_info.value.code == "unsupported_prompt_form"
    assert exc_info.value.param == "prompt"


def test_completions_echo_rejects_token_id_prompt() -> None:
    with pytest.raises(OpenAIProtocolError) as exc_info:
        CompletionsRequestAdapter().parse(
            {"model": "m", "prompt": [1, 2], "echo": True},
            request_id="req_echo_tokens",
        )
    assert exc_info.value.code == "unsupported_echo_token_prompt"


@pytest.mark.parametrize("stop", ["", ["ok", ""], ["1", "2", "3", "4", "5"], [1]])
def test_completions_stop_validation_is_strict(stop: object) -> None:
    with pytest.raises(OpenAIProtocolError) as exc_info:
        CompletionsRequestAdapter().parse(
            {"model": "m", "prompt": "raw", "stop": stop},
            request_id="req_bad_stop",
        )
    assert exc_info.value.code == "invalid_stop"


def test_completions_invalid_stream_options_reject() -> None:
    with pytest.raises(OpenAIProtocolError) as exc_info:
        CompletionsRequestAdapter().parse(
            {
                "model": "m",
                "prompt": "raw",
                "stream": True,
                "stream_options": {"include_usage": "yes"},
            },
            request_id="req_bad_stream_options",
        )
    assert exc_info.value.code == "invalid_stream_options"
