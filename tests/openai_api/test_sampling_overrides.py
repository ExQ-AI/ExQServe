from __future__ import annotations

from exqserve.core.sampling import SamplingOverride, SamplingOverridePolicy
from exqserve.protocol.openai.chat import ChatRequestAdapter
from exqserve.protocol.openai.completions import CompletionsRequestAdapter
from exqserve.protocol.openai.responses import ResponsesRequestAdapter


def _policy() -> SamplingOverridePolicy:
    return SamplingOverridePolicy(
        (
            SamplingOverride("temperature", 0.7, False),
            SamplingOverride("top_p", 0.9, True),
            SamplingOverride("min_p", 0.05, False),
        )
    )


def test_sampling_override_fallback_and_force_preserve_request_presence() -> None:
    policy = _policy()
    parsed = ChatRequestAdapter(sampling_overrides=policy).parse(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.4,
        },
        request_id="chat",
    )
    assert parsed.serving.sampling is not None
    assert parsed.serving.sampling.temperature == 0.2
    assert parsed.serving.sampling.top_p == 0.9
    assert parsed.serving.sampling.min_p == 0.05


def test_sampling_override_can_create_sampling_for_request_with_no_sampler_fields() -> None:
    policy = SamplingOverridePolicy((SamplingOverride("temperature", 0.6, False),))
    parsed = ChatRequestAdapter(sampling_overrides=policy).parse(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
        request_id="chat-default",
    )
    assert parsed.serving.sampling is not None
    assert parsed.serving.sampling.temperature == 0.6


def test_sampling_override_policy_is_shared_by_all_openai_request_adapters() -> None:
    policy = SamplingOverridePolicy((SamplingOverride("top_k", 17, True),))

    chat = ChatRequestAdapter(sampling_overrides=policy).parse(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "top_k": 2,
        },
        request_id="chat",
    )
    responses = ResponsesRequestAdapter(sampling_overrides=policy).parse(
        {
            "model": "m",
            "input": "hi",
            "max_output_tokens": 8,
            "top_k": 2,
        },
        request_id="resp",
    )
    completions = CompletionsRequestAdapter(sampling_overrides=policy).parse(
        {"model": "m", "prompt": "hi", "max_tokens": 8, "top_k": 2},
        request_id="completion",
    )

    assert chat.serving.sampling is not None and chat.serving.sampling.top_k == 17
    assert responses.serving.sampling is not None and responses.serving.sampling.top_k == 17
    assert completions.raw.sampling is not None and completions.raw.sampling.top_k == 17
