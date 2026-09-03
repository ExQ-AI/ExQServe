from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.core.tokens import NativeTokenSpan
from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    LoRAAdapterConfig,
    RuntimeCancelled,
    RuntimeCapabilities,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeModelMetadata,
    RuntimeRenderedPrompt,
    RuntimeSamplingConfig,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)


def test_runtime_capabilities_are_immutable_booleans() -> None:
    caps = RuntimeCapabilities(True, True, True, True, True, True)
    assert caps.cache_usage is True
    with pytest.raises(FrozenInstanceError):
        caps.cancellation = False  # type: ignore[misc]


def test_runtime_model_metadata_is_truthful_and_immutable() -> None:
    known = RuntimeModelMetadata(
        max_context_tokens=131072,
        architecture=" Qwen3_5ForConditionalGeneration ",
        backend_context_tokens=131077,
        generation_headroom_tokens=5,
    )
    unknown = RuntimeModelMetadata()

    assert known.max_context_tokens == 131072
    assert known.architecture == "Qwen3_5ForConditionalGeneration"
    assert known.backend_context_tokens == 131077
    assert known.generation_headroom_tokens == 5
    assert unknown.max_context_tokens is None
    assert unknown.architecture is None
    assert unknown.backend_context_tokens is None
    assert unknown.generation_headroom_tokens is None
    with pytest.raises(ValueError, match="positive"):
        RuntimeModelMetadata(max_context_tokens=0)
    with pytest.raises(TypeError, match="integer"):
        RuntimeModelMetadata(max_context_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="architecture"):
        RuntimeModelMetadata(architecture="   ")
    with pytest.raises(TypeError, match="architecture"):
        RuntimeModelMetadata(architecture=123)  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        known.max_context_tokens = 1  # type: ignore[misc]


def test_lora_adapter_config_is_immutable_normalized_and_validated() -> None:
    adapter = LoRAAdapterConfig(" /loras/code ", 0.75)
    assert adapter.directory == "/loras/code"
    assert adapter.scaling == 0.75
    with pytest.raises(ValueError, match="directory"):
        LoRAAdapterConfig("   ")
    with pytest.raises(ValueError, match="non-negative"):
        LoRAAdapterConfig("/loras/code", -0.1)
    with pytest.raises(ValueError, match="finite"):
        LoRAAdapterConfig("/loras/code", float("nan"))
    with pytest.raises(FrozenInstanceError):
        adapter.scaling = 1.0  # type: ignore[misc]


def test_load_config_supports_q8_and_fp16_cache_without_profile_fields() -> None:
    q8 = ExLlamaV3LoadConfig("/models/qwen", cache_tokens=4096)
    fp16 = ExLlamaV3LoadConfig(
        "/models/qwen",
        cache_tokens=4096,
        cache_key_bits=None,
        cache_value_bits=None,
    )

    assert (q8.cache_key_bits, q8.cache_value_bits) == (8, 8)
    assert (fp16.cache_key_bits, fp16.cache_value_bits) == (None, None)
    assert q8.mtp_enabled is False
    assert q8.mtp_draft_tokens == 4
    assert q8.mtp_cache_bits == 4
    assert q8.dynamic_draft_tokens is False
    assert q8.draft_confidence == 0.4
    assert q8.draft_model_directory is None
    assert q8.draft_tokens == 4
    assert q8.draft_cache_bits == 4
    assert q8.lora_adapters == ()
    assert q8.autosplit_no_forward is False
    assert q8.cuda_malloc_async is False
    assert q8.qc_staging is None
    assert q8.max_requeue_tokens is None
    assert q8.device_ids is None
    assert q8.chat_template is None
    assert q8.sysmem_kv_cache_mb == 0
    assert q8.sysmem_recurrent_cache_mb == 4096
    assert q8.ngram_match_min == 0
    assert q8.ngram_draft_size == 4
    assert q8.moe_cpu_offload_layers == 0
    assert q8.moe_cpu_split_experts == 0
    assert q8.draft_moe_cpu_offload_layers == 0
    assert q8.vision_offload is False
    assert q8.moe_cpu_threads is None
    selected = ExLlamaV3LoadConfig("/m", 256, device_ids=(0, 2), tp_output_device=2)
    assert selected.device_ids == (0, 2)

    with pytest.raises(ValueError, match="multiple of 256"):
        ExLlamaV3LoadConfig("/m", cache_tokens=257)
    with pytest.raises(ValueError, match="both"):
        ExLlamaV3LoadConfig("/m", 256, cache_key_bits=8, cache_value_bits=None)
    with pytest.raises(ValueError, match="2..8"):
        ExLlamaV3LoadConfig("/m", 256, cache_key_bits=1, cache_value_bits=8)
    with pytest.raises(TypeError, match="mtp_enabled"):
        ExLlamaV3LoadConfig("/m", 256, mtp_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mtp_draft_tokens"):
        ExLlamaV3LoadConfig("/m", 256, mtp_draft_tokens=0)
    with pytest.raises(ValueError, match="mtp_cache_bits"):
        ExLlamaV3LoadConfig("/m", 256, mtp_cache_bits=1)
    with pytest.raises(TypeError, match="dynamic_draft_tokens"):
        ExLlamaV3LoadConfig("/m", 256, dynamic_draft_tokens=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="draft_confidence"):
        ExLlamaV3LoadConfig("/m", 256, draft_confidence=0.0)
    with pytest.raises(ValueError, match="draft_confidence"):
        ExLlamaV3LoadConfig("/m", 256, draft_confidence=1.0)
    with pytest.raises(ValueError, match="requires MTP"):
        ExLlamaV3LoadConfig("/m", 256, dynamic_draft_tokens=True)
    with pytest.raises(TypeError, match="chat_template"):
        ExLlamaV3LoadConfig("/m", 256, chat_template=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chat_template"):
        ExLlamaV3LoadConfig("/m", 256, chat_template="   ")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ExLlamaV3LoadConfig("/m", 256, mtp_enabled=True, draft_model_directory="/draft")
    with pytest.raises(ValueError, match="draft_tokens"):
        ExLlamaV3LoadConfig("/m", 256, draft_tokens=0)
    with pytest.raises(ValueError, match="draft_cache_bits"):
        ExLlamaV3LoadConfig("/m", 256, draft_cache_bits=1)
    with pytest.raises(ValueError, match="draft_model_directory"):
        ExLlamaV3LoadConfig("/m", 256, draft_model_directory="   ")
    with pytest.raises(TypeError, match="lora_adapters"):
        ExLlamaV3LoadConfig("/m", 256, lora_adapters=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="autosplit_no_forward"):
        ExLlamaV3LoadConfig("/m", 256, autosplit_no_forward=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="cuda_malloc_async"):
        ExLlamaV3LoadConfig("/m", 256, cuda_malloc_async=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="qc_staging"):
        ExLlamaV3LoadConfig("/m", 256, qc_staging=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="qc_staging"):
        ExLlamaV3LoadConfig("/m", 256, qc_staging=3)
    with pytest.raises(TypeError, match="max_requeue_tokens"):
        ExLlamaV3LoadConfig("/m", 256, max_requeue_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_requeue_tokens"):
        ExLlamaV3LoadConfig("/m", 256, max_requeue_tokens=0)
    with pytest.raises(TypeError, match="sysmem_kv_cache_mb"):
        ExLlamaV3LoadConfig("/m", 256, sysmem_kv_cache_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sysmem_kv_cache_mb"):
        ExLlamaV3LoadConfig("/m", 256, sysmem_kv_cache_mb=-1)
    with pytest.raises(TypeError, match="sysmem_recurrent_cache_mb"):
        ExLlamaV3LoadConfig("/m", 256, sysmem_recurrent_cache_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sysmem_recurrent_cache_mb"):
        ExLlamaV3LoadConfig("/m", 256, sysmem_recurrent_cache_mb=0)
    with pytest.raises(TypeError, match="ngram_match_min"):
        ExLlamaV3LoadConfig("/m", 256, ngram_match_min=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ngram_match_min"):
        ExLlamaV3LoadConfig("/m", 256, ngram_match_min=-1)
    with pytest.raises(TypeError, match="ngram_draft_size"):
        ExLlamaV3LoadConfig("/m", 256, ngram_draft_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ngram_draft_size"):
        ExLlamaV3LoadConfig("/m", 256, ngram_draft_size=0)
    with pytest.raises(ValueError, match="cannot be combined"):
        ExLlamaV3LoadConfig("/m", 256, ngram_match_min=2, mtp_enabled=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        ExLlamaV3LoadConfig("/m", 256, ngram_match_min=2, draft_model_directory="/draft")
    with pytest.raises(ValueError, match="not supported with n-gram"):
        ExLlamaV3LoadConfig("/m", 256, ngram_match_min=2, dynamic_draft_tokens=True)
    with pytest.raises(TypeError, match="moe_cpu_offload_layers"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_offload_layers=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="moe_cpu_offload_layers"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_offload_layers=-1)
    with pytest.raises(TypeError, match="moe_cpu_split_experts"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_split_experts=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="moe_cpu_split_experts"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_split_experts=-1)
    with pytest.raises(TypeError, match="draft_moe_cpu_offload_layers"):
        ExLlamaV3LoadConfig("/m", 256, draft_moe_cpu_offload_layers=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="draft_moe_cpu_offload_layers"):
        ExLlamaV3LoadConfig("/m", 256, draft_moe_cpu_offload_layers=-1)
    with pytest.raises(TypeError, match="moe_cpu_threads"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_threads=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="moe_cpu_threads"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_threads=0)
    with pytest.raises(ValueError, match="layer-split"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_offload_layers=4, tensor_parallel=True)
    with pytest.raises(ValueError, match="vision_offload requires vision_enabled"):
        ExLlamaV3LoadConfig("/m", 256, vision_offload=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_offload_layers=4, moe_cpu_split_experts=2)
    with pytest.raises(ValueError, match="cannot be combined"):
        ExLlamaV3LoadConfig(
            "/m",
            256,
            mtp_enabled=True,
            moe_cpu_split_experts=2,
            draft_moe_cpu_offload_layers=1,
        )
    external_draft = ExLlamaV3LoadConfig(
        "/m",
        256,
        draft_model_directory="/draft",
        moe_cpu_split_experts=2,
        draft_moe_cpu_offload_layers=1,
    )
    assert external_draft.moe_cpu_split_experts == 2
    assert external_draft.draft_moe_cpu_offload_layers == 1
    with pytest.raises(ValueError, match="layer-split"):
        ExLlamaV3LoadConfig("/m", 256, moe_cpu_split_experts=2, tensor_parallel=True)
    with pytest.raises(ValueError, match="requires MTP or an external draft model"):
        ExLlamaV3LoadConfig("/m", 256, draft_moe_cpu_offload_layers=1)
    with pytest.raises(ValueError, match="layer-split"):
        ExLlamaV3LoadConfig(
            "/m",
            256,
            mtp_enabled=True,
            draft_moe_cpu_offload_layers=1,
            tensor_parallel=True,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        ExLlamaV3LoadConfig("/m", 256, device_ids=())
    with pytest.raises(ValueError, match="non-negative"):
        ExLlamaV3LoadConfig("/m", 256, device_ids=(-1,))
    with pytest.raises(ValueError, match="duplicates"):
        ExLlamaV3LoadConfig("/m", 256, device_ids=(0, 0))
    with pytest.raises(ValueError, match="negative reserve"):
        ExLlamaV3LoadConfig("/m", 256, reserve_per_device_gb=(-1.0,), device_ids=(0,))
    with pytest.raises(ValueError, match="included in device_ids"):
        ExLlamaV3LoadConfig("/m", 256, device_ids=(0, 2), tp_output_device=1)
    assert ExLlamaV3LoadConfig("/m", 256, mtp_enabled=True, mtp_cache_bits=None).mtp_cache_bits is None


def test_sampling_config_validates_common_combo_subset() -> None:
    sampling = RuntimeSamplingConfig(
        temperature=0.8,
        min_p=0.05,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.05,
        frequency_penalty=0.1,
        presence_penalty=-0.1,
        repetition_penalty_range=512,
        repetition_decay=64,
        temperature_last=True,
        adaptive_target=0.5,
        adaptive_decay=0.8,
        logit_bias=((10, 2.5), (20, -3.0)),
    )
    assert sampling.top_k == 40
    assert sampling.repetition_penalty_range == 512
    assert sampling.repetition_decay == 64
    assert sampling.adaptive_target == 0.5
    assert sampling.adaptive_decay == 0.8
    assert sampling.logit_bias == ((10, 2.5), (20, -3.0))
    assert sampling.temperature_last is True

    with pytest.raises(ValueError, match="temperature"):
        RuntimeSamplingConfig(temperature=-0.1)
    with pytest.raises(ValueError, match="top_p"):
        RuntimeSamplingConfig(top_p=1.1)
    with pytest.raises(ValueError, match="finite"):
        RuntimeSamplingConfig(temperature=float("nan"))
    with pytest.raises(TypeError, match="temperature_last"):
        RuntimeSamplingConfig(temperature_last=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="repetition_penalty_range"):
        RuntimeSamplingConfig(repetition_penalty_range=-1)
    with pytest.raises(ValueError, match="repetition_decay"):
        RuntimeSamplingConfig(repetition_decay=-1)
    with pytest.raises(ValueError, match="adaptive_target"):
        RuntimeSamplingConfig(adaptive_target=1.1)
    with pytest.raises(ValueError, match="adaptive_decay"):
        RuntimeSamplingConfig(adaptive_decay=1.0)
    with pytest.raises(ValueError, match="logit_bias"):
        RuntimeSamplingConfig(logit_bias=((1, 0.0), (1, 1.0)))
    with pytest.raises(ValueError, match="logit_bias"):
        RuntimeSamplingConfig(logit_bias=((-1, 0.0),))
    with pytest.raises(ValueError, match="finite"):
        RuntimeSamplingConfig(logit_bias=((1, float("nan")),))


def test_generation_request_is_immutable_and_protocol_neutral() -> None:
    request = RuntimeGenerationRequest(
        request_id="req-1",
        input_ids=(1, 2, 3),
        max_new_tokens=32,
        seed=7,
        stop_conditions=("<stop>", 2),
        sampling=None,
    )

    assert request.input_ids == (1, 2, 3)
    assert request.output_json_schema is None
    assert request.output_json_trigger is None
    constrained = RuntimeGenerationRequest("req-json", (1,), 1, output_json_schema='{"type":"object"}')
    assert constrained.output_json_schema == '{"type":"object"}'
    triggered = RuntimeGenerationRequest(
        "req-trigger",
        (1,),
        1,
        output_json_schema='{"type":"object"}',
        output_json_trigger="</think>",
    )
    assert triggered.output_json_trigger == "</think>"
    with pytest.raises(TypeError, match="output_json_schema"):
        RuntimeGenerationRequest("req", (1,), 1, output_json_schema={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="output_json_schema"):
        RuntimeGenerationRequest("req", (1,), 1, output_json_schema="   ")
    with pytest.raises(ValueError, match="output_json_trigger"):
        RuntimeGenerationRequest("req", (1,), 1, output_json_trigger="</think>")
    with pytest.raises(ValueError, match="output_json_trigger"):
        RuntimeGenerationRequest(
            "req",
            (1,),
            1,
            output_json_schema='{"type":"object"}',
            output_json_trigger="   ",
        )
    with pytest.raises(TypeError, match="output_json_trigger"):
        RuntimeGenerationRequest(
            "req",
            (1,),
            1,
            output_json_schema='{"type":"object"}',
            output_json_trigger=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="input_ids"):
        RuntimeGenerationRequest("req", (), 1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        RuntimeGenerationRequest("req", (1,), 0)
    with pytest.raises(ValueError, match="stop"):
        RuntimeGenerationRequest("req", (1,), 1, stop_conditions=("",))


def test_runtime_generation_request_preserves_guarantee_and_fallback_policy() -> None:
    request = RuntimeGenerationRequest(
        "req",
        (1,),
        4,
        output_json_schema='{"type":"object"}',
        generation_guarantee=GenerationGuarantee.SCHEMA,
        constraint_fallback_policy=ConstraintFallbackPolicy.FAIL_CLOSED,
    )

    assert request.generation_guarantee is GenerationGuarantee.SCHEMA
    assert request.constraint_fallback_policy is ConstraintFallbackPolicy.FAIL_CLOSED

    with pytest.raises(TypeError, match="generation_guarantee"):
        RuntimeGenerationRequest(
            "req",
            (1,),
            4,
            generation_guarantee="schema",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="constraint_fallback_policy"):
        RuntimeGenerationRequest(
            "req",
            (1,),
            4,
            constraint_fallback_policy="fail_closed",  # type: ignore[arg-type]
        )


def test_runtime_finished_effective_guarantee_requires_activation() -> None:
    usage = TokenUsage(input_tokens=1, output_tokens=1)
    timing = RuntimeTiming()
    finished = RuntimeFinished(
        "req",
        RuntimeStopReason.FILTER,
        usage,
        timing,
        hard_constraint_installed=True,
        hard_constraint_activated=True,
        effective_generation_guarantee=GenerationGuarantee.FORMAT,
    )
    assert finished.effective_generation_guarantee is GenerationGuarantee.FORMAT

    with pytest.raises(ValueError, match="activated hard constraint"):
        RuntimeFinished(
            "req",
            RuntimeStopReason.EOS,
            usage,
            timing,
            hard_constraint_installed=True,
            hard_constraint_activated=False,
            effective_generation_guarantee=GenerationGuarantee.SCHEMA,
        )


def test_timing_preserves_unknown_and_rejects_invalid_measurements() -> None:
    timing = RuntimeTiming(queue_seconds=None, prefill_seconds=0.2, generation_seconds=1.0)
    assert timing.queue_seconds is None

    with pytest.raises(ValueError, match="non-negative"):
        RuntimeTiming(queue_seconds=-0.1)
    with pytest.raises(ValueError, match="finite"):
        RuntimeTiming(prefill_seconds=float("inf"))


def test_rendered_prompt_contains_plain_python_token_ids() -> None:
    rendered = RuntimeRenderedPrompt("prompt", (1, 2, 3))
    assert rendered.input_ids == (1, 2, 3)
    with pytest.raises(ValueError, match="input_ids"):
        RuntimeRenderedPrompt("prompt", ())


def test_runtime_event_vocabulary_is_safe_and_immutable() -> None:
    usage = TokenUsage(input_tokens=3, cached_input_tokens=2, output_tokens=4)
    timing = RuntimeTiming(0.1, 0.2, 0.3)
    error = CanonicalError(ErrorCategory.RUNTIME_FAILURE, "backend", "failed", retryable=False)

    events = (
        RuntimeStarted("req"),
        RuntimeTextDelta("req", "hello"),
        RuntimeFinished("req", RuntimeStopReason.EOS, usage, timing),
        RuntimeCancelled("req"),
        RuntimeFailed("req", error),
    )

    assert events[2].usage == usage
    assert RuntimeStopReason.OTHER.value == "other"
    with pytest.raises(ValueError, match="text"):
        RuntimeTextDelta("req", "")


def test_runtime_text_delta_validates_native_token_provenance_spans() -> None:
    span = NativeTokenSpan(1, 12, 248058, "<tool_call>")
    delta = RuntimeTextDelta("req", "a<tool_call>b", (1, 248058, 2), (span,), True)
    assert delta.native_token_provenance is True
    assert delta.native_token_spans == (span,)

    with pytest.raises(ValueError, match="require native_token_provenance"):
        RuntimeTextDelta("req", "a<tool_call>b", (1, 248058, 2), (span,))
    with pytest.raises(ValueError, match="does not match"):
        RuntimeTextDelta(
            "req",
            "a<tool_call>b",
            (1, 248058, 2),
            (NativeTokenSpan(0, 11, 248058, "<tool_call>"),),
            True,
        )
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        RuntimeTextDelta(
            "req",
            "<think></think>",
            (248068, 248069),
            (
                NativeTokenSpan(7, 15, 248069, "</think>"),
                NativeTokenSpan(0, 7, 248068, "<think>"),
            ),
            True,
        )
