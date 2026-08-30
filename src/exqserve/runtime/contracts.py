"""CPU-safe runtime contracts independent of any concrete inference backend."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from exqserve.core.errors import CanonicalError
from exqserve.core.usage import TokenUsage

_PAGE_SIZE = 256


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str):
        raise TypeError("request_id must be a string")
    if not request_id.strip():
        raise ValueError("request_id must not be empty")


def _validate_token_ids(input_ids: tuple[int, ...]) -> None:
    if not isinstance(input_ids, tuple):
        raise TypeError("input_ids must be a tuple")
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    for token_id in input_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError("input_ids must contain only integers")
        if token_id < 0:
            raise ValueError("input_ids must be non-negative")


def _validate_finite(name: str, value: float) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    cancellation: bool
    template_rendering: bool
    tokenization: bool
    seed: bool
    cache_usage: bool
    quantized_kv_cache: bool
    vision: bool = False

    def __post_init__(self) -> None:
        for name in (
            "cancellation",
            "template_rendering",
            "tokenization",
            "seed",
            "cache_usage",
            "quantized_kv_cache",
            "vision",
        ):
            _validate_bool(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class RuntimeModelMetadata:
    max_context_tokens: int | None = None
    architecture: str | None = None

    def __post_init__(self) -> None:
        if self.max_context_tokens is not None:
            if not isinstance(self.max_context_tokens, int) or isinstance(self.max_context_tokens, bool):
                raise TypeError("max_context_tokens must be an integer or None")
            if self.max_context_tokens <= 0:
                raise ValueError("max_context_tokens must be positive")
        if self.architecture is not None:
            if not isinstance(self.architecture, str):
                raise TypeError("architecture must be a string or None")
            normalized = self.architecture.strip()
            if not normalized:
                raise ValueError("architecture must not be empty")
            object.__setattr__(self, "architecture", normalized)


@dataclass(frozen=True, slots=True)
class LoRAAdapterConfig:
    directory: str
    scaling: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.directory, str):
            raise TypeError("directory must be a string")
        normalized = self.directory.strip()
        if not normalized:
            raise ValueError("directory must not be empty")
        object.__setattr__(self, "directory", normalized)
        _validate_finite("scaling", self.scaling)
        if float(self.scaling) < 0:
            raise ValueError("scaling must be non-negative")
        object.__setattr__(self, "scaling", float(self.scaling))


@dataclass(frozen=True, slots=True)
class ExLlamaV3LoadConfig:
    model_directory: str
    cache_tokens: int
    cache_key_bits: int | None = 8
    cache_value_bits: int | None = 8
    max_batch_size: int = 16
    max_chunk_size: int = 2048
    reserve_per_device_gb: tuple[float, ...] | None = None
    mtp_enabled: bool = False
    mtp_draft_tokens: int = 4
    mtp_cache_bits: int | None = 4
    autosplit_no_forward: bool = False
    cuda_malloc_async: bool = False
    qc_staging: int | None = None
    max_requeue_tokens: int | None = None
    vision_enabled: bool = False
    allow_remote_images: bool = False
    max_image_bytes: int = 20 * 1024 * 1024
    dynamic_draft_tokens: bool = False
    draft_confidence: float = 0.4
    draft_model_directory: str | None = None
    draft_tokens: int = 4
    draft_cache_bits: int | None = 4
    lora_adapters: tuple[LoRAAdapterConfig, ...] = ()
    tensor_parallel: bool = False
    tp_backend: str = "native"
    tp_output_device: int | None = None
    device_ids: tuple[int, ...] | None = None
    chat_template: str | None = None
    vision_cache_mb: int = 256
    sysmem_kv_cache_mb: int = 0
    sysmem_recurrent_cache_mb: int = 4096
    ngram_match_min: int = 0
    ngram_draft_size: int = 4
    moe_cpu_offload_layers: int = 0
    moe_cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_directory, str):
            raise TypeError("model_directory must be a string")
        if not self.model_directory.strip():
            raise ValueError("model_directory must not be empty")
        if not isinstance(self.cache_tokens, int) or isinstance(self.cache_tokens, bool):
            raise TypeError("cache_tokens must be an integer")
        if self.cache_tokens <= 0 or self.cache_tokens % _PAGE_SIZE != 0:
            raise ValueError("cache_tokens must be positive and a multiple of 256")

        key_bits = self.cache_key_bits
        value_bits = self.cache_value_bits
        if (key_bits is None) != (value_bits is None):
            raise ValueError("cache_key_bits and cache_value_bits must both be set or both be None")
        if key_bits is not None and value_bits is not None:
            for name, bits in (("cache_key_bits", key_bits), ("cache_value_bits", value_bits)):
                if not isinstance(bits, int) or isinstance(bits, bool):
                    raise TypeError(f"{name} must be an integer or None")
                if not 2 <= bits <= 8:
                    raise ValueError(f"{name} must be in the range 2..8")

        for name in ("max_batch_size", "max_chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.reserve_per_device_gb is not None:
            if not isinstance(self.reserve_per_device_gb, tuple):
                raise TypeError("reserve_per_device_gb must be a tuple or None")
            for reserve in self.reserve_per_device_gb:
                _validate_finite("reserve_per_device_gb", reserve)

        if self.device_ids is not None:
            if not isinstance(self.device_ids, tuple):
                raise TypeError("device_ids must be a tuple or None")
            if not self.device_ids:
                raise ValueError("device_ids must not be empty")
            seen_device_ids: set[int] = set()
            for device_id in self.device_ids:
                if not isinstance(device_id, int) or isinstance(device_id, bool):
                    raise TypeError("device_ids must contain integers")
                if device_id < 0:
                    raise ValueError("device_ids must contain non-negative integers")
                if device_id in seen_device_ids:
                    raise ValueError("device_ids must not contain duplicates")
                seen_device_ids.add(device_id)
                if (
                    self.reserve_per_device_gb is not None
                    and device_id < len(self.reserve_per_device_gb)
                    and self.reserve_per_device_gb[device_id] < 0
                ):
                    raise ValueError("selected device_ids cannot have a negative reserve")

        _validate_bool("mtp_enabled", self.mtp_enabled)
        _validate_bool("autosplit_no_forward", self.autosplit_no_forward)
        _validate_bool("cuda_malloc_async", self.cuda_malloc_async)
        _validate_bool("vision_enabled", self.vision_enabled)
        _validate_bool("allow_remote_images", self.allow_remote_images)
        _validate_bool("tensor_parallel", self.tensor_parallel)
        _validate_bool("dynamic_draft_tokens", self.dynamic_draft_tokens)
        if self.chat_template is not None:
            if not isinstance(self.chat_template, str):
                raise TypeError("chat_template must be a string or None")
            if not self.chat_template.strip():
                raise ValueError("chat_template must not be empty")
        _validate_finite("draft_confidence", self.draft_confidence)
        if not 0.0 < float(self.draft_confidence) < 1.0:
            raise ValueError("draft_confidence must be in the range (0, 1)")
        object.__setattr__(self, "draft_confidence", float(self.draft_confidence))
        if self.tp_backend not in {"native", "nccl"}:
            raise ValueError("tp_backend must be native or nccl")
        if self.tp_output_device is not None:
            if not isinstance(self.tp_output_device, int) or isinstance(self.tp_output_device, bool):
                raise TypeError("tp_output_device must be an integer or None")
            if self.tp_output_device < 0:
                raise ValueError("tp_output_device must be non-negative")
            if self.device_ids is not None and self.tp_output_device not in self.device_ids:
                raise ValueError("tp_output_device must be included in device_ids")
        if not isinstance(self.max_image_bytes, int) or isinstance(self.max_image_bytes, bool):
            raise TypeError("max_image_bytes must be an integer")
        if self.max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be positive")
        if not isinstance(self.vision_cache_mb, int) or isinstance(self.vision_cache_mb, bool):
            raise TypeError("vision_cache_mb must be an integer")
        if self.vision_cache_mb < 0:
            raise ValueError("vision_cache_mb must be non-negative")
        if not isinstance(self.sysmem_kv_cache_mb, int) or isinstance(
            self.sysmem_kv_cache_mb, bool
        ):
            raise TypeError("sysmem_kv_cache_mb must be an integer")
        if self.sysmem_kv_cache_mb < 0:
            raise ValueError("sysmem_kv_cache_mb must be non-negative")
        if not isinstance(self.sysmem_recurrent_cache_mb, int) or isinstance(
            self.sysmem_recurrent_cache_mb, bool
        ):
            raise TypeError("sysmem_recurrent_cache_mb must be an integer")
        if self.sysmem_recurrent_cache_mb <= 0:
            raise ValueError("sysmem_recurrent_cache_mb must be positive")
        if not isinstance(self.ngram_match_min, int) or isinstance(self.ngram_match_min, bool):
            raise TypeError("ngram_match_min must be an integer")
        if self.ngram_match_min < 0:
            raise ValueError("ngram_match_min must be non-negative")
        if not isinstance(self.ngram_draft_size, int) or isinstance(self.ngram_draft_size, bool):
            raise TypeError("ngram_draft_size must be an integer")
        if self.ngram_draft_size <= 0:
            raise ValueError("ngram_draft_size must be positive")
        if not isinstance(self.moe_cpu_offload_layers, int) or isinstance(
            self.moe_cpu_offload_layers, bool
        ):
            raise TypeError("moe_cpu_offload_layers must be an integer")
        if self.moe_cpu_offload_layers < 0:
            raise ValueError("moe_cpu_offload_layers must be non-negative")
        if self.moe_cpu_threads is not None:
            if not isinstance(self.moe_cpu_threads, int) or isinstance(self.moe_cpu_threads, bool):
                raise TypeError("moe_cpu_threads must be an integer or None")
            if self.moe_cpu_threads <= 0:
                raise ValueError("moe_cpu_threads must be positive or None")
        if self.draft_model_directory is not None:
            if not isinstance(self.draft_model_directory, str):
                raise TypeError("draft_model_directory must be a string or None")
            normalized_draft = self.draft_model_directory.strip()
            if not normalized_draft:
                raise ValueError("draft_model_directory must not be empty")
            object.__setattr__(self, "draft_model_directory", normalized_draft)
        if self.mtp_enabled and self.draft_model_directory is not None:
            raise ValueError("mtp_enabled and draft_model_directory are mutually exclusive")
        if self.ngram_match_min and (self.mtp_enabled or self.draft_model_directory is not None):
            raise ValueError("n-gram drafting cannot be combined with MTP or an external draft model")
        if self.ngram_match_min and self.dynamic_draft_tokens:
            raise ValueError("dynamic_draft_tokens is not supported with n-gram drafting")
        if self.dynamic_draft_tokens and not (self.mtp_enabled or self.draft_model_directory is not None):
            raise ValueError("dynamic_draft_tokens requires MTP or an external draft model")
        if self.moe_cpu_offload_layers and self.tensor_parallel:
            raise ValueError("moe_cpu_offload_layers requires layer-split mode, not tensor parallel")
        if not isinstance(self.draft_tokens, int) or isinstance(self.draft_tokens, bool):
            raise TypeError("draft_tokens must be an integer")
        if self.draft_tokens <= 0:
            raise ValueError("draft_tokens must be positive")
        if self.draft_cache_bits is not None:
            if not isinstance(self.draft_cache_bits, int) or isinstance(self.draft_cache_bits, bool):
                raise TypeError("draft_cache_bits must be an integer or None")
            if not 2 <= self.draft_cache_bits <= 8:
                raise ValueError("draft_cache_bits must be in the range 2..8 or None")
        if not isinstance(self.lora_adapters, tuple):
            raise TypeError("lora_adapters must be a tuple")
        if not all(isinstance(adapter, LoRAAdapterConfig) for adapter in self.lora_adapters):
            raise TypeError("lora_adapters must contain LoRAAdapterConfig values")
        if self.qc_staging is not None:
            if not isinstance(self.qc_staging, int) or isinstance(self.qc_staging, bool):
                raise TypeError("qc_staging must be an integer or None")
            if self.qc_staging not in (0, 1, 2):
                raise ValueError("qc_staging must be 0, 1, 2, or None")
        if self.max_requeue_tokens is not None:
            if not isinstance(self.max_requeue_tokens, int) or isinstance(self.max_requeue_tokens, bool):
                raise TypeError("max_requeue_tokens must be an integer or None")
            if self.max_requeue_tokens <= 0:
                raise ValueError("max_requeue_tokens must be positive or None")
        if not isinstance(self.mtp_draft_tokens, int) or isinstance(self.mtp_draft_tokens, bool):
            raise TypeError("mtp_draft_tokens must be an integer")
        if self.mtp_draft_tokens <= 0:
            raise ValueError("mtp_draft_tokens must be positive")
        if self.mtp_cache_bits is not None:
            if not isinstance(self.mtp_cache_bits, int) or isinstance(self.mtp_cache_bits, bool):
                raise TypeError("mtp_cache_bits must be an integer or None")
            if not 2 <= self.mtp_cache_bits <= 8:
                raise ValueError("mtp_cache_bits must be in the range 2..8 or None")


@dataclass(frozen=True, slots=True)
class RuntimeSamplingConfig:
    temperature: float = 1.0
    min_p: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    temperature_last: bool = False
    repetition_penalty_range: int = 100_000_000
    repetition_decay: int = 0
    adaptive_target: float = 1.0
    adaptive_decay: float = 0.9
    logit_bias: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "temperature",
            "min_p",
            "top_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "adaptive_target",
            "adaptive_decay",
        ):
            _validate_finite(name, getattr(self, name))

        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be between 0 and 1")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        _validate_bool("temperature_last", self.temperature_last)
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise TypeError("top_k must be an integer")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        for name in ("repetition_penalty_range", "repetition_decay"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.adaptive_target <= 1:
            raise ValueError("adaptive_target must be between 0 and 1")
        if not 0 <= self.adaptive_decay < 1:
            raise ValueError("adaptive_decay must be between 0 (inclusive) and 1 (exclusive)")
        if not isinstance(self.logit_bias, tuple):
            raise TypeError("logit_bias must be a tuple")
        normalized_bias: list[tuple[int, float]] = []
        seen_token_ids: set[int] = set()
        for entry in self.logit_bias:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("logit_bias entries must be (token_id, bias) tuples")
            token_id, bias = entry
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("logit_bias token ids must be integers")
            if token_id < 0 or token_id in seen_token_ids:
                raise ValueError("logit_bias token ids must be unique and non-negative")
            _validate_finite("logit_bias", bias)
            seen_token_ids.add(token_id)
            normalized_bias.append((token_id, float(bias)))
        object.__setattr__(self, "logit_bias", tuple(normalized_bias))


@dataclass(frozen=True, slots=True)
class RuntimeGenerationConstraint:
    trigger: str
    lark_grammar: str
    eos_after_completed: bool

    def __post_init__(self) -> None:
        for name in ("trigger", "lark_grammar"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        _validate_bool("eos_after_completed", self.eos_after_completed)


@dataclass(frozen=True, slots=True)
class RuntimeGenerationRequest:
    request_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int
    seed: int | None = None
    stop_conditions: tuple[str | int, ...] = ()
    sampling: RuntimeSamplingConfig | None = None
    prompt_attachments: tuple[object, ...] = ()
    output_json_schema: str | None = None
    output_json_trigger: str | None = None
    generation_constraint: RuntimeGenerationConstraint | None = None
    use_native_eos: bool = False

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        _validate_token_ids(self.input_ids)
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("seed must be an integer or None")
        if not isinstance(self.stop_conditions, tuple):
            raise TypeError("stop_conditions must be a tuple")
        for condition in self.stop_conditions:
            if isinstance(condition, str):
                if condition == "":
                    raise ValueError("string stop conditions must not be empty")
            elif isinstance(condition, int) and not isinstance(condition, bool):
                if condition < 0:
                    raise ValueError("integer stop conditions must be non-negative")
            else:
                raise TypeError("stop conditions must be strings or integers")
        if self.sampling is not None and not isinstance(self.sampling, RuntimeSamplingConfig):
            raise TypeError("sampling must be RuntimeSamplingConfig or None")
        if not isinstance(self.prompt_attachments, tuple):
            raise TypeError("prompt_attachments must be a tuple")
        if self.output_json_schema is not None:
            if not isinstance(self.output_json_schema, str):
                raise TypeError("output_json_schema must be a string or None")
            if not self.output_json_schema.strip():
                raise ValueError("output_json_schema must not be empty")
        if self.output_json_trigger is not None:
            if self.output_json_schema is None:
                raise ValueError("output_json_trigger requires output_json_schema")
            if not isinstance(self.output_json_trigger, str):
                raise TypeError("output_json_trigger must be a string or None")
            if not self.output_json_trigger.strip():
                raise ValueError("output_json_trigger must not be empty")
        if self.generation_constraint is not None and not isinstance(
            self.generation_constraint, RuntimeGenerationConstraint
        ):
            raise TypeError("generation_constraint must be a RuntimeGenerationConstraint or None")
        _validate_bool("use_native_eos", self.use_native_eos)
        if self.generation_constraint is not None and self.output_json_schema is not None:
            raise ValueError("generation_constraint cannot be combined with output_json_schema")


class RuntimeStopReason(str, Enum):
    EOS = "eos"
    STOP_STRING = "stop_string"
    LENGTH = "length"
    FILTER = "filter"
    LOOP = "loop"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class RuntimeTiming:
    queue_seconds: float | None = None
    prefill_seconds: float | None = None
    generation_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("queue_seconds", "prefill_seconds", "generation_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            _validate_finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeRenderedPrompt:
    text: str
    input_ids: tuple[int, ...]
    runtime_attachments: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_token_ids(self.input_ids)
        if not isinstance(self.runtime_attachments, tuple):
            raise TypeError("runtime_attachments must be a tuple")


@dataclass(frozen=True, slots=True)
class RuntimeStarted:
    request_id: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)


@dataclass(frozen=True, slots=True)
class RuntimeTextDelta:
    request_id: str
    text: str
    token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.text == "":
            raise ValueError("text delta must not be empty")
        if not isinstance(self.token_ids, tuple):
            raise TypeError("token_ids must be a tuple")
        for token_id in self.token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token_ids must contain only integers")
            if token_id < 0:
                raise ValueError("token_ids must be non-negative")


@dataclass(frozen=True, slots=True)
class RuntimeFinished:
    request_id: str
    reason: RuntimeStopReason
    usage: TokenUsage
    timing: RuntimeTiming
    stop_sequence: str | None = None
    backend_reason: str | None = None
    eos_token_id: int | None = None
    eos_token_text: str | None = None

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.reason, RuntimeStopReason):
            raise TypeError("reason must be RuntimeStopReason")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        if not isinstance(self.timing, RuntimeTiming):
            raise TypeError("timing must be RuntimeTiming")
        if self.stop_sequence is not None:
            if not isinstance(self.stop_sequence, str):
                raise TypeError("stop_sequence must be a string or None")
            if not self.stop_sequence:
                raise ValueError("stop_sequence must not be empty")
            if self.reason is not RuntimeStopReason.STOP_STRING:
                raise ValueError("stop_sequence is valid only for stop-string completions")
        if self.backend_reason is not None:
            if not isinstance(self.backend_reason, str):
                raise TypeError("backend_reason must be a string or None")
            if not self.backend_reason:
                raise ValueError("backend_reason must not be empty")
        if self.eos_token_id is not None:
            if not isinstance(self.eos_token_id, int) or isinstance(self.eos_token_id, bool):
                raise TypeError("eos_token_id must be an integer or None")
            if self.eos_token_id < 0:
                raise ValueError("eos_token_id must be non-negative")
        if self.eos_token_text is not None:
            if not isinstance(self.eos_token_text, str):
                raise TypeError("eos_token_text must be a string or None")
            if not self.eos_token_text:
                raise ValueError("eos_token_text must not be empty")


@dataclass(frozen=True, slots=True)
class RuntimeCancelled:
    request_id: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)


@dataclass(frozen=True, slots=True)
class RuntimeFailed:
    request_id: str
    error: CanonicalError

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.error, CanonicalError):
            raise TypeError("error must be CanonicalError")


RuntimeEvent = RuntimeStarted | RuntimeTextDelta | RuntimeFinished | RuntimeCancelled | RuntimeFailed


class RuntimeConstraintUnsupported(ValueError):
    """Raised when the active runtime cannot soundly enforce a generation constraint."""


class RuntimeInjectionUnavailable(RuntimeError):
    """Raised when a runtime session can no longer accept forced output."""


class RuntimeSessionLike(Protocol):
    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        ...

    def inject_text(self, text: str) -> None:
        ...

    async def cancel(self) -> None:
        ...
