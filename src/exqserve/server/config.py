"""Immutable configuration for the composed ExQServe server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from exqserve.control.request import RequestControlConfig
from exqserve.core.sampling import SamplingOverridePolicy
from exqserve.model.contracts import ToolConstraintMode
from exqserve.observability.capture import CaptureMode
from exqserve.runtime.contracts import ExLlamaV3LoadConfig, LoRAAdapterConfig


@dataclass(frozen=True, slots=True)
class ServerConfig:
    model_directory: Path
    host: str = "127.0.0.1"
    port: int = 8000
    cache_tokens: int = 32768
    cache_key_bits: int | None = 8
    cache_value_bits: int | None = 8
    max_batch_size: int = 8
    max_chunk_size: int = 2048
    reserve_per_device_gb: tuple[float, ...] | None = None
    max_in_flight: int = 8
    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    timeout_seconds: float | None = None
    default_api_output_tokens: int = 4096
    response_store_max_records: int = 1024
    capture_mode: CaptureMode = CaptureMode.OFF
    capture_path: Path | None = None
    served_model_id: str | None = None
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
    api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    protect_metrics: bool = True
    max_request_body_bytes: int = 16 * 1024 * 1024
    response_store_ttl_seconds: float = 3600.0
    response_store_max_bytes: int = 64 * 1024 * 1024
    model_root: Path | None = None
    draft_model: Path | None = None
    draft_tokens: int = 4
    draft_cache_bits: int | None = 4
    loras: tuple[Path, ...] = ()
    lora_scalings: tuple[float, ...] = ()
    sampling_overrides: SamplingOverridePolicy = field(default_factory=SamplingOverridePolicy)
    tensor_parallel: bool = False
    tp_backend: str = "native"
    tp_output_device: int | None = None
    device_ids: tuple[int, ...] | None = None
    dynamic_draft_tokens: bool = False
    draft_confidence: float = 0.4
    chat_template: Path | None = None
    max_injection_body_bytes: int = 64 * 1024
    vision_cache_mb: int = 256
    model_dialect: str = "auto"
    tool_constraint_mode: ToolConstraintMode = ToolConstraintMode.OFF
    sysmem_kv_cache_mb: int = 0
    sysmem_recurrent_cache_mb: int = 4096
    ngram_match_min: int = 0
    ngram_draft_size: int = 4
    moe_cpu_offload_layers: int = 0
    moe_cpu_threads: int | None = None
    tool_call_fanout_limit: int = 32
    _chat_template_text: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_directory, Path):
            raise TypeError("model_directory must be a pathlib.Path")
        if self.model_root is not None and not isinstance(self.model_root, Path):
            raise TypeError("model_root must be pathlib.Path or None")
        if self.draft_model is not None and not isinstance(self.draft_model, Path):
            raise TypeError("draft_model must be pathlib.Path or None")
        if self.chat_template is not None:
            if not isinstance(self.chat_template, Path):
                raise TypeError("chat_template must be pathlib.Path or None")
            try:
                template_text = self.chat_template.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"chat_template could not be read: {self.chat_template}") from exc
            if not template_text.strip():
                raise ValueError("chat_template must not be empty")
            object.__setattr__(self, "_chat_template_text", template_text)
        if not isinstance(self.loras, tuple) or not all(isinstance(path, Path) for path in self.loras):
            raise TypeError("loras must be a tuple of pathlib.Path values")
        if not isinstance(self.lora_scalings, tuple):
            raise TypeError("lora_scalings must be a tuple")
        if self.lora_scalings and len(self.lora_scalings) != len(self.loras):
            raise ValueError("lora_scalings must be omitted or match the number of loras")
        if not isinstance(self.sampling_overrides, SamplingOverridePolicy):
            raise TypeError("sampling_overrides must be SamplingOverridePolicy")
        if self.model_root is not None:
            try:
                initial_parent = self.model_directory.resolve().parent
                configured_root = self.model_root.resolve()
            except OSError as exc:
                raise ValueError("model_root could not be resolved") from exc
            if initial_parent != configured_root:
                raise ValueError("model_root must contain model_directory as a direct child")
        if not isinstance(self.host, str):
            raise TypeError("host must be a string")
        if not self.host.strip():
            raise ValueError("host must not be empty")
        if not isinstance(self.model_dialect, str):
            raise TypeError("model_dialect must be a string")
        normalized_dialect = self.model_dialect.strip()
        if not normalized_dialect:
            raise ValueError("model_dialect must not be empty")
        object.__setattr__(self, "model_dialect", normalized_dialect)
        if not isinstance(self.tool_constraint_mode, ToolConstraintMode):
            raise TypeError("tool_constraint_mode must be a ToolConstraintMode")
        if not isinstance(self.port, int) or isinstance(self.port, bool):
            raise TypeError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in the range 1..65535")
        if not isinstance(self.default_api_output_tokens, int) or isinstance(
            self.default_api_output_tokens, bool
        ):
            raise TypeError("default_api_output_tokens must be an integer")
        if self.default_api_output_tokens <= 0:
            raise ValueError("default_api_output_tokens must be positive")
        if not isinstance(self.response_store_max_records, int) or isinstance(
            self.response_store_max_records, bool
        ):
            raise TypeError("response_store_max_records must be an integer")
        if self.response_store_max_records <= 0:
            raise ValueError("response_store_max_records must be positive")
        if not isinstance(self.capture_mode, CaptureMode):
            raise TypeError("capture_mode must be CaptureMode")
        if self.capture_path is not None and not isinstance(self.capture_path, Path):
            raise TypeError("capture_path must be pathlib.Path or None")
        if self.capture_mode is CaptureMode.OFF and self.capture_path is not None:
            raise ValueError("capture_path requires metadata or full capture mode")
        if self.capture_mode is not CaptureMode.OFF and self.capture_path is None:
            raise ValueError("capture_path is required when capture is enabled")
        if self.served_model_id is not None:
            if not isinstance(self.served_model_id, str):
                raise TypeError("served_model_id must be a string or None")
            normalized_id = self.served_model_id.strip()
            if not normalized_id:
                raise ValueError("served_model_id must not be empty")
            object.__setattr__(self, "served_model_id", normalized_id)
        if not isinstance(self.api_keys, tuple):
            raise TypeError("api_keys must be a tuple")
        normalized_keys: list[str] = []
        for key in self.api_keys:
            if not isinstance(key, str):
                raise TypeError("api_keys must contain strings")
            normalized = key.strip()
            if not normalized:
                raise ValueError("api_keys must not contain empty values")
            if normalized not in normalized_keys:
                normalized_keys.append(normalized)
        object.__setattr__(self, "api_keys", tuple(normalized_keys))
        if not isinstance(self.protect_metrics, bool):
            raise TypeError("protect_metrics must be a boolean")
        for name in (
            "max_request_body_bytes",
            "response_store_max_bytes",
            "max_injection_body_bytes",
            "vision_cache_mb",
            "sysmem_kv_cache_mb",
            "sysmem_recurrent_cache_mb",
            "ngram_match_min",
            "ngram_draft_size",
            "moe_cpu_offload_layers",
            "tool_call_fanout_limit",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if name in {
                "vision_cache_mb",
                "sysmem_kv_cache_mb",
                "ngram_match_min",
                "moe_cpu_offload_layers",
            }:
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
                continue
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.moe_cpu_threads is not None:
            if not isinstance(self.moe_cpu_threads, int) or isinstance(self.moe_cpu_threads, bool):
                raise TypeError("moe_cpu_threads must be an integer or None")
            if self.moe_cpu_threads <= 0:
                raise ValueError("moe_cpu_threads must be positive or None")
        if not isinstance(self.response_store_ttl_seconds, int | float) or isinstance(
            self.response_store_ttl_seconds, bool
        ):
            raise TypeError("response_store_ttl_seconds must be a number")
        if self.response_store_ttl_seconds <= 0:
            raise ValueError("response_store_ttl_seconds must be positive")
        self.runtime_load_config()
        self.request_control_config()

    def effective_model_root(self) -> Path:
        return self.model_directory.parent if self.model_root is None else self.model_root

    def effective_served_model_id(self) -> str:
        if self.served_model_id is not None:
            return self.served_model_id
        model_id = self.model_directory.name.strip()
        if not model_id:
            raise ValueError("model_directory must have a basename for served model discovery")
        return model_id

    def effective_context_length(self, model_limit: int | None) -> int:
        if model_limit is not None:
            if not isinstance(model_limit, int) or isinstance(model_limit, bool):
                raise TypeError("model_limit must be an integer or None")
            if model_limit <= 0:
                raise ValueError("model_limit must be positive")
        limits = [self.cache_tokens]
        if model_limit is not None:
            limits.append(model_limit)
        if self.max_total_tokens is not None:
            limits.append(self.max_total_tokens)
        return min(limits)

    def runtime_load_config(self, model_directory: Path | None = None) -> ExLlamaV3LoadConfig:
        directory = self.model_directory if model_directory is None else model_directory
        if not isinstance(directory, Path):
            raise TypeError("model_directory must be pathlib.Path")
        return ExLlamaV3LoadConfig(
            str(directory),
            self.cache_tokens,
            cache_key_bits=self.cache_key_bits,
            cache_value_bits=self.cache_value_bits,
            max_batch_size=self.max_batch_size,
            max_chunk_size=self.max_chunk_size,
            reserve_per_device_gb=self.reserve_per_device_gb,
            mtp_enabled=self.mtp_enabled,
            mtp_draft_tokens=self.mtp_draft_tokens,
            mtp_cache_bits=self.mtp_cache_bits,
            autosplit_no_forward=self.autosplit_no_forward,
            cuda_malloc_async=self.cuda_malloc_async,
            qc_staging=self.qc_staging,
            max_requeue_tokens=self.max_requeue_tokens,
            vision_enabled=self.vision_enabled,
            allow_remote_images=self.allow_remote_images,
            max_image_bytes=self.max_image_bytes,
            dynamic_draft_tokens=self.dynamic_draft_tokens,
            draft_confidence=self.draft_confidence,
            draft_model_directory=None if self.draft_model is None else str(self.draft_model),
            draft_tokens=self.draft_tokens,
            draft_cache_bits=self.draft_cache_bits,
            tensor_parallel=self.tensor_parallel,
            tp_backend=self.tp_backend,
            tp_output_device=self.tp_output_device,
            device_ids=self.device_ids,
            chat_template=self._chat_template_text,
            vision_cache_mb=self.vision_cache_mb,
            sysmem_kv_cache_mb=self.sysmem_kv_cache_mb,
            sysmem_recurrent_cache_mb=self.sysmem_recurrent_cache_mb,
            ngram_match_min=self.ngram_match_min,
            ngram_draft_size=self.ngram_draft_size,
            moe_cpu_offload_layers=self.moe_cpu_offload_layers,
            moe_cpu_threads=self.moe_cpu_threads,
            lora_adapters=tuple(
                LoRAAdapterConfig(
                    str(path),
                    1.0 if not self.lora_scalings else self.lora_scalings[index],
                )
                for index, path in enumerate(self.loras)
            ),
        )

    def request_control_config(self, model_limit: int | None = None) -> RequestControlConfig:
        return RequestControlConfig(
            self.max_in_flight,
            self.max_prompt_tokens,
            self.max_output_tokens,
            self.effective_context_length(model_limit),
            self.timeout_seconds,
        )
