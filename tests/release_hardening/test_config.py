from pathlib import Path

import pytest

from exqserve.model.contracts import ToolConstraintMode
from exqserve.observability.capture import CaptureMode
from exqserve.server.config import ServerConfig


def test_server_config_defaults_are_generic_and_cpu_safe(tmp_path: Path) -> None:
    config = ServerConfig(tmp_path)

    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.cache_tokens == 32768
    assert config.cache_key_bits == 8
    assert config.cache_value_bits == 8
    assert config.max_batch_size == 8
    assert config.max_chunk_size == 2048
    assert config.max_in_flight == 8
    assert config.default_api_output_tokens is None
    assert config.response_store_max_records == 1024
    assert config.response_store_ttl_seconds == 3600.0
    assert config.response_store_max_bytes == 64 * 1024 * 1024
    assert config.max_request_body_bytes == 32 * 1024 * 1024
    assert config.max_injection_body_bytes == 64 * 1024
    assert config.vision_cache_mb == 256
    assert config.sysmem_kv_cache_mb == 0
    assert config.sysmem_recurrent_cache_mb == 4096
    assert config.ngram_match_min == 0
    assert config.ngram_draft_size == 4
    assert config.moe_cpu_offload_layers == 0
    assert config.moe_cpu_threads is None
    assert config.tool_constraint_mode is ToolConstraintMode.OFF
    assert config.api_keys == ()
    assert config.protect_metrics is True
    assert config.capture_mode is CaptureMode.OFF
    assert config.capture_path is None
    assert config.effective_served_model_id() == tmp_path.name
    assert config.effective_context_length(131072) == 32768

    runtime = config.runtime_load_config()
    assert runtime.model_directory == str(tmp_path)
    assert runtime.cache_tokens == 32768
    assert runtime.cache_key_bits == 8
    assert runtime.cache_value_bits == 8
    assert runtime.mtp_enabled is False
    assert runtime.mtp_draft_tokens == 4
    assert runtime.mtp_cache_bits == 4
    assert runtime.dynamic_draft_tokens is False
    assert runtime.draft_confidence == 0.4
    assert runtime.draft_model_directory is None
    assert runtime.draft_tokens == 4
    assert runtime.draft_cache_bits == 4
    assert runtime.autosplit_no_forward is False
    assert runtime.cuda_malloc_async is False
    assert runtime.qc_staging is None
    assert runtime.max_requeue_tokens is None
    assert runtime.tensor_parallel is False
    assert runtime.tp_backend == "native"
    assert runtime.tp_output_device is None
    assert runtime.device_ids is None
    assert runtime.chat_template is None
    assert runtime.vision_cache_mb == 256
    assert runtime.sysmem_kv_cache_mb == 0
    assert runtime.sysmem_recurrent_cache_mb == 4096
    assert runtime.ngram_match_min == 0
    assert runtime.ngram_draft_size == 4
    assert runtime.moe_cpu_offload_layers == 0
    assert runtime.moe_cpu_threads is None

    control = config.request_control_config()
    assert control.max_in_flight == 8
    assert control.max_prompt_tokens is None
    assert control.max_output_tokens is None
    assert control.max_total_tokens == 32768
    assert control.timeout_seconds is None

    store = config.response_store_options()
    assert store.max_records == 1024
    assert store.ttl_seconds == 3600.0
    assert store.max_total_bytes == 64 * 1024 * 1024

    tool_serving = config.tool_serving_options()
    assert tool_serving.constraint_mode is ToolConstraintMode.OFF
    assert tool_serving.fanout_limit == 32
    assert tool_serving.constrained_parallel_limit == 8


def test_server_config_snapshots_custom_chat_template_at_startup(tmp_path: Path) -> None:
    template = tmp_path / "custom.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    config = ServerConfig(tmp_path / "model", chat_template=template)

    assert config.chat_template == template
    assert config.runtime_load_config().chat_template == "{{ messages }}"

    template.write_text("changed", encoding="utf-8")
    assert config.runtime_load_config().chat_template == "{{ messages }}"

    with pytest.raises(ValueError, match="could not be read"):
        ServerConfig(tmp_path / "model", chat_template=tmp_path / "missing.jinja")

    empty = tmp_path / "empty.jinja"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        ServerConfig(tmp_path / "model", chat_template=empty)


def test_server_config_validates_dedicated_injection_body_limit(tmp_path: Path) -> None:
    config = ServerConfig(tmp_path / "model", max_injection_body_bytes=8192)
    assert config.max_injection_body_bytes == 8192

    with pytest.raises(TypeError, match="max_injection_body_bytes"):
        ServerConfig(tmp_path / "model", max_injection_body_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_injection_body_bytes"):
        ServerConfig(tmp_path / "model", max_injection_body_bytes=0)


def test_server_config_validates_vision_cache_budget(tmp_path: Path) -> None:
    config = ServerConfig(tmp_path / "model", vision_cache_mb=0)
    assert config.vision_cache_mb == 0
    assert config.runtime_load_config().vision_cache_mb == 0

    with pytest.raises(TypeError, match="vision_cache_mb"):
        ServerConfig(tmp_path / "model", vision_cache_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vision_cache_mb"):
        ServerConfig(tmp_path / "model", vision_cache_mb=-1)


def test_server_config_round_trips_sysmem_cache_budgets(tmp_path: Path) -> None:
    config = ServerConfig(
        tmp_path / "model",
        sysmem_kv_cache_mb=8192,
        sysmem_recurrent_cache_mb=2048,
    )
    runtime = config.runtime_load_config()
    assert runtime.sysmem_kv_cache_mb == 8192
    assert runtime.sysmem_recurrent_cache_mb == 2048

    with pytest.raises(TypeError, match="sysmem_kv_cache_mb"):
        ServerConfig(tmp_path / "model", sysmem_kv_cache_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sysmem_kv_cache_mb"):
        ServerConfig(tmp_path / "model", sysmem_kv_cache_mb=-1)
    with pytest.raises(TypeError, match="sysmem_recurrent_cache_mb"):
        ServerConfig(tmp_path / "model", sysmem_recurrent_cache_mb=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sysmem_recurrent_cache_mb"):
        ServerConfig(tmp_path / "model", sysmem_recurrent_cache_mb=0)


def test_server_config_round_trips_ngram_and_moe_cpu_settings(tmp_path: Path) -> None:
    config = ServerConfig(
        tmp_path / "model",
        ngram_match_min=3,
        ngram_draft_size=7,
        moe_cpu_offload_layers=12,
        moe_cpu_threads=6,
    )
    runtime = config.runtime_load_config()
    assert runtime.ngram_match_min == 3
    assert runtime.ngram_draft_size == 7
    assert runtime.moe_cpu_offload_layers == 12
    assert runtime.moe_cpu_threads == 6

    with pytest.raises(ValueError, match="cannot be combined"):
        ServerConfig(tmp_path / "model", ngram_match_min=2, mtp_enabled=True)
    with pytest.raises(ValueError, match="not supported with n-gram"):
        ServerConfig(tmp_path / "model", ngram_match_min=2, dynamic_draft_tokens=True)
    with pytest.raises(ValueError, match="layer-split"):
        ServerConfig(tmp_path / "model", moe_cpu_offload_layers=2, tensor_parallel=True)
    with pytest.raises(ValueError, match="moe_cpu_threads"):
        ServerConfig(tmp_path / "model", moe_cpu_threads=0)


def test_server_config_model_identity_override_and_context_ceiling(tmp_path: Path) -> None:
    config = ServerConfig(
        tmp_path,
        cache_tokens=65536,
        max_total_tokens=49152,
        served_model_id="  local-qwen  ",
    )

    assert config.effective_served_model_id() == "local-qwen"
    assert config.effective_context_length(131072) == 49152
    assert config.effective_context_length(None) == 49152
    assert config.request_control_config().max_total_tokens == 49152
    assert config.request_control_config(32768).max_total_tokens == 32768

    with pytest.raises(ValueError, match="served_model_id"):
        ServerConfig(tmp_path, served_model_id="   ")


def test_server_config_round_trips_static_mtp_settings(tmp_path: Path) -> None:
    config = ServerConfig(
        tmp_path,
        mtp_enabled=True,
        mtp_draft_tokens=6,
        mtp_cache_bits=None,
        dynamic_draft_tokens=True,
        draft_confidence=0.55,
        autosplit_no_forward=True,
        cuda_malloc_async=True,
        qc_staging=0,
        max_requeue_tokens=1024,
    )

    runtime = config.runtime_load_config()
    assert runtime.mtp_enabled is True
    assert runtime.mtp_draft_tokens == 6
    assert runtime.mtp_cache_bits is None
    assert runtime.dynamic_draft_tokens is True
    assert runtime.draft_confidence == 0.55
    assert runtime.autosplit_no_forward is True
    assert runtime.cuda_malloc_async is True
    assert runtime.qc_staging == 0
    assert runtime.max_requeue_tokens == 1024


def test_server_config_round_trips_tensor_parallel_settings(tmp_path: Path) -> None:
    config = ServerConfig(
        tmp_path,
        tensor_parallel=True,
        tp_backend="nccl",
        tp_output_device=1,
        device_ids=(0, 1),
    )

    runtime = config.runtime_load_config()
    assert runtime.tensor_parallel is True
    assert runtime.tp_backend == "nccl"
    assert runtime.tp_output_device == 1
    assert runtime.device_ids == (0, 1)

    with pytest.raises(ValueError, match="tp_backend"):
        ServerConfig(tmp_path, tensor_parallel=True, tp_backend="invalid")
    with pytest.raises(ValueError, match="tp_output_device"):
        ServerConfig(tmp_path, tensor_parallel=True, tp_output_device=-1)


def test_server_config_round_trips_external_draft_across_target_override(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    switched_target = tmp_path / "switched-target"
    config = ServerConfig(
        tmp_path,
        draft_model=draft,
        draft_tokens=5,
        draft_cache_bits=6,
    )

    runtime = config.runtime_load_config(switched_target)
    assert runtime.model_directory == str(switched_target)
    assert runtime.draft_model_directory == str(draft)
    assert runtime.draft_tokens == 5
    assert runtime.draft_cache_bits == 6


def test_server_config_round_trips_loras_across_target_override(tmp_path: Path) -> None:
    lora_a = tmp_path / "lora-a"
    lora_b = tmp_path / "lora-b"
    switched_target = tmp_path / "switched-target"
    config = ServerConfig(
        tmp_path,
        loras=(lora_a, lora_b),
        lora_scalings=(1.0, 0.8),
    )

    runtime = config.runtime_load_config(switched_target)
    assert runtime.model_directory == str(switched_target)
    assert [(adapter.directory, adapter.scaling) for adapter in runtime.lora_adapters] == [
        (str(lora_a), 1.0),
        (str(lora_b), 0.8),
    ]

    with pytest.raises(ValueError, match="number of loras"):
        ServerConfig(tmp_path, loras=(lora_a, lora_b), lora_scalings=(1.0,))
    with pytest.raises(ValueError, match="non-negative"):
        ServerConfig(tmp_path, loras=(lora_a,), lora_scalings=(-0.1,))


def test_server_config_accepts_fp16_cache_and_capture(tmp_path: Path) -> None:
    capture = tmp_path / "captures" / "requests.jsonl"
    config = ServerConfig(
        tmp_path,
        cache_key_bits=None,
        cache_value_bits=None,
        reserve_per_device_gb=(1.5, 2.0),
        capture_mode=CaptureMode.METADATA,
        capture_path=capture,
    )

    runtime = config.runtime_load_config()
    assert runtime.cache_key_bits is None
    assert runtime.cache_value_bits is None
    assert runtime.reserve_per_device_gb == (1.5, 2.0)
    assert config.capture_path == capture


def test_server_config_rejects_incoherent_capture_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="capture_path"):
        ServerConfig(tmp_path, capture_mode=CaptureMode.FULL)
    with pytest.raises(ValueError, match="capture_path"):
        ServerConfig(tmp_path, capture_path=tmp_path / "capture.jsonl")


def test_server_config_delegates_runtime_and_control_invariants(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiple of 256"):
        ServerConfig(tmp_path, cache_tokens=1000)
    with pytest.raises(ValueError, match="max_in_flight"):
        ServerConfig(tmp_path, max_in_flight=0)
    with pytest.raises(ValueError, match="port"):
        ServerConfig(tmp_path, port=0)
    with pytest.raises(ValueError, match="default_api_output_tokens"):
        ServerConfig(tmp_path, default_api_output_tokens=0)
    with pytest.raises(ValueError, match="tool_call_fanout_limit"):
        ServerConfig(tmp_path, tool_call_fanout_limit=0)
    with pytest.raises(TypeError, match="tool_call_fanout_limit"):
        ServerConfig(tmp_path, tool_call_fanout_limit=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="constrained_parallel_tool_call_limit"):
        ServerConfig(tmp_path, constrained_parallel_tool_call_limit=0)
    with pytest.raises(TypeError, match="constrained_parallel_tool_call_limit"):
        ServerConfig(tmp_path, constrained_parallel_tool_call_limit=True)  # type: ignore[arg-type]


def test_server_config_accepts_only_tool_constraint_enum_values(tmp_path: Path) -> None:
    assert (
        ServerConfig(tmp_path, tool_constraint_mode=ToolConstraintMode.SCHEMA).tool_constraint_mode
        is ToolConstraintMode.SCHEMA
    )
    with pytest.raises(TypeError, match="tool_constraint_mode"):
        ServerConfig(tmp_path, tool_constraint_mode="schema")  # type: ignore[arg-type]


def test_server_config_reasoning_budget_default_normalizes_disable_and_validates(tmp_path: Path) -> None:
    configured = ServerConfig(
        tmp_path, reasoning_budget_tokens=128, reasoning_budget_message="answer now "
    )
    budget = configured.reasoning_budget_default()
    assert budget.max_tokens == 128
    assert budget.message == "answer now "

    disabled = ServerConfig(tmp_path, reasoning_budget_tokens=-1)
    assert disabled.reasoning_budget_tokens is None
    assert disabled.reasoning_budget_default().max_tokens is None

    with pytest.raises(ValueError, match="reasoning_budget_tokens"):
        ServerConfig(tmp_path, reasoning_budget_tokens=-2)
    with pytest.raises(TypeError, match="reasoning_budget_tokens"):
        ServerConfig(tmp_path, reasoning_budget_tokens=True)  # type: ignore[arg-type]


def test_anthropic_compatibility_profile_validation(tmp_path: Path) -> None:
    assert (
        ServerConfig(
            tmp_path, anthropic_compatibility_profile="claude-code-2.1.251"
        ).anthropic_compatibility_profile
        == "claude-code-2.1.251"
    )
    with pytest.raises(ValueError, match="anthropic_compatibility_profile"):
        ServerConfig(tmp_path, anthropic_compatibility_profile="claude-code-latest")
