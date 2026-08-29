from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from exqserve.model.contracts import ToolConstraintMode
from exqserve.observability.capture import CaptureMode
from exqserve.server import cli
from exqserve.server.app import RuntimeUnavailableError


def test_cli_parses_runtime_control_and_capture_options(tmp_path: Path) -> None:
    capture = tmp_path / "captures.jsonl"
    config = cli.parse_config(
        [
            str(tmp_path),
            "--served-model-id",
            "local-qwen",
            "--tool-constraint-mode",
            "schema",
            "--model-root",
            str(tmp_path.parent),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--cache-tokens",
            "65536",
            "--kv-cache-bits",
            "fp16",
            "--max-batch-size",
            "4",
            "--max-chunk-size",
            "1024",
            "--sysmem-kv-cache-mb",
            "8192",
            "--sysmem-recurrent-cache-mb",
            "2048",
            "--mtp",
            "--mtp-draft-tokens",
            "6",
            "--mtp-cache-bits",
            "fp16",
            "--max-injection-body-bytes",
            "8192",
            "--dynamic-draft",
            "--draft-confidence",
            "0.55",
            "--autosplit-no-forward",
            "--cuda-malloc-async",
            "--qc-staging",
            "0",
            "--max-requeue-tokens",
            "1024",
            "--reserve-per-device-gb",
            "1.5",
            "--device-ids",
            "0,1",
            "--tensor-parallel",
            "--tp-backend",
            "nccl",
            "--tp-output-device",
            "1",
            "--max-in-flight",
            "3",
            "--max-prompt-tokens",
            "32000",
            "--max-output-tokens",
            "4096",
            "--max-total-tokens",
            "36000",
            "--timeout-seconds",
            "120",
            "--default-output-tokens",
            "2048",
            "--response-store-max-records",
            "64",
            "--capture-mode",
            "metadata",
            "--capture-path",
            str(capture),
        ]
    )

    assert config.model_directory == tmp_path
    assert config.served_model_id == "local-qwen"
    assert config.tool_constraint_mode is ToolConstraintMode.SCHEMA
    assert config.model_root == tmp_path.parent
    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.cache_tokens == 65536
    assert config.cache_key_bits is None
    assert config.cache_value_bits is None
    assert config.sysmem_kv_cache_mb == 8192
    assert config.sysmem_recurrent_cache_mb == 2048
    assert config.mtp_enabled is True
    assert config.mtp_draft_tokens == 6
    assert config.mtp_cache_bits is None
    assert config.max_injection_body_bytes == 8192
    assert config.dynamic_draft_tokens is True
    assert config.draft_confidence == 0.55
    assert config.autosplit_no_forward is True
    assert config.cuda_malloc_async is True
    assert config.qc_staging == 0
    assert config.max_requeue_tokens == 1024
    assert config.reserve_per_device_gb == (1.5,)
    assert config.device_ids == (0, 1)
    assert config.tensor_parallel is True
    assert config.tp_backend == "nccl"
    assert config.tp_output_device == 1
    assert config.max_in_flight == 3
    assert config.max_prompt_tokens == 32000
    assert config.max_output_tokens == 4096
    assert config.max_total_tokens == 36000
    assert config.timeout_seconds == 120.0
    assert config.default_api_output_tokens == 2048
    assert config.response_store_max_records == 64
    assert config.capture_mode is CaptureMode.METADATA
    assert config.capture_path == capture


def test_cli_and_yaml_parse_ngram_and_moe_cpu_options(tmp_path: Path) -> None:
    cli_config = cli.parse_config(
        [
            str(tmp_path / "model"),
            "--ngram-match-min",
            "3",
            "--ngram-draft-tokens",
            "7",
            "--moe-cpu-offload-layers",
            "12",
            "--moe-cpu-threads",
            "6",
        ]
    )
    assert cli_config.ngram_match_min == 3
    assert cli_config.ngram_draft_size == 7
    assert cli_config.moe_cpu_offload_layers == 12
    assert cli_config.moe_cpu_threads == 6

    config_path = tmp_path / "ngram-moe.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"model-directory: {tmp_path / 'model'}",
                "ngram-match-min: 4",
                "ngram-draft-tokens: 9",
                "moe-cpu-offload-layers: 16",
                "moe-cpu-threads: 8",
            ]
        ),
        encoding="utf-8",
    )
    yaml_config = cli.parse_config(["--config", str(config_path)])
    assert yaml_config.ngram_match_min == 4
    assert yaml_config.ngram_draft_size == 9
    assert yaml_config.moe_cpu_offload_layers == 16
    assert yaml_config.moe_cpu_threads == 8

    invalid_key = tmp_path / "invalid-ngram-key.yaml"
    invalid_key.write_text(
        f"model-directory: {tmp_path / 'model'}\nngram-draft-size: 9\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown configuration key: ngram-draft-size"):
        cli.parse_config(["--config", str(invalid_key)])


def test_cli_loads_complete_yaml_config_without_positional_model(tmp_path: Path) -> None:
    model = tmp_path / "Qwen3.8-27B"
    capture = tmp_path / "capture.jsonl"
    config_path = tmp_path / "exqserve.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"model-directory: {model}",
                "host: 0.0.0.0",
                "port: 9100",
                "cache-tokens: 65536",
                "kv-cache-bits: fp16",
                "max-batch-size: 2",
                "max-chunk-size: 1024",
                "sysmem-kv-cache-mb: 4096",
                "sysmem-recurrent-cache-mb: 3072",
                "tool-constraint-mode: format",
                "reserve-per-device-gb: [1.25, 2.5]",
                "device-ids: [0, 1]",
                "tensor-parallel: true",
                "tp-backend: native",
                "tp-output-device: 0",
                "mtp: true",
                "mtp-draft-tokens: 6",
                "mtp-cache-bits: 8",
                "max-injection-body-bytes: 12288",
                "dynamic-draft-tokens: true",
                "draft-confidence: 0.6",
                "autosplit-no-forward: true",
                "cuda-malloc-async: true",
                "qc-staging: 0",
                "vision: true",
                "allow-remote-images: true",
                "vision-cache-mb: 64",
                "public-metrics: true",
                "capture-mode: metadata",
                f"capture-path: {capture}",
            ]
        ),
        encoding="utf-8",
    )

    config = cli.parse_config(["--config", str(config_path)])

    assert config.model_directory == model
    assert config.host == "0.0.0.0"
    assert config.port == 9100
    assert config.cache_tokens == 65536
    assert config.cache_key_bits is None
    assert config.cache_value_bits is None
    assert config.max_batch_size == 2
    assert config.max_chunk_size == 1024
    assert config.sysmem_kv_cache_mb == 4096
    assert config.sysmem_recurrent_cache_mb == 3072
    assert config.tool_constraint_mode is ToolConstraintMode.FORMAT
    assert config.reserve_per_device_gb == (1.25, 2.5)
    assert config.device_ids == (0, 1)
    assert config.tensor_parallel is True
    assert config.tp_backend == "native"
    assert config.tp_output_device == 0
    assert config.mtp_enabled is True
    assert config.mtp_draft_tokens == 6
    assert config.mtp_cache_bits == 8
    assert config.max_injection_body_bytes == 12288
    assert config.dynamic_draft_tokens is True
    assert config.draft_confidence == 0.6
    assert config.autosplit_no_forward is True
    assert config.cuda_malloc_async is True
    assert config.qc_staging == 0
    assert config.vision_enabled is True
    assert config.allow_remote_images is True
    assert config.vision_cache_mb == 64
    assert config.protect_metrics is False
    assert config.capture_mode is CaptureMode.METADATA
    assert config.capture_path == capture


def test_cli_and_yaml_support_chat_template_override_with_cli_precedence(tmp_path: Path) -> None:
    yaml_template = tmp_path / "yaml-template.jinja"
    cli_template = tmp_path / "cli-template.jinja"
    yaml_template.write_text("YAML {{ messages }}", encoding="utf-8")
    cli_template.write_text("CLI {{ messages }}", encoding="utf-8")
    config_path = tmp_path / "exqserve.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"model-directory: {tmp_path / 'model'}",
                f"chat-template: {yaml_template}",
            ]
        ),
        encoding="utf-8",
    )

    yaml_config = cli.parse_config(["--config", str(config_path)])
    assert yaml_config.chat_template == yaml_template
    assert yaml_config.runtime_load_config().chat_template == "YAML {{ messages }}"

    cli_config = cli.parse_config(
        ["--config", str(config_path), "--chat-template", str(cli_template)]
    )
    assert cli_config.chat_template == cli_template
    assert cli_config.runtime_load_config().chat_template == "CLI {{ messages }}"


def test_cli_explicit_values_override_yaml_including_lists_and_booleans(tmp_path: Path) -> None:
    yaml_model = tmp_path / "yaml-model"
    cli_model = tmp_path / "cli-model"
    config_path = tmp_path / "exqserve.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"model-directory: {yaml_model}",
                "port: 9000",
                "reserve-per-device-gb: [1.0, 2.0]",
                "tool-constraint-mode: format",
                "mtp: true",
                "vision: true",
                "cuda-malloc-async: false",
                "public-metrics: true",
            ]
        ),
        encoding="utf-8",
    )

    config = cli.parse_config(
        [
            "--config",
            str(config_path),
            str(cli_model),
            "--port",
            "9200",
            "--reserve-per-device-gb",
            "0.5",
            "--tool-constraint-mode",
            "schema",
            "--no-mtp",
            "--no-vision",
            "--cuda-malloc-async",
            "--no-public-metrics",
        ]
    )

    assert config.model_directory == cli_model
    assert config.port == 9200
    assert config.reserve_per_device_gb == (0.5,)
    assert config.tool_constraint_mode is ToolConstraintMode.SCHEMA
    assert config.mtp_enabled is False
    assert config.vision_enabled is False
    assert config.cuda_malloc_async is True
    assert config.protect_metrics is True


def test_cli_yaml_integer_cache_precision_and_api_key_list_are_normalized(tmp_path: Path) -> None:
    config_path = tmp_path / "exqserve.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"model-directory: {tmp_path / 'model'}",
                "kv-cache-bits: 8",
                "mtp-cache-bits: 4",
                f"draft-model: {tmp_path / 'draft'}",
                "draft-tokens: 5",
                "draft-cache-bits: 6",
                "api-key: [alpha, beta]",
            ]
        ),
        encoding="utf-8",
    )

    config = cli.parse_config(["--config", str(config_path)])

    assert config.cache_key_bits == 8
    assert config.cache_value_bits == 8
    assert config.mtp_cache_bits == 4
    assert config.draft_model == tmp_path / "draft"
    assert config.draft_tokens == 5
    assert config.draft_cache_bits == 6
    assert config.api_keys == ("alpha", "beta")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("model-directory: /models/example\nunknown-option: 1\n", "unknown configuration key"),
        ("- model-directory\n- /models/example\n", "top-level mapping"),
        ("model-directory: [unterminated\n", "invalid YAML"),
    ],
)
def test_cli_rejects_invalid_yaml_config(tmp_path: Path, content: str, message: str) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        cli.parse_config(["--config", str(config_path)])


def test_cli_yaml_and_explicit_cli_support_multi_lora_with_scaling(tmp_path: Path) -> None:
    yaml_lora = tmp_path / "yaml-lora"
    cli_lora_a = tmp_path / "cli-lora-a"
    cli_lora_b = tmp_path / "cli-lora-b"
    config_path = tmp_path / "lora.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"model-directory: {tmp_path / 'model'}",
                f"lora: [{yaml_lora}]",
                "lora-scaling: [0.5]",
            ]
        ),
        encoding="utf-8",
    )

    yaml_config = cli.parse_config(["--config", str(config_path)])
    assert yaml_config.loras == (yaml_lora,)
    assert yaml_config.lora_scalings == (0.5,)

    cli_config = cli.parse_config(
        [
            "--config",
            str(config_path),
            "--lora",
            str(cli_lora_a),
            "--lora",
            str(cli_lora_b),
            "--lora-scaling",
            "1.0",
            "--lora-scaling",
            "0.8",
        ]
    )
    assert cli_config.loras == (cli_lora_a, cli_lora_b)
    assert cli_config.lora_scalings == (1.0, 0.8)
    assert [(item.directory, item.scaling) for item in cli_config.runtime_load_config().lora_adapters] == [
        (str(cli_lora_a), 1.0),
        (str(cli_lora_b), 0.8),
    ]

    default_scaling = cli.parse_config([str(tmp_path), "--lora", str(cli_lora_a)])
    assert default_scaling.runtime_load_config().lora_adapters[0].scaling == 1.0

    with pytest.raises(ValueError, match="number of loras"):
        cli.parse_config(
            [
                str(tmp_path),
                "--lora",
                str(cli_lora_a),
                "--lora",
                str(cli_lora_b),
                "--lora-scaling",
                "1.0",
            ]
        )


def test_cli_exposes_generic_external_draft_but_not_private_offload(tmp_path: Path) -> None:
    draft = tmp_path / "draft"
    config = cli.parse_config(
        [
            str(tmp_path),
            "--draft-model",
            str(draft),
            "--draft-tokens",
            "5",
            "--draft-cache-bits",
            "fp16",
        ]
    )
    assert config.draft_model == draft
    assert config.draft_tokens == 5
    assert config.draft_cache_bits is None
    runtime_config = config.runtime_load_config()
    assert runtime_config.draft_model_directory == str(draft)
    assert runtime_config.draft_tokens == 5
    assert runtime_config.draft_cache_bits is None

    with pytest.raises(SystemExit):
        cli.parse_config([str(tmp_path), "--draft-prefill-offload"])


def test_cli_rejects_mtp_with_external_draft(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.parse_config([str(tmp_path), "--mtp", "--draft-model", str(tmp_path / "draft")])


def test_cli_yaml_and_explicit_cli_sampler_preset_precedence(tmp_path: Path) -> None:
    yaml_preset = tmp_path / "yaml-sampler.yaml"
    cli_preset = tmp_path / "cli-sampler.yaml"
    yaml_preset.write_text("temperature:\n  override: 0.7\n", encoding="utf-8")
    cli_preset.write_text("temperature:\n  override: 0.2\n  force: true\n", encoding="utf-8")
    config_path = tmp_path / "server.yaml"
    config_path.write_text(
        f"model-directory: {tmp_path / 'model'}\nsampler-preset: {yaml_preset}\n",
        encoding="utf-8",
    )

    yaml_config = cli.parse_config(["--config", str(config_path)])
    assert yaml_config.sampling_overrides.overrides[0].value == 0.7
    assert yaml_config.sampling_overrides.overrides[0].force is False

    cli_config = cli.parse_config(
        ["--config", str(config_path), "--sampler-preset", str(cli_preset)]
    )
    assert cli_config.sampling_overrides.overrides[0].value == 0.2
    assert cli_config.sampling_overrides.overrides[0].force is True


def test_cli_sampler_preset_validation_happens_during_parse_config(tmp_path: Path) -> None:
    preset = tmp_path / "bad-sampler.yaml"
    preset.write_text("top_p:\n  override: 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid sampler override"):
        cli.parse_config([str(tmp_path), "--sampler-preset", str(preset)])


def test_cli_main_composes_once_and_runs_single_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    composed = SimpleNamespace(app=object())
    seen: dict[str, object] = {}

    def fake_compose(config):  # type: ignore[no-untyped-def]
        seen["config"] = config
        return composed

    def fake_run(app, *, host: str, port: int):  # type: ignore[no-untyped-def]
        seen["run"] = (app, host, port)

    monkeypatch.setattr(cli, "compose_server", fake_compose)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = cli.main([str(tmp_path), "--host", "127.0.0.1", "--port", "8123"])

    assert result == 0
    assert seen["run"] == (composed.app, "127.0.0.1", 8123)


def test_cli_dispatches_leading_download_without_composing_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_download_main(argv):  # type: ignore[no-untyped-def]
        seen.append(list(argv))
        return 7

    monkeypatch.setattr(cli, "_download_main", fake_download_main)
    result = cli.main(["download", "owner/model", "--output", "/tmp/model"])

    assert result == 7
    assert seen == [["owner/model", "--output", "/tmp/model"]]


def test_cli_missing_runtime_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_config):  # type: ignore[no-untyped-def]
        raise RuntimeUnavailableError("install runtime extra")

    monkeypatch.setattr(cli, "compose_server", fail)

    assert cli.main([str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "install runtime extra" in captured.err


def test_cli_loads_multiple_auth_keys_from_env_file_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "keys.txt"
    key_file.write_text("file-beta\n\n", encoding="utf-8")
    monkeypatch.setenv("EXQSERVE_API_KEY", "env-alpha")
    monkeypatch.delenv("EXQSERVE_API_KEY_FILE", raising=False)

    config = cli.parse_config(
        [
            str(tmp_path),
            "--api-key-file",
            str(key_file),
            "--api-key",
            "cli-gamma",
            "--public-metrics",
            "--max-request-body-bytes",
            "4096",
            "--response-store-ttl-seconds",
            "120",
            "--response-store-max-bytes",
            "8192",
        ]
    )

    assert config.api_keys == ("env-alpha", "file-beta", "cli-gamma")
    assert config.protect_metrics is False
    assert config.max_request_body_bytes == 4096
    assert config.response_store_ttl_seconds == 120.0
    assert config.response_store_max_bytes == 8192
    rendered = repr(config)
    assert "env-alpha" not in rendered
    assert "file-beta" not in rendered
    assert "cli-gamma" not in rendered


def test_cli_warns_when_binding_keyless_server_to_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("EXQSERVE_API_KEY", raising=False)
    monkeypatch.delenv("EXQSERVE_API_KEY_FILE", raising=False)
    composed = SimpleNamespace(app=object())
    monkeypatch.setattr(cli, "compose_server", lambda _config: composed)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_args, **_kwargs: None)

    assert cli.main([str(tmp_path), "--host", "0.0.0.0"]) == 0
    assert "non-loopback host without API-key authentication" in capsys.readouterr().err


def test_cli_empty_api_key_file_fails_without_starting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("EXQSERVE_API_KEY", raising=False)
    monkeypatch.delenv("EXQSERVE_API_KEY_FILE", raising=False)
    key_file = tmp_path / "empty-keys.txt"
    key_file.write_text("\n", encoding="utf-8")
    compose_calls = 0

    def fake_compose(_config):  # type: ignore[no-untyped-def]
        nonlocal compose_calls
        compose_calls += 1
        return SimpleNamespace(app=object())

    monkeypatch.setattr(cli, "compose_server", fake_compose)
    assert cli.main([str(tmp_path), "--api-key-file", str(key_file)]) == 2
    assert compose_calls == 0
    assert "API key file contains no keys" in capsys.readouterr().err


def test_cli_model_dialect_defaults_to_auto_and_accepts_explicit_id(tmp_path: Path) -> None:
    default_config = cli.parse_config([str(tmp_path)])
    explicit_config = cli.parse_config([str(tmp_path), "--model-dialect", "custom-agent"])

    assert default_config.model_dialect == "auto"
    assert explicit_config.model_dialect == "custom-agent"


def test_yaml_model_dialect_is_supported_and_cli_overrides_it(tmp_path: Path) -> None:
    config_path = tmp_path / "dialect.yaml"
    config_path.write_text(
        f"model-directory: {tmp_path}\nmodel-dialect: yaml-agent\n",
        encoding="utf-8",
    )

    yaml_config = cli.parse_config(["--config", str(config_path)])
    cli_config = cli.parse_config(
        ["--config", str(config_path), "--model-dialect", "cli-agent"]
    )

    assert yaml_config.model_dialect == "yaml-agent"
    assert cli_config.model_dialect == "cli-agent"
