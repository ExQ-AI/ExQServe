"""Command-line entry point for the composed ExQServe server."""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Never

import uvicorn
import yaml  # type: ignore[import-untyped]

from exqserve.core.sampling import SamplingOverridePolicy
from exqserve.downloader import main as _download_main
from exqserve.observability.capture import CaptureMode
from exqserve.server.app import RuntimeUnavailableError, compose_server
from exqserve.server.config import ServerConfig
from exqserve.server.sampling_overrides import load_sampling_override_policy

_PARSER_DEFAULTS: dict[str, object] = {
    "model_directory": None,
    "config": None,
    "served_model_id": None,
    "model_root": None,
    "host": "127.0.0.1",
    "port": 8000,
    "cache_tokens": 32768,
    "kv_cache_bits": "8",
    "max_batch_size": 8,
    "max_chunk_size": 2048,
    "mtp": False,
    "mtp_draft_tokens": 4,
    "mtp_cache_bits": "4",
    "draft_model": None,
    "draft_tokens": 4,
    "draft_cache_bits": "4",
    "lora": None,
    "lora_scaling": None,
    "sampler_preset": None,
    "reserve_per_device_gb": None,
    "device_ids": None,
    "tensor_parallel": False,
    "tp_backend": "native",
    "tp_output_device": None,
    "autosplit_no_forward": False,
    "cuda_malloc_async": False,
    "qc_staging": None,
    "max_requeue_tokens": None,
    "vision": False,
    "allow_remote_images": False,
    "max_image_bytes": 20 * 1024 * 1024,
    "max_in_flight": 8,
    "max_prompt_tokens": None,
    "max_output_tokens": None,
    "max_total_tokens": None,
    "timeout_seconds": None,
    "default_output_tokens": 4096,
    "response_store_max_records": 1024,
    "response_store_ttl_seconds": 3600.0,
    "response_store_max_bytes": 64 * 1024 * 1024,
    "max_request_body_bytes": 16 * 1024 * 1024,
    "api_key": None,
    "api_key_file": None,
    "public_metrics": False,
    "capture_mode": CaptureMode.OFF.value,
    "capture_path": None,
}

_BOOLEAN_CONFIG_KEYS = frozenset(
    {
        "mtp",
        "autosplit-no-forward",
        "tensor-parallel",
        "cuda-malloc-async",
        "vision",
        "allow-remote-images",
        "public-metrics",
    }
)
_PATH_CONFIG_KEYS = frozenset(
    {
        "model-directory",
        "model-root",
        "draft-model",
        "sampler-preset",
        "api-key-file",
        "capture-path",
    }
)


class _ConfigurationArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError(f"invalid configuration value: {message}")


def _build_parser(
    *,
    include_defaults: bool,
    configuration_mode: bool = False,
) -> argparse.ArgumentParser:
    parser_type = _ConfigurationArgumentParser if configuration_mode else argparse.ArgumentParser
    parser = parser_type(
        prog="exqserve",
        description="OpenAI- and Anthropic-compatible serving for ExLlamaV3.",
        epilog=(
            None
            if configuration_mode
            else "Other command: exqserve download REPO_ID --output LOCAL_DIR"
        ),
        argument_default=argparse.SUPPRESS,
        add_help=not configuration_mode,
    )
    parser.add_argument("model_directory", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--config",
        type=Path,
        help="Load server options from a YAML file; explicit CLI values take precedence.",
    )
    parser.add_argument("--served-model-id")
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Discover switchable models as immediate children of this directory.",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--cache-tokens", type=int)
    parser.add_argument(
        "--kv-cache-bits",
        choices=("fp16", "2", "3", "4", "5", "6", "7", "8"),
        help="KV cache precision for both key/value caches; use fp16 to disable quantization.",
    )
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--max-chunk-size", type=int)
    parser.add_argument(
        "--mtp",
        action=argparse.BooleanOptionalAction,
        help="Enable upstream ExLlamaV3 MTP drafting.",
    )
    parser.add_argument("--mtp-draft-tokens", type=int)
    parser.add_argument(
        "--mtp-cache-bits",
        choices=("fp16", "2", "3", "4", "5", "6", "7", "8"),
        help="MTP draft-cache precision; fp16 disables draft-cache quantization.",
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        help="Load a separate ExLlamaV3 draft model for speculative decoding.",
    )
    parser.add_argument("--draft-tokens", type=int)
    parser.add_argument(
        "--draft-cache-bits",
        choices=("fp16", "2", "3", "4", "5", "6", "7", "8"),
        help="External draft-cache precision; fp16 disables draft-cache quantization.",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        action="append",
        help="Load a PEFT LoRA adapter; repeat to load multiple adapters.",
    )
    parser.add_argument(
        "--lora-scaling",
        type=float,
        action="append",
        help="Independent LoRA scaling; repeat in the same order as --lora.",
    )
    parser.add_argument(
        "--sampler-preset",
        type=Path,
        help="Load a static sampler override preset YAML for OpenAI-compatible requests.",
    )
    parser.add_argument(
        "--reserve-per-device-gb",
        type=float,
        action="append",
        metavar="GB",
        help="Optional per-device VRAM reserve; repeat once per device.",
    )
    parser.add_argument(
        "--device-ids",
        metavar="ID[,ID...]",
        help="Comma-separated process-visible CUDA device indices to allow, e.g. 0,2.",
    )
    parser.add_argument(
        "--tensor-parallel",
        action=argparse.BooleanOptionalAction,
        help="Enable upstream ExLlamaV3 tensor parallelism across active CUDA devices.",
    )
    parser.add_argument(
        "--tp-backend",
        choices=("native", "nccl"),
        help="ExLlamaV3 tensor-parallel communication backend.",
    )
    parser.add_argument(
        "--tp-output-device",
        type=int,
        metavar="INDEX",
        help="CUDA device index on which ExLlamaV3 gathers TP output logits.",
    )
    parser.add_argument(
        "--autosplit-no-forward",
        action=argparse.BooleanOptionalAction,
        help="Skip ExLlamaV3 autosplit reference forwards to reduce peak VRAM during load.",
    )
    parser.add_argument(
        "--cuda-malloc-async",
        action=argparse.BooleanOptionalAction,
        help="Select Torch cudaMallocAsync before importing the ExLlamaV3 runtime.",
    )
    parser.add_argument(
        "--qc-staging",
        type=int,
        choices=(0, 1, 2),
        help=(
            "Override EXL3_QC_STAGING before importing ExLlamaV3; use 0 for the lowest-memory "
            "quantized-KV prefill path."
        ),
    )
    parser.add_argument(
        "--max-requeue-tokens",
        type=int,
        help="Optional ExLlamaV3 per-job requeue budget; leave unset to disable periodic requeue.",
    )
    parser.add_argument(
        "--vision",
        action=argparse.BooleanOptionalAction,
        help="Load the model's ExLlamaV3 vision component and accept image input.",
    )
    parser.add_argument(
        "--allow-remote-images",
        action=argparse.BooleanOptionalAction,
        help="Allow HTTP(S) image URLs. Data-image URLs work with --vision without this flag.",
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        help="Maximum encoded image payload size accepted per image.",
    )
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--default-output-tokens", type=int)
    parser.add_argument("--response-store-max-records", type=int)
    parser.add_argument("--response-store-ttl-seconds", type=float)
    parser.add_argument("--response-store-max-bytes", type=int)
    parser.add_argument("--max-request-body-bytes", type=int)
    parser.add_argument(
        "--api-key",
        action="append",
        help=(
            "Bearer API key; repeat for rotation. Prefer EXQSERVE_API_KEY or --api-key-file "
            "because command-line secrets may be visible in process listings."
        ),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Read newline-delimited Bearer API keys from a file.",
    )
    parser.add_argument(
        "--public-metrics",
        action=argparse.BooleanOptionalAction,
        help="Keep /metrics public even when API-key authentication is configured.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=tuple(mode.value for mode in CaptureMode),
    )
    parser.add_argument("--capture-path", type=Path)
    if include_defaults:
        parser.set_defaults(**_PARSER_DEFAULTS)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return _build_parser(include_defaults=True)


def _config_tokens(raw: Mapping[object, object]) -> list[str]:
    allowed_keys = {
        dest.replace("_", "-")
        for dest in _PARSER_DEFAULTS
        if dest not in {"config"}
    }
    tokens: list[str] = []

    model_value = raw.get("model-directory")
    if model_value is not None:
        if not isinstance(model_value, str):
            raise ValueError("configuration key 'model-directory' must be a string path")
        tokens.append(model_value)

    for raw_key, value in raw.items():
        if not isinstance(raw_key, str):
            raise ValueError(  # noqa: TRY004 - normalize user configuration failures
                "configuration keys must be strings"
            )
        key = raw_key
        if key not in allowed_keys:
            raise ValueError(f"unknown configuration key: {raw_key}")
        if key == "model-directory":
            continue
        if key in _BOOLEAN_CONFIG_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"configuration key '{key}' must be a boolean")
            tokens.append(f"--{key}" if value else f"--no-{key}")
            continue
        if key == "device-ids":
            if not isinstance(value, list) or not value or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                raise ValueError("configuration key 'device-ids' must be a non-empty integer list")
            tokens.extend(("--device-ids", ",".join(str(item) for item in value)))
            continue
        if key in {"reserve-per-device-gb", "lora-scaling"}:
            if not isinstance(value, list) or not value:
                raise ValueError(f"configuration key '{key}' must be a non-empty list")
            for item in value:
                if not isinstance(item, int | float) or isinstance(item, bool):
                    raise ValueError(  # noqa: TRY004 - normalize user configuration failures
                        f"configuration key '{key}' must contain numbers"
                    )
                tokens.extend((f"--{key}", str(item)))
            continue
        if key == "lora":
            values = [value] if isinstance(value, str) else value
            if not isinstance(values, list) or not values or not all(
                isinstance(item, str) for item in values
            ):
                raise ValueError("configuration key 'lora' must be a string path or string list")
            for item in values:
                tokens.extend(("--lora", item))
            continue
        if key == "api-key":
            values = [value] if isinstance(value, str) else value
            if not isinstance(values, list) or not values or not all(
                isinstance(item, str) for item in values
            ):
                raise ValueError("configuration key 'api-key' must be a string or string list")
            for item in values:
                tokens.extend((f"--{key}", item))
            continue
        if key in _PATH_CONFIG_KEYS:
            if not isinstance(value, str):
                raise ValueError(f"configuration key '{key}' must be a string path")
        elif isinstance(value, bool) or value is None or isinstance(value, list | dict):
            raise ValueError(f"configuration key '{key}' must be a scalar value")
        tokens.extend((f"--{key}", str(value)))

    return tokens


def _load_config_file(path: Path) -> dict[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in configuration file: {path}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(  # noqa: TRY004 - normalize user configuration failures
            "configuration file must contain a top-level mapping"
        )

    config_parser = _build_parser(include_defaults=False, configuration_mode=True)
    parsed = vars(config_parser.parse_args(_config_tokens(raw)))
    parsed.pop("config", None)
    if parsed.get("model_directory") is None:
        parsed.pop("model_directory", None)
    return parsed


def _cache_bits(value: str) -> tuple[int | None, int | None]:
    if value == "fp16":
        return None, None
    bits = int(value)
    return bits, bits


def _device_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("device_ids must be a comma-separated list of CUDA device indices")
    try:
        device_ids = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("device_ids must contain integer CUDA device indices") from exc
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError("device_ids must contain non-negative CUDA device indices")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("device_ids must not contain duplicates")
    return device_ids


def _read_api_key_file(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"API key file contains no keys: {path}")
    return values


def _configured_api_keys(args: argparse.Namespace) -> tuple[str, ...]:
    values: list[str] = []
    env_key = os.environ.get("EXQSERVE_API_KEY", "").strip()
    if env_key:
        values.append(env_key)
    env_file = os.environ.get("EXQSERVE_API_KEY_FILE", "").strip()
    if env_file:
        values.extend(_read_api_key_file(Path(env_file)))
    if args.api_key_file is not None:
        values.extend(_read_api_key_file(args.api_key_file))
    if args.api_key:
        values.extend(args.api_key)
    return tuple(values)


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def parse_config(argv: Sequence[str] | None = None) -> ServerConfig:
    parser = _build_parser(include_defaults=False)
    cli_values = vars(parser.parse_args(argv))
    config_path = cli_values.pop("config", None)
    if cli_values.get("model_directory") is None:
        cli_values.pop("model_directory", None)
    file_values = {} if config_path is None else _load_config_file(config_path)
    defaults = {key: value for key, value in _PARSER_DEFAULTS.items() if key != "config"}
    resolved = defaults | file_values | cli_values
    if resolved["model_directory"] is None:
        parser.error("model_directory is required (or set model-directory in --config)")
    args = argparse.Namespace(**resolved)

    key_bits, value_bits = _cache_bits(args.kv_cache_bits)
    mtp_bits, _ = _cache_bits(args.mtp_cache_bits)
    draft_bits, _ = _cache_bits(args.draft_cache_bits)
    reserve = None if args.reserve_per_device_gb is None else tuple(args.reserve_per_device_gb)
    device_ids = _device_ids(args.device_ids)
    sampling_overrides = (
        load_sampling_override_policy(args.sampler_preset)
        if args.sampler_preset is not None
        else SamplingOverridePolicy()
    )
    return ServerConfig(
        args.model_directory,
        args.host,
        args.port,
        args.cache_tokens,
        key_bits,
        value_bits,
        args.max_batch_size,
        args.max_chunk_size,
        reserve,
        args.max_in_flight,
        args.max_prompt_tokens,
        args.max_output_tokens,
        args.max_total_tokens,
        args.timeout_seconds,
        args.default_output_tokens,
        args.response_store_max_records,
        CaptureMode(args.capture_mode),
        args.capture_path,
        args.served_model_id,
        args.mtp,
        args.mtp_draft_tokens,
        mtp_bits,
        args.autosplit_no_forward,
        args.cuda_malloc_async,
        args.qc_staging,
        args.max_requeue_tokens,
        args.vision,
        args.allow_remote_images,
        args.max_image_bytes,
        _configured_api_keys(args),
        not args.public_metrics,
        args.max_request_body_bytes,
        args.response_store_ttl_seconds,
        args.response_store_max_bytes,
        args.model_root,
        args.draft_model,
        args.draft_tokens,
        draft_bits,
        () if args.lora is None else tuple(args.lora),
        () if args.lora_scaling is None else tuple(args.lora_scaling),
        sampling_overrides,
        args.tensor_parallel,
        args.tp_backend,
        args.tp_output_device,
        device_ids,
    )


def main(argv: Sequence[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv and resolved_argv[0] == "download":
        return _download_main(resolved_argv[1:])

    try:
        config = parse_config(resolved_argv)
    except (OSError, ValueError) as exc:
        print(f"exqserve: configuration failed: {exc}", file=sys.stderr)
        return 2
    if not config.api_keys and not _is_loopback_host(config.host):
        print(
            "exqserve: warning: serving on a non-loopback host without API-key authentication",
            file=sys.stderr,
        )
    try:
        composed = compose_server(config)
    except RuntimeUnavailableError as exc:
        print(f"exqserve: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI converts startup failures into concise local errors
        print(f"exqserve: startup failed: {exc}", file=sys.stderr)
        return 2

    uvicorn.run(composed.app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
