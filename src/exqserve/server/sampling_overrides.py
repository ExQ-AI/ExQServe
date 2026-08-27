"""Safe loader for static OpenAI sampler-override presets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from exqserve.core.sampling import (
    SamplingOverride,
    SamplingOverridePolicy,
    SamplingOverrideValue,
)
from exqserve.runtime.contracts import RuntimeSamplingConfig

_FULL_CONTEXT_PENALTY_RANGE = 100_000_000
_PUBLIC_TO_CANONICAL = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "penalty_range": "repetition_penalty_range",
    "repetition_decay": "repetition_decay",
    "temperature_last": "temperature_last",
    "adaptive_target": "adaptive_target",
    "adaptive_decay": "adaptive_decay",
    "logit_bias": "logit_bias",
}


def _normalize_logit_bias(value: object) -> tuple[tuple[int, float], ...]:
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - normalize user preset schema failures
            "sampler override 'logit_bias.override' must be a mapping"
        )
    parsed: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_token_id, raw_bias in value.items():
        if isinstance(raw_token_id, bool):
            raise ValueError(  # noqa: TRY004 - normalize user preset schema failures
                "sampler override logit_bias token ids must be decimal integers"
            )
        if isinstance(raw_token_id, int):
            token_id = raw_token_id
        elif isinstance(raw_token_id, str):
            try:
                token_id = int(raw_token_id)
            except ValueError as exc:
                raise ValueError("sampler override logit_bias token ids must be decimal integers") from exc
        else:
            raise ValueError(  # noqa: TRY004 - normalize user preset schema failures
                "sampler override logit_bias token ids must be decimal integers"
            )
        if token_id < 0 or token_id in seen:
            raise ValueError("sampler override logit_bias token ids must be unique and non-negative")
        if not isinstance(raw_bias, int | float) or isinstance(raw_bias, bool):
            raise ValueError(  # noqa: TRY004 - normalize user preset schema failures
                "sampler override logit_bias values must be numbers"
            )
        bias = float(raw_bias)
        if not math.isfinite(bias) or not -100 <= bias <= 100:
            raise ValueError("sampler override logit_bias values must be finite and between -100 and 100")
        seen.add(token_id)
        parsed.append((token_id, bias))
    return tuple(parsed)


def _normalize_override_value(public_name: str, value: object) -> SamplingOverrideValue:
    canonical = _PUBLIC_TO_CANONICAL[public_name]
    normalized: object = value
    if public_name == "penalty_range":
        if not isinstance(value, int) or isinstance(value, bool) or value < -1:
            raise ValueError("sampler override penalty_range must be -1 or a non-negative integer")
        normalized = _FULL_CONTEXT_PENALTY_RANGE if value == -1 else value
    elif public_name == "logit_bias":
        normalized = _normalize_logit_bias(value)

    try:
        kwargs = cast(dict[str, Any], {canonical: normalized})
        candidate = RuntimeSamplingConfig(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid sampler override value for '{public_name}'") from exc
    validated = getattr(candidate, canonical)
    if not isinstance(validated, int | float | bool | tuple):
        raise TypeError(f"unexpected validated sampler override type for '{public_name}'")
    return validated


def load_sampling_override_policy(path: Path) -> SamplingOverridePolicy:
    if not isinstance(path, Path):
        raise TypeError("sampler preset path must be pathlib.Path")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in sampler preset: {path}") from exc
    if raw is None:
        raise ValueError("sampler preset must contain a top-level mapping")
    if not isinstance(raw, Mapping):
        raise ValueError(  # noqa: TRY004 - normalize user preset schema failures
            "sampler preset must contain a top-level mapping"
        )

    overrides: list[SamplingOverride] = []
    for raw_name, raw_block in raw.items():
        if not isinstance(raw_name, str) or raw_name not in _PUBLIC_TO_CANONICAL:
            raise ValueError(f"unknown sampler override field: {raw_name}")
        if not isinstance(raw_block, Mapping):
            raise ValueError(  # noqa: TRY004 - normalize user preset schema failures
                f"sampler override '{raw_name}' must be a mapping"
            )
        block_keys = set(raw_block)
        if "override" not in block_keys:
            raise ValueError(f"sampler override '{raw_name}' requires 'override'")
        unknown = block_keys - {"override", "force"}
        if unknown:
            raise ValueError(f"sampler override '{raw_name}' has unknown keys: {sorted(unknown)!r}")
        force = raw_block.get("force", False)
        if not isinstance(force, bool):
            raise ValueError(  # noqa: TRY004 - normalize user preset schema failures
                f"sampler override '{raw_name}.force' must be boolean"
            )
        overrides.append(
            SamplingOverride(
                _PUBLIC_TO_CANONICAL[raw_name],
                _normalize_override_value(raw_name, raw_block.get("override")),
                force,
            )
        )
    return SamplingOverridePolicy(tuple(overrides))
