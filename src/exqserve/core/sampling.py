"""Immutable sampler-override value contracts shared by configuration and protocols."""

from __future__ import annotations

from dataclasses import dataclass

SamplingOverrideValue = int | float | bool | tuple[tuple[int, float], ...]

_SUPPORTED_SAMPLING_OVERRIDE_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty_range",
        "repetition_decay",
        "temperature_last",
        "adaptive_target",
        "adaptive_decay",
        "logit_bias",
    }
)


@dataclass(frozen=True, slots=True)
class SamplingOverride:
    field: str
    value: SamplingOverrideValue
    force: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.field, str):
            raise TypeError("field must be a string")
        if self.field not in _SUPPORTED_SAMPLING_OVERRIDE_FIELDS:
            raise ValueError(f"unsupported sampling override field: {self.field}")
        if not isinstance(self.force, bool):
            raise TypeError("force must be a boolean")


@dataclass(frozen=True, slots=True)
class SamplingOverridePolicy:
    overrides: tuple[SamplingOverride, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.overrides, tuple):
            raise TypeError("overrides must be a tuple")
        seen: set[str] = set()
        for override in self.overrides:
            if not isinstance(override, SamplingOverride):
                raise TypeError("overrides must contain SamplingOverride values")
            if override.field in seen:
                raise ValueError(f"duplicate sampling override field: {override.field}")
            seen.add(override.field)

    @property
    def enabled(self) -> bool:
        return bool(self.overrides)
