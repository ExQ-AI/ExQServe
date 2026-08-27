"""Protocol-neutral measured generation timing values."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationTiming:
    queue_seconds: float | None = None
    prefill_seconds: float | None = None
    generation_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("queue_seconds", "prefill_seconds", "generation_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number or None")
            measured = float(value)
            if not math.isfinite(measured) or measured < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, measured)
