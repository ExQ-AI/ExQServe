"""Protocol-neutral reasoning policy values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReasoningMode(str, Enum):
    DEFAULT = "default"
    ENABLED = "enabled"
    DISABLED = "disabled"


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAXIMUM = "maximum"


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    mode: ReasoningMode = ReasoningMode.DEFAULT
    effort: ReasoningEffort | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReasoningMode):
            raise TypeError("mode must be a ReasoningMode")
        if self.effort is not None and not isinstance(self.effort, ReasoningEffort):
            raise TypeError("effort must be a ReasoningEffort or None")
