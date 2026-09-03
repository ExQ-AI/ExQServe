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


class ReasoningBudgetMode(str, Enum):
    INHERIT = "inherit"
    DISABLE = "disable"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class ReasoningBudgetOverride:
    mode: ReasoningBudgetMode = ReasoningBudgetMode.INHERIT
    max_tokens: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReasoningBudgetMode):
            raise TypeError("mode must be a ReasoningBudgetMode")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None")
        if self.mode is ReasoningBudgetMode.EXPLICIT:
            if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
                raise TypeError("explicit reasoning budget max_tokens must be an integer")
            if self.max_tokens < 0:
                raise ValueError("explicit reasoning budget max_tokens must be non-negative")
            return
        if self.max_tokens is not None:
            raise ValueError("non-explicit reasoning budget must not carry max_tokens")
        if self.message is not None:
            raise ValueError("non-explicit reasoning budget must not carry a message")


@dataclass(frozen=True, slots=True)
class ReasoningBudgetDefault:
    max_tokens: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
                raise TypeError("max_tokens must be an integer or None")
            if self.max_tokens < 0:
                raise ValueError("max_tokens must be non-negative or None")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
