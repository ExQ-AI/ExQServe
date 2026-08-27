"""Protocol-neutral semantic generation events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from exqserve.core.errors import CanonicalError
from exqserve.core.items import ToolCallItem
from exqserve.core.timing import GenerationTiming
from exqserve.core.usage import TokenUsage


def _validate_request_id(request_id: str) -> None:
    if not request_id.strip():
        raise ValueError("request_id must not be empty")


def _validate_index(index: int) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("index must be an integer")
    if index < 0:
        raise ValueError("index must be non-negative")


def _validate_tool_identity(call_id: str, name: str | None = None) -> None:
    if not call_id.strip():
        raise ValueError("call_id must not be empty")
    if name is not None and not name.strip():
        raise ValueError("name must not be empty")


class CompletionReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"


@dataclass(frozen=True, slots=True)
class GenerationStarted:
    request_id: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)


@dataclass(frozen=True, slots=True)
class GenerationCompleted:
    request_id: str
    reason: CompletionReason
    usage: TokenUsage | None = None
    stop_sequence: str | None = None

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.reason, CompletionReason):
            raise TypeError("reason must be a CompletionReason")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage or None")
        if self.stop_sequence is not None:
            if not isinstance(self.stop_sequence, str):
                raise TypeError("stop_sequence must be a string or None")
            if not self.stop_sequence:
                raise ValueError("stop_sequence must not be empty")
            if self.reason is not CompletionReason.STOP:
                raise ValueError("stop_sequence is valid only for stop completions")


@dataclass(frozen=True, slots=True)
class GenerationCancelled:
    request_id: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)


@dataclass(frozen=True, slots=True)
class GenerationFailed:
    request_id: str
    error: CanonicalError

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.error, CanonicalError):
            raise TypeError("error must be a CanonicalError")


@dataclass(frozen=True, slots=True)
class TextStarted:
    request_id: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)


@dataclass(frozen=True, slots=True)
class TextDelta:
    request_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if self.text == "":
            raise ValueError("text delta must not be empty")


@dataclass(frozen=True, slots=True)
class TextCompleted:
    request_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class ReasoningStarted:
    request_id: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    request_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if self.text == "":
            raise ValueError("reasoning delta must not be empty")


@dataclass(frozen=True, slots=True)
class ReasoningCompleted:
    request_id: str
    text: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    request_id: str
    call_id: str
    name: str
    index: int

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        _validate_tool_identity(self.call_id, self.name)
        _validate_index(self.index)


@dataclass(frozen=True, slots=True)
class ToolCallArgumentsDelta:
    request_id: str
    call_id: str
    delta: str
    index: int

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        _validate_tool_identity(self.call_id)
        _validate_index(self.index)
        if self.delta == "":
            raise ValueError("tool-call arguments delta must not be empty")


@dataclass(frozen=True, slots=True)
class ToolCallCompleted:
    request_id: str
    call: ToolCallItem

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.call, ToolCallItem):
            raise TypeError("call must be a ToolCallItem")


@dataclass(frozen=True, slots=True)
class TimingUpdated:
    request_id: str
    timing: GenerationTiming

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.timing, GenerationTiming):
            raise TypeError("timing must be GenerationTiming")


@dataclass(frozen=True, slots=True)
class UsageUpdated:
    request_id: str
    usage: TokenUsage

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")


GenerationEvent = (
    GenerationStarted
    | GenerationCompleted
    | GenerationCancelled
    | GenerationFailed
    | TextStarted
    | TextDelta
    | TextCompleted
    | ReasoningStarted
    | ReasoningDelta
    | ReasoningCompleted
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ToolCallCompleted
    | TimingUpdated
    | UsageUpdated
)
