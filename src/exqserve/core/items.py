"""Canonical protocol-neutral items for text-first Agent interactions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class MessageItem:
    role: MessageRole
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class TextContentPart:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class ImageContentPart:
    source: str
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.detail is not None:
            if not isinstance(self.detail, str):
                raise TypeError("detail must be a string or None")
            if self.detail not in {"auto", "low", "high"}:
                raise ValueError("detail must be auto, low, high, or None")


type MessageContentPart = TextContentPart | ImageContentPart


@dataclass(frozen=True, slots=True)
class MultimodalMessageItem:
    role: MessageRole
    parts: tuple[MessageContentPart, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole")
        if self.role is not MessageRole.USER:
            raise ValueError("multimodal messages are supported only for the user role")
        if not isinstance(self.parts, tuple):
            raise TypeError("parts must be a tuple")
        if not self.parts:
            raise ValueError("parts must not be empty")
        if not all(isinstance(part, TextContentPart | ImageContentPart) for part in self.parts):
            raise TypeError("parts must contain only text or image content parts")
        if not any(isinstance(part, ImageContentPart) for part in self.parts):
            raise ValueError("multimodal messages must contain at least one image part")


@dataclass(frozen=True, slots=True)
class ReasoningItem:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class ToolCallItem:
    call_id: str
    name: str
    arguments_json: str
    index: int

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not isinstance(self.arguments_json, str):
            raise TypeError("arguments_json must be a string")
        if not isinstance(self.index, int) or isinstance(self.index, bool):
            raise TypeError("index must be an integer")
        if self.index < 0:
            raise ValueError("index must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolResultItem:
    call_id: str
    text: str
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.is_error, bool):
            raise TypeError("is_error must be a bool")


@dataclass(frozen=True, slots=True)
class MultimodalToolResultItem:
    call_id: str
    parts: tuple[MessageContentPart, ...]
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not isinstance(self.parts, tuple):
            raise TypeError("parts must be a tuple")
        if not self.parts:
            raise ValueError("parts must not be empty")
        if not all(isinstance(part, TextContentPart | ImageContentPart) for part in self.parts):
            raise TypeError("parts must contain only text or image content parts")
        if not any(isinstance(part, ImageContentPart) for part in self.parts):
            raise ValueError("multimodal tool results must contain at least one image part")
        if not isinstance(self.is_error, bool):
            raise TypeError("is_error must be a bool")


@dataclass(frozen=True, slots=True)
class RawPromptItem:
    """Protocol-neutral raw continuation prompt, without chat-role semantics."""

    text: str | None = None
    token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.token_ids is None):
            raise ValueError("exactly one of text or token_ids must be set")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or None")
        if self.token_ids is not None:
            if not isinstance(self.token_ids, tuple):
                raise TypeError("token_ids must be a tuple or None")
            if not self.token_ids:
                raise ValueError("token_ids must not be empty")
            if not all(
                isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
                for token_id in self.token_ids
            ):
                raise TypeError("token_ids must contain only non-negative integers")


CanonicalItem = (
    MessageItem
    | MultimodalMessageItem
    | ReasoningItem
    | ToolCallItem
    | ToolResultItem
    | MultimodalToolResultItem
)
