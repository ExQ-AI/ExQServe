"""Protocol-neutral generation request envelope."""

from __future__ import annotations

from dataclasses import dataclass

from exqserve.core.items import CanonicalItem, RawPromptItem


@dataclass(frozen=True, slots=True)
class CanonicalRequest:
    request_id: str
    model: str
    items: tuple[CanonicalItem, ...]

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not isinstance(self.items, tuple):
            raise TypeError("items must be a tuple")
        if not all(isinstance(item, CanonicalItem) for item in self.items):
            raise TypeError("items must contain only CanonicalItem values")


@dataclass(frozen=True, slots=True)
class RawPromptRequest:
    request_id: str
    model: str
    items: tuple[RawPromptItem, ...]

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not isinstance(self.items, tuple):
            raise TypeError("items must be a tuple")
        if len(self.items) != 1 or not isinstance(self.items[0], RawPromptItem):
            raise ValueError("raw prompt request must contain exactly one RawPromptItem")
