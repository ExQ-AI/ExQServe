"""Protocol-neutral metadata for the model currently served by ExQServe."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServedModelInfo:
    id: str
    created: int
    context_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("model id must be a string")
        if not self.id.strip():
            raise ValueError("model id must not be empty")
        if not isinstance(self.created, int) or isinstance(self.created, bool):
            raise TypeError("created must be an integer")
        if self.created < 0:
            raise ValueError("created must be non-negative")
        if not isinstance(self.context_length, int) or isinstance(self.context_length, bool):
            raise TypeError("context_length must be an integer")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
