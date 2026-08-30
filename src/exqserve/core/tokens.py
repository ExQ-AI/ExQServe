"""Token provenance values shared across runtime and model parsing layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NativeTokenSpan:
    """One tokenizer-emitted native piece with verified offsets in visible text."""

    start: int
    end: int
    token_id: int
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or isinstance(self.start, bool):
            raise TypeError("start must be an integer")
        if not isinstance(self.end, int) or isinstance(self.end, bool):
            raise TypeError("end must be an integer")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if not isinstance(self.token_id, int) or isinstance(self.token_id, bool):
            raise TypeError("token_id must be an integer")
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text:
            raise ValueError("text must not be empty")
