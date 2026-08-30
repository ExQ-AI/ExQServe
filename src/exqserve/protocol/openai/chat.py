"""Compatibility facade for OpenAI Chat Completions codecs."""

from __future__ import annotations

from .chat_output import ChatAccumulator, ChatStreamSerializer
from .chat_request import ChatRequestAdapter

__all__ = [
    "ChatAccumulator",
    "ChatRequestAdapter",
    "ChatStreamSerializer",
]
