"""Compatibility facade for OpenAI Responses codecs."""

from __future__ import annotations

from .responses_output import ResponsesAccumulator, ResponsesStreamSerializer, build_response_object
from .responses_request import ParsedResponsesRequest, ResponsesRequestAdapter

__all__ = [
    "ParsedResponsesRequest",
    "ResponsesAccumulator",
    "ResponsesRequestAdapter",
    "ResponsesStreamSerializer",
    "build_response_object",
]
