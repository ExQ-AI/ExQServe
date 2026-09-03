"""Shared immutable vocabulary for generation-time semantic guarantees."""

from __future__ import annotations

from enum import Enum


class GenerationGuarantee(str, Enum):
    NONE = "none"
    FORMAT = "format"
    SCHEMA = "schema"
    UNKNOWN = "unknown"


class ConstraintFallbackPolicy(str, Enum):
    FAIL_CLOSED = "fail_closed"
    ALLOW_VALIDATION_ONLY = "allow_validation_only"
