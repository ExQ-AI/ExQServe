"""Shared translation helpers for runtime terminal metadata."""

from __future__ import annotations

from exqserve.core.events import CompletionReason, TimingUpdated
from exqserve.core.timing import GenerationTiming
from exqserve.runtime.contracts import RuntimeStopReason, RuntimeTiming


def completion_reason_from_runtime(reason: RuntimeStopReason) -> CompletionReason:
    return CompletionReason.LENGTH if reason is RuntimeStopReason.LENGTH else CompletionReason.STOP


def timing_event_from_runtime(request_id: str, timing: RuntimeTiming) -> TimingUpdated | None:
    if not any(
        value is not None
        for value in (timing.queue_seconds, timing.prefill_seconds, timing.generation_seconds)
    ):
        return None
    return TimingUpdated(
        request_id,
        GenerationTiming(
            queue_seconds=timing.queue_seconds,
            prefill_seconds=timing.prefill_seconds,
            generation_seconds=timing.generation_seconds,
        ),
    )
