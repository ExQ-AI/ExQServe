"""Shared translation helpers for runtime terminal metadata."""

from __future__ import annotations

from exqserve.core.events import CompletionReason, TimingUpdated
from exqserve.core.timing import GenerationTiming
from exqserve.runtime.contracts import RuntimeStopReason, RuntimeTiming


def completion_reason_from_runtime(reason: RuntimeStopReason) -> CompletionReason:
    if not isinstance(reason, RuntimeStopReason):
        raise TypeError("reason must be a RuntimeStopReason")
    if reason is RuntimeStopReason.LENGTH:
        return CompletionReason.LENGTH
    if reason is RuntimeStopReason.FILTER:
        return CompletionReason.FILTER
    if reason in {RuntimeStopReason.EOS, RuntimeStopReason.STOP_STRING}:
        return CompletionReason.STOP
    raise ValueError(f"runtime stop reason {reason.value!r} is not success-compatible")


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
