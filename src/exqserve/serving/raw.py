"""Protocol-neutral raw-prompt serving path for document continuation."""

from __future__ import annotations

import asyncio
import hashlib
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol, Self

from exqserve.control.request import RequestRejected, RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    UsageUpdated,
)
from exqserve.core.items import RawPromptItem
from exqserve.runtime.contracts import (
    RuntimeCancelled,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeRenderedPrompt,
    RuntimeStarted,
    RuntimeTextDelta,
)
from exqserve.serving.contracts import RawServingRequest, ServingRejected
from exqserve.serving.preprocessing import RendererLanePool
from exqserve.serving.runtime_events import timing_event_from_runtime
from exqserve.serving.terminal import (
    TerminalDecision,
    TerminalDisposition,
    TerminalEvidence,
    TerminalPrimaryOwner,
)


class RawPromptTokenizer(Protocol):
    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        ...


class ControlledRawSession(Protocol):
    terminal_reason: RequestTerminalReason | None

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        ...

    async def cancel(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED,
    ) -> None:
        ...


class RawRequestLease(Protocol):
    async def submit(self, request: RuntimeGenerationRequest) -> ControlledRawSession:
        ...

    async def release(self) -> None:
        ...


class RawRequestController(Protocol):
    async def acquire(self, request_id: str) -> RawRequestLease:
        ...


@dataclass(frozen=True, slots=True)
class RawCompiledPrompt:
    text: str
    input_ids: tuple[int, ...]
    prompt_hash: str
    stop_conditions: tuple[str | int, ...]


def _prompt_hash(input_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token_id in input_ids:
        encoded = str(token_id).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _safe_error(
    category: ErrorCategory,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> CanonicalError:
    return CanonicalError(category, code, message, retryable)


class RawServingEngine:
    def __init__(
        self,
        tokenizer: RawPromptTokenizer | None,
        controller: RawRequestController,
        *,
        output_limit_resolver: Callable[[int, int | None], int] | None = None,
        preprocessing_pool: RendererLanePool | None = None,
    ) -> None:
        if tokenizer is None and preprocessing_pool is None:
            raise ValueError("tokenizer or preprocessing_pool is required")
        self._tokenizer = tokenizer
        self._preprocessing_pool = preprocessing_pool
        self._controller = controller
        self._output_limit_resolver = output_limit_resolver

    async def submit(self, request: RawServingRequest) -> RawServingSession:
        if not isinstance(request, RawServingRequest):
            raise TypeError("request must be a RawServingRequest")
        try:
            lease = await self._controller.acquire(request.input.request_id)
        except RequestRejected as exc:
            raise ServingRejected(exc.error) from exc

        controlled: ControlledRawSession | None = None
        try:
            prompt = request.input.items[0]
            assert isinstance(prompt, RawPromptItem)

            if prompt.text is not None:
                prompt_text = prompt.text
                try:
                    pool = self._preprocessing_pool
                    if pool is not None:
                        rendered = await pool.run(
                            "raw_text",
                            lambda lane: lane.renderer.tokenize_text(prompt_text),
                        )
                    else:
                        tokenizer = self._tokenizer
                        if tokenizer is None:
                            raise RuntimeError("raw tokenizer is unavailable")
                        rendered = tokenizer.tokenize_text(prompt.text)
                except (TypeError, ValueError) as exc:
                    raise ServingRejected(
                        _safe_error(
                            ErrorCategory.INVALID_REQUEST,
                            "raw_prompt_tokenization_failed",
                            "Raw prompt cannot be tokenized for the selected model.",
                        )
                    ) from exc
                except Exception as exc:
                    raise ServingRejected(
                        _safe_error(
                            ErrorCategory.INTERNAL,
                            "serving_internal_error",
                            "Raw prompt tokenization failed internally.",
                        )
                    ) from exc
                text = prompt_text
                input_ids = rendered.input_ids
            else:
                assert prompt.token_ids is not None
                text = ""
                input_ids = prompt.token_ids

            compiled = RawCompiledPrompt(
                text,
                input_ids,
                _prompt_hash(input_ids),
                request.stop_conditions,
            )
            max_output_tokens = request.max_output_tokens
            if max_output_tokens is None:
                if self._output_limit_resolver is None:
                    raise ServingRejected(
                        _safe_error(
                            ErrorCategory.INTERNAL,
                            "serving_internal_error",
                            "Automatic output token resolution is unavailable.",
                        )
                    )
                try:
                    max_output_tokens = self._output_limit_resolver(len(input_ids), max_output_tokens)
                except RequestRejected as exc:
                    raise ServingRejected(exc.error) from exc
            runtime_request = RuntimeGenerationRequest(
                request.input.request_id,
                input_ids,
                max_output_tokens,
                request.seed,
                request.stop_conditions,
                request.sampling,
            )
            try:
                controlled = await lease.submit(runtime_request)
            except RequestRejected as exc:
                raise ServingRejected(exc.error) from exc
            except Exception as exc:
                raise ServingRejected(
                    _safe_error(
                        ErrorCategory.RUNTIME_FAILURE,
                        "runtime_submission_failed",
                        "Inference runtime submission failed.",
                    )
                ) from exc

            try:
                return RawServingSession(request.input.request_id, controlled, compiled)
            except asyncio.CancelledError:
                try:
                    await controlled.cancel(RequestTerminalReason.CLIENT_CANCELLED)
                finally:
                    raise
            except Exception:  # noqa: BLE001 - runtime ownership must roll back on any wrapper failure
                try:
                    await controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
                finally:
                    raise
        finally:
            if controlled is None:
                await lease.release()


class RawServingSession:
    def __init__(
        self,
        request_id: str,
        controlled: ControlledRawSession,
        compiled_prompt: RawCompiledPrompt,
    ) -> None:
        self._request_id = request_id
        self._controlled = controlled
        self._runtime_iterator = controlled.__aiter__()
        self._compiled_prompt = compiled_prompt
        self._pending: deque[GenerationEvent] = deque()
        self._text_parts: list[str] = []
        self._text_started = False
        self._terminal_evidence = TerminalEvidence()
        self._terminal = False
        self._cancelled = False

    @property
    def compiled_prompt(self) -> RawCompiledPrompt:
        return self._compiled_prompt

    @property
    def terminal_decision(self) -> TerminalDecision | None:
        return self._terminal_evidence.decision

    def __aiter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self._terminal:
            await self.cancel()
        return False

    async def cancel(self) -> None:
        if self._terminal or self._cancelled:
            return
        self._cancelled = True
        await self._controlled.cancel(RequestTerminalReason.CLIENT_CANCELLED)

    def _start_text(self) -> None:
        if self._text_started:
            return
        self._text_started = True
        self._pending.append(TextStarted(self._request_id))

    def _emit_terminal_decision(self, decision: TerminalDecision) -> None:
        if decision.disposition is TerminalDisposition.FAILURE:
            error = decision.canonical_error
            if error is None:  # pragma: no cover - TerminalDecision validates this invariant.
                raise RuntimeError("failure terminal decision is missing canonical error")
            self._pending.append(GenerationFailed(self._request_id, error))
        elif decision.disposition is TerminalDisposition.CANCELLATION:
            self._pending.append(GenerationCancelled(self._request_id))
        else:
            raise RuntimeError("failure/cancellation emitter received a successful decision")
        self._terminal_evidence.commit_decision(decision)
        self._terminal = True

    def _finish(self, event: RuntimeFinished) -> None:
        self._terminal_evidence.record_controlled_reason(self._controlled.terminal_reason)
        self._terminal_evidence.record_runtime_finished(event)
        decision = self._terminal_evidence.resolve()
        if decision.disposition is not TerminalDisposition.SUCCESS:
            self._emit_terminal_decision(decision)
            return
        reason = decision.completion_reason
        if reason is None:  # pragma: no cover - TerminalDecision validates success.
            raise RuntimeError("successful terminal decision is missing completion reason")
        text_started_event = None if self._text_started else TextStarted(self._request_id)
        text_completed_event = TextCompleted(self._request_id, "".join(self._text_parts))
        timing_event = timing_event_from_runtime(self._request_id, event.timing)
        usage_event = UsageUpdated(self._request_id, event.usage)
        completed_event = GenerationCompleted(self._request_id, reason, event.usage)

        pending_checkpoint = len(self._pending)
        text_started_checkpoint = self._text_started
        try:
            if text_started_event is not None:
                self._text_started = True
                self._pending.append(text_started_event)
            self._pending.append(text_completed_event)
            if timing_event is not None:
                self._pending.append(timing_event)
            self._pending.append(usage_event)
            self._pending.append(completed_event)
            self._terminal_evidence.commit_decision(decision)
        except Exception:
            while len(self._pending) > pending_checkpoint:
                self._pending.pop()
            self._text_started = text_started_checkpoint
            raise
        self._terminal = True

    def _cancel_event(self) -> None:
        self._terminal_evidence.record_controlled_reason(self._controlled.terminal_reason)
        if self._terminal_evidence.causal_owner is not TerminalPrimaryOwner.LIFECYCLE_TERMINATION:
            self._terminal_evidence.record_runtime_cancelled()
        self._emit_terminal_decision(self._terminal_evidence.resolve())

    def _process_runtime(self, event: RuntimeEvent) -> None:
        if isinstance(event, RuntimeStarted):
            self._pending.append(GenerationStarted(self._request_id))
        elif isinstance(event, RuntimeTextDelta):
            self._start_text()
            self._text_parts.append(event.text)
            self._pending.append(TextDelta(self._request_id, event.text))
        elif isinstance(event, RuntimeFinished):
            self._finish(event)
        elif isinstance(event, RuntimeFailed):
            self._terminal_evidence.record_controlled_reason(self._controlled.terminal_reason)
            self._terminal_evidence.record_runtime_failure(event.error)
            self._emit_terminal_decision(self._terminal_evidence.resolve())
        elif isinstance(event, RuntimeCancelled):
            self._cancel_event()

    async def __anext__(self) -> GenerationEvent:
        while True:
            if self._pending:
                return self._pending.popleft()
            if self._terminal:
                raise StopAsyncIteration
            try:
                runtime_event = await anext(self._runtime_iterator)
            except StopAsyncIteration:
                if not self._terminal:
                    self._terminal_evidence.record_controlled_reason(self._controlled.terminal_reason)
                    self._terminal_evidence.record_runtime_failure(
                        _safe_error(
                            ErrorCategory.RUNTIME_FAILURE,
                            "runtime_stream_ended",
                            "Inference runtime stream ended without a terminal event.",
                        )
                    )
                    self._emit_terminal_decision(self._terminal_evidence.resolve())
                continue
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - B1 converts unknown terminal failures into typed evidence.
                if not self._terminal:
                    self._terminal_evidence.record_controlled_reason(self._controlled.terminal_reason)
                    self._terminal_evidence.record_unknown_failure(
                        _safe_error(
                            ErrorCategory.INTERNAL,
                            "runtime_stream_exception",
                            "Inference runtime stream failed without a typed terminal event.",
                        )
                    )
                    self._emit_terminal_decision(self._terminal_evidence.resolve())
                continue
            try:
                self._process_runtime(runtime_event)
            except Exception:  # noqa: BLE001 - B1 converts unknown terminal failures into typed evidence.
                if not self._terminal:
                    self._terminal_evidence.record_controlled_reason(self._controlled.terminal_reason)
                    self._terminal_evidence.record_unknown_failure(
                        _safe_error(
                            ErrorCategory.INTERNAL,
                            "terminal_processing_failed",
                            "Inference terminal processing failed unexpectedly.",
                        )
                    )
                    self._emit_terminal_decision(self._terminal_evidence.resolve())
