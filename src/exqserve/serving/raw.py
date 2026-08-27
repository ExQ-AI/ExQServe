"""Protocol-neutral raw-prompt serving path for document continuation."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import AsyncIterator
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
from exqserve.serving.runtime_events import (
    completion_reason_from_runtime,
    timing_event_from_runtime,
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


class RawRequestController(Protocol):
    async def submit(self, request: RuntimeGenerationRequest) -> ControlledRawSession:
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
    def __init__(self, tokenizer: RawPromptTokenizer, controller: RawRequestController) -> None:
        self._tokenizer = tokenizer
        self._controller = controller

    async def submit(self, request: RawServingRequest) -> RawServingSession:
        if not isinstance(request, RawServingRequest):
            raise TypeError("request must be a RawServingRequest")
        prompt = request.input.items[0]
        assert isinstance(prompt, RawPromptItem)

        if prompt.text is not None:
            try:
                rendered = self._tokenizer.tokenize_text(prompt.text)
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
            text = prompt.text
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
        runtime_request = RuntimeGenerationRequest(
            request.input.request_id,
            input_ids,
            request.max_output_tokens,
            request.seed,
            request.stop_conditions,
            request.sampling,
        )
        try:
            controlled = await self._controller.submit(runtime_request)
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
        return RawServingSession(request.input.request_id, controlled, compiled)


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
        self._terminal = False
        self._cancelled = False

    @property
    def compiled_prompt(self) -> RawCompiledPrompt:
        return self._compiled_prompt

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

    def _finish(self, event: RuntimeFinished) -> None:
        self._start_text()
        self._pending.append(TextCompleted(self._request_id, "".join(self._text_parts)))
        timing_event = timing_event_from_runtime(self._request_id, event.timing)
        if timing_event is not None:
            self._pending.append(timing_event)
        self._pending.append(UsageUpdated(self._request_id, event.usage))
        reason = completion_reason_from_runtime(event.reason)
        self._pending.append(GenerationCompleted(self._request_id, reason, event.usage))
        self._terminal = True

    def _cancel_event(self) -> None:
        if self._controlled.terminal_reason is RequestTerminalReason.TIMEOUT:
            self._pending.append(
                GenerationFailed(
                    self._request_id,
                    _safe_error(
                        ErrorCategory.RUNTIME_FAILURE,
                        "request_timeout",
                        "Inference request exceeded its serving deadline.",
                        retryable=True,
                    ),
                )
            )
        else:
            self._pending.append(GenerationCancelled(self._request_id))
        self._terminal = True

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
            self._pending.append(GenerationFailed(self._request_id, event.error))
            self._terminal = True
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
                    self._pending.append(
                        GenerationFailed(
                            self._request_id,
                            _safe_error(
                                ErrorCategory.RUNTIME_FAILURE,
                                "runtime_stream_ended",
                                "Inference runtime stream ended without a terminal event.",
                            ),
                        )
                    )
                    self._terminal = True
                continue
            self._process_runtime(runtime_event)
