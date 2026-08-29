"""Protocol-neutral request admission and lifecycle control."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Self

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.runtime.contracts import (
    RuntimeCancelled,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeInjectionUnavailable,
    RuntimeSessionLike,
)


@dataclass(frozen=True, slots=True)
class RequestControlConfig:
    max_in_flight: int
    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_in_flight",
            "max_prompt_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer or None")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.timeout_seconds is not None:
            if not isinstance(self.timeout_seconds, int | float) or isinstance(
                self.timeout_seconds, bool
            ):
                raise TypeError("timeout_seconds must be a number or None")
            if not math.isfinite(float(self.timeout_seconds)):
                raise ValueError("timeout_seconds must be finite")
            if self.timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive")


class RequestTerminalReason(str, Enum):
    COMPLETED = "completed"
    RUNTIME_FAILED = "runtime_failed"
    RUNTIME_CANCELLED = "runtime_cancelled"
    CLIENT_CANCELLED = "client_cancelled"
    APPLICATION_CANCELLED = "application_cancelled"
    TIMEOUT = "timeout"
    SERVER_SHUTDOWN = "server_shutdown"


class RequestRejected(Exception):
    def __init__(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        self.error = error
        super().__init__(error.message)


class RequestInjectionNotFound(LookupError):
    """Raised when no active generation owns the requested request id."""


class RequestInjectionConflict(RuntimeError):
    """Raised when a known generation is already terminating."""


class RuntimeSubmitter(Protocol):
    def submit(self, request: RuntimeGenerationRequest) -> RuntimeSessionLike:
        ...


type _ReleaseCallback = Callable[["ControlledSession"], Awaitable[None]]


def _rejection(
    category: ErrorCategory,
    code: str,
    message: str,
    *,
    retryable: bool,
) -> RequestRejected:
    return RequestRejected(
        CanonicalError(
            category=category,
            code=code,
            message=message,
            retryable=retryable,
        )
    )


class ControlledSession:
    """Lifecycle wrapper around one already-admitted runtime session."""

    def __init__(
        self,
        runtime_session: RuntimeSessionLike,
        *,
        request_id: str,
        injection_allowed: bool,
        timeout_seconds: float | None,
        release: _ReleaseCallback,
    ) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(injection_allowed, bool):
            raise TypeError("injection_allowed must be a bool")
        self._runtime_session = runtime_session
        self._request_id = request_id
        self._injection_allowed = injection_allowed
        self._iterator = runtime_session.__aiter__()
        self._release = release
        self._released = False
        self._cancel_called = False
        self._iteration_terminal = False
        self._terminal_reason: RequestTerminalReason | None = None
        loop = asyncio.get_running_loop()
        self._deadline = None if timeout_seconds is None else loop.time() + float(timeout_seconds)

    @property
    def terminal_reason(self) -> RequestTerminalReason | None:
        return self._terminal_reason

    @property
    def request_id(self) -> str:
        return self._request_id

    def __aiter__(self) -> ControlledSession:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self._iteration_terminal and not self._cancel_called:
            await self.cancel(RequestTerminalReason.CLIENT_CANCELLED)
        return False

    async def _release_once(self) -> None:
        if self._released:
            return
        self._released = True
        await self._release(self)

    async def _cancel_runtime_once(self) -> None:
        if self._cancel_called:
            return
        self._cancel_called = True
        await self._runtime_session.cancel()

    def inject_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if text == "":
            raise ValueError("text must not be empty")
        if not self._injection_allowed:
            raise RequestInjectionConflict("Text injection is unavailable for structured-output requests.")
        if self._iteration_terminal or self._cancel_called or self._released:
            raise RequestInjectionConflict("The requested generation is already terminating.")
        try:
            self._runtime_session.inject_text(text)
        except RuntimeInjectionUnavailable as exc:
            raise RequestInjectionConflict("The requested generation is already terminating.") from exc

    async def cancel(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED,
    ) -> None:
        allowed = {
            RequestTerminalReason.CLIENT_CANCELLED,
            RequestTerminalReason.APPLICATION_CANCELLED,
            RequestTerminalReason.TIMEOUT,
            RequestTerminalReason.SERVER_SHUTDOWN,
        }
        if not isinstance(reason, RequestTerminalReason):
            raise TypeError("reason must be a RequestTerminalReason")
        if reason not in allowed:
            raise ValueError("reason is not valid for explicit request cancellation")
        if self._iteration_terminal or self._cancel_called:
            return
        if self._terminal_reason is None:
            self._terminal_reason = reason
        await self._cancel_runtime_once()
        await self._release_once()

    async def _next_runtime_event(self) -> RuntimeEvent:
        return await anext(self._iterator)

    async def _next_with_deadline(self) -> RuntimeEvent:
        if self._deadline is None or self._cancel_called:
            return await self._next_runtime_event()

        remaining = self._deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            if self._terminal_reason is None:
                self._terminal_reason = RequestTerminalReason.TIMEOUT
            await self._cancel_runtime_once()
            await self._release_once()
            return await self._next_runtime_event()

        next_task = asyncio.create_task(self._next_runtime_event())
        deadline_task = asyncio.create_task(asyncio.sleep(remaining))
        done, _ = await asyncio.wait(
            {next_task, deadline_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if next_task in done:
            deadline_task.cancel()
            try:
                await deadline_task
            except asyncio.CancelledError:
                pass
            return await next_task

        if self._terminal_reason is None:
            self._terminal_reason = RequestTerminalReason.TIMEOUT
        await self._cancel_runtime_once()
        await self._release_once()
        # Runtime cancellation must wake the already-pending iterator operation.
        # Do not cancel that task or mutate backend iterator state.
        return await next_task

    async def __anext__(self) -> RuntimeEvent:
        if self._iteration_terminal:
            raise StopAsyncIteration

        try:
            event = await self._next_with_deadline()
        except StopAsyncIteration:
            self._iteration_terminal = True
            await self._release_once()
            raise
        except asyncio.CancelledError:
            if self._terminal_reason is None:
                self._terminal_reason = RequestTerminalReason.CLIENT_CANCELLED
            self._iteration_terminal = True
            try:
                await self._cancel_runtime_once()
            finally:
                await self._release_once()
            raise
        except BaseException:
            self._iteration_terminal = True
            await self._release_once()
            raise

        if isinstance(event, RuntimeFinished):
            if self._terminal_reason is None:
                self._terminal_reason = RequestTerminalReason.COMPLETED
            self._iteration_terminal = True
            await self._release_once()
        elif isinstance(event, RuntimeFailed):
            if self._terminal_reason is None:
                self._terminal_reason = RequestTerminalReason.RUNTIME_FAILED
            self._iteration_terminal = True
            await self._release_once()
        elif isinstance(event, RuntimeCancelled):
            if self._terminal_reason is None:
                self._terminal_reason = RequestTerminalReason.RUNTIME_CANCELLED
            self._iteration_terminal = True
            await self._release_once()

        return event


class RequestController:
    """Immediate admission/rejection without a second request queue."""

    def __init__(self, runtime: RuntimeSubmitter, config: RequestControlConfig) -> None:
        if not isinstance(config, RequestControlConfig):
            raise TypeError("config must be a RequestControlConfig")
        self._runtime = runtime
        self._config = config
        self._lock = asyncio.Lock()
        self._in_flight = 0
        self._closed = False
        self._sessions: set[ControlledSession] = set()
        self._sessions_by_request_id: dict[str, ControlledSession] = {}

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def _validate_limits(self, request: RuntimeGenerationRequest) -> None:
        prompt_count = len(request.input_ids)
        output_count = request.max_new_tokens

        if (
            self._config.max_prompt_tokens is not None
            and prompt_count > self._config.max_prompt_tokens
        ):
            raise _rejection(
                ErrorCategory.CONTEXT_LENGTH,
                "prompt_limit_exceeded",
                "Prompt token limit exceeded.",
                retryable=False,
            )
        if (
            self._config.max_output_tokens is not None
            and output_count > self._config.max_output_tokens
        ):
            raise _rejection(
                ErrorCategory.INVALID_REQUEST,
                "output_limit_exceeded",
                "Requested output token limit exceeded.",
                retryable=False,
            )
        if (
            self._config.max_total_tokens is not None
            and prompt_count + output_count > self._config.max_total_tokens
        ):
            raise _rejection(
                ErrorCategory.CONTEXT_LENGTH,
                "total_context_limit_exceeded",
                "Total requested context limit exceeded.",
                retryable=False,
            )

    async def _release(self, session: ControlledSession) -> None:
        async with self._lock:
            if session not in self._sessions:
                return
            self._sessions.remove(session)
            if self._sessions_by_request_id.get(session.request_id) is session:
                del self._sessions_by_request_id[session.request_id]
            self._in_flight -= 1
            if self._in_flight < 0:  # pragma: no cover - defensive invariant
                raise RuntimeError("request-control in-flight count became negative")

    async def submit(self, request: RuntimeGenerationRequest) -> ControlledSession:
        if not isinstance(request, RuntimeGenerationRequest):
            raise TypeError("request must be a RuntimeGenerationRequest")
        self._validate_limits(request)

        async with self._lock:
            if self._closed:
                raise _rejection(
                    ErrorCategory.OVERLOADED,
                    "server_shutting_down",
                    "Server is shutting down.",
                    retryable=True,
                )
            if request.request_id in self._sessions_by_request_id:
                raise _rejection(
                    ErrorCategory.INVALID_REQUEST,
                    "duplicate_request_id",
                    "An active request already uses this request id.",
                    retryable=False,
                )
            if self._in_flight >= self._config.max_in_flight:
                raise _rejection(
                    ErrorCategory.OVERLOADED,
                    "server_overloaded",
                    "Server is at capacity.",
                    retryable=True,
                )

            self._in_flight += 1
            try:
                runtime_session = self._runtime.submit(request)
                controlled = ControlledSession(
                    runtime_session,
                    request_id=request.request_id,
                    injection_allowed=request.output_json_schema is None,
                    timeout_seconds=self._config.timeout_seconds,
                    release=self._release,
                )
            except BaseException:
                self._in_flight -= 1
                raise
            self._sessions.add(controlled)
            self._sessions_by_request_id[request.request_id] = controlled
            return controlled

    async def inject_text(self, request_id: str, text: str) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if text == "":
            raise ValueError("text must not be empty")

        async with self._lock:
            session = self._sessions_by_request_id.get(request_id)
        if session is None:
            raise RequestInjectionNotFound(request_id)
        session.inject_text(text)

    async def close(self) -> None:
        async with self._lock:
            if self._closed and not self._sessions:
                return
            self._closed = True
            sessions = tuple(self._sessions)

        if sessions:
            await asyncio.gather(
                *(session.cancel(RequestTerminalReason.SERVER_SHUTDOWN) for session in sessions)
            )
