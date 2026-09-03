"""Protocol-neutral request admission and lifecycle control."""

from __future__ import annotations

import asyncio
import logging
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
    RuntimeUnavailable,
)

logger = logging.getLogger(__name__)


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

    def resolve_output_limit(self, prompt_tokens: int, requested: int | None) -> int:
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
            raise TypeError("prompt_tokens must be an integer")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if requested is not None:
            if not isinstance(requested, int) or isinstance(requested, bool):
                raise TypeError("requested must be an integer or None")
            if requested <= 0:
                raise ValueError("requested must be positive or None")
            return requested

        candidates: list[int] = []
        if self.max_total_tokens is not None:
            candidates.append(self.max_total_tokens - prompt_tokens)
        if self.max_output_tokens is not None:
            candidates.append(self.max_output_tokens)
        if not candidates:
            raise ValueError(
                "automatic output token resolution requires max_total_tokens or max_output_tokens"
            )
        resolved = min(candidates)
        if resolved <= 0:
            raise _rejection(
                ErrorCategory.CONTEXT_LENGTH,
                "total_context_limit_exceeded",
                "Prompt leaves no room for model output within the served context.",
                retryable=False,
            )
        return resolved


class RequestTerminalReason(str, Enum):
    COMPLETED = "completed"
    RUNTIME_FAILED = "runtime_failed"
    RUNTIME_CANCELLED = "runtime_cancelled"
    CLIENT_CANCELLED = "client_cancelled"
    APPLICATION_CANCELLED = "application_cancelled"
    TIMEOUT = "timeout"
    SERVER_SHUTDOWN = "server_shutdown"
    MODEL_SWITCH = "model_switch"


class RequestRejected(Exception):
    def __init__(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        self.error = error
        super().__init__(error.message)


class RequestInjectionNotFound(LookupError):
    """Raised when no active generation owns the requested request id."""


class RequestInjectionConflict(RuntimeError):
    """Raised when text injection cannot be applied to a known generation."""


class RequestInjectionTerminating(RequestInjectionConflict):
    """Raised when text injection loses a race with generation termination."""


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


class _RequestLeaseState(str, Enum):
    RESERVED = "reserved"
    SUBMITTED = "submitted"
    RELEASED = "released"


class RequestLease:
    """One max-in-flight reservation spanning preprocessing through runtime release."""

    def __init__(self, controller: RequestController, request_id: str) -> None:
        self._controller = controller
        self._request_id = request_id
        self._state = _RequestLeaseState.RESERVED
        self._session: ControlledSession | None = None

    @property
    def request_id(self) -> str:
        return self._request_id

    async def submit(self, request: RuntimeGenerationRequest) -> ControlledSession:
        if not isinstance(request, RuntimeGenerationRequest):
            raise TypeError("request must be a RuntimeGenerationRequest")
        if request.request_id != self._request_id:
            raise ValueError("runtime request id must match the reserved request id")
        return await self._controller._submit_reserved(self, request)

    async def release(self) -> None:
        await self._controller._release_reserved(self)


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
            raise RequestInjectionConflict(
                "Text injection is unavailable while a generation constraint is active."
            )
        if self._iteration_terminal or self._cancel_called or self._released:
            raise RequestInjectionTerminating("The requested generation is already terminating.")
        try:
            self._runtime_session.inject_text(text)
        except RuntimeInjectionUnavailable as exc:
            raise RequestInjectionTerminating("The requested generation is already terminating.") from exc

    async def cancel(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED,
    ) -> None:
        allowed = {
            RequestTerminalReason.CLIENT_CANCELLED,
            RequestTerminalReason.APPLICATION_CANCELLED,
            RequestTerminalReason.TIMEOUT,
            RequestTerminalReason.SERVER_SHUTDOWN,
            RequestTerminalReason.MODEL_SWITCH,
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
        try:
            done, _ = await asyncio.wait(
                {next_task, deadline_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            deadline_task.cancel()
            await asyncio.gather(deadline_task, return_exceptions=True)
            raise

        if next_task in done:
            deadline_task.cancel()
            await asyncio.gather(deadline_task, return_exceptions=True)
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
        self._leases_by_request_id: dict[str, RequestLease] = {}
        self._sessions: set[ControlledSession] = set()
        self._sessions_by_request_id: dict[str, ControlledSession] = {}
        self._drained = asyncio.Event()
        self._drained.set()

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

    def _release_lease_locked(self, lease: RequestLease) -> None:
        if self._leases_by_request_id.get(lease.request_id) is not lease:
            return
        if lease._state is _RequestLeaseState.RELEASED:
            return

        session = lease._session
        if session is not None:
            self._sessions.discard(session)
            if self._sessions_by_request_id.get(lease.request_id) is session:
                del self._sessions_by_request_id[lease.request_id]

        del self._leases_by_request_id[lease.request_id]
        lease._session = None
        lease._state = _RequestLeaseState.RELEASED
        self._in_flight -= 1
        if self._in_flight < 0:  # pragma: no cover - defensive invariant
            raise RuntimeError("request-control in-flight count became negative")
        if self._in_flight == 0:
            self._drained.set()

    async def acquire(self, request_id: str) -> RequestLease:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")

        async with self._lock:
            if self._closed:
                raise _rejection(
                    ErrorCategory.OVERLOADED,
                    "server_shutting_down",
                    "Server is shutting down.",
                    retryable=True,
                )
            if request_id in self._leases_by_request_id:
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

            lease = RequestLease(self, request_id)
            self._leases_by_request_id[request_id] = lease
            self._in_flight += 1
            self._drained.clear()
            return lease

    async def _release_reserved(self, lease: RequestLease) -> None:
        async with self._lock:
            if self._leases_by_request_id.get(lease.request_id) is not lease:
                return
            if lease._state is _RequestLeaseState.SUBMITTED:
                return
            self._release_lease_locked(lease)

    async def _release(self, session: ControlledSession) -> None:
        async with self._lock:
            lease = self._leases_by_request_id.get(session.request_id)
            if lease is None or lease._session is not session:
                return
            self._release_lease_locked(lease)

    async def _submit_reserved(
        self, lease: RequestLease, request: RuntimeGenerationRequest
    ) -> ControlledSession:
        try:
            self._validate_limits(request)
        except BaseException:
            await self._release_reserved(lease)
            raise

        runtime_session: RuntimeSessionLike | None = None
        setup_error: BaseException | None = None
        async with self._lock:
            if self._leases_by_request_id.get(lease.request_id) is not lease:
                raise RuntimeError("request lease is no longer active")
            if lease._state is not _RequestLeaseState.RESERVED:
                raise RuntimeError("request lease has already been submitted")
            if self._closed:
                self._release_lease_locked(lease)
                raise _rejection(
                    ErrorCategory.OVERLOADED,
                    "server_shutting_down",
                    "Server is shutting down.",
                    retryable=True,
                )

            try:
                runtime_session = self._runtime.submit(request)
                controlled = ControlledSession(
                    runtime_session,
                    request_id=request.request_id,
                    injection_allowed=(
                        request.output_json_schema is None and request.generation_constraint is None
                    ),
                    timeout_seconds=self._config.timeout_seconds,
                    release=self._release,
                )
            except RuntimeUnavailable as exc:
                self._release_lease_locked(lease)
                raise RequestRejected(exc.error) from exc
            except BaseException as exc:  # noqa: BLE001 - runtime ownership rollback boundary
                self._release_lease_locked(lease)
                setup_error = exc
            else:
                lease._state = _RequestLeaseState.SUBMITTED
                lease._session = controlled
                self._sessions.add(controlled)
                self._sessions_by_request_id[request.request_id] = controlled
                return controlled

        if setup_error is not None:
            if runtime_session is not None:
                try:
                    await runtime_session.cancel()
                except Exception as cancel_exc:  # noqa: BLE001 - preserve original setup failure
                    logger.warning(
                        "runtime session rollback cancellation failed request_id=%s: %s",
                        lease.request_id,
                        cancel_exc,
                    )
            raise setup_error
        raise RuntimeError("runtime session setup ended without a result")  # pragma: no cover

    async def submit(self, request: RuntimeGenerationRequest) -> ControlledSession:
        """Compatibility entry point for callers that have no preprocessing phase."""
        if not isinstance(request, RuntimeGenerationRequest):
            raise TypeError("request must be a RuntimeGenerationRequest")
        lease = await self.acquire(request.request_id)
        try:
            return await lease.submit(request)
        except BaseException:
            await lease.release()
            raise

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

    async def close(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.SERVER_SHUTDOWN,
    ) -> None:
        if not isinstance(reason, RequestTerminalReason):
            raise TypeError("reason must be a RequestTerminalReason")
        if reason not in {RequestTerminalReason.SERVER_SHUTDOWN, RequestTerminalReason.MODEL_SWITCH}:
            raise ValueError("controller close reason must be server shutdown or model switch")
        async with self._lock:
            if self._closed and self._in_flight == 0:
                return
            self._closed = True
            sessions = tuple(self._sessions)

        if sessions:
            await asyncio.gather(*(session.cancel(reason) for session in sessions))
        await self._drained.wait()
