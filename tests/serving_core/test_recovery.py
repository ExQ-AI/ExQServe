from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from exqserve.core.errors import CanonicalError, ErrorCategory, FailureCause
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import MessageItem, MessageRole, ToolCallItem
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import CompiledPrompt, TemplateRequest
from exqserve.runtime.contracts import (
    RuntimeCapabilities,
    RuntimeReadinessResult,
    RuntimeSamplingConfig,
)
from exqserve.serving.contracts import ServingVisibilityMode
from exqserve.serving.recovery import (
    AttemptRecoveryDecision,
    AttemptRecoveryEvidence,
    AttemptRecoveryKind,
    PublicationState,
    RecoveringServingSession,
    RecoverySkipReason,
    classify_attempt_recovery,
)
from exqserve.state.session import StatefulServingSession
from exqserve.state.store import InMemoryResponseStore


def _error(code: str, cause: FailureCause | None) -> CanonicalError:
    return CanonicalError(
        ErrorCategory.MODEL_FAILURE,
        code,
        "failed",
        retryable=False,
        cause=cause,
    )


def _evidence(
    code: str = "tool_call_incomplete",
    cause: FailureCause | None = FailureCause.OUTPUT_EOS,
    **overrides: object,
) -> AttemptRecoveryEvidence:
    values: dict[str, object] = {
        "error": _error(code, cause),
        "visibility_mode": ServingVisibilityMode.BUFFERED,
        "publication_state": PublicationState.UNPUBLISHED,
        "attempt_ordinal": 1,
        "max_extra_attempts": 1,
        "fresh_attempt_replay": True,
        "recovery_readiness": True,
    }
    values.update(overrides)
    return AttemptRecoveryEvidence(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("code", "cause"),
    [
        ("tool_call_incomplete", FailureCause.OUTPUT_EOS),
        ("protocol_ambiguity", FailureCause.OUTPUT_EOS),
        ("protocol_ambiguity", FailureCause.PARSER_AMBIGUITY_LIMIT),
        ("tool_call_invalid", FailureCause.MODEL_TOOL_OUTPUT_INVALID),
    ],
)
def test_allowlisted_unpublished_model_failures_regenerate_once(
    code: str,
    cause: FailureCause,
) -> None:
    assert classify_attempt_recovery(_evidence(code, cause)) == AttemptRecoveryDecision(
        AttemptRecoveryKind.REGENERATE_MODEL_OUTPUT
    )


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"max_extra_attempts": 0}, RecoverySkipReason.DISABLED),
        ({"attempt_ordinal": 2}, RecoverySkipReason.ATTEMPT_BUDGET_EXHAUSTED),
        ({"visibility_mode": ServingVisibilityMode.UNKNOWN}, RecoverySkipReason.VISIBILITY_UNKNOWN),
        ({"publication_state": PublicationState.PUBLISHED}, RecoverySkipReason.SEMANTIC_ALREADY_PUBLISHED),
        ({"seed": 7}, RecoverySkipReason.DETERMINISTIC_MODEL_OUTPUT),
        ({"sampling": RuntimeSamplingConfig(temperature=0)}, RecoverySkipReason.DETERMINISTIC_MODEL_OUTPUT),
        ({"fresh_attempt_replay": False}, RecoverySkipReason.RUNTIME_REPLAY_UNSUPPORTED),
        ({"request_replay_safe": False}, RecoverySkipReason.INJECTION_REPLAY_UNSUPPORTED),
        (
            {"has_prompt_attachments": True, "prompt_attachment_replay": False},
            RecoverySkipReason.ATTACHMENT_REPLAY_UNSUPPORTED,
        ),
    ],
)
def test_model_regeneration_fail_closed_guards(
    override: dict[str, object],
    reason: RecoverySkipReason,
) -> None:
    decision = classify_attempt_recovery(_evidence(**override))
    assert decision == AttemptRecoveryDecision(AttemptRecoveryKind.NONE, reason)


@pytest.mark.parametrize(
    ("code", "cause"),
    [
        ("tool_call_incomplete", FailureCause.OUTPUT_LENGTH),
        ("protocol_ambiguity", FailureCause.CONSTRAINT_FAILURE),
        ("restart_required", FailureCause.RESTART_REQUIRED),
        ("runtime_stream_exception", None),
        ("structured_output_invalid", FailureCause.OUTPUT_EOS),
    ],
)
def test_hard_or_unknown_failures_never_regenerate(code: str, cause: FailureCause | None) -> None:
    decision = classify_attempt_recovery(_evidence(code, cause))
    assert decision == AttemptRecoveryDecision(
        AttemptRecoveryKind.NONE,
        RecoverySkipReason.FAILURE_NOT_ALLOWLISTED,
    )


def test_runtime_recovering_uses_same_budget_even_for_deterministic_request() -> None:
    evidence = _evidence(
        "runtime_recovering",
        FailureCause.RUNTIME_RECOVERING,
        seed=17,
    )
    assert classify_attempt_recovery(evidence) == AttemptRecoveryDecision(
        AttemptRecoveryKind.WAIT_RUNTIME_AND_RETRY
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"fresh_attempt_replay": False}, RecoverySkipReason.RUNTIME_REPLAY_UNSUPPORTED),
        ({"recovery_readiness": False}, RecoverySkipReason.RUNTIME_READINESS_UNSUPPORTED),
    ],
)
def test_runtime_recovering_requires_replay_and_readiness_capabilities(
    overrides: dict[str, object],
    reason: RecoverySkipReason,
) -> None:
    evidence = _evidence("runtime_recovering", FailureCause.RUNTIME_RECOVERING, **overrides)
    assert classify_attempt_recovery(evidence) == AttemptRecoveryDecision(
        AttemptRecoveryKind.NONE,
        reason,
    )


def test_attempt_recovery_evidence_rejects_more_than_one_extra_attempt() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        _evidence(max_extra_attempts=2)


_COMPILED = CompiledPrompt(
    text="prompt",
    input_ids=(1,),
    prompt_hash="a" * 64,
    stop_conditions=(),
    template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
)


class _Attempt:
    def __init__(self, events: tuple[GenerationEvent, ...]) -> None:
        self.compiled_prompt = _COMPILED
        self._events = events
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        async def stream() -> AsyncIterator[GenerationEvent]:
            for event in self._events:
                yield event

        return stream()

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Lease:
    def __init__(self) -> None:
        self.deadline: float | None = None
        self.release_calls = 0
        self.is_active = True

    async def release(self) -> None:
        self.release_calls += 1
        self.is_active = False


_CAPABILITIES = RuntimeCapabilities(
    cancellation=True,
    template_rendering=True,
    tokenization=True,
    seed=True,
    cache_usage=True,
    quantized_kv_cache=True,
    fresh_attempt_replay=True,
    recovery_readiness=True,
)


async def _ready(_: float | None) -> RuntimeReadinessResult:
    return RuntimeReadinessResult.READY


def test_buffered_recovery_discards_failed_attempt_and_publishes_only_success() -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                ReasoningStarted("req"),
                ReasoningDelta("req", "hidden"),
                GenerationFailed("req", _error("tool_call_incomplete", FailureCause.OUTPUT_EOS)),
            )
        )
        second = _Attempt(
            (
                GenerationStarted("req"),
                TextStarted("req"),
                TextDelta("req", "ok"),
                GenerationCompleted("req", CompletionReason.STOP),
            )
        )
        starts = 0

        async def start_attempt() -> _Attempt:
            nonlocal starts
            starts += 1
            return second

        lease = _Lease()
        session = RecoveringServingSession(
            first,
            attempt_factory=start_attempt,
            lease=lease,
            visibility_mode=ServingVisibilityMode.BUFFERED,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        events = [event async for event in session]

        assert events == list(second._events)
        assert starts == 1
        assert lease.release_calls == 1
        assert session.diagnostics.attempts_started == 2
        assert session.diagnostics.recovery_attempts == 1
        assert session.diagnostics.recovered is True
        assert tuple(record.status.value for record in session.diagnostics.attempt_records) == (
            "failed",
            "completed",
        )
        assert tuple(record.discarded for record in session.diagnostics.attempt_records) == (True, False)

    asyncio.run(scenario())


def test_buffered_second_failure_is_the_only_outward_terminal() -> None:
    async def scenario() -> None:
        first_failure = GenerationFailed(
            "req", _error("tool_call_incomplete", FailureCause.OUTPUT_EOS)
        )
        second_failure = GenerationFailed(
            "req", _error("protocol_ambiguity", FailureCause.OUTPUT_EOS)
        )
        first = _Attempt((GenerationStarted("req"), TextCompleted("req", "discarded"), first_failure))
        second = _Attempt((GenerationStarted("req"), TextCompleted("req", "also-discarded"), second_failure))

        async def start_attempt() -> _Attempt:
            return second

        session = RecoveringServingSession(
            first,
            attempt_factory=start_attempt,
            lease=_Lease(),
            visibility_mode=ServingVisibilityMode.BUFFERED,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        events = [event async for event in session]

        assert events == [second_failure]
        assert session.diagnostics.recovery_attempts == 1
        assert session.diagnostics.recovered is False
        assert session.diagnostics.recovery_kind is AttemptRecoveryKind.REGENERATE_MODEL_OUTPUT
        assert session.diagnostics.final_skip_reason is RecoverySkipReason.ATTEMPT_BUDGET_EXHAUSTED

    asyncio.run(scenario())


def test_streaming_public_reasoning_prevents_transparent_retry() -> None:
    async def scenario() -> None:
        failure = GenerationFailed(
            "req",
            _error("tool_call_incomplete", FailureCause.OUTPUT_EOS),
        )
        first = _Attempt(
            (
                GenerationStarted("req"),
                ReasoningStarted("req"),
                ReasoningDelta("req", "visible"),
                failure,
            )
        )

        async def must_not_retry() -> _Attempt:
            raise AssertionError("published streaming attempt must not retry")

        lease = _Lease()
        session = RecoveringServingSession(
            first,
            attempt_factory=must_not_retry,
            lease=lease,
            visibility_mode=ServingVisibilityMode.STREAMING,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        events = [event async for event in session]

        assert events == list(first._events)
        assert session.diagnostics.recovery_attempts == 0
        assert session.diagnostics.final_skip_reason is RecoverySkipReason.SEMANTIC_ALREADY_PUBLISHED
        assert lease.release_calls == 1

    asyncio.run(scenario())


def test_streaming_recovery_discards_uncommitted_tool_fragments() -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":', 0),
                GenerationFailed(
                    "req",
                    _error("tool_call_invalid", FailureCause.MODEL_TOOL_OUTPUT_INVALID),
                ),
            )
        )
        second = _Attempt(
            (
                GenerationStarted("req"),
                TextStarted("req"),
                TextDelta("req", "recovered"),
                GenerationCompleted("req", CompletionReason.STOP),
            )
        )

        async def start_attempt() -> _Attempt:
            return second

        lease = _Lease()
        session = RecoveringServingSession(
            first,
            attempt_factory=start_attempt,
            lease=lease,
            visibility_mode=ServingVisibilityMode.STREAMING,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        events = [event async for event in session]

        assert events == [
            GenerationStarted("req"),
            TextStarted("req"),
            TextDelta("req", "recovered"),
            GenerationCompleted("req", CompletionReason.STOP),
        ]
        assert not any(isinstance(event, ToolCallStarted | ToolCallArgumentsDelta) for event in events)
        assert session.diagnostics.recovered is True
        assert lease.release_calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "terminal",
    (
        GenerationFailed(
            "req",
            _error("tool_call_invalid", FailureCause.MODEL_TOOL_OUTPUT_INVALID),
        ),
        GenerationCancelled("req"),
    ),
)
def test_streaming_final_terminal_never_flushes_unvalidated_tool_fragments(
    terminal: GenerationEvent,
) -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", "{\"id\":", 0),
                terminal,
            )
        )

        async def must_not_retry() -> _Attempt:
            raise AssertionError("final terminal must not create another attempt")

        session = RecoveringServingSession(
            first,
            attempt_factory=must_not_retry,
            lease=_Lease(),
            visibility_mode=ServingVisibilityMode.STREAMING,
            max_extra_attempts=1,
            seed=7 if isinstance(terminal, GenerationFailed) else None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        events = [event async for event in session]

        assert events == [GenerationStarted("req"), terminal]
        assert not any(isinstance(event, ToolCallStarted | ToolCallArgumentsDelta) for event in events)

    asyncio.run(scenario())


def test_pre_session_recovery_publishes_first_external_start_from_attempt_two() -> None:
    async def scenario() -> None:
        second = _Attempt(
            (
                GenerationStarted("req"),
                TextStarted("req"),
                TextDelta("req", "ok"),
                GenerationCompleted("req", CompletionReason.STOP),
            )
        )
        session = RecoveringServingSession(
            second,
            attempt_factory=lambda: _never_attempt(),
            lease=_Lease(),
            visibility_mode=ServingVisibilityMode.STREAMING,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
            initial_attempt_ordinal=2,
            initial_attempts_started=2,
            initial_recovery_attempts=1,
        )

        events = [event async for event in session]
        assert events == list(second._events)
        assert sum(isinstance(event, GenerationStarted) for event in events) == 1
        assert session.input_token_count == len(_COMPILED.input_ids)

    async def _never_attempt() -> _Attempt:
        raise AssertionError("attempt budget is already consumed")

    asyncio.run(scenario())


def test_streaming_valid_tool_fragments_flush_in_order_at_commit() -> None:
    async def scenario() -> None:
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        events = (
            GenerationStarted("req"),
            ToolCallStarted("req", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
            ToolCallCompleted("req", call),
            GenerationCompleted("req", CompletionReason.TOOL_CALLS),
        )
        session = RecoveringServingSession(
            _Attempt(events),
            attempt_factory=lambda: _never_attempt(),
            lease=_Lease(),
            visibility_mode=ServingVisibilityMode.STREAMING,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )

        assert [event async for event in session] == list(events)
        assert session.diagnostics.recovery_attempts == 0

    async def _never_attempt() -> _Attempt:
        raise AssertionError("valid committed Tool must not retry")

    asyncio.run(scenario())


def test_disabled_recovery_preserves_existing_stream_order_exactly() -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                GenerationFailed(
                    "req",
                    _error("tool_call_invalid", FailureCause.MODEL_TOOL_OUTPUT_INVALID),
                ),
            )
        )

        async def must_not_retry() -> _Attempt:
            raise AssertionError("disabled recovery must not create another attempt")

        lease = _Lease()
        session = RecoveringServingSession(
            first,
            attempt_factory=must_not_retry,
            lease=lease,
            visibility_mode=ServingVisibilityMode.STREAMING,
            max_extra_attempts=0,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        events = [event async for event in session]

        assert events == list(first._events)
        assert session.diagnostics.recovery_attempts == 0
        assert lease.release_calls == 1

    asyncio.run(scenario())


def test_runtime_readiness_deadline_preserves_request_timeout_semantics() -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                GenerationFailed("req", _error("runtime_recovering", FailureCause.RUNTIME_RECOVERING)),
            )
        )

        async def must_not_retry() -> _Attempt:
            raise AssertionError("deadline must prevent another attempt")

        async def deadline(_: float | None) -> RuntimeReadinessResult:
            return RuntimeReadinessResult.DEADLINE

        lease = _Lease()
        lease.deadline = 123.0
        session = RecoveringServingSession(
            first,
            attempt_factory=must_not_retry,
            lease=lease,
            visibility_mode=ServingVisibilityMode.BUFFERED,
            max_extra_attempts=1,
            seed=7,
            sampling=RuntimeSamplingConfig(temperature=0),
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=deadline,
        )
        events = [event async for event in session]

        assert len(events) == 1
        assert isinstance(events[0], GenerationFailed)
        assert events[0].error.code == "request_timeout"
        assert events[0].error.cause is None
        assert session.diagnostics.recovery_attempts == 0
        assert lease.release_calls == 1

    asyncio.run(scenario())


def test_runtime_wait_never_starts_retry_after_external_lease_release() -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                GenerationFailed("req", _error("runtime_recovering", FailureCause.RUNTIME_RECOVERING)),
            )
        )
        lease = _Lease()

        async def closed(_: float | None) -> RuntimeReadinessResult:
            lease.is_active = False
            return RuntimeReadinessResult.CLOSED

        async def must_not_retry() -> _Attempt:
            raise AssertionError("inactive lease must prevent another attempt")

        session = RecoveringServingSession(
            first,
            attempt_factory=must_not_retry,
            lease=lease,
            visibility_mode=ServingVisibilityMode.BUFFERED,
            max_extra_attempts=1,
            seed=7,
            sampling=RuntimeSamplingConfig(temperature=0),
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=closed,
        )
        events = [event async for event in session]

        assert len(events) == 1
        assert isinstance(events[0], GenerationCancelled)
        assert session.diagnostics.recovery_attempts == 0
        assert lease.release_calls == 1

    asyncio.run(scenario())


def test_cancel_during_retry_submit_window_prevents_second_attempt_publication() -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                GenerationFailed("req", _error("tool_call_incomplete", FailureCause.OUTPUT_EOS)),
            )
        )
        second = _Attempt((GenerationStarted("req"), GenerationCompleted("req", CompletionReason.STOP)))
        factory_started = asyncio.Event()
        factory_release = asyncio.Event()

        async def delayed_attempt() -> _Attempt:
            factory_started.set()
            await factory_release.wait()
            return second

        lease = _Lease()
        session = RecoveringServingSession(
            first,
            attempt_factory=delayed_attempt,
            lease=lease,
            visibility_mode=ServingVisibilityMode.BUFFERED,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        consume = asyncio.create_task(_collect(session))
        await factory_started.wait()
        await session.cancel()
        factory_release.set()
        events = await consume

        assert events == [GenerationCancelled("req")]
        assert second.cancel_calls == 1
        assert lease.release_calls == 1

    async def _collect(session: RecoveringServingSession) -> list[GenerationEvent]:
        return [event async for event in session]

    asyncio.run(scenario())


def test_buffered_recovery_exposes_only_successful_attempt_usage() -> None:
    async def scenario() -> None:
        discarded_usage = TokenUsage(input_tokens=10, output_tokens=99)
        published_usage = TokenUsage(input_tokens=10, output_tokens=3)
        first = _Attempt(
            (
                GenerationStarted("req"),
                UsageUpdated("req", discarded_usage),
                GenerationFailed("req", _error("tool_call_incomplete", FailureCause.OUTPUT_EOS)),
            )
        )
        second = _Attempt(
            (
                GenerationStarted("req"),
                UsageUpdated("req", published_usage),
                GenerationCompleted("req", CompletionReason.STOP, usage=published_usage),
            )
        )

        async def start_attempt() -> _Attempt:
            return second

        session = RecoveringServingSession(
            first,
            attempt_factory=start_attempt,
            lease=_Lease(),
            visibility_mode=ServingVisibilityMode.BUFFERED,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        events = [event async for event in session]
        usages = [event.usage for event in events if isinstance(event, UsageUpdated)]

        assert usages == [published_usage]
        assert not any(usage is discarded_usage for usage in usages)
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].usage == published_usage

    asyncio.run(scenario())


def test_buffered_recovery_discards_failed_attempt_from_response_state() -> None:
    async def scenario() -> None:
        first = _Attempt(
            (
                GenerationStarted("req"),
                TextCompleted("req", "discarded"),
                GenerationFailed("req", _error("tool_call_incomplete", FailureCause.OUTPUT_EOS)),
            )
        )
        second = _Attempt(
            (
                GenerationStarted("req"),
                TextCompleted("req", "kept"),
                GenerationCompleted("req", CompletionReason.STOP),
            )
        )

        async def start_attempt() -> _Attempt:
            return second

        recovering = RecoveringServingSession(
            first,
            attempt_factory=start_attempt,
            lease=_Lease(),
            visibility_mode=ServingVisibilityMode.BUFFERED,
            max_extra_attempts=1,
            seed=None,
            sampling=None,
            capabilities=_CAPABILITIES,
            has_prompt_attachments=False,
            runtime_waiter=_ready,
        )
        store = InMemoryResponseStore()
        session = StatefulServingSession(
            recovering,
            store,
            response_id="resp_recovered",
            model="m",
            base_context=(),
            current_input=(MessageItem(MessageRole.USER, "question"),),
            store_response=True,
        )
        _ = [event async for event in session]
        record = await store.get("resp_recovered")

        assert record is not None
        assert await store.materialize("resp_recovered") == (
            MessageItem(MessageRole.USER, "question"),
            MessageItem(MessageRole.ASSISTANT, "kept"),
        )

    asyncio.run(scenario())
