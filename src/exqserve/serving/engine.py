"""Protocol-independent serving orchestration and runtime/model bridges."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Self

from exqserve.agent._json import InvalidJsonError, parse_json_strict
from exqserve.agent.reasoning import (
    ReasoningBudgetDefault,
    ReasoningBudgetMode,
    ReasoningMode,
    ReasoningPolicy,
)
from exqserve.agent.structured_output import StructuredOutputSpec, validate_structured_output
from exqserve.agent.tools import ToolPolicy
from exqserve.agent.validation import validate_tool_calls, validate_tool_history
from exqserve.control.request import (
    RequestInjectionConflict,
    RequestInjectionTerminating,
    RequestRejected,
    RequestTerminalReason,
)
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    ReasoningCompleted,
    ReasoningDelta,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.tokens import NativeTokenSpan
from exqserve.model.contracts import (
    CompiledPrompt,
    NativeTokenAwareIncrementalParser,
    NativeTokenProvenanceError,
    ReasoningControlSpec,
    RenderedPrompt,
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
    TemplateToolResponse,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
    has_exposed_strict_tool,
)
from exqserve.runtime.contracts import (
    RuntimeConstraintUnsupported,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationConstraint,
    RuntimeGenerationRequest,
    RuntimeRenderedPrompt,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
)
from exqserve.serving.contracts import (
    IncrementalParserLike,
    PromptCompilerLike,
    ServingRejected,
    ServingRequest,
)
from exqserve.serving.runtime_events import (
    completion_reason_from_runtime,
    timing_event_from_runtime,
)
from exqserve.serving.tool_batch import ToolCallBatchGate, tool_validation_failure

logger = logging.getLogger(__name__)

class RuntimeTemplateRenderer(Protocol):
    def tokenize_encoded_prompt(self, text: str) -> RuntimeRenderedPrompt:
        ...

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
        protect_literal_tokens: bool = False,
    ) -> RuntimeRenderedPrompt:
        ...


def _tool_call_dict(call: TemplateToolCall) -> dict[str, object]:
    try:
        arguments = parse_json_strict(call.arguments_json)
    except InvalidJsonError as exc:
        raise ValueError("tool-call arguments must contain strict JSON") from exc
    if not isinstance(arguments, dict):
        raise TypeError("tool-call arguments must be a JSON object")
    return {"type": "function", "function": {"name": call.name, "arguments": arguments}}


def _tool_response_dict(response: TemplateToolResponse) -> dict[str, object]:
    try:
        value = parse_json_strict(response.response_json)
    except InvalidJsonError as exc:
        raise ValueError("tool-response payload must contain strict JSON") from exc
    return {"name": response.name, "response": value}


def _message_dict(message: TemplateMessage) -> dict[str, object]:
    content: object
    if isinstance(message.content, tuple):
        content_parts: list[dict[str, object]] = []
        for part in message.content:
            if isinstance(part, TemplateTextPart):
                content_parts.append({"type": "text", "text": part.text})
            elif isinstance(part, TemplateImagePart):
                image_part: dict[str, object] = {"type": "image", "image": part.source}
                if part.detail is not None:
                    image_part["detail"] = part.detail
                content_parts.append(image_part)
            else:  # pragma: no cover - template validation prevents this
                raise TypeError(f"unsupported template content part: {type(part).__name__}")
        content = content_parts
    else:
        content = message.content

    result: dict[str, object] = {"role": message.role, "content": content}
    if message.reasoning_content is not None:
        result["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        result["tool_calls"] = [_tool_call_dict(call) for call in message.tool_calls]
    if message.tool_responses:
        result["tool_responses"] = [
            _tool_response_dict(response) for response in message.tool_responses
        ]
    if message.name is not None:
        result["name"] = message.name
    return result


def _tool_dict(tool: TemplateTool) -> dict[str, object]:
    try:
        parameters = parse_json_strict(tool.parameters_json)
    except InvalidJsonError as exc:
        raise ValueError("tool schema must contain strict JSON") from exc
    if not isinstance(parameters, dict):
        raise TypeError("tool schema must be a JSON object")
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


class RuntimeTemplateAdapter:
    """Bridge model template values into a structural runtime renderer."""

    def __init__(self, renderer: RuntimeTemplateRenderer) -> None:
        self._renderer = renderer

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        if not isinstance(request, TemplateRequest):
            raise TypeError("request must be a TemplateRequest")
        messages = [_message_dict(message) for message in request.messages]
        tools = [_tool_dict(tool) for tool in request.tools] if request.tools else None
        if request.protect_literal_tokens:
            rendered = self._renderer.render_chat_template(
                messages,
                tools,
                dict(request.template_kwargs),
                add_generation_prompt=request.add_generation_prompt,
                protect_literal_tokens=True,
            )
        else:
            rendered = self._renderer.render_chat_template(
                messages,
                tools,
                dict(request.template_kwargs),
                add_generation_prompt=request.add_generation_prompt,
            )
        return RenderedPrompt(rendered.text, rendered.input_ids, rendered.runtime_attachments)

    def tokenize_encoded_prompt(self, text: str) -> RenderedPrompt:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        rendered = self._renderer.tokenize_encoded_prompt(text)
        return RenderedPrompt(rendered.text, rendered.input_ids, rendered.runtime_attachments)


class ControlledSessionLike(Protocol):
    terminal_reason: RequestTerminalReason | None

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        ...

    def inject_text(self, text: str) -> None:
        ...

    async def cancel(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED,
    ) -> None:
        ...


class RequestControllerLike(Protocol):
    async def submit(self, request: RuntimeGenerationRequest) -> ControlledSessionLike:
        ...


type ParserFactory = Callable[[str, ReasoningPolicy, ToolPolicy], IncrementalParserLike]
type ToolConstraintFactory = Callable[[ToolPolicy], ToolGenerationConstraint | None]
type ReasoningControlFactory = Callable[[ReasoningPolicy, ToolPolicy], ReasoningControlSpec | None]
type ReasoningControlTokenizer = Callable[[str], tuple[int, ...]]
type OutputLimitResolver = Callable[[int, int | None], int]


class _ReasoningBudgetSource(str, Enum):
    REQUEST = "request"
    SERVER_DEFAULT = "server_default"


@dataclass(frozen=True, slots=True)
class _EffectiveReasoningBudget:
    max_tokens: int
    message: str
    source: _ReasoningBudgetSource
    control: ReasoningControlSpec
    close_token_id: int

    @property
    def forced_text(self) -> str:
        return self.message + self.control.close_sequence


class _ReasoningBudgetState(str, Enum):
    DISABLED = "disabled"
    COUNTING = "counting"
    FORCE_REQUESTED = "force_requested"
    DONE = "done"


def _safe_error(
    category: ErrorCategory,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> CanonicalError:
    return CanonicalError(category, code, message, retryable)


class ServingEngine:
    def __init__(
        self,
        compiler: PromptCompilerLike,
        parser_factory: ParserFactory,
        controller: RequestControllerLike,
        tool_constraint_factory: ToolConstraintFactory | None = None,
        tool_call_fanout_limit: int = 32,
        constrained_parallel_tool_call_limit: int = 8,
        output_limit_resolver: OutputLimitResolver | None = None,
        reasoning_control_factory: ReasoningControlFactory | None = None,
        reasoning_control_tokenizer: ReasoningControlTokenizer | None = None,
        reasoning_budget_default: ReasoningBudgetDefault | None = None,
    ) -> None:
        if not isinstance(tool_call_fanout_limit, int) or isinstance(tool_call_fanout_limit, bool):
            raise TypeError("tool_call_fanout_limit must be an integer")
        if tool_call_fanout_limit <= 0:
            raise ValueError("tool_call_fanout_limit must be positive")
        if not isinstance(constrained_parallel_tool_call_limit, int) or isinstance(
            constrained_parallel_tool_call_limit, bool
        ):
            raise TypeError("constrained_parallel_tool_call_limit must be an integer")
        if constrained_parallel_tool_call_limit <= 0:
            raise ValueError("constrained_parallel_tool_call_limit must be positive")
        self._compiler = compiler
        self._parser_factory = parser_factory
        self._controller = controller
        self._tool_constraint_factory = tool_constraint_factory
        self._tool_call_fanout_limit = tool_call_fanout_limit
        if reasoning_budget_default is not None and not isinstance(
            reasoning_budget_default, ReasoningBudgetDefault
        ):
            raise TypeError("reasoning_budget_default must be ReasoningBudgetDefault or None")
        self._constrained_parallel_tool_call_limit = constrained_parallel_tool_call_limit
        self._output_limit_resolver = output_limit_resolver
        self._reasoning_control_factory = reasoning_control_factory
        self._reasoning_control_tokenizer = reasoning_control_tokenizer
        self._reasoning_budget_default = reasoning_budget_default or ReasoningBudgetDefault()
        self._compile_lock = asyncio.Lock()

    def _reasoning_budget_rejected(
        self, code: str, message: str, *, unsupported: bool = False
    ) -> ServingRejected:
        return ServingRejected(
            _safe_error(
                ErrorCategory.UNSUPPORTED_CAPABILITY if unsupported else ErrorCategory.INVALID_REQUEST,
                code,
                message,
            )
        )

    def _resolve_reasoning_budget(
        self, request: ServingRequest, *, active_generation_constraint: bool
    ) -> _EffectiveReasoningBudget | None:
        override = request.reasoning_budget
        if request.reasoning.mode is ReasoningMode.DISABLED:
            if override.mode is ReasoningBudgetMode.EXPLICIT:
                raise self._reasoning_budget_rejected(
                    "reasoning_budget_requires_reasoning",
                    "An explicit reasoning budget requires reasoning to be enabled.",
                )
            return None
        if override.mode is ReasoningBudgetMode.DISABLE:
            return None

        if override.mode is ReasoningBudgetMode.EXPLICIT:
            assert override.max_tokens is not None
            source = _ReasoningBudgetSource.REQUEST
            max_tokens = override.max_tokens
            message = (
                self._reasoning_budget_default.message
                if override.message is None
                else override.message
            )
        else:
            if self._reasoning_budget_default.max_tokens is None:
                return None
            source = _ReasoningBudgetSource.SERVER_DEFAULT
            max_tokens = self._reasoning_budget_default.max_tokens
            message = self._reasoning_budget_default.message

        if active_generation_constraint:
            if source is _ReasoningBudgetSource.REQUEST:
                raise self._reasoning_budget_rejected(
                    "reasoning_budget_incompatible_with_constraint",
                    "Reasoning budget fallback cannot be combined with an active generation constraint.",
                )
            logger.info(
                "reasoning budget skipped request_id=%s source=server_default reason=active_generation_constraint",
                request.input.request_id,
            )
            return None

        if self._reasoning_control_factory is None or self._reasoning_control_tokenizer is None:
            if source is _ReasoningBudgetSource.REQUEST:
                raise self._reasoning_budget_rejected(
                    "reasoning_budget_unsupported",
                    "Reasoning budget is unsupported by the selected model/runtime.",
                    unsupported=True,
                )
            logger.info(
                "reasoning budget skipped request_id=%s source=server_default reason=no_control_capability",
                request.input.request_id,
            )
            return None

        try:
            control = self._reasoning_control_factory(request.reasoning, request.tools)
        except Exception as exc:
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INTERNAL,
                    "serving_internal_error",
                    "Reasoning-control initialization failed internally.",
                )
            ) from exc
        if control is None or not control.initially_in_reasoning:
            if source is _ReasoningBudgetSource.REQUEST:
                raise self._reasoning_budget_rejected(
                    "reasoning_budget_unsupported",
                    "Reasoning budget fallback requires a generation that starts in reasoning.",
                    unsupported=True,
                )
            logger.info(
                "reasoning budget skipped request_id=%s source=server_default reason=not_initially_reasoning",
                request.input.request_id,
            )
            return None

        try:
            close_ids = self._reasoning_control_tokenizer(control.close_sequence)
        except Exception as exc:
            if source is _ReasoningBudgetSource.REQUEST:
                raise self._reasoning_budget_rejected(
                    "reasoning_budget_unsupported",
                    "Reasoning close sequence cannot be verified by the active tokenizer.",
                    unsupported=True,
                ) from exc
            logger.warning(
                "reasoning budget skipped request_id=%s source=server_default reason=close_tokenization_failed",
                request.input.request_id,
            )
            return None
        if len(close_ids) != 1:
            if source is _ReasoningBudgetSource.REQUEST:
                raise self._reasoning_budget_rejected(
                    "reasoning_budget_unsupported",
                    "Reasoning budget fallback requires an atomic single-token reasoning close.",
                    unsupported=True,
                )
            logger.info(
                "reasoning budget skipped request_id=%s source=server_default reason=non_atomic_close tokens=%d",
                request.input.request_id,
                len(close_ids),
            )
            return None

        logger.info(
            "reasoning budget enabled request_id=%s backend=fallback source=%s budget=%d",
            request.input.request_id,
            source.value,
            max_tokens,
        )
        return _EffectiveReasoningBudget(max_tokens, message, source, control, close_ids[0])

    def _compile_request(self, request: ServingRequest) -> CompiledPrompt:
        if not isinstance(request, ServingRequest):
            raise TypeError("request must be a ServingRequest")

        history_result = validate_tool_history(request.input.items)
        if not history_result.is_valid:
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INVALID_REQUEST,
                    "invalid_tool_history",
                    "Canonical tool history is invalid.",
                )
            )

        try:
            return self._compiler.compile(request.input, request.reasoning, request.tools)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Prompt compilation rejected for request %s: %s: %s",
                request.input.request_id,
                type(exc).__name__,
                exc,
            )
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INVALID_REQUEST,
                    "prompt_compilation_failed",
                    "Canonical input cannot be compiled for the selected model.",
                )
            ) from exc
        except Exception as exc:
            logger.exception("Serving prompt compilation failed for request %s", request.input.request_id)
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INTERNAL,
                    "serving_internal_error",
                    "Serving prompt compilation failed internally.",
                )
            ) from exc

    async def _compile_request_async(self, request: ServingRequest) -> CompiledPrompt:
        async with self._compile_lock:
            task = asyncio.create_task(asyncio.to_thread(self._compile_request, request))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # The worker thread cannot be cancelled. Keep the compiler lease
                # until it exits so a cancelled request cannot overlap a later
                # tokenizer/vision compilation on the same runtime objects.
                with contextlib.suppress(Exception):
                    await task
                raise

    async def count_input_tokens(self, request: ServingRequest) -> int:
        return len((await self._compile_request_async(request)).input_ids)

    async def submit(self, request: ServingRequest) -> ServingSession:
        if (
            self._tool_constraint_factory is None
            and has_exposed_strict_tool(request.tools)
        ):
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INVALID_REQUEST,
                    "tool_constraint_unsupported",
                    "Strict function tools are not supported by the selected model dialect.",
                )
            )
        compiled = await self._compile_request_async(request)
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
                max_output_tokens = self._output_limit_resolver(
                    len(compiled.input_ids), max_output_tokens
                )
            except RequestRejected as exc:
                raise ServingRejected(exc.error) from exc

        try:
            parser = self._parser_factory(
                request.input.request_id, request.reasoning, request.tools
            )
        except Exception as exc:
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INTERNAL,
                    "serving_internal_error",
                    "Serving parser initialization failed internally.",
                )
            ) from exc

        tool_constraint = None
        if self._tool_constraint_factory is not None:
            try:
                tool_constraint = self._tool_constraint_factory(request.tools)
            except ToolConstraintUnsupported as exc:
                logger.warning(
                    "Tool constraint compilation rejected for request %s: %s",
                    request.input.request_id,
                    exc,
                )
                raise ServingRejected(
                    _safe_error(
                        ErrorCategory.INVALID_REQUEST,
                        "tool_constraint_unsupported",
                        "Tool schema or policy cannot be represented by the configured constrained-generation mode.",
                    )
                ) from exc
            except (TypeError, ValueError) as exc:
                raise ServingRejected(
                    _safe_error(
                        ErrorCategory.INVALID_REQUEST,
                        "tool_constraint_invalid",
                        "Tool constraint configuration is invalid for this request.",
                    )
                ) from exc
            except Exception as exc:
                raise ServingRejected(
                    _safe_error(
                        ErrorCategory.INTERNAL,
                        "serving_internal_error",
                        "Tool constraint initialization failed internally.",
                    )
                ) from exc

        schema_hint = None
        schema_trigger = None
        if request.structured_output is not None:
            if compiled.raw_output_is_text_only:
                schema_hint = request.structured_output.schema.canonical_json
            elif compiled.structured_output_trigger is not None:
                schema_hint = request.structured_output.schema.canonical_json
                schema_trigger = compiled.structured_output_trigger

        if tool_constraint is not None and schema_hint is not None:
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INVALID_REQUEST,
                    "tool_constraint_conflict",
                    "Constrained tool generation cannot be combined with structured output in one request.",
                )
            )

        reasoning_budget = self._resolve_reasoning_budget(
            request,
            active_generation_constraint=schema_hint is not None or tool_constraint is not None,
        )

        runtime_request = RuntimeGenerationRequest(
            request_id=request.input.request_id,
            input_ids=compiled.input_ids,
            max_new_tokens=max_output_tokens,
            seed=request.seed,
            stop_conditions=(*compiled.stop_conditions, *request.stop_conditions),
            sampling=request.sampling,
            prompt_attachments=compiled.runtime_attachments,
            output_json_schema=schema_hint,
            output_json_trigger=schema_trigger,
            generation_constraint=(
                None
                if tool_constraint is None
                else RuntimeGenerationConstraint(
                    tool_constraint.trigger,
                    tool_constraint.lark_grammar,
                    tool_constraint.eos_after_completed,
                )
            ),
            use_native_eos=compiled.use_native_eos,
        )
        try:
            controlled = await self._controller.submit(runtime_request)
        except RuntimeConstraintUnsupported as exc:
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INVALID_REQUEST,
                    "tool_constraint_unsupported",
                    "Tool schema or policy cannot be represented by the active constrained-generation runtime.",
                )
            ) from exc
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

        session = ServingSession(
            request.input.request_id,
            controlled,
            parser,
            compiled,
            request.tools,
            request.structured_output,
            request.stop_conditions,
            self._tool_call_fanout_limit,
            request.tools.allow_parallel and tool_constraint is not None,
            self._constrained_parallel_tool_call_limit,
            reasoning_budget,
        )
        await session._initialize_reasoning_budget()
        return session


class ServingSession:
    """Canonical semantic stream over one controlled runtime generation."""

    def __init__(
        self,
        request_id: str,
        controlled: ControlledSessionLike,
        parser: IncrementalParserLike,
        compiled_prompt: CompiledPrompt,
        tool_policy: ToolPolicy,
        structured_output: StructuredOutputSpec | None,
        requested_stop_conditions: tuple[str | int, ...] = (),
        tool_call_fanout_limit: int = 32,
        atomic_parallel_tools: bool = False,
        constrained_parallel_tool_call_limit: int = 8,
        reasoning_budget: _EffectiveReasoningBudget | None = None,
    ) -> None:
        self._request_id = request_id
        self._controlled = controlled
        self._runtime_iterator = controlled.__aiter__()
        self._parser = parser
        self._compiled_prompt = compiled_prompt
        self._tool_policy = tool_policy
        self._structured_output = structured_output
        self._requested_stop_sequences = frozenset(
            condition for condition in requested_stop_conditions if isinstance(condition, str)
        )
        self._tool_batch = ToolCallBatchGate(
            tool_policy,
            tool_call_fanout_limit=tool_call_fanout_limit,
            atomic_parallel_tools=atomic_parallel_tools,
            constrained_parallel_tool_call_limit=constrained_parallel_tool_call_limit,
        )
        self._pending: deque[GenerationEvent] = deque()
        self._terminal = False
        self._parser_finished = False
        self._text_parts: list[str] = []
        self._runtime_trace: list[dict[str, object]] | None = None
        self._reasoning_budget = reasoning_budget
        self._reasoning_budget_state = (
            _ReasoningBudgetState.COUNTING
            if reasoning_budget is not None
            else _ReasoningBudgetState.DISABLED
        )
        self._reasoning_budget_token_count = 0
        self._reasoning_budget_overshoot_lower_bound = 0
        self._reasoning_budget_duplicate_close_pending = False
        self._reasoning_budget_duplicate_close_consumed_in_delta = False
        self._reasoning_budget_post_force_reasoning_tail = ""

    async def _reasoning_budget_failure(self, code: str, message: str) -> None:
        if self._terminal:
            return
        error = _safe_error(ErrorCategory.RUNTIME_FAILURE, code, message)
        self._discard_atomic_tool_batch()
        self._terminal = True
        self._reasoning_budget_state = _ReasoningBudgetState.DONE
        try:
            await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
        except Exception:
            logger.exception(
                "runtime cancellation failed after reasoning-budget failure request_id=%s",
                self._request_id,
            )
        self._pending.append(GenerationFailed(self._request_id, error))

    def _disable_reasoning_budget(self, reason: str) -> None:
        budget = self._reasoning_budget
        if budget is not None:
            logger.warning(
                "reasoning budget disabled request_id=%s source=%s reason=%s observed_tokens=%d",
                self._request_id,
                budget.source.value,
                reason,
                self._reasoning_budget_token_count,
            )
        self._reasoning_budget_state = _ReasoningBudgetState.DISABLED

    async def _force_reasoning_close(self) -> None:
        if self._reasoning_budget_state is not _ReasoningBudgetState.COUNTING:
            return
        budget = self._reasoning_budget
        if budget is None:
            self._reasoning_budget_state = _ReasoningBudgetState.DISABLED
            return
        try:
            self._controlled.inject_text(budget.forced_text)
        except RequestInjectionTerminating:
            logger.info(
                "reasoning budget close lost terminal race request_id=%s source=%s observed_tokens=%d",
                self._request_id,
                budget.source.value,
                self._reasoning_budget_token_count,
            )
            self._reasoning_budget_state = _ReasoningBudgetState.DONE
            return
        except RequestInjectionConflict:
            if budget.source is _ReasoningBudgetSource.SERVER_DEFAULT:
                self._disable_reasoning_budget("injection_conflict")
                return
            await self._reasoning_budget_failure(
                "reasoning_budget_enforcement_failed",
                "Reasoning budget close could not be applied safely.",
            )
            return
        except Exception:
            logger.exception("reasoning budget injection failed request_id=%s", self._request_id)
            if budget.source is _ReasoningBudgetSource.SERVER_DEFAULT:
                self._disable_reasoning_budget("injection_failed")
                return
            await self._reasoning_budget_failure(
                "reasoning_budget_enforcement_failed",
                "Reasoning budget close failed internally.",
            )
            return
        self._reasoning_budget_state = _ReasoningBudgetState.FORCE_REQUESTED
        logger.info(
            "reasoning budget close queued request_id=%s source=%s budget=%d observed_tokens=%d",
            self._request_id,
            budget.source.value,
            budget.max_tokens,
            self._reasoning_budget_token_count,
        )

    async def _initialize_reasoning_budget(self) -> None:
        budget = self._reasoning_budget
        if budget is not None and budget.max_tokens == 0:
            await self._force_reasoning_close()

    def _reasoning_budget_close_spans(
        self, event: RuntimeTextDelta
    ) -> tuple[NativeTokenSpan, ...]:
        budget = self._reasoning_budget
        if (
            budget is None
            or not event.native_token_provenance
            or event.native_token_spans is None
        ):
            return ()
        return tuple(
            span
            for span in event.native_token_spans
            if span.token_id == budget.close_token_id
            and span.text == budget.control.close_sequence
        )

    def _strip_reasoning_budget_close(
        self,
        event: RuntimeTextDelta,
        target: NativeTokenSpan,
    ) -> RuntimeTextDelta | None:
        budget = self._reasoning_budget
        spans = event.native_token_spans
        if budget is None or spans is None:
            return event

        target_occurrence = 0
        found_target = False
        for span in spans:
            if span.token_id != budget.close_token_id:
                continue
            if span is target:
                found_target = True
                break
            target_occurrence += 1
        if not found_target:
            return event

        token_ids = list(event.token_ids)
        seen = 0
        remove_index: int | None = None
        for index, token_id in enumerate(token_ids):
            if token_id != budget.close_token_id:
                continue
            if seen == target_occurrence:
                remove_index = index
                break
            seen += 1
        if remove_index is None:
            return event
        del token_ids[remove_index]

        removed_chars = target.end - target.start
        text = event.text[: target.start] + event.text[target.end :]
        if not text:
            return None

        adjusted_spans: list[NativeTokenSpan] = []
        for span in spans:
            if span is target:
                continue
            if span.start >= target.end:
                adjusted_spans.append(
                    NativeTokenSpan(
                        span.start - removed_chars,
                        span.end - removed_chars,
                        span.token_id,
                        span.text,
                    )
                )
            else:
                adjusted_spans.append(span)
        return RuntimeTextDelta(
            event.request_id,
            text,
            tuple(token_ids),
            tuple(adjusted_spans),
            True,
        )

    async def _prepare_reasoning_budget_delta(
        self, event: RuntimeTextDelta
    ) -> RuntimeTextDelta | None:
        budget = self._reasoning_budget
        self._reasoning_budget_duplicate_close_consumed_in_delta = False
        if budget is None:
            return event

        close_spans = self._reasoning_budget_close_spans(event)
        if (
            self._reasoning_budget_state is _ReasoningBudgetState.FORCE_REQUESTED
            and len(close_spans) >= 2
        ):
            self._reasoning_budget_duplicate_close_consumed_in_delta = True
            logger.info(
                "reasoning budget suppressed merged duplicate close request_id=%s token_id=%d",
                self._request_id,
                budget.close_token_id,
            )
            return self._strip_reasoning_budget_close(event, close_spans[1])

        if not self._reasoning_budget_duplicate_close_pending:
            return event

        if close_spans:
            self._reasoning_budget_duplicate_close_pending = False
            logger.info(
                "reasoning budget suppressed queued duplicate close request_id=%s token_id=%d",
                self._request_id,
                budget.close_token_id,
            )
            return self._strip_reasoning_budget_close(event, close_spans[0])

        if (
            event.text == budget.control.close_sequence
            and event.token_ids == (budget.close_token_id,)
        ):
            self._reasoning_budget_duplicate_close_pending = False
            logger.info(
                "reasoning budget suppressed exact queued duplicate close request_id=%s token_id=%d",
                self._request_id,
                budget.close_token_id,
            )
            return None
        return event

    async def _apply_reasoning_budget(
        self, event: RuntimeTextDelta, semantic_events: tuple[GenerationEvent, ...]
    ) -> None:
        budget = self._reasoning_budget
        state = self._reasoning_budget_state
        if budget is None or state in {
            _ReasoningBudgetState.DISABLED,
            _ReasoningBudgetState.DONE,
        }:
            return
        if state is _ReasoningBudgetState.FORCE_REQUESTED and budget.message:
            for semantic in semantic_events:
                if isinstance(semantic, ReasoningDelta):
                    self._reasoning_budget_post_force_reasoning_tail = (
                        self._reasoning_budget_post_force_reasoning_tail + semantic.text
                    )[-len(budget.message) :]
        if any(isinstance(semantic, ReasoningCompleted) for semantic in semantic_events):
            if state is _ReasoningBudgetState.FORCE_REQUESTED:
                if budget.message and not self._reasoning_budget_post_force_reasoning_tail.endswith(
                    budget.message
                ):
                    await self._reasoning_budget_failure(
                        "reasoning_budget_enforcement_failed",
                        "Reasoning budget message lost a race with a natural reasoning close.",
                    )
                    return
                self._reasoning_budget_duplicate_close_pending = (
                    not budget.message
                    and not self._reasoning_budget_duplicate_close_consumed_in_delta
                )
                logger.info(
                    "reasoning budget close completed request_id=%s observed_trigger_tokens=%d "
                    "overshoot_lower_bound_tokens=%d",
                    self._request_id,
                    self._reasoning_budget_token_count,
                    self._reasoning_budget_overshoot_lower_bound,
                )
            else:
                logger.info(
                    "reasoning budget completed naturally request_id=%s observed_tokens=%d",
                    self._request_id,
                    self._reasoning_budget_token_count,
                )
            self._reasoning_budget_state = _ReasoningBudgetState.DONE
            self._reasoning_budget_post_force_reasoning_tail = ""
            return
        if self._reasoning_budget_state is _ReasoningBudgetState.FORCE_REQUESTED:
            self._reasoning_budget_overshoot_lower_bound += len(event.token_ids)
            return
        if self._reasoning_budget_state is not _ReasoningBudgetState.COUNTING:
            return
        if event.text and not event.token_ids:
            if budget.source is _ReasoningBudgetSource.SERVER_DEFAULT:
                self._disable_reasoning_budget("token_ids_unavailable")
                return
            await self._reasoning_budget_failure(
                "reasoning_budget_accounting_unavailable",
                "Reasoning budget accounting became unavailable during generation.",
            )
            return
        self._reasoning_budget_token_count += len(event.token_ids)
        if self._reasoning_budget_token_count >= budget.max_tokens:
            await self._force_reasoning_close()

    @property
    def compiled_prompt(self) -> CompiledPrompt:
        return self._compiled_prompt

    def enable_runtime_trace(self) -> None:
        if self._runtime_trace is None:
            self._runtime_trace = []

    @property
    def runtime_trace(self) -> tuple[dict[str, object], ...]:
        if self._runtime_trace is None:
            return ()
        return tuple(dict(entry) for entry in self._runtime_trace)

    def __aiter__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not self._terminal:
            await self.cancel()
        return False

    def _discard_atomic_tool_batch(self) -> None:
        self._tool_batch.abort()

    def _flush_atomic_tool_batch(self) -> None:
        self._pending.extend(self._tool_batch.commit_events())

    async def cancel(self) -> None:
        if self._terminal:
            return
        self._discard_atomic_tool_batch()
        await self._controlled.cancel(RequestTerminalReason.CLIENT_CANCELLED)

    async def _model_failure(self, code: str, message: str) -> None:
        if self._terminal:
            return
        error = _safe_error(ErrorCategory.MODEL_FAILURE, code, message)
        self._discard_atomic_tool_batch()
        self._terminal = True
        try:
            await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
        except Exception:
            logger.exception(
                "runtime cancellation failed after local model failure request_id=%s",
                self._request_id,
            )
        self._pending.append(GenerationFailed(self._request_id, error))

    async def _process_semantic(self, event: GenerationEvent) -> None:
        if self._terminal:
            return
        if isinstance(event, TextDelta):
            self._text_parts.append(event.text)
            self._pending.append(event)
            return
        if isinstance(event, ToolCallStarted):
            decision = self._tool_batch.on_started(event)
            if decision.failure is not None:
                await self._model_failure(decision.failure.code, decision.failure.message)
                return
            self._pending.extend(decision.events)
            return
        if isinstance(event, ToolCallArgumentsDelta):
            decision = self._tool_batch.on_arguments_delta(event)
            if decision.failure is not None:
                await self._model_failure(decision.failure.code, decision.failure.message)
                return
            self._pending.extend(decision.events)
            return
        if isinstance(event, ToolCallCompleted):
            decision = self._tool_batch.on_completed(event)
            if decision.failure is not None:
                await self._model_failure(decision.failure.code, decision.failure.message)
                return
            self._pending.extend(decision.events)
            return
        self._pending.append(event)

    async def _fail_native_token_provenance(self) -> None:
        error = _safe_error(
            ErrorCategory.RUNTIME_FAILURE,
            "output_token_provenance_unavailable",
            "Inference output provenance was insufficient to classify a Qwen structural marker safely.",
            retryable=True,
        )
        self._discard_atomic_tool_batch()
        self._terminal = True
        try:
            await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
        except Exception:
            logger.exception(
                "runtime cancellation failed after local provenance failure request_id=%s",
                self._request_id,
            )
        self._pending.append(GenerationFailed(self._request_id, error))

    async def _finish_parser_events(self) -> bool:
        if self._parser_finished:
            return False
        self._parser_finished = True
        try:
            finish = self._parser.finish()
        except NativeTokenProvenanceError:
            await self._fail_native_token_provenance()
            return False
        for event in finish.events:
            await self._process_semantic(event)
            if self._terminal:
                return finish.incomplete_tool_call
        return finish.incomplete_tool_call

    async def _handle_runtime_finished(self, event: RuntimeFinished) -> None:
        incomplete_tool = await self._finish_parser_events()
        if self._terminal:
            return
        if incomplete_tool:
            logger.warning(
                "model output ended with an incomplete tool call request_id=%s",
                self._request_id,
            )
            message = "Model output ended with an incomplete tool call."
            if event.reason is RuntimeStopReason.LENGTH:
                message = "Model output reached the output token limit with an incomplete tool call."
            await self._model_failure("tool_call_incomplete", message)
            return

        completed_calls = self._tool_batch.completed_calls
        final_tool_validation = validate_tool_calls(completed_calls, self._tool_policy)
        if not final_tool_validation.is_valid:
            logger.warning(
                "model output violates requested tool policy request_id=%s issues=%s",
                self._request_id,
                ",".join(issue.code.value for issue in final_tool_validation.issues),
            )
            code, message = tool_validation_failure(final_tool_validation)
            await self._model_failure(code, message)
            return

        if not incomplete_tool and not completed_calls and self._structured_output is not None:
            structured = validate_structured_output("".join(self._text_parts), self._structured_output)
            if not structured.is_valid:
                await self._model_failure(
                    "structured_output_invalid",
                    "Model output did not satisfy the requested structured-output schema.",
                )
                return

        reason = (
            CompletionReason.TOOL_CALLS
            if completed_calls
            else completion_reason_from_runtime(event.reason)
        )

        self._flush_atomic_tool_batch()
        timing_event = timing_event_from_runtime(self._request_id, event.timing)
        if timing_event is not None:
            self._pending.append(timing_event)
        self._pending.append(UsageUpdated(self._request_id, event.usage))
        exposed_stop_sequence = (
            event.stop_sequence
            if reason is CompletionReason.STOP
            and event.stop_sequence in self._requested_stop_sequences
            else None
        )
        self._pending.append(
            GenerationCompleted(
                self._request_id,
                reason,
                event.usage,
                exposed_stop_sequence,
            )
        )
        self._terminal = True

    async def _handle_runtime_failure(self, event: RuntimeFailed) -> None:
        await self._finish_parser_events()
        if self._terminal:
            return
        self._discard_atomic_tool_batch()
        self._pending.append(GenerationFailed(self._request_id, event.error))
        self._terminal = True

    async def _handle_runtime_cancelled(self) -> None:
        await self._finish_parser_events()
        if self._terminal:
            return
        self._discard_atomic_tool_batch()
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

    async def _process_runtime(self, event: RuntimeEvent) -> None:
        if self._runtime_trace is not None:
            if isinstance(event, RuntimeTextDelta):
                spans = (
                    None
                    if event.native_token_spans is None
                    else [
                        {
                            "start": span.start,
                            "end": span.end,
                            "token_id": span.token_id,
                            "text": span.text,
                        }
                        for span in event.native_token_spans
                    ]
                )
                self._runtime_trace.append(
                    {
                        "type": "text_delta",
                        "text": event.text,
                        "token_ids": list(event.token_ids),
                        "native_token_provenance": event.native_token_provenance,
                        "native_token_spans": spans,
                    }
                )
            elif isinstance(event, RuntimeFinished):
                self._runtime_trace.append(
                    {
                        "type": "finished",
                        "reason": event.reason.value,
                        "backend_reason": event.backend_reason,
                        "stop_sequence": event.stop_sequence,
                        "eos_token_id": event.eos_token_id,
                        "eos_token_text": event.eos_token_text,
                    }
                )

        if isinstance(event, RuntimeStarted):
            self._pending.append(GenerationStarted(self._request_id))
            return
        if isinstance(event, RuntimeTextDelta):
            prepared_event = await self._prepare_reasoning_budget_delta(event)
            if self._terminal:
                return
            if prepared_event is None:
                return
            event = prepared_event
            try:
                if (
                    isinstance(self._parser, NativeTokenAwareIncrementalParser)
                    and event.native_token_provenance
                ):
                    semantic_events = self._parser.feed_with_native_tokens(
                        event.text,
                        event.native_token_spans,
                    )
                else:
                    semantic_events = self._parser.feed(event.text)
            except NativeTokenProvenanceError:
                await self._fail_native_token_provenance()
                return
            await self._apply_reasoning_budget(event, semantic_events)
            if self._terminal:
                return
            for semantic in semantic_events:
                await self._process_semantic(semantic)
                if self._terminal:
                    return
            return
        if isinstance(event, RuntimeFinished):
            await self._handle_runtime_finished(event)
            return
        if isinstance(event, RuntimeFailed):
            await self._handle_runtime_failure(event)
            return
        await self._handle_runtime_cancelled()

    async def __anext__(self) -> GenerationEvent:
        while True:
            if self._pending:
                return self._pending.popleft()
            if self._terminal:
                raise StopAsyncIteration

            try:
                runtime_event = await anext(self._runtime_iterator)
            except StopAsyncIteration:
                await self._finish_parser_events()
                if self._pending:
                    continue
                if not self._terminal:
                    self._discard_atomic_tool_batch()
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

            await self._process_runtime(runtime_event)
