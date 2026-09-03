"""Protocol-independent serving orchestration and runtime/model bridges."""

from __future__ import annotations

import asyncio
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
from exqserve.agent.structured_output import (
    StructuredOutputSpec,
    validate_structured_output,
    violates_structured_constraint_guarantee,
)
from exqserve.agent.tools import ToolPolicy
from exqserve.agent.validation import validate_tool_calls, validate_tool_history
from exqserve.control.request import (
    RequestInjectionConflict,
    RequestInjectionTerminating,
    RequestRejected,
    RequestTerminalReason,
)
from exqserve.core.errors import (
    CanonicalError,
    ErrorCategory,
    FailureCause,
    SemanticCommitClass,
    commit_aware_error,
)
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
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.core.items import (
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    TextContentPart,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.core.tokens import NativeTokenSpan
from exqserve.model.contracts import (
    CompiledPrompt,
    NativeTokenAwareIncrementalParser,
    NativeTokenProvenanceError,
    ParserAmbiguityDetail,
    ParserTerminalIssue,
    ParserTerminalIssueKind,
    ReasoningControlSpec,
    RenderedPrompt,
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
    TemplateToolResponse,
    ToolConstraintGuarantee,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
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
    BestEffortMidSystemLowering,
    IncrementalParserLike,
    MidSystemCapability,
    MidSystemPolicy,
    PromptCompilerLike,
    ServingRejected,
    ServingRequest,
)
from exqserve.serving.guarantees import RequestGuaranteeResolver, guarantee_satisfies
from exqserve.serving.preprocessing import RendererLanePool, await_task_termination
from exqserve.serving.runtime_events import timing_event_from_runtime
from exqserve.serving.terminal import (
    TerminalDecision,
    TerminalDisposition,
    TerminalEvidence,
    TerminalPrimaryOwner,
)
from exqserve.serving.tool_batch import (
    ToolCallBatchGate,
    is_model_tool_output_invalid,
    tool_validation_failure,
    violates_tool_constraint_guarantee,
)

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
        structural_marker_texts: tuple[str, ...] = (),
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

    def __init__(
        self,
        renderer: RuntimeTemplateRenderer,
        structural_marker_texts: tuple[str, ...] = (),
    ) -> None:
        self._renderer = renderer
        self._structural_marker_texts = structural_marker_texts

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        if not isinstance(request, TemplateRequest):
            raise TypeError("request must be a TemplateRequest")
        messages = [_message_dict(message) for message in request.messages]
        tools = [_tool_dict(tool) for tool in request.tools] if request.tools else None
        if self._structural_marker_texts:
            rendered = self._renderer.render_chat_template(
                messages,
                tools,
                dict(request.template_kwargs),
                add_generation_prompt=request.add_generation_prompt,
                protect_literal_tokens=True,
                structural_marker_texts=self._structural_marker_texts,
            )
        elif request.protect_literal_tokens:
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


class RequestLeaseLike(Protocol):
    async def submit(self, request: RuntimeGenerationRequest) -> ControlledSessionLike:
        ...

    async def release(self) -> None:
        ...


class RequestControllerLike(Protocol):
    async def acquire(self, request_id: str) -> RequestLeaseLike:
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


def _is_instruction_item(item: object) -> bool:
    return isinstance(item, MessageItem) and item.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}


def _mid_system_counts(items: tuple[object, ...], leading_end: int) -> tuple[int, int]:
    message_count = 0
    section_count = 0
    in_section = False
    for item in items[leading_end:]:
        is_system = isinstance(item, MessageItem) and item.role is MessageRole.SYSTEM
        if is_system:
            message_count += 1
            if not in_section:
                section_count += 1
        in_section = is_system
    return message_count, section_count


def _render_mid_system_reminder(section: tuple[MessageItem, ...]) -> str:
    return "\n\n".join(
        f"<system-reminder>\n{item.text}\n</system-reminder>" for item in section
    )


def _invalid_mid_system_attachment(source: CanonicalRequest) -> ServingRejected:
    logger.warning(
        "mid-system in-place lowering found invalid predecessor request_id=%s",
        source.request_id,
    )
    return ServingRejected(
        _safe_error(
            ErrorCategory.INVALID_REQUEST,
            "mid_conversation_system_invalid_placement",
            "A mid-conversation system section could not be attached to the preceding user turn.",
        )
    )


def _lower_mid_system_in_place(source: CanonicalRequest, leading_end: int) -> CanonicalRequest:
    items = source.items
    lowered = list(items[:leading_end])
    position = leading_end

    while position < len(items):
        item = items[position]
        if not (isinstance(item, MessageItem) and item.role is MessageRole.SYSTEM):
            lowered.append(item)
            position += 1
            continue

        section: list[MessageItem] = []
        while position < len(items):
            candidate = items[position]
            if not (isinstance(candidate, MessageItem) and candidate.role is MessageRole.SYSTEM):
                break
            section.append(candidate)
            position += 1

        if not lowered:
            raise _invalid_mid_system_attachment(source)
        reminder = _render_mid_system_reminder(tuple(section))
        predecessor = lowered[-1]
        if isinstance(predecessor, MessageItem) and predecessor.role is MessageRole.USER:
            lowered[-1] = MessageItem(MessageRole.USER, predecessor.text + "\n\n" + reminder)
            continue
        if isinstance(predecessor, MultimodalMessageItem) and predecessor.role is MessageRole.USER:
            lowered[-1] = MultimodalMessageItem(
                MessageRole.USER,
                predecessor.parts + (TextContentPart("\n\n" + reminder),),
            )
            continue
        if isinstance(predecessor, ToolResultItem | MultimodalToolResultItem):
            lowered.append(MessageItem(MessageRole.USER, reminder))
            continue
        raise _invalid_mid_system_attachment(source)

    return CanonicalRequest(source.request_id, source.model, tuple(lowered))


def _normalize_mid_system_input(
    request: ServingRequest,
    capability: MidSystemCapability,
    best_effort_lowering: BestEffortMidSystemLowering,
) -> CanonicalRequest:
    source = request.input
    items = source.items
    leading_end = 0
    while leading_end < len(items) and _is_instruction_item(items[leading_end]):
        leading_end += 1

    message_count, section_count = _mid_system_counts(items, leading_end)
    if message_count == 0 or request.mid_system_policy is MidSystemPolicy.LEGACY_UNSPECIFIED:
        return source
    if capability is MidSystemCapability.INLINE:
        return source
    if request.mid_system_policy is MidSystemPolicy.STRICT:
        logger.info(
            "mid-system normalization rejected request_id=%s policy=%s capability=%s sections=%d messages=%d",
            source.request_id,
            request.mid_system_policy.value,
            capability.value,
            section_count,
            message_count,
        )
        raise ServingRejected(
            _safe_error(
                ErrorCategory.UNSUPPORTED_CAPABILITY,
                "mid_conversation_system_unsupported",
                "The active model dialect cannot preserve mid-conversation system authority; use an explicit compatibility profile for best-effort lowering.",
            )
        )

    if best_effort_lowering is BestEffortMidSystemLowering.IN_PLACE_USER_META:
        effective = _lower_mid_system_in_place(source, leading_end)
    else:
        late_system = tuple(
            item
            for item in items[leading_end:]
            if isinstance(item, MessageItem) and item.role is MessageRole.SYSTEM
        )
        retained_tail = tuple(
            item
            for item in items[leading_end:]
            if not (isinstance(item, MessageItem) and item.role is MessageRole.SYSTEM)
        )
        effective = CanonicalRequest(
            source.request_id,
            source.model,
            items[:leading_end] + late_system + retained_tail,
        )

    logger.info(
        "mid-system normalization applied request_id=%s policy=%s capability=%s best_effort_lowering=%s sections=%d messages=%d",
        source.request_id,
        request.mid_system_policy.value,
        capability.value,
        best_effort_lowering.value,
        section_count,
        message_count,
    )
    return effective


class ServingEngine:
    def __init__(
        self,
        compiler: PromptCompilerLike | None,
        parser_factory: ParserFactory,
        controller: RequestControllerLike,
        tool_constraint_factory: ToolConstraintFactory | None = None,
        tool_call_fanout_limit: int = 32,
        constrained_parallel_tool_call_limit: int = 8,
        output_limit_resolver: OutputLimitResolver | None = None,
        reasoning_control_factory: ReasoningControlFactory | None = None,
        reasoning_control_tokenizer: ReasoningControlTokenizer | None = None,
        reasoning_budget_default: ReasoningBudgetDefault | None = None,
        preprocessing_pool: RendererLanePool | None = None,
        mid_system_capability: MidSystemCapability = MidSystemCapability.LEADING_ONLY,
        best_effort_mid_system_lowering: BestEffortMidSystemLowering = (
            BestEffortMidSystemLowering.MERGED_LEADING
        ),
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
        if compiler is None and preprocessing_pool is None:
            raise ValueError("compiler or preprocessing_pool is required")
        self._compiler = compiler
        self._preprocessing_pool = preprocessing_pool
        self._parser_factory = parser_factory
        self._controller = controller
        self._guarantee_resolver = RequestGuaranteeResolver(tool_constraint_factory)
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
        if not isinstance(mid_system_capability, MidSystemCapability):
            raise TypeError("mid_system_capability must be a MidSystemCapability")
        if not isinstance(best_effort_mid_system_lowering, BestEffortMidSystemLowering):
            raise TypeError(
                "best_effort_mid_system_lowering must be a BestEffortMidSystemLowering"
            )
        self._mid_system_capability = mid_system_capability
        self._best_effort_mid_system_lowering = best_effort_mid_system_lowering
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

    def _compile_request_with_compiler(
        self, compiler: PromptCompilerLike, request: ServingRequest
    ) -> CompiledPrompt:
        if not isinstance(request, ServingRequest):
            raise TypeError("request must be a ServingRequest")

        effective_input = _normalize_mid_system_input(
            request,
            self._mid_system_capability,
            self._best_effort_mid_system_lowering,
        )
        history_result = validate_tool_history(effective_input.items)
        if not history_result.is_valid:
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INVALID_REQUEST,
                    "invalid_tool_history",
                    "Canonical tool history is invalid.",
                )
            )

        try:
            return compiler.compile(effective_input, request.reasoning, request.tools)
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

    def _compile_request(self, request: ServingRequest) -> CompiledPrompt:
        compiler = self._compiler
        if compiler is None:
            raise RuntimeError("direct compiler is unavailable while preprocessing pool is active")
        return self._compile_request_with_compiler(compiler, request)

    async def _compile_request_async(
        self, request: ServingRequest, *, kind: str = "chat"
    ) -> CompiledPrompt:
        pool = self._preprocessing_pool
        if pool is not None:
            return await pool.run(
                kind,
                lambda lane: self._compile_request_with_compiler(lane.compiler, request),
            )

        async with self._compile_lock:
            task = asyncio.create_task(asyncio.to_thread(self._compile_request, request))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # The worker thread cannot be cancelled. Keep the compiler lease
                # until it exits so a cancelled request cannot overlap a later
                # tokenizer/vision compilation on the same runtime objects.
                await await_task_termination(task)
                raise

    async def count_input_tokens(self, request: ServingRequest) -> int:
        try:
            lease = await self._controller.acquire(request.input.request_id)
        except RequestRejected as exc:
            raise ServingRejected(exc.error) from exc
        try:
            compiled = await self._compile_request_async(request, kind="count_tokens")
            return len(compiled.input_ids)
        finally:
            await lease.release()

    async def submit(self, request: ServingRequest) -> ServingSession:
        try:
            self._guarantee_resolver.ensure_tool_request_supported(request.tools)
        except ToolConstraintUnsupported as exc:
            raise ServingRejected(
                _safe_error(
                    ErrorCategory.INVALID_REQUEST,
                    "tool_constraint_unsupported",
                    str(exc),
                )
            ) from exc

        try:
            lease = await self._controller.acquire(request.input.request_id)
        except RequestRejected as exc:
            raise ServingRejected(exc.error) from exc

        controlled: ControlledSessionLike | None = None
        try:
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

            try:
                tool_plan = self._guarantee_resolver.resolve_tool_policy(request.tools)
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
            tool_constraint = tool_plan.constraint

            structured_plan = self._guarantee_resolver.resolve_structured_output(
                request.structured_output,
                raw_output_is_text_only=compiled.raw_output_is_text_only,
                structured_output_trigger=compiled.structured_output_trigger,
            )
            if structured_plan is not None and not structured_plan.is_supported:
                raise ServingRejected(
                    _safe_error(
                        ErrorCategory.INVALID_REQUEST,
                        "structured_output_constraint_unsupported",
                        "Requested structured-output generation guarantee is not supported by the selected model/runtime constraint path.",
                    )
                )
            schema_hint = None if structured_plan is None else structured_plan.schema_json
            schema_trigger = None if structured_plan is None else structured_plan.trigger

            if tool_constraint is not None and schema_hint is not None:
                raise ServingRejected(
                    _safe_error(
                        ErrorCategory.INVALID_REQUEST,
                        "tool_constraint_conflict",
                        "Constrained tool generation cannot be combined with structured output in one request.",
                    )
                )

            if tool_constraint is not None:
                runtime_guarantee = tool_plan.runtime_guarantee
                runtime_fallback = tool_plan.fallback_policy
            elif structured_plan is not None and schema_hint is not None:
                runtime_guarantee = structured_plan.planned_guarantee
                runtime_fallback = structured_plan.fallback_policy
            else:
                runtime_guarantee = GenerationGuarantee.NONE
                runtime_fallback = ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY

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
                generation_guarantee=runtime_guarantee,
                constraint_fallback_policy=runtime_fallback,
                use_native_eos=compiled.use_native_eos,
            )
            try:
                controlled = await lease.submit(runtime_request)
            except RuntimeConstraintUnsupported as exc:
                if schema_hint is not None:
                    raise ServingRejected(
                        _safe_error(
                            ErrorCategory.INVALID_REQUEST,
                            "structured_output_constraint_unsupported",
                            "Requested structured-output generation guarantee is not supported by the selected model/runtime constraint path.",
                        )
                    ) from exc
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

            try:
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
                    tool_constraint=tool_constraint,
                )
                await session._initialize_reasoning_budget()
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
            return session
        finally:
            if controlled is None:
                await lease.release()


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
        tool_constraint: ToolGenerationConstraint | None = None,
    ) -> None:
        self._request_id = request_id
        self._controlled = controlled
        self._runtime_iterator = controlled.__aiter__()
        self._parser = parser
        self._compiled_prompt = compiled_prompt
        self._tool_policy = tool_policy
        self._structured_output = structured_output
        self._tool_constraint = tool_constraint
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
        self._commit_class = SemanticCommitClass.NO_SEMANTIC_COMMIT
        self._terminal_evidence = TerminalEvidence()
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

    @property
    def input_token_count(self) -> int:
        return len(self._compiled_prompt.input_ids)

    @property
    def commit_class(self) -> SemanticCommitClass:
        return self._commit_class

    @property
    def terminal_decision(self) -> TerminalDecision | None:
        return self._terminal_evidence.decision

    def _observe_semantic_commit(self, event: GenerationEvent) -> None:
        if isinstance(event, ToolCallCompleted):
            self._commit_class = SemanticCommitClass.TOOL_COMPLETED
            return
        if self._commit_class is SemanticCommitClass.TOOL_COMPLETED:
            return
        if isinstance(event, ToolCallStarted | ToolCallArgumentsDelta):
            self._commit_class = SemanticCommitClass.PARTIAL_TOOL_COMMITTED
            return
        if self._commit_class is SemanticCommitClass.PARTIAL_TOOL_COMMITTED:
            return
        if isinstance(event, TextDelta | ReasoningDelta):
            self._commit_class = SemanticCommitClass.CONTENT_COMMITTED
            return

    def _queue_event(self, event: GenerationEvent) -> None:
        self._observe_semantic_commit(event)
        self._pending.append(event)

    def _queue_events(self, events: tuple[GenerationEvent, ...]) -> None:
        for event in events:
            self._queue_event(event)

    def _committed_error(self, error: CanonicalError) -> CanonicalError:
        return commit_aware_error(error, self._commit_class)

    def _record_controlled_terminal_reason(self) -> None:
        self._terminal_evidence.record_controlled_reason(self._controlled.terminal_reason)

    def _emit_recorded_failure_or_cancellation(self) -> TerminalDecision:
        decision = self._terminal_evidence.resolve(error_transform=self._committed_error)
        self._abort_tool_batch()
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
        return decision

    def _tool_constraint_guarantee(self, tool_name: str) -> ToolConstraintGuarantee:
        if self._tool_constraint is None:
            return ToolConstraintGuarantee.NONE
        return self._tool_constraint.guarantee_for_tool(tool_name)

    def _tool_requires_schema_guarantee(self, tool_name: str) -> bool:
        for tool in self._tool_policy.tools:
            if tool.name == tool_name:
                return tool.strict
        return False

    def _fail_tool_constraint_unavailable(self) -> None:
        if self._terminal:
            return
        error = self._committed_error(
            CanonicalError(
                ErrorCategory.INVALID_REQUEST,
                "tool_constraint_unsupported",
                "Requested strict Tool generation guarantee was not established by the selected model/runtime constraint path.",
                False,
            )
        )
        self._record_controlled_terminal_reason()
        self._terminal_evidence.record_semantic_failure(error)
        self._terminal_evidence.record_constraint_failure(error)
        self._emit_recorded_failure_or_cancellation()

    async def _reasoning_budget_failure(self, code: str, message: str) -> None:
        if self._terminal:
            return
        error = self._committed_error(_safe_error(ErrorCategory.RUNTIME_FAILURE, code, message))
        self._record_controlled_terminal_reason()
        self._terminal_evidence.record_unknown_failure(error)
        decision = self._emit_recorded_failure_or_cancellation()
        self._reasoning_budget_state = _ReasoningBudgetState.DONE
        if decision.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL:
            try:
                await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
            except Exception:
                logger.exception(
                    "runtime cancellation failed after reasoning-budget failure request_id=%s",
                    self._request_id,
                )

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

    def _abort_tool_batch(self) -> None:
        self._tool_batch.abort()

    def _commit_tool_batch(self) -> None:
        self._queue_events(self._tool_batch.commit_events())

    async def cancel(self) -> None:
        if self._terminal:
            return
        self._abort_tool_batch()
        await self._controlled.cancel(RequestTerminalReason.CLIENT_CANCELLED)

    def _fail_structured_constraint_unavailable(self, message: str) -> None:
        if self._terminal:
            return
        error = self._committed_error(
            CanonicalError(
                ErrorCategory.INVALID_REQUEST,
                "structured_output_constraint_unsupported",
                message,
                False,
            )
        )
        self._record_controlled_terminal_reason()
        self._terminal_evidence.record_semantic_failure(error)
        self._terminal_evidence.record_constraint_failure(error)
        self._emit_recorded_failure_or_cancellation()

    async def _model_failure(
        self,
        code: str,
        message: str,
        *,
        cause: FailureCause | None = None,
    ) -> None:
        if self._terminal:
            return
        error = self._committed_error(
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                code,
                message,
                False,
                cause,
            )
        )
        self._record_controlled_terminal_reason()
        self._terminal_evidence.record_semantic_failure(error)
        if cause is FailureCause.CONSTRAINT_FAILURE:
            self._terminal_evidence.record_constraint_failure(error)
        decision = self._emit_recorded_failure_or_cancellation()
        if decision.primary_owner in {
            TerminalPrimaryOwner.CONSTRAINT_INTEGRITY,
            TerminalPrimaryOwner.SEMANTIC_CONTRACT,
        }:
            try:
                await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
            except Exception:
                logger.exception(
                    "runtime cancellation failed after local model failure request_id=%s",
                    self._request_id,
                )

    async def _process_semantic(self, event: GenerationEvent) -> None:
        if self._terminal:
            return
        if isinstance(event, TextDelta):
            self._text_parts.append(event.text)
            self._queue_event(event)
            return
        if isinstance(event, ToolCallStarted):
            decision = self._tool_batch.on_started(event)
            if decision.failure is not None:
                await self._model_failure(decision.failure.code, decision.failure.message)
                return
            self._queue_events(decision.events)
            return
        if isinstance(event, ToolCallArgumentsDelta):
            decision = self._tool_batch.on_arguments_delta(event)
            if decision.failure is not None:
                await self._model_failure(decision.failure.code, decision.failure.message)
                return
            self._queue_events(decision.events)
            return
        if isinstance(event, ToolCallCompleted):
            decision = self._tool_batch.on_completed(event)
            if decision.failure is not None:
                guarantee = self._tool_constraint_guarantee(event.call.name)
                cause = (
                    FailureCause.CONSTRAINT_FAILURE
                    if violates_tool_constraint_guarantee(decision.failure, guarantee)
                    else FailureCause.MODEL_TOOL_OUTPUT_INVALID
                    if is_model_tool_output_invalid(decision.failure, guarantee)
                    else None
                )
                await self._model_failure(
                    decision.failure.code,
                    decision.failure.message,
                    cause=cause,
                )
                return
            self._queue_events(decision.events)
            return
        self._queue_event(event)

    async def _parser_integrity_failure(
        self,
        *,
        category: ErrorCategory,
        code: str,
        message: str,
        issue: str,
    ) -> None:
        if self._terminal:
            return
        error = self._committed_error(_safe_error(category, code, message, retryable=False))
        self._record_controlled_terminal_reason()
        self._terminal_evidence.record_parser_issue(issue)
        self._terminal_evidence.record_parser_integrity_failure(error)
        decision = self._emit_recorded_failure_or_cancellation()
        if decision.primary_owner is TerminalPrimaryOwner.PARSER_INTEGRITY:
            try:
                await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
            except Exception:
                logger.exception(
                    "runtime cancellation failed after parser integrity failure request_id=%s",
                    self._request_id,
                )

    async def _fail_native_token_provenance(self) -> None:
        await self._parser_integrity_failure(
            category=ErrorCategory.RUNTIME_FAILURE,
            code="output_token_provenance_unavailable",
            message="Inference output provenance was insufficient to classify a structural marker safely.",
            issue="native_token_provenance",
        )

    async def _finish_parser_events(self) -> ParserTerminalIssue | None:
        if self._parser_finished:
            return None
        self._parser_finished = True
        try:
            finish = self._parser.finish()
        except NativeTokenProvenanceError:
            await self._fail_native_token_provenance()
            return None
        except Exception:
            logger.exception("parser finish failed request_id=%s", self._request_id)
            await self._parser_integrity_failure(
                category=ErrorCategory.INTERNAL,
                code="parser_finish_failed",
                message="Incremental parser failed while finalizing model output.",
                issue="parser_finish_exception",
            )
            return None
        terminal_issue = finish.terminal_issue
        if terminal_issue is not None:
            detail = (
                terminal_issue.kind.value
                if terminal_issue.ambiguity_detail is None
                else f"{terminal_issue.kind.value}:{terminal_issue.ambiguity_detail.value}"
            )
            self._terminal_evidence.record_parser_issue(detail)
        for event in finish.events:
            await self._process_semantic(event)
            if self._terminal:
                return terminal_issue
        return terminal_issue

    def _early_parser_terminal_issue(self) -> ParserTerminalIssue | None:
        if not isinstance(self._parser, NativeTokenAwareIncrementalParser):
            return None
        issue = self._parser.early_terminal_issue
        if issue is not None and not isinstance(issue, ParserTerminalIssue):
            raise TypeError("early_terminal_issue must be a ParserTerminalIssue or None")
        return issue

    async def _handle_parser_terminal_issue(
        self,
        issue: ParserTerminalIssue,
        *,
        runtime_event: RuntimeFinished | None = None,
    ) -> None:
        detail_text = (
            issue.kind.value
            if issue.ambiguity_detail is None
            else f"{issue.kind.value}:{issue.ambiguity_detail.value}"
        )
        self._terminal_evidence.record_parser_issue(detail_text)
        if issue.kind is ParserTerminalIssueKind.INCOMPLETE_TOOL:
            if runtime_event is None:
                raise RuntimeError("incomplete Tool terminal issue requires a runtime terminal event")
            constraint_integrity_active = (
                runtime_event.hard_constraint_installed
                and runtime_event.hard_constraint_activated
                and runtime_event.effective_generation_guarantee
                in {GenerationGuarantee.FORMAT, GenerationGuarantee.SCHEMA}
            )
            cause = (
                FailureCause.OUTPUT_LENGTH
                if runtime_event.reason is RuntimeStopReason.LENGTH
                else FailureCause.CONSTRAINT_FAILURE
                if runtime_event.reason in {RuntimeStopReason.EOS, RuntimeStopReason.FILTER}
                and constraint_integrity_active
                else FailureCause.OUTPUT_EOS
                if runtime_event.reason is RuntimeStopReason.EOS
                else None
            )
            message = "Model output ended with an incomplete tool call."
            if runtime_event.reason is RuntimeStopReason.LENGTH:
                message = "Model output reached the output token limit with an incomplete tool call."
            await self._model_failure("tool_call_incomplete", message, cause=cause)
            return

        if issue.kind is not ParserTerminalIssueKind.PROTOCOL_AMBIGUITY:
            raise RuntimeError(f"unsupported parser terminal issue: {issue.kind.value}")
        detail = issue.ambiguity_detail
        if detail is ParserAmbiguityDetail.HOLD_LIMIT:
            cause = FailureCause.PARSER_AMBIGUITY_LIMIT
            message = "Model output exceeded the bounded protocol-ambiguity hold limit."
        else:
            constraint_integrity_active = (
                runtime_event is not None
                and runtime_event.hard_constraint_installed
                and runtime_event.hard_constraint_activated
                and runtime_event.effective_generation_guarantee
                in {GenerationGuarantee.FORMAT, GenerationGuarantee.SCHEMA}
            )
            reason = None if runtime_event is None else runtime_event.reason
            cause = (
                FailureCause.OUTPUT_LENGTH
                if reason is RuntimeStopReason.LENGTH
                else FailureCause.CONSTRAINT_FAILURE
                if reason in {RuntimeStopReason.EOS, RuntimeStopReason.FILTER}
                and constraint_integrity_active
                else FailureCause.OUTPUT_EOS
                if reason is RuntimeStopReason.EOS
                else None
            )
            message = "Model output ended with unresolved protocol ambiguity."
        await self._model_failure("protocol_ambiguity", message, cause=cause)

    async def _handle_runtime_finished(self, event: RuntimeFinished) -> None:
        self._record_controlled_terminal_reason()
        self._terminal_evidence.record_runtime_finished(event)

        terminal_issue = await self._finish_parser_events()
        if self._terminal:
            return
        if terminal_issue is not None:
            await self._handle_parser_terminal_issue(terminal_issue, runtime_event=event)
            return
        if self._terminal_evidence.causal_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION:
            self._emit_recorded_failure_or_cancellation()
            return

        hard_constraint_active = (
            event.hard_constraint_installed and event.hard_constraint_activated
        )

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

        if completed_calls:
            for call in completed_calls:
                if not self._tool_requires_schema_guarantee(call.name):
                    continue
                branch_guarantee = self._tool_constraint_guarantee(call.name)
                if (
                    not hard_constraint_active
                    or not guarantee_satisfies(
                        branch_guarantee,
                        GenerationGuarantee.SCHEMA,
                    )
                ):
                    self._fail_tool_constraint_unavailable()
                    return

        if not completed_calls and self._structured_output is not None:
            requested_guarantee = self._structured_output.requested_guarantee
            if (
                requested_guarantee is not GenerationGuarantee.NONE
                and not guarantee_satisfies(
                    event.effective_generation_guarantee,
                    requested_guarantee,
                )
            ):
                self._fail_structured_constraint_unavailable(
                    "Requested structured-output generation guarantee was not activated by the selected model/runtime constraint path."
                )
                return
            structured = validate_structured_output("".join(self._text_parts), self._structured_output)
            if not structured.is_valid:
                constraint_contradiction = (
                    hard_constraint_active
                    and violates_structured_constraint_guarantee(
                        structured,
                        event.effective_generation_guarantee,
                    )
                )
                await self._model_failure(
                    "structured_output_invalid",
                    "Model output did not satisfy the requested structured-output schema.",
                    cause=(
                        FailureCause.CONSTRAINT_FAILURE
                        if constraint_contradiction
                        else None
                    ),
                )
                return

        success_reason = CompletionReason.TOOL_CALLS if completed_calls else None
        decision = self._terminal_evidence.resolve(success_reason=success_reason)
        if decision.disposition is not TerminalDisposition.SUCCESS:
            self._emit_recorded_failure_or_cancellation()
            return
        reason = decision.completion_reason
        if reason is None:  # pragma: no cover - TerminalDecision validates success.
            raise RuntimeError("successful terminal decision is missing completion reason")

        timing_event = timing_event_from_runtime(self._request_id, event.timing)
        usage_event = UsageUpdated(self._request_id, event.usage)
        exposed_stop_sequence = (
            event.stop_sequence
            if reason is CompletionReason.STOP
            and event.stop_sequence in self._requested_stop_sequences
            else None
        )
        completed_event = GenerationCompleted(
            self._request_id,
            reason,
            event.usage,
            exposed_stop_sequence,
        )

        pending_checkpoint = len(self._pending)
        commit_class_checkpoint = self._commit_class
        try:
            # These events remain local to this session until __anext__ returns.
            # If authority commit fails, roll the local publication queue back before
            # the normal UNKNOWN_INTERNAL fallback is allowed to terminate the request.
            self._commit_tool_batch()
            if timing_event is not None:
                self._pending.append(timing_event)
            self._pending.append(usage_event)
            self._pending.append(completed_event)
            self._terminal_evidence.commit_decision(decision)
        except Exception:
            while len(self._pending) > pending_checkpoint:
                self._pending.pop()
            self._commit_class = commit_class_checkpoint
            raise
        self._terminal = True

    async def _handle_runtime_failure(self, event: RuntimeFailed) -> None:
        self._record_controlled_terminal_reason()
        self._terminal_evidence.record_runtime_failure(self._committed_error(event.error))
        await self._finish_parser_events()
        if self._terminal:
            return
        self._emit_recorded_failure_or_cancellation()

    async def _handle_runtime_cancelled(self) -> None:
        self._record_controlled_terminal_reason()
        if self._terminal_evidence.causal_owner is not TerminalPrimaryOwner.LIFECYCLE_TERMINATION:
            self._terminal_evidence.record_runtime_cancelled()
        await self._finish_parser_events()
        if self._terminal:
            return
        self._emit_recorded_failure_or_cancellation()

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
            except Exception:
                logger.exception("parser feed failed request_id=%s", self._request_id)
                await self._parser_integrity_failure(
                    category=ErrorCategory.INTERNAL,
                    code="parser_feed_failed",
                    message="Incremental parser failed while consuming model output.",
                    issue="parser_feed_exception",
                )
                return
            try:
                early_terminal_issue = self._early_parser_terminal_issue()
            except Exception:
                logger.exception("parser terminal evidence failed request_id=%s", self._request_id)
                await self._parser_integrity_failure(
                    category=ErrorCategory.INTERNAL,
                    code="parser_terminal_evidence_failed",
                    message="Incremental parser exposed invalid terminal evidence.",
                    issue="parser_terminal_evidence_exception",
                )
                return
            if early_terminal_issue is not None:
                for semantic in semantic_events:
                    await self._process_semantic(semantic)
                    if self._terminal:
                        return
                await self._handle_parser_terminal_issue(early_terminal_issue)
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
                self._record_controlled_terminal_reason()
                error = self._committed_error(
                    _safe_error(
                        ErrorCategory.RUNTIME_FAILURE,
                        "runtime_stream_ended",
                        "Inference runtime stream ended without a terminal event.",
                    )
                )
                self._terminal_evidence.record_runtime_failure(error)
                await self._finish_parser_events()
                if self._pending:
                    continue
                if not self._terminal:
                    self._emit_recorded_failure_or_cancellation()
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("runtime stream iteration failed request_id=%s", self._request_id)
                self._record_controlled_terminal_reason()
                error = self._committed_error(
                    _safe_error(
                        ErrorCategory.INTERNAL,
                        "runtime_stream_exception",
                        "Inference runtime stream failed without a typed terminal event.",
                    )
                )
                self._terminal_evidence.record_unknown_failure(error)
                self._emit_recorded_failure_or_cancellation()
                continue

            try:
                await self._process_runtime(runtime_event)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("serving terminal processing failed request_id=%s", self._request_id)
                if not self._terminal:
                    self._record_controlled_terminal_reason()
                    self._terminal_evidence.record_unknown_failure(
                        self._committed_error(
                            _safe_error(
                                ErrorCategory.INTERNAL,
                                "terminal_processing_failed",
                                "Inference terminal processing failed unexpectedly.",
                            )
                        )
                    )
                    self._emit_recorded_failure_or_cancellation()
