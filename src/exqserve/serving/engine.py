"""Protocol-independent serving orchestration and runtime/model bridges."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Protocol, Self

from exqserve.agent._json import InvalidJsonError, parse_json_strict
from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.structured_output import StructuredOutputSpec, validate_structured_output
from exqserve.agent.tools import ToolPolicy
from exqserve.agent.validation import (
    ValidationCode,
    ValidationResult,
    validate_tool_calls,
    validate_tool_history,
)
from exqserve.control.request import RequestRejected, RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import ToolCallItem
from exqserve.model.contracts import (
    CompiledPrompt,
    NativeTokenAwareIncrementalParser,
    NativeTokenProvenanceError,
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

logger = logging.getLogger(__name__)

_TOOL_CALL_INVALID_CODES = frozenset(
    {
        ValidationCode.INVALID_JSON,
        ValidationCode.DUPLICATE_JSON_KEY,
        ValidationCode.JSON_VALUE_NOT_OBJECT,
        ValidationCode.SCHEMA_VALIDATION_FAILED,
        ValidationCode.DUPLICATE_TOOL_CALL_ID,
        ValidationCode.INVALID_TOOL_CALL_ORDER,
    }
)


def _tool_validation_failure(result: ValidationResult) -> tuple[str, str]:
    if result.is_valid:
        raise ValueError("tool validation result must contain at least one issue")
    if any(issue.code in _TOOL_CALL_INVALID_CODES for issue in result.issues):
        return "tool_call_invalid", "Model produced an invalid tool call."
    return "tool_policy_violation", "Model output violated the requested tool policy."


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
    ) -> None:
        if not isinstance(tool_call_fanout_limit, int) or isinstance(tool_call_fanout_limit, bool):
            raise TypeError("tool_call_fanout_limit must be an integer")
        if tool_call_fanout_limit <= 0:
            raise ValueError("tool_call_fanout_limit must be positive")
        self._compiler = compiler
        self._parser_factory = parser_factory
        self._controller = controller
        self._tool_constraint_factory = tool_constraint_factory
        self._tool_call_fanout_limit = tool_call_fanout_limit
        self._compile_lock = asyncio.Lock()

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

        runtime_request = RuntimeGenerationRequest(
            request_id=request.input.request_id,
            input_ids=compiled.input_ids,
            max_new_tokens=request.max_output_tokens,
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

        return ServingSession(
            request.input.request_id,
            controlled,
            parser,
            compiled,
            request.tools,
            request.structured_output,
            request.stop_conditions,
            self._tool_call_fanout_limit,
        )


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
    ) -> None:
        if not isinstance(tool_call_fanout_limit, int) or isinstance(tool_call_fanout_limit, bool):
            raise TypeError("tool_call_fanout_limit must be an integer")
        if tool_call_fanout_limit <= 0:
            raise ValueError("tool_call_fanout_limit must be positive")
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
        self._tool_call_fanout_limit = tool_call_fanout_limit
        self._pending: deque[GenerationEvent] = deque()
        self._terminal = False
        self._parser_finished = False
        self._text_parts: list[str] = []
        self._accepted_call_ids: set[str] = set()
        self._completed_calls: list[ToolCallItem] = []
        self._runtime_trace: list[dict[str, object]] | None = None

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

    async def cancel(self) -> None:
        if self._terminal:
            return
        await self._controlled.cancel(RequestTerminalReason.CLIENT_CANCELLED)

    async def _model_failure(self, code: str, message: str) -> None:
        if self._terminal:
            return
        await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
        self._pending.append(
            GenerationFailed(
                self._request_id,
                _safe_error(ErrorCategory.MODEL_FAILURE, code, message),
            )
        )
        self._terminal = True

    async def _process_semantic(self, event: GenerationEvent) -> None:
        if self._terminal:
            return
        if isinstance(event, TextDelta):
            self._text_parts.append(event.text)
            self._pending.append(event)
            return
        if isinstance(event, ToolCallStarted):
            if event.call_id in self._accepted_call_ids:
                await self._model_failure(
                    "tool_call_stream_invalid",
                    "Model produced a duplicate tool-call start event.",
                )
                return
            if len(self._accepted_call_ids) >= self._tool_call_fanout_limit:
                logger.warning(
                    "model exceeded tool-call fanout limit request_id=%s limit=%d",
                    self._request_id,
                    self._tool_call_fanout_limit,
                )
                await self._model_failure(
                    "tool_policy_violation",
                    "Model output exceeded the server tool-call policy.",
                )
                return
            self._accepted_call_ids.add(event.call_id)
            self._pending.append(event)
            return
        if isinstance(event, ToolCallArgumentsDelta):
            if event.call_id not in self._accepted_call_ids:
                await self._model_failure(
                    "tool_call_stream_invalid",
                    "Model produced tool arguments before an accepted tool call start.",
                )
                return
            self._pending.append(event)
            return
        if isinstance(event, ToolCallCompleted):
            if event.call.call_id not in self._accepted_call_ids:
                await self._model_failure(
                    "tool_call_stream_invalid",
                    "Model completed a tool call that was not accepted.",
                )
                return
            candidate_calls = (*self._completed_calls, event.call)
            validation = validate_tool_calls(candidate_calls, self._tool_policy)
            if not validation.is_valid:
                logger.warning(
                    "model produced invalid completed tool call request_id=%s issues=%s",
                    self._request_id,
                    ",".join(issue.code.value for issue in validation.issues),
                )
                code, message = _tool_validation_failure(validation)
                await self._model_failure(code, message)
                return
            self._completed_calls.append(event.call)
            self._pending.append(event)
            return
        self._pending.append(event)

    async def _fail_native_token_provenance(self) -> None:
        await self._controlled.cancel(RequestTerminalReason.APPLICATION_CANCELLED)
        self._pending.append(
            GenerationFailed(
                self._request_id,
                _safe_error(
                    ErrorCategory.RUNTIME_FAILURE,
                    "output_token_provenance_unavailable",
                    "Inference output provenance was insufficient to classify a Qwen structural marker safely.",
                    retryable=True,
                ),
            )
        )
        self._terminal = True

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
            await self._model_failure(
                "tool_call_incomplete",
                "Model output ended with an incomplete tool call.",
            )
            return

        final_tool_validation = validate_tool_calls(tuple(self._completed_calls), self._tool_policy)
        if not final_tool_validation.is_valid:
            logger.warning(
                "model output violates requested tool policy request_id=%s issues=%s",
                self._request_id,
                ",".join(issue.code.value for issue in final_tool_validation.issues),
            )
            code, message = _tool_validation_failure(final_tool_validation)
            await self._model_failure(code, message)
            return

        if not incomplete_tool and not self._completed_calls and self._structured_output is not None:
            structured = validate_structured_output("".join(self._text_parts), self._structured_output)
            if not structured.is_valid:
                await self._model_failure(
                    "structured_output_invalid",
                    "Model output did not satisfy the requested structured-output schema.",
                )
                return

        reason = (
            CompletionReason.TOOL_CALLS
            if self._completed_calls
            else completion_reason_from_runtime(event.reason)
        )

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
        self._pending.append(GenerationFailed(self._request_id, event.error))
        self._terminal = True

    async def _handle_runtime_cancelled(self) -> None:
        await self._finish_parser_events()
        if self._terminal:
            return
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
