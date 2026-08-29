"""Gemma 4 model dialect: HF prompt compilation and native Agent stream parsing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

from exqserve.agent._json import (
    InvalidJsonError,
    JsonValue,
    canonical_json_dumps,
    parse_json_strict,
)
from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.tools import FunctionTool, ToolChoiceMode, ToolPolicy
from exqserve.core.events import (
    GenerationEvent,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import (
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import (
    ModelCapabilities,
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
    TemplateToolResponse,
    ToolConstraintMode,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
)
from exqserve.model.hf_template import HFTemplatePromptCompiler
from exqserve.model.tool_constraints import (
    constraint_schema,
    exposed_tools,
    lark_literal,
    schema_lark,
)

GEMMA4_CAPABILITIES = ModelCapabilities(
    reasoning=True,
    tool_calling=True,
    parallel_tool_calls=True,
    system_role=True,
    developer_role=False,
    reasoning_history=True,
    vision=True,
)

_GEMMA4_STOP_CONDITIONS = ("<turn|>", "<|tool_response>")
_THOUGHT_OPEN = "<|channel>thought\n"
_THOUGHT_CLOSE = "<channel|>"
_TOOL_OPEN = "<|tool_call>"
_TOOL_CLOSE = "<tool_call|>"
_STRING_DELIMITER = '<|"|>'
_PLAIN_MARKERS = (_THOUGHT_OPEN, _THOUGHT_CLOSE, _TOOL_OPEN)


def gemma4_tool_constraint(
    tool_policy: ToolPolicy,
    mode: ToolConstraintMode,
) -> ToolGenerationConstraint | None:
    if not isinstance(mode, ToolConstraintMode):
        raise TypeError("mode must be a ToolConstraintMode")
    if mode is ToolConstraintMode.OFF:
        return None

    tools = tuple(sorted(exposed_tools(tool_policy), key=lambda item: item.name))
    if not tools:
        return None

    lines = ["%llguidance {}", "start: WS? tool WS?"]
    lines.append("tool: " + " | ".join(f"tool_{index}" for index in range(len(tools))))
    for index, tool in enumerate(tools):
        if tool.name != tool.name.strip() or "{" in tool.name:
            raise ToolConstraintUnsupported(
                f"Gemma constrained generation cannot represent tool name {tool.name!r}"
            )
        schema: dict[str, JsonValue] = (
            {"type": "object"}
            if mode is ToolConstraintMode.FORMAT
            else constraint_schema(tool.parameters)
        )
        lines.append(
            f"tool_{index}: {lark_literal(f'call:{tool.name}')} WS? "
            f'{schema_lark(schema)} WS? "<tool_call|>"'
        )
    lines.append("WS: /[ \\t\\r\\n]+/")
    return ToolGenerationConstraint(
        trigger=_TOOL_OPEN,
        lark_grammar="\n".join(lines),
        eos_after_completed=not tool_policy.allow_parallel,
    )


def _reasoning_kwargs(
    policy: ReasoningPolicy,
    *,
    preserve_thinking: bool,
) -> tuple[tuple[str, bool], ...]:
    values: dict[str, bool] = {}
    if policy.mode is ReasoningMode.ENABLED:
        values["enable_thinking"] = True
    elif policy.mode is ReasoningMode.DISABLED:
        values["enable_thinking"] = False
    if preserve_thinking:
        values["preserve_thinking"] = True
    return tuple(sorted(values.items()))


def _exposed_tools(policy: ToolPolicy) -> tuple[TemplateTool, ...]:
    selected: tuple[FunctionTool, ...]
    if policy.choice.mode is ToolChoiceMode.NONE:
        selected = ()
    elif policy.choice.mode is ToolChoiceMode.NAMED:
        selected = tuple(tool for tool in policy.tools if tool.name == policy.choice.name)
    else:
        selected = policy.tools
    return tuple(
        TemplateTool(
            name=tool.name,
            description=tool.description,
            parameters_json=tool.parameters.canonical_json,
        )
        for tool in sorted(selected, key=lambda item: item.name)
    )


def _tool_response_json(text: str) -> str:
    try:
        value = parse_json_strict(text)
    except InvalidJsonError:
        value = text
    return canonical_json_dumps(value)


class Gemma4PromptCompiler(HFTemplatePromptCompiler):
    """Compile canonical Agent history through Gemma 4's official HF chat template."""

    capabilities = GEMMA4_CAPABILITIES
    stop_conditions = _GEMMA4_STOP_CONDITIONS

    def prepare(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> TemplateRequest:
        if not isinstance(request, CanonicalRequest):
            raise TypeError("request must be a CanonicalRequest")
        if not isinstance(reasoning, ReasoningPolicy):
            raise TypeError("reasoning must be a ReasoningPolicy")
        if not isinstance(tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy")

        messages: list[TemplateMessage] = []
        items = request.items
        position = 0
        leading_instructions: list[str] = []
        while position < len(items):
            item = items[position]
            if not isinstance(item, MessageItem) or item.role not in {
                MessageRole.SYSTEM,
                MessageRole.DEVELOPER,
            }:
                break
            leading_instructions.append(item.text)
            position += 1

        if leading_instructions:
            messages.append(TemplateMessage("system", "\n\n".join(leading_instructions)))

        reasoning_parts: list[str] = []
        assistant_text: str | None = None
        assistant_calls: list[TemplateToolCall] = []
        assistant_call_ids: list[str] = []
        assistant_responses: list[TemplateToolResponse] = []
        responded_call_ids: set[str] = set()
        known_calls: dict[str, str] = {}
        has_reasoning_history = any(isinstance(item, ReasoningItem) for item in items)

        def flush_assistant() -> None:
            nonlocal reasoning_parts, assistant_text, assistant_calls, assistant_call_ids
            nonlocal assistant_responses, responded_call_ids
            if (
                not reasoning_parts
                and assistant_text is None
                and not assistant_calls
                and not assistant_responses
            ):
                return
            messages.append(
                TemplateMessage(
                    role="assistant",
                    content=assistant_text or "",
                    reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
                    tool_calls=tuple(assistant_calls),
                    tool_responses=tuple(assistant_responses),
                )
            )
            reasoning_parts = []
            assistant_text = None
            assistant_calls = []
            assistant_call_ids = []
            assistant_responses = []
            responded_call_ids = set()

        for item in items[position:]:
            if isinstance(item, MessageItem):
                if item.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                    raise ValueError("Gemma 4 system/developer messages must appear at the beginning")
                if item.role is MessageRole.USER:
                    flush_assistant()
                    messages.append(TemplateMessage("user", item.text))
                    continue
                if item.role is MessageRole.ASSISTANT:
                    if assistant_responses:
                        flush_assistant()
                    assistant_text = item.text if assistant_text is None else assistant_text + item.text
                    continue
                raise ValueError(f"unsupported Gemma 4 message role: {item.role.value}")

            if isinstance(item, MultimodalMessageItem):
                flush_assistant()
                multimodal_parts: list[TemplateTextPart | TemplateImagePart] = []
                for part in item.parts:
                    if isinstance(part, TextContentPart):
                        multimodal_parts.append(TemplateTextPart(part.text))
                    elif isinstance(part, ImageContentPart):
                        multimodal_parts.append(TemplateImagePart(part.source, part.detail))
                    else:  # pragma: no cover - canonical validation prevents this
                        raise TypeError(f"unsupported multimodal part: {type(part).__name__}")
                messages.append(TemplateMessage("user", tuple(multimodal_parts)))
                continue

            if isinstance(item, ReasoningItem):
                if assistant_responses:
                    flush_assistant()
                if assistant_text is not None or assistant_calls:
                    raise ValueError("assistant reasoning must precede assistant text and tool calls")
                # Newer official Gemma 4 templates preserve reasoning history while
                # older model-bundled templates ignore/strip this field. Supplying it
                # keeps one canonical history compatible with both template revisions.
                reasoning_parts.append(item.text)
                continue

            if isinstance(item, ToolCallItem):
                if assistant_responses:
                    flush_assistant()
                if item.call_id in known_calls:
                    raise ValueError(f"duplicate tool call id in history: {item.call_id!r}")
                if item.index != len(assistant_calls):
                    raise ValueError("tool call index must match order within the assistant turn")
                known_calls[item.call_id] = item.name
                assistant_call_ids.append(item.call_id)
                assistant_calls.append(TemplateToolCall(item.name, item.arguments_json))
                continue

            if isinstance(item, ToolResultItem):
                tool_name = known_calls.get(item.call_id)
                if tool_name is None:
                    raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
                if item.call_id not in assistant_call_ids:
                    raise ValueError("Gemma 4 tool results must follow their assistant tool call turn")
                if item.call_id in responded_call_ids:
                    raise ValueError(f"duplicate tool result for call id: {item.call_id!r}")
                responded_call_ids.add(item.call_id)
                assistant_responses.append(
                    TemplateToolResponse(tool_name, _tool_response_json(item.text))
                )
                continue

            if isinstance(item, MultimodalToolResultItem):
                raise TypeError("Gemma 4 multimodal tool results are not yet supported")

            raise TypeError(f"unsupported canonical item: {type(item).__name__}")

        flush_assistant()
        return TemplateRequest(
            messages=tuple(messages),
            tools=_exposed_tools(tool_policy),
            template_kwargs=_reasoning_kwargs(
                reasoning,
                preserve_thinking=has_reasoning_history,
            ),
        )

    def _raw_output_is_text_only(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> bool:
        del tool_policy
        return reasoning.mode is not ReasoningMode.ENABLED and not template_request.tools

    def _structured_output_trigger(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> str | None:
        del tool_policy
        if reasoning.mode is ReasoningMode.ENABLED and not template_request.tools:
            return _THOUGHT_CLOSE
        return None


@dataclass(frozen=True, slots=True)
class Gemma4ParserFinish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise TypeError("events must be a tuple")
        if not isinstance(self.incomplete_tool_call, bool):
            raise TypeError("incomplete_tool_call must be a bool")


def _deterministic_call_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0gemma4\0{index}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _longest_partial_marker_suffix(text: str) -> int:
    longest = 0
    for marker in _PLAIN_MARKERS:
        limit = min(len(text), len(marker) - 1)
        for size in range(1, limit + 1):
            if marker.startswith(text[-size:]):
                longest = max(longest, size)
    return longest


class _GemmaArgumentReader:
    def __init__(self, text: str) -> None:
        self._text = text
        self._position = 0

    def parse(self) -> dict[str, JsonValue]:
        value = self._parse_object()
        self._skip_space()
        if self._position != len(self._text):
            raise ValueError("unexpected trailing Gemma 4 tool-call argument text")
        return value

    def _skip_space(self) -> None:
        while self._position < len(self._text) and self._text[self._position].isspace():
            self._position += 1

    def _consume(self, literal: str) -> None:
        if not self._text.startswith(literal, self._position):
            raise ValueError(f"expected {literal!r} in Gemma 4 tool-call arguments")
        self._position += len(literal)

    def _parse_string(self) -> str:
        self._consume(_STRING_DELIMITER)
        end = self._text.find(_STRING_DELIMITER, self._position)
        if end < 0:
            raise ValueError("unterminated Gemma 4 tool-call string")
        value = self._text[self._position : end]
        self._position = end + len(_STRING_DELIMITER)
        return value

    def _parse_key(self) -> str:
        self._skip_space()
        if self._text.startswith(_STRING_DELIMITER, self._position):
            key = self._parse_string()
        else:
            colon = self._text.find(":", self._position)
            if colon < 0:
                raise ValueError("Gemma 4 tool-call object key is missing ':'")
            key = self._text[self._position : colon].strip()
            self._position = colon
        if not key:
            raise ValueError("Gemma 4 tool-call object key must not be empty")
        self._consume(":")
        return key

    def _parse_object(self) -> dict[str, JsonValue]:
        self._skip_space()
        self._consume("{")
        result: dict[str, JsonValue] = {}
        self._skip_space()
        if self._text.startswith("}", self._position):
            self._position += 1
            return result
        while True:
            key = self._parse_key()
            if key in result:
                raise ValueError(f"duplicate Gemma 4 tool-call argument key: {key!r}")
            result[key] = self._parse_value()
            self._skip_space()
            if self._text.startswith("}", self._position):
                self._position += 1
                return result
            self._consume(",")

    def _parse_array(self) -> list[JsonValue]:
        self._skip_space()
        self._consume("[")
        result: list[JsonValue] = []
        self._skip_space()
        if self._text.startswith("]", self._position):
            self._position += 1
            return result
        while True:
            result.append(self._parse_value())
            self._skip_space()
            if self._text.startswith("]", self._position):
                self._position += 1
                return result
            self._consume(",")

    def _parse_scalar(self) -> JsonValue:
        start = self._position
        while self._position < len(self._text):
            if self._text[self._position] in ",}]":
                break
            self._position += 1
        token = self._text[start : self._position].strip()
        if not token:
            raise ValueError("empty Gemma 4 tool-call scalar")
        try:
            return parse_json_strict(token)
        except InvalidJsonError as exc:
            raise ValueError("invalid Gemma 4 tool-call scalar") from exc

    def _parse_value(self) -> JsonValue:
        self._skip_space()
        if self._position >= len(self._text):
            raise ValueError("missing Gemma 4 tool-call value")
        if self._text.startswith(_STRING_DELIMITER, self._position):
            return self._parse_string()
        character = self._text[self._position]
        if character == "{":
            return self._parse_object()
        if character == "[":
            return self._parse_array()
        return self._parse_scalar()


def _parse_tool_body(body: str) -> tuple[str, str]:
    body = body.lstrip()
    if not body.startswith("call:"):
        raise ValueError("Gemma 4 tool call must begin with 'call:'")
    object_start = body.find("{", len("call:"))
    if object_start < 0:
        raise ValueError("Gemma 4 tool call is missing an argument object")
    name = body[len("call:") : object_start].strip()
    if not name or any(character.isspace() or character in "{}<>" for character in name):
        raise ValueError("invalid Gemma 4 tool name")

    argument_text = body[object_start:]
    try:
        arguments = parse_json_strict(argument_text)
    except InvalidJsonError:
        arguments = _GemmaArgumentReader(argument_text).parse()
    return name, canonical_json_dumps(cast(dict[str, JsonValue], arguments))


class Gemma4IncrementalParser:
    """Convert Gemma 4 native channels/tool calls into canonical events incrementally."""

    def __init__(self, request_id: str, *, start_in_reasoning: bool = False) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(start_in_reasoning, bool):
            raise TypeError("start_in_reasoning must be a bool")
        self._request_id = request_id
        self._buffer = ""
        self._mode = "reasoning" if start_in_reasoning else "text"
        self._tool_return_mode = self._mode
        self._text_open = False
        self._text_value = ""
        self._reasoning_open = False
        self._reasoning_value = ""
        self._call_index = 0
        self._had_incomplete_tool = False
        self._finished = False

    def _emit_content(self, text: str, events: list[GenerationEvent]) -> None:
        if not text:
            return
        if self._mode == "reasoning":
            if not self._reasoning_open:
                events.append(ReasoningStarted(self._request_id))
                self._reasoning_open = True
                self._reasoning_value = ""
            self._reasoning_value += text
            events.append(ReasoningDelta(self._request_id, text))
            return
        if not self._text_open:
            events.append(TextStarted(self._request_id))
            self._text_open = True
            self._text_value = ""
        self._text_value += text
        events.append(TextDelta(self._request_id, text))

    def _close_channel(self, events: list[GenerationEvent]) -> None:
        if self._mode == "reasoning" and self._reasoning_open:
            events.append(ReasoningCompleted(self._request_id, self._reasoning_value))
            self._reasoning_open = False
            self._reasoning_value = ""
        elif self._mode == "text" and self._text_open:
            events.append(TextCompleted(self._request_id, self._text_value))
            self._text_open = False
            self._text_value = ""

    def _enter_tool(self, events: list[GenerationEvent]) -> None:
        self._close_channel(events)
        self._tool_return_mode = self._mode
        self._mode = "tool"

    def _process_plain(self, events: list[GenerationEvent]) -> bool:
        matches = [
            (position, marker)
            for marker in _PLAIN_MARKERS
            if (position := self._buffer.find(marker)) >= 0
        ]
        if matches:
            position, marker = min(matches, key=lambda item: item[0])
            self._emit_content(self._buffer[:position], events)
            self._buffer = self._buffer[position + len(marker) :]
            if marker == _TOOL_OPEN:
                self._enter_tool(events)
            elif marker == _THOUGHT_OPEN:
                if self._mode != "reasoning":
                    self._close_channel(events)
                    self._mode = "reasoning"
            elif self._mode == "reasoning":
                self._close_channel(events)
                self._mode = "text"
            return True

        held = _longest_partial_marker_suffix(self._buffer)
        safe_length = len(self._buffer) - held
        if safe_length > 0:
            self._emit_content(self._buffer[:safe_length], events)
            self._buffer = self._buffer[safe_length:]
        return False

    def _process_tool(self, events: list[GenerationEvent]) -> bool:
        close_at = self._buffer.find(_TOOL_CLOSE)
        if close_at < 0:
            return False
        body = self._buffer[:close_at]
        self._buffer = self._buffer[close_at + len(_TOOL_CLOSE) :]
        try:
            name, arguments_json = _parse_tool_body(body)
        except ValueError:
            self._had_incomplete_tool = True
        else:
            call_id = _deterministic_call_id(self._request_id, self._call_index)
            events.append(ToolCallStarted(self._request_id, call_id, name, self._call_index))
            events.append(
                ToolCallArgumentsDelta(
                    self._request_id,
                    call_id,
                    arguments_json,
                    self._call_index,
                )
            )
            events.append(
                ToolCallCompleted(
                    self._request_id,
                    ToolCallItem(call_id, name, arguments_json, self._call_index),
                )
            )
            self._call_index += 1
        self._mode = self._tool_return_mode
        return True

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Gemma 4 parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        self._buffer += chunk
        events: list[GenerationEvent] = []
        while True:
            progressed = (
                self._process_tool(events) if self._mode == "tool" else self._process_plain(events)
            )
            if not progressed:
                break
        return tuple(events)

    def finish(self) -> Gemma4ParserFinish:
        if self._finished:
            return Gemma4ParserFinish((), self._had_incomplete_tool)
        events: list[GenerationEvent] = []
        if self._mode == "tool":
            self._had_incomplete_tool = True
            self._buffer = ""
            self._mode = self._tool_return_mode
        elif self._buffer:
            if any(marker.startswith(self._buffer) for marker in _PLAIN_MARKERS):
                if _TOOL_OPEN.startswith(self._buffer):
                    self._had_incomplete_tool = True
            else:
                self._emit_content(self._buffer, events)
            self._buffer = ""
        self._close_channel(events)
        self._finished = True
        return Gemma4ParserFinish(tuple(events), self._had_incomplete_tool)
