"""GLM-5 model dialect: HF prompt compilation and GLM reasoning/tool parsing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    ReasoningItem,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import (
    ModelCapabilities,
    ParserTerminalIssue,
    TemplateMessage,
    TemplateRequest,
    TemplateTool,
    TemplateToolCall,
    incomplete_tool_terminal_issue,
)
from exqserve.model.hf_template import HFTemplatePromptCompiler

GLM5_CAPABILITIES = ModelCapabilities(
    reasoning=True,
    tool_calling=True,
    parallel_tool_calls=True,
    system_role=True,
    developer_role=False,
    reasoning_history=True,
    vision=False,
)

_GLM5_STOP_CONDITIONS = ("<|endoftext|>", "<|user|>", "<|observation|>")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_ARG_KEY_OPEN = "<arg_key>"
_ARG_KEY_CLOSE = "</arg_key>"
_ARG_VALUE_OPEN = "<arg_value>"
_ARG_VALUE_CLOSE = "</arg_value>"
_PLAIN_MARKERS = (_THINK_OPEN, _THINK_CLOSE, _TOOL_OPEN)


def _reasoning_kwargs(policy: ReasoningPolicy) -> tuple[tuple[str, bool], ...]:
    if policy.mode is ReasoningMode.ENABLED:
        return (("enable_thinking", True),)
    if policy.mode is ReasoningMode.DISABLED:
        return (("enable_thinking", False),)
    return ()


def _exposed_tools(policy: ToolPolicy) -> tuple[TemplateTool, ...]:
    selected: tuple[FunctionTool, ...]
    if policy.choice.mode is ToolChoiceMode.NONE:
        selected = ()
    elif policy.choice.mode is ToolChoiceMode.NAMED:
        selected = tuple(tool for tool in policy.tools if tool.name == policy.choice.name)
    else:
        selected = policy.tools
    return tuple(
        TemplateTool(tool.name, tool.description, tool.parameters.canonical_json)
        for tool in sorted(selected, key=lambda item: item.name)
    )


class Glm5PromptCompiler(HFTemplatePromptCompiler):
    """Compile canonical Agent history through the official GLM-5 HF template."""

    capabilities = GLM5_CAPABILITIES
    stop_conditions = _GLM5_STOP_CONDITIONS

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
        known_calls: dict[str, str] = {}

        def flush_assistant() -> None:
            nonlocal reasoning_parts, assistant_text, assistant_calls
            if not reasoning_parts and assistant_text is None and not assistant_calls:
                return
            messages.append(
                TemplateMessage(
                    role="assistant",
                    content=assistant_text or "",
                    reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
                    tool_calls=tuple(assistant_calls),
                )
            )
            reasoning_parts = []
            assistant_text = None
            assistant_calls = []

        for item in items[position:]:
            if isinstance(item, MessageItem):
                if item.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                    raise ValueError("GLM-5 system/developer messages must appear at the beginning")
                if item.role is MessageRole.USER:
                    flush_assistant()
                    messages.append(TemplateMessage("user", item.text))
                    continue
                if item.role is MessageRole.ASSISTANT:
                    assistant_text = item.text if assistant_text is None else assistant_text + item.text
                    continue
                raise ValueError(f"unsupported GLM-5 message role: {item.role.value}")

            if isinstance(item, MultimodalMessageItem | MultimodalToolResultItem):
                raise TypeError("GLM-5 ExLlamaV3 text architecture does not support multimodal input")

            if isinstance(item, ReasoningItem):
                if (
                    item.starts_new_assistant_segment
                    and assistant_text is not None
                    and assistant_text.strip()
                    and not assistant_calls
                ):
                    flush_assistant()
                if assistant_text is not None or assistant_calls:
                    raise ValueError("assistant reasoning must precede assistant text and tool calls")
                reasoning_parts.append(item.text)
                continue

            if isinstance(item, ToolCallItem):
                if item.call_id in known_calls:
                    raise ValueError(f"duplicate tool call id in history: {item.call_id!r}")
                if item.index != len(assistant_calls):
                    raise ValueError("tool call index must match order within the assistant turn")
                known_calls[item.call_id] = item.name
                assistant_calls.append(TemplateToolCall(item.name, item.arguments_json))
                continue

            if isinstance(item, ToolResultItem):
                flush_assistant()
                tool_name = known_calls.get(item.call_id)
                if tool_name is None:
                    raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
                messages.append(
                    TemplateMessage(
                        role="tool",
                        content=item.text,
                        tool_call_id=item.call_id,
                        name=tool_name,
                    )
                )
                continue

            raise TypeError(f"unsupported canonical item: {type(item).__name__}")

        flush_assistant()
        return TemplateRequest(
            messages=tuple(messages),
            tools=_exposed_tools(tool_policy),
            template_kwargs=_reasoning_kwargs(reasoning),
        )

    def _raw_output_is_text_only(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> bool:
        del tool_policy
        return reasoning.mode is ReasoningMode.DISABLED and not template_request.tools

    def _structured_output_trigger(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> str | None:
        del tool_policy
        if reasoning.mode is not ReasoningMode.DISABLED and not template_request.tools:
            return _THINK_CLOSE
        return None


@dataclass(frozen=True, slots=True)
class Glm5ParserFinish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        return incomplete_tool_terminal_issue(self.incomplete_tool_call)


def _deterministic_call_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0glm5\0{index}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _longest_partial_suffix(text: str, markers: tuple[str, ...]) -> int:
    longest = 0
    for marker in markers:
        limit = min(len(text), len(marker) - 1)
        for size in range(1, limit + 1):
            if marker.startswith(text[-size:]):
                longest = max(longest, size)
    return longest


def _resolve_local_schema_ref(reference: str, root: JsonValue) -> JsonValue | None:
    if not reference.startswith("#/"):
        return None
    current = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _schema_types(
    schema: JsonValue,
    root: JsonValue,
    seen_references: frozenset[str] = frozenset(),
) -> set[str] | None:
    if not isinstance(schema, dict):
        return None
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if reference in seen_references:
            return None
        resolved = _resolve_local_schema_ref(reference, root)
        if resolved is None:
            return None
        return _schema_types(resolved, root, seen_references | {reference})
    declared = schema.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        return {item for item in declared if isinstance(item, str)}
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, list) or not branches:
            continue
        combined: set[str] = set()
        for branch in branches:
            branch_types = _schema_types(branch, root, seen_references)
            if branch_types is None:
                return None
            combined.update(branch_types)
        return combined
    return None


type _ParameterTypes = dict[str, dict[str, frozenset[str] | None]]


@dataclass(frozen=True, slots=True)
class _Glm5ParserContext:
    tools_enabled: bool
    parameter_types: _ParameterTypes


def _parser_context_from_tools(tools: tuple[TemplateTool, ...]) -> _Glm5ParserContext:
    parameter_types: _ParameterTypes = {}
    for tool in tools:
        try:
            schema = parse_json_strict(tool.parameters_json)
        except InvalidJsonError:
            parameter_types[tool.name] = {}
            continue
        if not isinstance(schema, dict):
            parameter_types[tool.name] = {}
            continue
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            parameter_types[tool.name] = {}
            continue
        parameter_types[tool.name] = {
            name: None
            if (types := _schema_types(property_schema, schema)) is None
            else frozenset(types)
            for name, property_schema in properties.items()
        }
    return _Glm5ParserContext(bool(tools), parameter_types)


def glm5_parser_context(tool_policy: ToolPolicy) -> _Glm5ParserContext:
    """Derive parser-only tool schema state directly from the current request policy."""
    if not isinstance(tool_policy, ToolPolicy):
        raise TypeError("tool_policy must be a ToolPolicy")
    return _parser_context_from_tools(_exposed_tools(tool_policy))


def _decode_argument(raw: str, schema_types: frozenset[str] | None) -> JsonValue:
    stripped = raw.strip()
    if schema_types is not None and "string" in schema_types and schema_types <= {"string", "null"}:
        if stripped == "null" and "null" in schema_types:
            return None
        return raw
    try:
        return parse_json_strict(stripped)
    except InvalidJsonError:
        return raw


def _parse_tool_body(
    body: str,
    parameter_types: _ParameterTypes,
) -> tuple[str, str]:
    first_key = body.find(_ARG_KEY_OPEN)
    if first_key < 0:
        name = body.strip()
        if not name or any(character.isspace() or character in "<>" for character in name):
            raise ValueError("invalid GLM-5 zero-argument tool call")
        return name, "{}"

    name = body[:first_key].strip()
    if not name or any(character.isspace() or character in "<>" for character in name):
        raise ValueError("invalid GLM-5 tool name")

    arguments: dict[str, JsonValue] = {}
    cursor = first_key
    tool_parameter_types = parameter_types.get(name, {})
    while cursor < len(body):
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if cursor == len(body):
            break
        if not body.startswith(_ARG_KEY_OPEN, cursor):
            raise ValueError("unexpected text between GLM-5 tool arguments")
        key_start = cursor + len(_ARG_KEY_OPEN)
        key_end = body.find(_ARG_KEY_CLOSE, key_start)
        if key_end < 0:
            raise ValueError("GLM-5 tool argument key is incomplete")
        key = body[key_start:key_end].strip()
        if not key or key in arguments:
            raise ValueError("invalid or duplicate GLM-5 tool argument key")
        cursor = key_end + len(_ARG_KEY_CLOSE)
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if not body.startswith(_ARG_VALUE_OPEN, cursor):
            raise ValueError("GLM-5 tool argument value is missing")
        value_start = cursor + len(_ARG_VALUE_OPEN)
        value_end = body.find(_ARG_VALUE_CLOSE, value_start)
        if value_end < 0:
            raise ValueError("GLM-5 tool argument value is incomplete")
        raw_value = body[value_start:value_end]
        arguments[key] = _decode_argument(raw_value, tool_parameter_types.get(key))
        cursor = value_end + len(_ARG_VALUE_CLOSE)

    return name, canonical_json_dumps(arguments)


class Glm5IncrementalParser:
    """Parse GLM-5's glm45 reasoning envelope plus glm47-style tool calls."""

    def __init__(
        self,
        request_id: str,
        *,
        start_in_reasoning: bool,
        tools: tuple[TemplateTool, ...] = (),
        parser_context: _Glm5ParserContext | None = None,
    ) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(start_in_reasoning, bool):
            raise TypeError("start_in_reasoning must be a bool")
        if parser_context is not None and tools:
            raise ValueError("provide tools or parser_context, not both")
        if parser_context is None:
            parser_context = _parser_context_from_tools(tools)
        self._request_id = request_id
        self._buffer = ""
        self._mode = "reasoning" if start_in_reasoning else "text"
        self._text_open = False
        self._text_value = ""
        self._reasoning_open = False
        self._reasoning_value = ""
        self._tool_body = ""
        self._tool_return_mode = "text"
        self._call_index = 0
        self._had_incomplete_tool = False
        self._finished = False
        self._parameter_types = parser_context.parameter_types
        self._tools_enabled = parser_context.tools_enabled

    def _emit_content(self, text: str, events: list[GenerationEvent]) -> None:
        if not text:
            return
        if self._mode == "reasoning":
            if not self._reasoning_open:
                events.append(ReasoningStarted(self._request_id))
                self._reasoning_open = True
            self._reasoning_value += text
            events.append(ReasoningDelta(self._request_id, text))
            return
        if not self._text_open:
            events.append(TextStarted(self._request_id))
            self._text_open = True
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
        self._tool_return_mode = "text"
        self._mode = "tool"
        self._tool_body = ""

    def _finish_tool(self, events: list[GenerationEvent]) -> None:
        try:
            name, arguments_json = _parse_tool_body(self._tool_body, self._parameter_types)
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
        self._tool_body = ""
        self._mode = self._tool_return_mode

    def _process_plain(self, events: list[GenerationEvent]) -> bool:
        markers = (_THINK_OPEN, _THINK_CLOSE, _TOOL_OPEN) if self._tools_enabled else (_THINK_OPEN, _THINK_CLOSE)
        matches = [
            (position, marker)
            for marker in markers
            if (position := self._buffer.find(marker)) >= 0
        ]
        if matches:
            position, marker = min(matches, key=lambda item: item[0])
            if position > 0:
                self._emit_content(self._buffer[:position], events)
                self._buffer = self._buffer[position:]
                return True
            self._buffer = self._buffer[len(marker) :]
            if marker == _THINK_OPEN:
                self._close_channel(events)
                self._mode = "reasoning"
            elif marker == _THINK_CLOSE:
                self._close_channel(events)
                self._mode = "text"
            else:
                self._enter_tool(events)
            return True

        keep = _longest_partial_suffix(self._buffer, markers)
        emit_length = len(self._buffer) - keep
        if emit_length > 0:
            self._emit_content(self._buffer[:emit_length], events)
            self._buffer = self._buffer[emit_length:]
            return True
        return False

    def _process_tool(self, events: list[GenerationEvent]) -> bool:
        close_at = self._buffer.find(_TOOL_CLOSE)
        if close_at >= 0:
            self._tool_body += self._buffer[:close_at]
            self._buffer = self._buffer[close_at + len(_TOOL_CLOSE) :]
            self._finish_tool(events)
            return True
        keep = _longest_partial_suffix(self._buffer, (_TOOL_CLOSE,))
        consume = len(self._buffer) - keep
        if consume > 0:
            self._tool_body += self._buffer[:consume]
            self._buffer = self._buffer[consume:]
            return True
        return False

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished GLM-5 parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        self._buffer += chunk
        events: list[GenerationEvent] = []
        while True:
            progressed = self._process_tool(events) if self._mode == "tool" else self._process_plain(events)
            if not progressed:
                break
        return tuple(events)

    def finish(self) -> Glm5ParserFinish:
        if self._finished:
            return Glm5ParserFinish((), self._had_incomplete_tool)
        events: list[GenerationEvent] = []
        if self._mode == "tool":
            self._had_incomplete_tool = True
            self._buffer = ""
            self._tool_body = ""
            self._mode = self._tool_return_mode
        else:
            if self._buffer:
                if any(marker.startswith(self._buffer) for marker in (_THINK_OPEN, _THINK_CLOSE)):
                    pass
                elif self._tools_enabled and _TOOL_OPEN.startswith(self._buffer):
                    self._had_incomplete_tool = True
                else:
                    self._emit_content(self._buffer, events)
                self._buffer = ""
            self._close_channel(events)
        self._finished = True
        return Glm5ParserFinish(tuple(events), self._had_incomplete_tool)
