"""Qwen3.8 model dialect: deterministic prompt compilation and streaming parsing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from exqserve.agent._json import (
    InvalidJsonError,
    JsonValue,
    canonical_json_dumps,
    parse_json_strict,
)
from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode, ReasoningPolicy
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
    ToolConstraintMode,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
)
from exqserve.model.hf_template import HFTemplatePromptCompiler
from exqserve.model.tool_constraints import (
    exposed_tools,
    lark_literal,
    qwen_parameter_schema,
    qwen_property_schema,
    schema_lark,
)

QWEN38_CAPABILITIES = ModelCapabilities(
    reasoning=True,
    tool_calling=True,
    parallel_tool_calls=True,
    system_role=True,
    developer_role=False,
    reasoning_history=True,
    vision=True,
)

_QWEN_TOOL_TRIGGER = "<tool_call>"
_QWEN_STRUCTURAL_WS_MAX = 8
_QWEN_NATIVE_STRING_ALLOWED_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "type",
        "enum",
        "const",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
_QWEN_PARAMETER_CLOSE = "</parameter>"
_QWEN_RAW_STRING_RULE = 'qwen_raw_string[suffix="</parameter>"]: /[\\s\\S]*/'


def _qwen_native_string_lark(schema: dict[str, JsonValue]) -> tuple[str | None, bool]:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "string" in schema_type:
            raise ToolConstraintUnsupported(
                "mixed string/non-string unions are not supported in Qwen schema mode"
            )
        return None, False
    if schema_type != "string":
        return None, False

    unsupported = sorted(set(schema) - _QWEN_NATIVE_STRING_ALLOWED_KEYS)
    if unsupported:
        raise ToolConstraintUnsupported(
            "unsupported Qwen native string schema keyword: " + unsupported[0]
        )

    allowed_values: set[str] | None = None
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list):
            raise ToolConstraintUnsupported("Qwen string enum must be an array")
        allowed_values = {item for item in enum if isinstance(item, str)}
        if not allowed_values:
            raise ToolConstraintUnsupported("Qwen string enum has no string values")

    if "const" in schema:
        const = schema["const"]
        if not isinstance(const, str):
            raise ToolConstraintUnsupported("Qwen string const must be a string")
        allowed_values = {const} if allowed_values is None else allowed_values & {const}
        if not allowed_values:
            raise ToolConstraintUnsupported("Qwen string enum and const have no common value")

    if allowed_values is not None:
        for value in allowed_values:
            if value != value.strip() or _QWEN_PARAMETER_CLOSE in value:
                raise ToolConstraintUnsupported(
                    "Qwen native string enum/const value cannot be represented losslessly"
                )
        alternatives = " | ".join(lark_literal(value) for value in sorted(allowed_values))
        return f"({alternatives})", False
    return "qwen_raw_string", True


def qwen_tool_constraint(
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

    lines = ["%llguidance {}", 'start: WS? function WS? "</tool_call>"']
    lines.append("function: " + " | ".join(f"function_{index}" for index in range(len(tools))))
    uses_raw_string = False

    if mode is ToolConstraintMode.FORMAT:
        lines.extend(
            [
                'parameter: "<parameter=" NAME ">" value "</parameter>" WS?',
                "value: VALUE_CHAR*",
                "NAME: /[^\\s<>]+/",
                "VALUE_CHAR: /[^<]/",
            ]
        )

    for index, tool in enumerate(tools):
        if not _valid_tag_name(tool.name):
            raise ToolConstraintUnsupported(
                f"Qwen constrained generation cannot represent tool name {tool.name!r}"
            )
        open_tag = lark_literal(f"<function={tool.name}>")
        if mode is ToolConstraintMode.FORMAT:
            lines.append(f'function_{index}: {open_tag} WS? parameter* "</function>"')
            continue

        schema = qwen_parameter_schema(tool.parameters)
        properties = schema.get("properties", {})
        assert isinstance(properties, dict)
        required = schema.get("required", [])
        assert isinstance(required, list)
        required_names = set(required)
        parameter_rules: list[str] = []
        for parameter_index, name in enumerate(sorted(properties)):
            if not _valid_tag_name(name):
                raise ToolConstraintUnsupported(
                    f"Qwen constrained generation cannot represent parameter name {name!r}"
                )
            property_schema = properties[name]
            assert isinstance(property_schema, dict)
            property_schema = qwen_property_schema(schema, property_schema)
            value_lark, raw_string = _qwen_native_string_lark(property_schema)
            uses_raw_string = uses_raw_string or raw_string
            if value_lark is None:
                value_lark = schema_lark(property_schema)
            rule_name = f"function_{index}_parameter_{parameter_index}"
            suffix = "" if name in required_names else "?"
            parameter_rules.append(rule_name + suffix)
            if raw_string:
                lines.append(
                    f"{rule_name}: {lark_literal(f'<parameter={name}>')} WS? "
                    f"{value_lark} WS?"
                )
            else:
                lines.append(
                    f"{rule_name}: {lark_literal(f'<parameter={name}>')} WS? "
                    f'{value_lark} WS? "</parameter>" WS?'
                )
        parameter_body = " ".join(parameter_rules)
        if parameter_body:
            parameter_body += " "
        lines.append(f'function_{index}: {open_tag} WS? {parameter_body}"</function>"')

    if uses_raw_string:
        lines.append(_QWEN_RAW_STRING_RULE)
    lines.append(f"WS: /[ \\t\\r\\n]{{1,{_QWEN_STRUCTURAL_WS_MAX}}}/")
    return ToolGenerationConstraint(
        trigger=_QWEN_TOOL_TRIGGER,
        lark_grammar="\n".join(lines),
        eos_after_completed=not tool_policy.allow_parallel,
    )


def _reasoning_kwargs(
    policy: ReasoningPolicy,
    *,
    preserve_thinking: bool,
) -> tuple[tuple[str, str | bool], ...]:
    values: dict[str, str | bool] = {}
    if policy.mode is ReasoningMode.ENABLED:
        values["enable_thinking"] = True
    elif policy.mode is ReasoningMode.DISABLED:
        values["enable_thinking"] = False

    if policy.effort is not None:
        effort_map = {
            ReasoningEffort.LOW: "low",
            ReasoningEffort.MEDIUM: "medium",
            ReasoningEffort.HIGH: "xhigh",
            ReasoningEffort.XHIGH: "xhigh",
            ReasoningEffort.MAXIMUM: "xhigh",
        }
        values["reasoning_effort"] = effort_map[policy.effort]

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


class QwenPromptCompiler(HFTemplatePromptCompiler):
    capabilities = QWEN38_CAPABILITIES
    use_native_eos = True

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
        has_reasoning_history = any(isinstance(item, ReasoningItem) for item in items)

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
                    raise ValueError("Qwen system/developer messages must appear at the beginning")
                if item.role is MessageRole.USER:
                    flush_assistant()
                    messages.append(TemplateMessage("user", item.text))
                    continue
                if item.role is MessageRole.ASSISTANT:
                    if assistant_text is None:
                        assistant_text = item.text
                    else:
                        assistant_text += item.text
                    continue
                raise ValueError(f"unsupported Qwen message role: {item.role.value}")

            if isinstance(item, MultimodalMessageItem):
                flush_assistant()
                content_parts: list[TemplateTextPart | TemplateImagePart] = []
                for part in item.parts:
                    if isinstance(part, TextContentPart):
                        content_parts.append(TemplateTextPart(part.text))
                    elif isinstance(part, ImageContentPart):
                        content_parts.append(TemplateImagePart(part.source, part.detail))
                    else:  # pragma: no cover - canonical value validation prevents this
                        raise TypeError(f"unsupported multimodal part: {type(part).__name__}")
                messages.append(TemplateMessage("user", tuple(content_parts)))
                continue

            if isinstance(item, ReasoningItem):
                if assistant_calls:
                    raise ValueError("assistant reasoning must precede assistant text and tool calls")
                if assistant_text is not None:
                    if assistant_text.strip():
                        raise ValueError("assistant reasoning must precede assistant text and tool calls")
                    assistant_text = None
                reasoning_parts.append(item.text)
                continue

            if isinstance(item, ToolCallItem):
                if item.call_id in known_calls:
                    raise ValueError(f"duplicate tool call id in history: {item.call_id!r}")
                if item.index != len(assistant_calls):
                    raise ValueError("tool call index must match order within the assistant turn")
                known_calls[item.call_id] = item.name
                assistant_calls.append(
                    TemplateToolCall(name=item.name, arguments_json=item.arguments_json)
                )
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

            if isinstance(item, MultimodalToolResultItem):
                flush_assistant()
                tool_name = known_calls.get(item.call_id)
                if tool_name is None:
                    raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
                tool_content_parts: list[TemplateTextPart | TemplateImagePart] = []
                for part in item.parts:
                    if isinstance(part, TextContentPart):
                        tool_content_parts.append(TemplateTextPart(part.text))
                    elif isinstance(part, ImageContentPart):
                        tool_content_parts.append(TemplateImagePart(part.source, part.detail))
                    else:  # pragma: no cover - canonical validation prevents this
                        raise TypeError(f"unsupported multimodal tool result part: {type(part).__name__}")
                messages.append(
                    TemplateMessage(
                        role="tool",
                        content=tuple(tool_content_parts),
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
            template_kwargs=_reasoning_kwargs(
                reasoning,
                preserve_thinking=has_reasoning_history,
            ),
            protect_literal_tokens=True,
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
            return "</think>"
        return None


@dataclass(frozen=True, slots=True)
class QwenParserFinish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise TypeError("events must be a tuple")
        if not isinstance(self.incomplete_tool_call, bool):
            raise TypeError("incomplete_tool_call must be a bool")


_PLAIN_MARKERS = ("<think>", "</think>", "<tool_call>")
_FUNCTION_OPEN = "<function="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_OPEN = "<parameter="
_PARAMETER_CLOSE = "</parameter>"
_TOOL_CLOSE = "</tool_call>"
_LITERAL_MARKER_QUOTES = frozenset({"'", '"', "`"})


@dataclass(slots=True)
class _MarkdownCodeContext:
    """Track Qwen backtick-delimited source spans across runtime chunks."""

    delimiter_width: int | None = None
    pending_backticks: int = 0

    def _commit_pending_backticks(self) -> None:
        width = self.pending_backticks
        if width == 0:
            return
        self.pending_backticks = 0
        if self.delimiter_width is None:
            self.delimiter_width = width
        elif width == self.delimiter_width:
            self.delimiter_width = None

    def classify_marker(self, text: str, marker: str, *, final: bool = False) -> tuple[bool, bool]:
        self._commit_pending_backticks()
        if self.delimiter_width is None:
            return False, False

        delimiter = "`" * self.delimiter_width
        if text.find(delimiter, len(marker)) >= 0:
            return True, False
        return (False, False) if final else (False, True)

    def observe(self, text: str) -> None:
        for character in text:
            if character == "`":
                self.pending_backticks += 1
                continue
            self._commit_pending_backticks()


def _marker_is_directly_quoted(
    text: str,
    marker: str,
    previous_content_character: str | None,
) -> tuple[bool, bool]:
    left = previous_content_character
    if left not in _LITERAL_MARKER_QUOTES:
        return False, False
    right_at = len(marker)
    if right_at >= len(text):
        return False, True
    return text[right_at] == left, False


def _is_pending_tool_candidate(text: str) -> bool:
    if "<tool_call>".startswith(text):
        return True
    if not text.startswith("<tool_call>"):
        return False
    candidate = text[len("<tool_call>") :].lstrip()
    return not candidate or _FUNCTION_OPEN.startswith(candidate)


def _longest_partial_marker_suffix(text: str) -> int:
    longest = 0
    for marker in _PLAIN_MARKERS:
        limit = min(len(text), len(marker) - 1)
        for size in range(1, limit + 1):
            if marker.startswith(text[-size:]):
                longest = max(longest, size)
    return longest


def _valid_tag_name(value: str) -> bool:
    return bool(value) and not any(character.isspace() or character in "<>" for character in value)


def _deterministic_call_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0{index}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _parameter_value_json(value_text: str, *, string_parameter: bool = False) -> str:
    stripped = value_text.strip()
    try:
        value = parse_json_strict(stripped)
    except InvalidJsonError:
        value = stripped
    else:
        if string_parameter and not isinstance(value, str):
            value = stripped
    return canonical_json_dumps(value)


def _find_parameter_close(text: str, value_start: int) -> int:
    """Find an envelope close outside a double-quoted JSON/string literal."""

    in_string = False
    escaped = False
    position = value_start
    while position < len(text):
        character = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            position += 1
            continue
        if character == '"':
            in_string = True
            position += 1
            continue
        if text.startswith(_PARAMETER_CLOSE, position):
            return position
        position += 1
    return -1


def _qwen_string_parameters(tool_policy: ToolPolicy | None) -> dict[str, frozenset[str]]:
    if tool_policy is None:
        return {}
    if not isinstance(tool_policy, ToolPolicy):
        raise TypeError("tool_policy must be a ToolPolicy or None")

    result: dict[str, frozenset[str]] = {}
    for tool in tool_policy.tools:
        schema = parse_json_strict(tool.parameters.canonical_json)
        if not isinstance(schema, dict):
            continue
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            continue
        names = frozenset(
            name
            for name, property_schema in properties.items()
            if isinstance(name, str)
            and isinstance(property_schema, dict)
            and property_schema.get("type") == "string"
        )
        if names:
            result[tool.name] = names
    return result


class QwenIncrementalParser:
    """Incrementally convert Qwen model-native text into canonical semantic events."""

    def __init__(
        self,
        request_id: str,
        *,
        start_in_reasoning: bool = False,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(start_in_reasoning, bool):
            raise TypeError("start_in_reasoning must be a bool")
        self._string_parameters = _qwen_string_parameters(tool_policy)
        self._request_id = request_id
        self._buffer = ""
        self._mode = "reasoning" if start_in_reasoning else "text"
        self._text_open = False
        self._text_value = ""
        self._reasoning_open = False
        self._reasoning_value = ""
        self._call_index = 0
        self._tool_return_mode = "text"
        self._tool_state = "function"
        self._tool_name: str | None = None
        self._tool_call_id: str | None = None
        self._tool_started = False
        self._tool_argument_parts: list[str] = []
        self._tool_arguments_json: str | None = None
        self._had_incomplete_tool = False
        self._last_content_character: str | None = None
        self._literal_context = _MarkdownCodeContext()
        self._finished = False

    def _emit_content(self, text: str, events: list[GenerationEvent]) -> None:
        if not text:
            return
        self._literal_context.observe(text)
        self._last_content_character = text[-1]
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

    def _close_current_channel(self, events: list[GenerationEvent]) -> None:
        if self._mode == "reasoning" and self._reasoning_open:
            events.append(ReasoningCompleted(self._request_id, self._reasoning_value))
            self._reasoning_open = False
            self._reasoning_value = ""
        elif self._mode == "text" and self._text_open:
            events.append(TextCompleted(self._request_id, self._text_value))
            self._text_open = False
            self._text_value = ""

    def _enter_tool(self, events: list[GenerationEvent]) -> None:
        self._close_current_channel(events)
        self._tool_return_mode = self._mode
        self._mode = "tool"
        self._tool_state = "function"
        self._tool_name = None
        self._tool_call_id = None
        self._tool_started = False
        self._tool_argument_parts = []
        self._tool_arguments_json = None

    def _restore_after_tool(self) -> None:
        self._mode = self._tool_return_mode
        self._tool_state = "function"
        self._tool_name = None
        self._tool_call_id = None
        self._tool_started = False
        self._tool_argument_parts = []
        self._tool_arguments_json = None

    def _process_plain(self, events: list[GenerationEvent], *, final: bool = False) -> bool:
        match: tuple[int, str] | None = None
        for marker in _PLAIN_MARKERS:
            position = self._buffer.find(marker)
            if position >= 0 and (match is None or position < match[0]):
                match = (position, marker)

        if match is not None:
            position, marker = match
            if position > 0:
                self._emit_content(self._buffer[:position], events)
                self._buffer = self._buffer[position:]
                return True

            code_literal, code_pending = self._literal_context.classify_marker(
                self._buffer,
                marker,
                final=final,
            )
            if code_pending:
                return False
            if code_literal:
                self._emit_content(marker, events)
                self._buffer = self._buffer[len(marker) :]
                return True

            is_literal, needs_more_text = _marker_is_directly_quoted(
                self._buffer,
                marker,
                self._last_content_character,
            )
            if needs_more_text and not final:
                return False
            if is_literal:
                self._emit_content(marker, events)
                self._buffer = self._buffer[len(marker) :]
                return True

            if marker == "<tool_call>":
                after_marker = self._buffer[len(marker) :]
                candidate = after_marker.lstrip()
                if candidate.startswith(_FUNCTION_OPEN):
                    self._buffer = after_marker
                    self._enter_tool(events)
                    return True
                if not candidate or _FUNCTION_OPEN.startswith(candidate):
                    return False
                self._emit_content(marker, events)
                self._buffer = after_marker
                return True

            self._buffer = self._buffer[len(marker) :]
            if marker == "<think>":
                if self._mode != "reasoning":
                    self._close_current_channel(events)
                    self._mode = "reasoning"
            else:
                if self._mode == "reasoning":
                    self._close_current_channel(events)
                    self._mode = "text"
            return True

        held = _longest_partial_marker_suffix(self._buffer)
        safe_length = len(self._buffer) - held
        if safe_length > 0:
            self._emit_content(self._buffer[:safe_length], events)
            self._buffer = self._buffer[safe_length:]
        return False

    def _consume_tool_whitespace(self) -> bool:
        stripped = self._buffer.lstrip()
        if len(stripped) == len(self._buffer):
            return False
        self._buffer = stripped
        return True

    def _mark_tool_malformed(self) -> None:
        self._had_incomplete_tool = True
        self._tool_state = "malformed"

    def _ensure_tool_started(self, events: list[GenerationEvent]) -> None:
        if self._tool_started:
            return
        assert self._tool_name is not None
        assert self._tool_call_id is not None
        events.append(
            ToolCallStarted(
                request_id=self._request_id,
                call_id=self._tool_call_id,
                name=self._tool_name,
                index=self._call_index,
            )
        )
        self._tool_started = True

    def _emit_tool_delta(self, delta: str, events: list[GenerationEvent]) -> None:
        assert self._tool_call_id is not None
        events.append(
            ToolCallArgumentsDelta(
                request_id=self._request_id,
                call_id=self._tool_call_id,
                delta=delta,
                index=self._call_index,
            )
        )

    def _complete_function_arguments(self, events: list[GenerationEvent]) -> None:
        self._ensure_tool_started(events)
        if not self._tool_argument_parts:
            self._tool_arguments_json = "{}"
            self._emit_tool_delta("{}", events)
        else:
            self._tool_arguments_json = "".join(self._tool_argument_parts) + "}"
            self._emit_tool_delta("}", events)
        self._tool_state = "outer"

    def _process_tool_function(self) -> bool:
        if self._consume_tool_whitespace():
            return True
        if not self._buffer:
            return False
        if _FUNCTION_OPEN.startswith(self._buffer):
            return False
        if not self._buffer.startswith(_FUNCTION_OPEN):
            self._mark_tool_malformed()
            return True

        header_end = self._buffer.find(">", len(_FUNCTION_OPEN))
        if header_end < 0:
            return False
        name = self._buffer[len(_FUNCTION_OPEN) : header_end]
        if not _valid_tag_name(name):
            self._mark_tool_malformed()
            return True

        self._tool_name = name
        self._tool_call_id = _deterministic_call_id(self._request_id, self._call_index)
        self._buffer = self._buffer[header_end + 1 :]
        self._tool_state = "parameters"
        return True

    def _process_tool_parameters(self, events: list[GenerationEvent]) -> bool:
        if self._consume_tool_whitespace():
            return True
        if not self._buffer:
            return False

        if self._buffer.startswith(_FUNCTION_CLOSE):
            self._buffer = self._buffer[len(_FUNCTION_CLOSE) :]
            self._complete_function_arguments(events)
            return True
        if _FUNCTION_CLOSE.startswith(self._buffer):
            return False

        if self._buffer.startswith(_PARAMETER_OPEN):
            header_end = self._buffer.find(">", len(_PARAMETER_OPEN))
            if header_end < 0:
                return False
            parameter_name = self._buffer[len(_PARAMETER_OPEN) : header_end]
            if not _valid_tag_name(parameter_name):
                self._mark_tool_malformed()
                return True

            value_start = header_end + 1
            close_at = _find_parameter_close(self._buffer, value_start)
            if close_at < 0:
                return False
            assert self._tool_name is not None
            string_parameter = parameter_name in self._string_parameters.get(
                self._tool_name, frozenset()
            )
            value_json = _parameter_value_json(
                self._buffer[value_start:close_at],
                string_parameter=string_parameter,
            )
            prefix = "{" if not self._tool_argument_parts else ","
            fragment = f"{prefix}{canonical_json_dumps(parameter_name)}:{value_json}"
            self._ensure_tool_started(events)
            self._tool_argument_parts.append(fragment)
            self._emit_tool_delta(fragment, events)
            self._buffer = self._buffer[close_at + len(_PARAMETER_CLOSE) :]
            return True
        if _PARAMETER_OPEN.startswith(self._buffer):
            return False

        self._mark_tool_malformed()
        return True

    def _process_tool_outer(self, events: list[GenerationEvent]) -> bool:
        if self._consume_tool_whitespace():
            return True
        if not self._buffer:
            return False
        if self._buffer.startswith(_TOOL_CLOSE):
            assert self._tool_name is not None
            assert self._tool_call_id is not None
            assert self._tool_arguments_json is not None
            self._buffer = self._buffer[len(_TOOL_CLOSE) :]
            events.append(
                ToolCallCompleted(
                    self._request_id,
                    ToolCallItem(
                        call_id=self._tool_call_id,
                        name=self._tool_name,
                        arguments_json=self._tool_arguments_json,
                        index=self._call_index,
                    ),
                )
            )
            self._call_index += 1
            self._restore_after_tool()
            return True
        if _TOOL_CLOSE.startswith(self._buffer):
            return False
        self._mark_tool_malformed()
        return True

    def _process_tool_malformed(self) -> bool:
        close_at = self._buffer.find(_TOOL_CLOSE)
        if close_at < 0:
            return False
        self._buffer = self._buffer[close_at + len(_TOOL_CLOSE) :]
        self._restore_after_tool()
        return True

    def _process_tool(self, events: list[GenerationEvent]) -> bool:
        if self._tool_state == "function":
            return self._process_tool_function()
        if self._tool_state == "parameters":
            return self._process_tool_parameters(events)
        if self._tool_state == "outer":
            return self._process_tool_outer(events)
        return self._process_tool_malformed()

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Qwen parser")
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

    def finish(self) -> QwenParserFinish:
        if self._finished:
            return QwenParserFinish((), self._had_incomplete_tool)

        events: list[GenerationEvent] = []
        if self._mode == "tool":
            self._had_incomplete_tool = True
            self._buffer = ""
            self._restore_after_tool()
        else:
            while self._buffer and self._process_plain(events, final=True):
                pass
            if self._buffer:
                quoted_partial_marker = (
                    self._last_content_character in {"'", '"', "`"}
                    and any(marker.startswith(self._buffer) for marker in _PLAIN_MARKERS)
                )
                if quoted_partial_marker:
                    self._emit_content(self._buffer, events)
                elif _TOOL_CLOSE.startswith(self._buffer) or _is_pending_tool_candidate(self._buffer):
                    self._had_incomplete_tool = True
                elif any(marker.startswith(self._buffer) for marker in ("<think>", "</think>")):
                    pass
                else:
                    self._emit_content(self._buffer, events)
                self._buffer = ""
            self._close_current_channel(events)

        self._finished = True
        return QwenParserFinish(tuple(events), self._had_incomplete_tool)
