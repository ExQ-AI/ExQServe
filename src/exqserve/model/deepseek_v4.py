"""DeepSeek-V4 model dialect: native prompt encoding and DSML stream parsing."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
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
    ChatTemplateAdapter,
    CompiledPrompt,
    ModelCapabilities,
    TemplateMessage,
    TemplateRequest,
    TemplateTool,
    TemplateToolCall,
)

DEEPSEEK_V4_CAPABILITIES = ModelCapabilities(
    reasoning=True,
    tool_calling=True,
    parallel_tool_calls=True,
    system_role=True,
    developer_role=False,
    reasoning_history=True,
    vision=False,
)

_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_ASSISTANT = "<｜Assistant｜>"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_DSML = "｜DSML｜"
_TOOL_BLOCK_OPEN = f"<{_DSML}tool_calls>"
_TOOL_BLOCK_CLOSE = f"</{_DSML}tool_calls>"
_MALFORMED_TOOL_BLOCK_OPEN = f"<{_DSML}toolcalls>"
_INVOKE_OPEN_PREFIX = f'<{_DSML}invoke name="'
_INVOKE_CLOSE = f"</{_DSML}invoke>"
_PARAMETER_OPEN_PREFIX = f'<{_DSML}parameter name="'
_PARAMETER_CLOSE = f"</{_DSML}parameter>"
_TOOL_RESULT_OPEN = "<tool_result>"
_TOOL_RESULT_CLOSE = "</tool_result>"
_DEEPSEEK_V4_STOP_CONDITIONS = (_EOS,)
_TOOL_OPEN_MARKERS = ("\n\n" + _TOOL_BLOCK_OPEN, _TOOL_BLOCK_OPEN)
_MALFORMED_TOOL_OPEN_MARKERS = (
    "\n\n" + _MALFORMED_TOOL_BLOCK_OPEN,
    _MALFORMED_TOOL_BLOCK_OPEN,
)
_BARE_INVOKE_MARKERS = ("\n\n" + _INVOKE_OPEN_PREFIX, _INVOKE_OPEN_PREFIX)
_TOOL_START_MARKERS = _TOOL_OPEN_MARKERS + _MALFORMED_TOOL_OPEN_MARKERS + _BARE_INVOKE_MARKERS

_HIGH_EFFORT_PREFIX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to "
    "resolve the root cause, rigorously stress-testing your logic against all potential paths, "
    "edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, "
    "considered alternative, and rejected hypothesis to ensure absolutely no assumption is left "
    "unchecked.\n\n"
)
_MAX_EFFORT_PREFIX = (
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: "
    "exhaustively decompose the problem into its most fundamental components, trace every causal "
    "chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles "
    "and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
)

_TOOL_INSTRUCTIONS = f'''## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{_DSML}tool_calls>" block like the following:

<{_DSML}tool_calls>
<{_DSML}invoke name="$TOOL_NAME">
<{_DSML}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{_DSML}parameter>
...
</{_DSML}invoke>
<{_DSML}invoke name="$TOOL_NAME2">
...
</{_DSML}invoke>
</{_DSML}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {_THINK_OPEN}), you MUST output your complete reasoning inside {_THINK_OPEN}...{_THINK_CLOSE} BEFORE any tool calls or final response.

Otherwise, output directly after {_THINK_CLOSE} with tool calls or final response.

### Available Tool Schemas

{{tool_schemas}}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n'''


@dataclass(frozen=True, slots=True)
class _Call:
    name: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class _Turn:
    role: str
    content: str = ""
    reasoning: str | None = None
    calls: tuple[_Call, ...] = ()
    tool_result_user: bool = False


@dataclass(frozen=True, slots=True)
class DeepSeekV4ParserContext:
    tools_enabled: bool
    tool_properties: dict[str, frozenset[str]]


def _prompt_hash(input_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token_id in input_ids:
        encoded = str(token_id).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


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
        for tool in selected
    )


def _parser_context_from_tools(tools: tuple[TemplateTool, ...]) -> DeepSeekV4ParserContext:
    properties: dict[str, frozenset[str]] = {}
    for tool in tools:
        try:
            schema = parse_json_strict(tool.parameters_json)
        except InvalidJsonError:
            properties[tool.name] = frozenset()
            continue
        if not isinstance(schema, dict):
            properties[tool.name] = frozenset()
            continue
        declared = schema.get("properties")
        if not isinstance(declared, dict):
            properties[tool.name] = frozenset()
            continue
        properties[tool.name] = frozenset(
            key for key in declared if isinstance(key, str) and key
        )
    return DeepSeekV4ParserContext(bool(tools), properties)


def _effort_prefix(policy: ReasoningPolicy) -> str:
    if policy.mode is ReasoningMode.DISABLED:
        return ""
    if policy.effort is None:
        return _HIGH_EFFORT_PREFIX
    if policy.effort in {
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    }:
        return _HIGH_EFFORT_PREFIX
    if policy.effort is ReasoningEffort.MAXIMUM:
        return _MAX_EFFORT_PREFIX
    return ""


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _tool_schema_lines(tools: tuple[TemplateTool, ...]) -> str:
    lines: list[str] = []
    for tool in tools:
        try:
            parameters = parse_json_strict(tool.parameters_json)
        except InvalidJsonError as exc:
            raise ValueError("DeepSeek-V4 tool schema must contain strict JSON") from exc
        if not isinstance(parameters, dict):
            raise TypeError("DeepSeek-V4 tool schema must be a JSON object")
        definition: dict[str, object] = {"name": tool.name}
        if tool.description is not None:
            definition["description"] = tool.description
        definition["parameters"] = parameters
        lines.append(_json_dump(definition))
    return "\n".join(lines)


def _encode_call(call: _Call) -> str:
    try:
        arguments = parse_json_strict(call.arguments_json)
    except InvalidJsonError as exc:
        raise ValueError("DeepSeek-V4 tool-call arguments must contain strict JSON") from exc
    if not isinstance(arguments, dict):
        raise TypeError("DeepSeek-V4 tool-call arguments must be a JSON object")
    parts = [f'{_INVOKE_OPEN_PREFIX}{call.name}">']
    for key, value in arguments.items():
        if not isinstance(key, str) or not key:
            raise ValueError("DeepSeek-V4 tool argument names must be non-empty strings")
        is_string = isinstance(value, str)
        encoded_value = value if is_string else _json_dump(value)
        parts.append(
            f'{_PARAMETER_OPEN_PREFIX}{key}" string="{"true" if is_string else "false"}">'
            f"{encoded_value}{_PARAMETER_CLOSE}"
        )
    parts.append(_INVOKE_CLOSE)
    return "\n".join(parts)


def _encode_calls(calls: tuple[_Call, ...]) -> str:
    encoded = "\n".join(_encode_call(call) for call in calls)
    return f"\n\n{_TOOL_BLOCK_OPEN}\n{encoded}\n{_TOOL_BLOCK_CLOSE}"


def _canonical_turns(request: CanonicalRequest) -> tuple[_Turn, ...]:
    turns: list[_Turn] = []
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
        turns.append(_Turn("system", "\n\n".join(leading_instructions)))

    reasoning_parts: list[str] = []
    assistant_text: str | None = None
    assistant_calls: list[_Call] = []
    known_calls: dict[str, tuple[str, int]] = {}
    pending_results: list[ToolResultItem] = []

    def append_user(content: str) -> None:
        if turns and turns[-1].role == "user" and turns[-1].tool_result_user:
            previous = turns[-1]
            turns[-1] = _Turn(
                "user",
                f"{previous.content}\n\n{content}",
                tool_result_user=True,
            )
            return
        turns.append(_Turn("user", content))

    def flush_assistant() -> None:
        nonlocal reasoning_parts, assistant_text, assistant_calls
        if not reasoning_parts and assistant_text is None and not assistant_calls:
            return
        turns.append(
            _Turn(
                "assistant",
                assistant_text or "",
                "".join(reasoning_parts) if reasoning_parts else None,
                tuple(assistant_calls),
            )
        )
        reasoning_parts = []
        assistant_text = None
        assistant_calls = []

    def flush_results() -> None:
        nonlocal pending_results
        if not pending_results:
            return
        ordered = sorted(pending_results, key=lambda result: known_calls[result.call_id][1])
        content = "\n\n".join(
            f"{_TOOL_RESULT_OPEN}{result.text}{_TOOL_RESULT_CLOSE}" for result in ordered
        )
        turns.append(_Turn("user", content, tool_result_user=True))
        pending_results = []

    for item in items[position:]:
        if not isinstance(item, ToolResultItem):
            flush_results()

        if isinstance(item, MessageItem):
            if item.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                raise ValueError("DeepSeek-V4 system/developer messages must appear at the beginning")
            if item.role is MessageRole.USER:
                flush_assistant()
                append_user(item.text)
                continue
            if item.role is MessageRole.ASSISTANT:
                assistant_text = item.text if assistant_text is None else assistant_text + item.text
                continue
            raise ValueError(f"unsupported DeepSeek-V4 message role: {item.role.value}")

        if isinstance(item, MultimodalMessageItem | MultimodalToolResultItem):
            raise TypeError("DeepSeek-V4 text architecture does not support multimodal input")

        if isinstance(item, ReasoningItem):
            if assistant_text is not None or assistant_calls:
                raise ValueError("assistant reasoning must precede assistant text and tool calls")
            reasoning_parts.append(item.text)
            continue

        if isinstance(item, ToolCallItem):
            if item.call_id in known_calls:
                raise ValueError(f"duplicate tool call id in history: {item.call_id!r}")
            if item.index != len(assistant_calls):
                raise ValueError("tool call index must match order within the assistant turn")
            known_calls[item.call_id] = (item.name, item.index)
            assistant_calls.append(_Call(item.name, item.arguments_json))
            continue

        if isinstance(item, ToolResultItem):
            flush_assistant()
            if item.call_id not in known_calls:
                raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
            pending_results.append(item)
            continue

        raise TypeError(f"unsupported canonical item: {type(item).__name__}")

    flush_results()
    flush_assistant()
    return tuple(turns)


def _template_snapshot(turns: tuple[_Turn, ...], tools: tuple[TemplateTool, ...]) -> TemplateRequest:
    messages = tuple(
        TemplateMessage(
            role=turn.role,
            content=turn.content,
            reasoning_content=turn.reasoning,
            tool_calls=tuple(TemplateToolCall(call.name, call.arguments_json) for call in turn.calls),
        )
        for turn in turns
    )
    return TemplateRequest(messages=messages, tools=tools, template_kwargs=())


def _render_prompt(
    turns: tuple[_Turn, ...],
    tools: tuple[TemplateTool, ...],
    reasoning: ReasoningPolicy,
) -> str:
    mutable = list(turns)
    if tools and (not mutable or mutable[0].role != "system"):
        mutable.insert(0, _Turn("system", ""))
    if not mutable:
        raise ValueError("DeepSeek-V4 prompt history must not be empty")

    thinking = reasoning.mode is not ReasoningMode.DISABLED
    preserve_reasoning = bool(tools)
    last_user = max((index for index, turn in enumerate(mutable) if turn.role == "user"), default=-1)

    prompt = _BOS + _effort_prefix(reasoning)
    for index, turn in enumerate(mutable):
        if turn.role == "system":
            prompt += turn.content
            if tools:
                prompt += "\n\n" + _TOOL_INSTRUCTIONS.format(tool_schemas=_tool_schema_lines(tools))
            continue

        if turn.role == "user":
            prompt += _USER + turn.content
            next_is_assistant = index + 1 < len(mutable) and mutable[index + 1].role == "assistant"
            if next_is_assistant or index == len(mutable) - 1:
                prompt += _ASSISTANT
                if thinking and (preserve_reasoning or index >= last_user):
                    prompt += _THINK_OPEN
                else:
                    prompt += _THINK_CLOSE
            continue

        if turn.role != "assistant":
            raise ValueError(f"unsupported DeepSeek-V4 turn role: {turn.role!r}")

        if thinking and (preserve_reasoning or index > last_user):
            prompt += (turn.reasoning or "") + _THINK_CLOSE
        prompt += turn.content
        if turn.calls:
            prompt += _encode_calls(turn.calls)
        prompt += _EOS

    return prompt


class DeepSeekV4PromptCompiler:
    capabilities = DEEPSEEK_V4_CAPABILITIES
    stop_conditions = _DEEPSEEK_V4_STOP_CONDITIONS

    def __init__(self, template_adapter: ChatTemplateAdapter) -> None:
        self._template_adapter = template_adapter
        self._parser_contexts: OrderedDict[str, DeepSeekV4ParserContext] = OrderedDict()
        self._parser_context_lock = threading.Lock()

    def take_parser_context(self, request_id: str) -> DeepSeekV4ParserContext | None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        with self._parser_context_lock:
            return self._parser_contexts.pop(request_id, None)

    def compile(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> CompiledPrompt:
        if not isinstance(request, CanonicalRequest):
            raise TypeError("request must be a CanonicalRequest")
        if not isinstance(reasoning, ReasoningPolicy):
            raise TypeError("reasoning must be a ReasoningPolicy")
        if not isinstance(tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy")
        turns = _canonical_turns(request)
        tools = _exposed_tools(tool_policy)
        parser_context = _parser_context_from_tools(tools)
        with self._parser_context_lock:
            self._parser_contexts[request.request_id] = parser_context
            self._parser_contexts.move_to_end(request.request_id)
            while len(self._parser_contexts) > 256:
                self._parser_contexts.popitem(last=False)
        text = _render_prompt(turns, tools, reasoning)
        rendered = self._template_adapter.tokenize_encoded_prompt(text)
        snapshot = _template_snapshot(turns, tools)
        return CompiledPrompt(
            text=rendered.text,
            input_ids=rendered.input_ids,
            prompt_hash=_prompt_hash(rendered.input_ids),
            stop_conditions=self.stop_conditions,
            template_request=snapshot,
            runtime_attachments=rendered.runtime_attachments,
            raw_output_is_text_only=reasoning.mode is ReasoningMode.DISABLED and not tools,
            structured_output_trigger=(
                _THINK_CLOSE if reasoning.mode is not ReasoningMode.DISABLED and not tools else None
            ),
        )


@dataclass(frozen=True, slots=True)
class DeepSeekV4ParserFinish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool


def _deterministic_call_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0deepseek-v4\0{index}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _longest_partial_suffix(text: str, markers: tuple[str, ...]) -> int:
    longest = 0
    for marker in markers:
        limit = min(len(text), len(marker) - 1)
        for size in range(1, limit + 1):
            if marker.startswith(text[-size:]):
                longest = max(longest, size)
    return longest


def _valid_name(value: str) -> bool:
    return bool(value) and not any(character.isspace() or character in '<>"' for character in value)


def _unwrap_wrapper_arguments(
    tool_name: str,
    arguments: dict[str, JsonValue],
    tool_properties: dict[str, frozenset[str]],
) -> dict[str, JsonValue]:
    properties = tool_properties.get(tool_name)
    if not properties:
        return arguments
    for wrapper in ("arguments", "input"):
        if set(arguments) != {wrapper} or wrapper in properties:
            continue
        inner: JsonValue = arguments[wrapper]
        if isinstance(inner, str):
            try:
                inner = parse_json_strict(inner)
            except InvalidJsonError:
                continue
        if isinstance(inner, dict) and set(inner).issubset(properties):
            return inner
    return arguments


def _parse_invoke_body(
    body: str,
    tool_properties: dict[str, frozenset[str]],
) -> tuple[str, str]:
    if not body.startswith(_INVOKE_OPEN_PREFIX):
        raise ValueError("DeepSeek-V4 invoke is missing its opening tag")
    name_start = len(_INVOKE_OPEN_PREFIX)
    name_end = body.find('">', name_start)
    if name_end < 0:
        raise ValueError("DeepSeek-V4 invoke name is incomplete")
    name = body[name_start:name_end]
    if not _valid_name(name):
        raise ValueError("invalid DeepSeek-V4 tool name")
    cursor = name_end + 2
    if cursor < len(body) and body[cursor] == "\n":
        cursor += 1
    if not body.endswith(_INVOKE_CLOSE):
        raise ValueError("DeepSeek-V4 invoke is incomplete")
    payload = body[cursor : -len(_INVOKE_CLOSE)]
    payload = payload.removesuffix("\n")

    if _PARAMETER_OPEN_PREFIX not in payload:
        stripped = payload.strip()
        if not stripped:
            return name, "{}"
        try:
            direct = parse_json_strict(stripped)
        except InvalidJsonError as exc:
            raise ValueError("DeepSeek-V4 direct tool payload must be a JSON object") from exc
        if not isinstance(direct, dict):
            raise ValueError("DeepSeek-V4 direct tool payload must be a JSON object")
        return name, canonical_json_dumps(_unwrap_wrapper_arguments(name, direct, tool_properties))

    arguments: dict[str, JsonValue] = {}
    cursor = 0
    while cursor < len(payload):
        while cursor < len(payload) and payload[cursor].isspace():
            cursor += 1
        if cursor == len(payload):
            break
        if not payload.startswith(_PARAMETER_OPEN_PREFIX, cursor):
            raise ValueError("unexpected text between DeepSeek-V4 parameters")
        key_start = cursor + len(_PARAMETER_OPEN_PREFIX)
        key_end = payload.find('" string="', key_start)
        if key_end < 0:
            raise ValueError("DeepSeek-V4 parameter name is incomplete")
        key = payload[key_start:key_end]
        if not key or key in arguments:
            raise ValueError("invalid or duplicate DeepSeek-V4 parameter name")
        flag_start = key_end + len('" string="')
        flag_end = payload.find('">', flag_start)
        if flag_end < 0:
            raise ValueError("DeepSeek-V4 parameter string flag is incomplete")
        string_flag = payload[flag_start:flag_end]
        if string_flag not in {"true", "false"}:
            raise ValueError("invalid DeepSeek-V4 parameter string flag")
        value_start = flag_end + 2
        value_end = payload.find(_PARAMETER_CLOSE, value_start)
        if value_end < 0:
            raise ValueError("DeepSeek-V4 parameter value is incomplete")
        raw_value = payload[value_start:value_end]
        if string_flag == "true":
            value: JsonValue = raw_value
        else:
            try:
                value = parse_json_strict(raw_value.strip())
            except InvalidJsonError as exc:
                raise ValueError(
                    "DeepSeek-V4 non-string parameter must contain valid JSON"
                ) from exc
        arguments[key] = value
        cursor = value_end + len(_PARAMETER_CLOSE)

    return name, canonical_json_dumps(_unwrap_wrapper_arguments(name, arguments, tool_properties))


def _parse_tool_block(
    block: str,
    tool_properties: dict[str, frozenset[str]],
) -> list[tuple[str, str]]:
    if not block.startswith(_TOOL_BLOCK_OPEN) or not block.endswith(_TOOL_BLOCK_CLOSE):
        raise ValueError("DeepSeek-V4 tool block is incomplete")
    body = block[len(_TOOL_BLOCK_OPEN) : -len(_TOOL_BLOCK_CLOSE)]
    cursor = 0
    calls: list[tuple[str, str]] = []
    while cursor < len(body):
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if cursor == len(body):
            break
        if not body.startswith(_INVOKE_OPEN_PREFIX, cursor):
            raise ValueError("unexpected text inside DeepSeek-V4 tool block")
        close = body.find(_INVOKE_CLOSE, cursor)
        if close < 0:
            raise ValueError("DeepSeek-V4 invoke is incomplete")
        end = close + len(_INVOKE_CLOSE)
        calls.append(_parse_invoke_body(body[cursor:end], tool_properties))
        cursor = end
    if not calls:
        raise ValueError("DeepSeek-V4 tool block must contain at least one invoke")
    return calls


class DeepSeekV4IncrementalParser:
    """Chunk-safe parser for DeepSeek-V4 thinking and DSML tool-call output."""

    def __init__(
        self,
        request_id: str,
        *,
        start_in_reasoning: bool = True,
        parser_context: DeepSeekV4ParserContext | None = None,
    ) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(start_in_reasoning, bool):
            raise TypeError("start_in_reasoning must be a bool")
        if parser_context is None:
            parser_context = DeepSeekV4ParserContext(False, {})
        if not isinstance(parser_context, DeepSeekV4ParserContext):
            raise TypeError("parser_context must be a DeepSeekV4ParserContext or None")
        self._request_id = request_id
        self._tools_enabled = parser_context.tools_enabled
        self._tool_properties = parser_context.tool_properties
        self._buffer = ""
        self._mode = "reasoning" if start_in_reasoning else "text"
        self._reasoning_open = False
        self._reasoning_value = ""
        self._text_open = False
        self._text_value = ""
        self._call_index = 0
        self._in_tool = False
        self._bare_tool = False
        self._had_incomplete_tool = False
        self._finished = False

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

    def _emit_tool_block(self, block: str, events: list[GenerationEvent]) -> None:
        self._close_channel(events)
        try:
            calls = _parse_tool_block(block, self._tool_properties)
            if any(name not in self._tool_properties for name, _ in calls):
                raise ValueError("DeepSeek-V4 tool call references an undeclared tool")
        except ValueError:
            self._had_incomplete_tool = True
            return
        for name, arguments_json in calls:
            index = self._call_index
            call_id = _deterministic_call_id(self._request_id, index)
            events.append(ToolCallStarted(self._request_id, call_id, name, index))
            events.append(ToolCallArgumentsDelta(self._request_id, call_id, arguments_json, index))
            events.append(
                ToolCallCompleted(
                    self._request_id,
                    ToolCallItem(call_id, name, arguments_json, index),
                )
            )
            self._call_index += 1
        self._mode = "text"

    def feed(self, text: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            return ()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return ()
        self._buffer += text
        events: list[GenerationEvent] = []

        markers: tuple[str, ...] = (_THINK_OPEN, _THINK_CLOSE, _EOS)
        if self._tools_enabled:
            markers += _TOOL_START_MARKERS

        while self._buffer:
            if self._in_tool:
                close_at = self._buffer.find(_TOOL_BLOCK_CLOSE)
                if close_at < 0:
                    return tuple(events)
                end = close_at + len(_TOOL_BLOCK_CLOSE)
                if self._bare_tool:
                    payload = self._buffer[:close_at].rstrip()
                    block = f"{_TOOL_BLOCK_OPEN}\n{payload}\n{_TOOL_BLOCK_CLOSE}"
                else:
                    block = self._buffer[:end]
                self._buffer = self._buffer[end:]
                self._in_tool = False
                self._bare_tool = False
                self._emit_tool_block(block, events)
                continue

            positions = [(self._buffer.find(marker), marker) for marker in markers]
            positions = [(position, marker) for position, marker in positions if position >= 0]
            if not positions:
                hold = _longest_partial_suffix(self._buffer, markers)
                emit = self._buffer[:-hold] if hold else self._buffer
                self._buffer = self._buffer[-hold:] if hold else ""
                self._emit_content(emit, events)
                break

            position, marker = min(positions, key=lambda item: item[0])
            if position > 0:
                self._emit_content(self._buffer[:position], events)
                self._buffer = self._buffer[position:]
                continue

            if marker == _THINK_OPEN:
                self._buffer = self._buffer[len(_THINK_OPEN) :]
                if self._mode != "reasoning":
                    self._close_channel(events)
                    self._mode = "reasoning"
                continue
            if marker == _THINK_CLOSE:
                self._buffer = self._buffer[len(_THINK_CLOSE) :]
                if self._mode == "reasoning":
                    self._close_channel(events)
                self._mode = "text"
                continue
            if marker == _EOS:
                self._buffer = self._buffer[len(_EOS) :]
                self._close_channel(events)
                continue

            leading = "\n\n" if marker.startswith("\n\n") else ""
            if leading:
                self._buffer = self._buffer[len(leading) :]
            malformed_wrapper = marker in _MALFORMED_TOOL_OPEN_MARKERS
            if malformed_wrapper:
                if not self._buffer.startswith(_MALFORMED_TOOL_BLOCK_OPEN):
                    self._had_incomplete_tool = True
                    break
                self._buffer = self._buffer[len(_MALFORMED_TOOL_BLOCK_OPEN) :]
            self._close_channel(events)
            self._mode = "text"
            self._in_tool = True
            self._bare_tool = malformed_wrapper or marker in _BARE_INVOKE_MARKERS

        return tuple(events)

    def finish(self) -> DeepSeekV4ParserFinish:
        if self._finished:
            return DeepSeekV4ParserFinish((), self._had_incomplete_tool)
        self._finished = True
        events: list[GenerationEvent] = []
        if self._in_tool:
            if self._bare_tool:
                payload = self._buffer.rstrip()
                block = f"{_TOOL_BLOCK_OPEN}\n{payload}\n{_TOOL_BLOCK_CLOSE}"
                self._emit_tool_block(block, events)
            else:
                self._had_incomplete_tool = True
            self._buffer = ""
            self._in_tool = False
            self._bare_tool = False
        elif self._tools_enabled and self._buffer.startswith(_TOOL_BLOCK_OPEN):
            self._had_incomplete_tool = True
            self._buffer = ""
        elif self._buffer:
            partial_tool_open = (
                _longest_partial_suffix(self._buffer, _TOOL_START_MARKERS)
                if self._tools_enabled
                else 0
            )
            if partial_tool_open:
                content = self._buffer[:-partial_tool_open]
                self._emit_content(content, events)
                self._had_incomplete_tool = True
            else:
                self._emit_content(self._buffer, events)
            self._buffer = ""
        self._close_channel(events)
        return DeepSeekV4ParserFinish(tuple(events), self._had_incomplete_tool)
