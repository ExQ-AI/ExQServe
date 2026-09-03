"""OpenAI Chat Completions request codec over serving-core semantics."""

from __future__ import annotations

import json

from exqserve.agent.schema import JsonSchema, validate_strict_function_schema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.items import (
    CanonicalItem,
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.core.sampling import SamplingOverridePolicy
from exqserve.protocol.openai.common import (
    OpenAIProtocol,
    ParsedOpenAIRequest,
    invalid_request,
    parse_reasoning_effort,
    parse_sampling,
    parse_stop,
)
from exqserve.serving.contracts import ServingRequest


def _as_dict(value: object, *, code: str, param: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise invalid_request(code, f"{param} must be an object.", param)
    return value


def _text_content(value: object, *, param: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise invalid_request("invalid_message_content", "Message content must be text.", param)
    parts: list[str] = []
    for index, raw_part in enumerate(value):
        part = _as_dict(
            raw_part,
            code="invalid_message_content",
            param=f"{param}[{index}]",
        )
        if part.get("type") != "text":
            raise invalid_request(
                "unsupported_content_part",
                "Only text content is supported for this message role.",
                f"{param}[{index}].type",
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise invalid_request(
                "invalid_message_content",
                "Text content part must contain a string.",
                f"{param}[{index}].text",
            )
        parts.append(text)
    return "".join(parts)


def _user_content(value: object, *, param: str) -> str | tuple[TextContentPart | ImageContentPart, ...]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise invalid_request(
            "invalid_message_content",
            "User message content must be text or a non-empty content-part array.",
            param,
        )

    parts: list[TextContentPart | ImageContentPart] = []
    has_image = False
    for index, raw_part in enumerate(value):
        part_param = f"{param}[{index}]"
        part = _as_dict(raw_part, code="invalid_message_content", param=part_param)
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise invalid_request(
                    "invalid_message_content",
                    "Text content part must contain a string.",
                    f"{part_param}.text",
                )
            parts.append(TextContentPart(text))
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                raise invalid_request(
                    "invalid_image_url",
                    "image_url content part must contain an image_url object.",
                    f"{part_param}.image_url",
                )
            source = image_url.get("url")
            detail = image_url.get("detail")
            if not isinstance(source, str) or not source.strip():
                raise invalid_request(
                    "invalid_image_url",
                    "image_url.url must be a non-empty string.",
                    f"{part_param}.image_url.url",
                )
            if detail is not None and detail not in {"auto", "low", "high"}:
                raise invalid_request(
                    "invalid_image_detail",
                    "image_url.detail must be auto, low, or high.",
                    f"{part_param}.image_url.detail",
                )
            parts.append(ImageContentPart(source, detail if isinstance(detail, str) else None))
            has_image = True
            continue
        raise invalid_request(
            "unsupported_content_part",
            "Only text and image_url user content parts are supported.",
            f"{part_param}.type",
        )

    if not has_image:
        return "".join(part.text for part in parts if isinstance(part, TextContentPart))
    return tuple(parts)


def _assistant_tool_calls(value: object, *, message_index: int) -> tuple[ToolCallItem, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise invalid_request("invalid_tool_calls", "tool_calls must be an array.", "messages")
    calls: list[ToolCallItem] = []
    for index, raw_call in enumerate(value):
        call = _as_dict(
            raw_call,
            code="invalid_tool_calls",
            param=f"messages[{message_index}].tool_calls[{index}]",
        )
        if call.get("type") != "function":
            raise invalid_request(
                "unsupported_tool_type",
                "Only function tool calls are supported in V1.",
                f"messages[{message_index}].tool_calls[{index}].type",
            )
        call_id = call.get("id")
        function = call.get("function")
        if not isinstance(call_id, str) or not call_id.strip() or not isinstance(function, dict):
            raise invalid_request("invalid_tool_calls", "Function tool call is malformed.", "messages")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
            raise invalid_request("invalid_tool_calls", "Function tool call is malformed.", "messages")
        calls.append(ToolCallItem(call_id, name, arguments, index))
    return tuple(calls)


def _parse_messages(value: object) -> tuple[CanonicalItem, ...]:
    if not isinstance(value, list) or not value:
        raise invalid_request("invalid_messages", "messages must be a non-empty array.", "messages")
    items: list[CanonicalItem] = []
    role_map = {
        "system": MessageRole.SYSTEM,
        "developer": MessageRole.DEVELOPER,
    }
    for index, raw_message in enumerate(value):
        message = _as_dict(raw_message, code="invalid_messages", param=f"messages[{index}]")
        role = message.get("role")
        if role in role_map:
            content = _text_content(message.get("content"), param=f"messages[{index}].content")
            assert content is not None
            items.append(MessageItem(role_map[role], content))
            continue
        if role == "user":
            user_content = _user_content(message.get("content"), param=f"messages[{index}].content")
            if isinstance(user_content, str):
                items.append(MessageItem(MessageRole.USER, user_content))
            else:
                items.append(MultimodalMessageItem(MessageRole.USER, user_content))
            continue
        if role == "assistant":
            if "function_call" in message:
                raise invalid_request(
                    "unsupported_function_call",
                    "Deprecated function_call messages are not supported in V1.",
                    f"messages[{index}].function_call",
                )
            reasoning = message.get("reasoning_content")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise invalid_request(
                        "invalid_reasoning_content",
                        "reasoning_content must be a string.",
                        f"messages[{index}].reasoning_content",
                    )
                items.append(ReasoningItem(reasoning))
            content = _text_content(
                message.get("content"),
                param=f"messages[{index}].content",
                allow_none=True,
            )
            if content is not None:
                items.append(MessageItem(MessageRole.ASSISTANT, content))
            items.extend(_assistant_tool_calls(message.get("tool_calls"), message_index=index))
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise invalid_request(
                    "invalid_tool_result",
                    "Tool message requires tool_call_id.",
                    f"messages[{index}].tool_call_id",
                )
            content = _text_content(message.get("content"), param=f"messages[{index}].content")
            assert content is not None
            items.append(ToolResultItem(call_id, content))
            continue
        raise invalid_request(
            "unsupported_message_role",
            "Unsupported Chat message role.",
            f"messages[{index}].role",
        )
    return tuple(items)


def _schema_from_object(value: object, *, param: str) -> JsonSchema:
    if not isinstance(value, dict):
        raise invalid_request("invalid_json_schema", "JSON Schema must be an object.", param)
    try:
        return JsonSchema(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise invalid_request("invalid_json_schema", "JSON Schema is invalid.", param) from exc


def _parse_tools(value: object) -> tuple[FunctionTool, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise invalid_request("invalid_tools", "tools must be an array.", "tools")
    tools: list[FunctionTool] = []
    for index, raw_tool in enumerate(value):
        tool = _as_dict(raw_tool, code="invalid_tools", param=f"tools[{index}]")
        if tool.get("type") != "function":
            raise invalid_request(
                "unsupported_tool_type",
                "Only function tools are supported in V1.",
                f"tools[{index}].type",
            )
        function = tool.get("function")
        if not isinstance(function, dict):
            raise invalid_request("invalid_tools", "Function tool declaration is malformed.", f"tools[{index}]")
        name = function.get("name")
        description = function.get("description")
        strict = function.get("strict", False)
        if not isinstance(name, str) or not name.strip():
            raise invalid_request("invalid_tools", "Function tool name is required.", f"tools[{index}].function.name")
        if description is not None and not isinstance(description, str):
            raise invalid_request("invalid_tools", "Function description must be text.", f"tools[{index}].function.description")
        if not isinstance(strict, bool):
            raise invalid_request("invalid_tools", "Function strict must be boolean.", f"tools[{index}].function.strict")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        schema = _schema_from_object(parameters, param=f"tools[{index}].function.parameters")
        if strict:
            try:
                validate_strict_function_schema(schema)
            except (TypeError, ValueError) as exc:
                raise invalid_request(
                    "invalid_json_schema",
                    f"Strict function schema is invalid: {exc}",
                    f"tools[{index}].function.parameters",
                ) from exc
        try:
            tools.append(FunctionTool(name, description, schema, strict))
        except (TypeError, ValueError) as exc:
            raise invalid_request("invalid_tools", "Function tool declaration is invalid.", f"tools[{index}]") from exc
    return tuple(tools)


def _parse_choice(value: object) -> ToolChoice:
    if value is None:
        return ToolChoice(ToolChoiceMode.AUTO)
    if isinstance(value, str):
        modes = {
            "none": ToolChoiceMode.NONE,
            "auto": ToolChoiceMode.AUTO,
            "required": ToolChoiceMode.REQUIRED,
        }
        mode = modes.get(value)
        if mode is None:
            raise invalid_request("invalid_tool_choice", "Unsupported tool_choice.", "tool_choice")
        return ToolChoice(mode)
    if isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return ToolChoice(ToolChoiceMode.NAMED, function["name"])
    raise invalid_request("invalid_tool_choice", "Unsupported tool_choice.", "tool_choice")


def _parse_tool_policy(body: dict[str, object]) -> ToolPolicy:
    tools = _parse_tools(body.get("tools"))
    choice = _parse_choice(body.get("tool_choice"))
    allow_parallel = body.get("parallel_tool_calls", True)
    if not isinstance(allow_parallel, bool):
        raise invalid_request("invalid_parallel_tool_calls", "parallel_tool_calls must be boolean.", "parallel_tool_calls")
    try:
        return ToolPolicy(tools, choice, allow_parallel)
    except (TypeError, ValueError) as exc:
        raise invalid_request("invalid_tool_choice", "tool_choice is incompatible with declared tools.", "tool_choice") from exc


def _parse_output_limit(body: dict[str, object], default_value: int | None) -> int:
    modern = body.get("max_completion_tokens")
    legacy = body.get("max_tokens")
    if modern is not None and legacy is not None and modern != legacy:
        raise invalid_request(
            "conflicting_max_tokens",
            "max_completion_tokens and max_tokens must match when both are supplied.",
        )
    value = modern if modern is not None else legacy
    if value is None:
        value = default_value
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise invalid_request("invalid_max_tokens", "A positive output token limit is required.")
    return value


def _parse_structured_output(value: object) -> StructuredOutputSpec | None:
    if value is None:
        return None
    format_value = _as_dict(value, code="invalid_response_format", param="response_format")
    format_type = format_value.get("type")
    if format_type == "text":
        return None
    if format_type == "json_object":
        return StructuredOutputSpec(_schema_from_object({"type": "object"}, param="response_format"))
    if format_type == "json_schema":
        definition = format_value.get("json_schema")
        if not isinstance(definition, dict):
            raise invalid_request("invalid_response_format", "json_schema definition is required.", "response_format.json_schema")
        return StructuredOutputSpec(
            _schema_from_object(definition.get("schema"), param="response_format.json_schema.schema")
        )
    raise invalid_request("invalid_response_format", "Unsupported response_format.", "response_format.type")


class ChatRequestAdapter:
    def __init__(
        self,
        default_max_output_tokens: int | None = None,
        sampling_overrides: SamplingOverridePolicy | None = None,
    ) -> None:
        if default_max_output_tokens is not None and (
            not isinstance(default_max_output_tokens, int)
            or isinstance(default_max_output_tokens, bool)
            or default_max_output_tokens <= 0
        ):
            raise ValueError("default_max_output_tokens must be a positive integer or None")
        if sampling_overrides is not None and not isinstance(sampling_overrides, SamplingOverridePolicy):
            raise TypeError("sampling_overrides must be SamplingOverridePolicy or None")
        self._default_output_limit = default_max_output_tokens
        self._sampling_overrides = sampling_overrides

    def parse(self, body: dict[str, object], *, request_id: str) -> ParsedOpenAIRequest:
        if not isinstance(body, dict):
            raise TypeError("body must be a dictionary")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise invalid_request("invalid_model", "model must be a non-empty string.", "model")
        n = body.get("n", 1)
        if isinstance(n, bool) or n != 1:
            raise invalid_request("unsupported_n", "Only n=1 is supported in V1.", "n")
        if body.get("logprobs") not in {None, False}:
            raise invalid_request("unsupported_logprobs", "logprobs are not supported in V1.", "logprobs")

        items = _parse_messages(body.get("messages"))
        policy = _parse_tool_policy(body)
        reasoning = parse_reasoning_effort(body.get("reasoning_effort"), param="reasoning_effort")
        output_limit = _parse_output_limit(body, self._default_output_limit)
        structured = _parse_structured_output(body.get("response_format"))
        sampling = parse_sampling(body, self._sampling_overrides)
        seed = body.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise invalid_request("invalid_seed", "seed must be an integer.", "seed")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise invalid_request("invalid_stream", "stream must be boolean.", "stream")
        include_usage = False
        stream_options = body.get("stream_options")
        if stream_options is not None:
            options = _as_dict(stream_options, code="invalid_stream_options", param="stream_options")
            include_value = options.get("include_usage", False)
            if not isinstance(include_value, bool):
                raise invalid_request(
                    "invalid_stream_options",
                    "stream_options.include_usage must be boolean.",
                    "stream_options.include_usage",
                )
            include_usage = include_value

        serving = ServingRequest(
            CanonicalRequest(request_id, model, items),
            reasoning,
            policy,
            output_limit,
            structured,
            seed,
            sampling,
            parse_stop(body.get("stop")),
        )
        return ParsedOpenAIRequest(serving, model, stream, OpenAIProtocol.CHAT, include_usage)
