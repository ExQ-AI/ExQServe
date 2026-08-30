"""OpenAI Responses request codec over item-native serving-core semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass

from exqserve.agent.schema import JsonSchema, validate_strict_function_schema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.items import (
    CanonicalItem,
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
from exqserve.core.sampling import SamplingOverridePolicy
from exqserve.protocol.openai.common import (
    OpenAIProtocol,
    invalid_request,
    parse_reasoning_effort,
    parse_sampling,
)
from exqserve.serving.contracts import ServingRequest


def _as_dict(value: object, *, code: str, param: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise invalid_request(code, f"{param} must be an object.", param)
    return value


def _schema_from_object(value: object, *, param: str) -> JsonSchema:
    if not isinstance(value, dict):
        raise invalid_request("invalid_json_schema", "JSON Schema must be an object.", param)
    try:
        return JsonSchema(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise invalid_request("invalid_json_schema", "JSON Schema is invalid.", param) from exc


def _message_text(value: object, *, param: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise invalid_request("invalid_input_message", "Message content must be text.", param)
    parts: list[str] = []
    allowed_types = {"text", "input_text", "output_text"}
    for index, raw_part in enumerate(value):
        part = _as_dict(raw_part, code="invalid_input_message", param=f"{param}[{index}]")
        part_type = part.get("type")
        if part_type not in allowed_types:
            raise invalid_request(
                "unsupported_content_part",
                "Only text content is supported for this message role.",
                f"{param}[{index}].type",
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise invalid_request(
                "invalid_input_message",
                "Text content part must contain a string.",
                f"{param}[{index}].text",
            )
        parts.append(text)
    return "".join(parts)


def _user_message_content(
    value: object,
    *,
    param: str,
) -> str | tuple[TextContentPart | ImageContentPart, ...]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise invalid_request(
            "invalid_input_message",
            "User message content must be text or a non-empty content-part array.",
            param,
        )

    parts: list[TextContentPart | ImageContentPart] = []
    has_image = False
    allowed_text_types = {"text", "input_text", "output_text"}
    for index, raw_part in enumerate(value):
        part_param = f"{param}[{index}]"
        part = _as_dict(raw_part, code="invalid_input_message", param=part_param)
        part_type = part.get("type")
        if part_type in allowed_text_types:
            text = part.get("text")
            if not isinstance(text, str):
                raise invalid_request(
                    "invalid_input_message",
                    "Text content part must contain a string.",
                    f"{part_param}.text",
                )
            parts.append(TextContentPart(text))
            continue
        if part_type == "input_image":
            source = part.get("image_url")
            detail = part.get("detail")
            if not isinstance(source, str) or not source.strip():
                if "file_id" in part:
                    raise invalid_request(
                        "unsupported_image_file",
                        "input_image.file_id is not supported; provide image_url instead.",
                        f"{part_param}.file_id",
                    )
                raise invalid_request(
                    "invalid_image_url",
                    "input_image.image_url must be a non-empty string.",
                    f"{part_param}.image_url",
                )
            if detail is not None and detail not in {"auto", "low", "high"}:
                raise invalid_request(
                    "invalid_image_detail",
                    "input_image.detail must be auto, low, or high.",
                    f"{part_param}.detail",
                )
            parts.append(ImageContentPart(source, detail if isinstance(detail, str) else None))
            has_image = True
            continue
        raise invalid_request(
            "unsupported_content_part",
            "Only text and input_image user content parts are supported.",
            f"{part_param}.type",
        )

    if not has_image:
        return "".join(part.text for part in parts if isinstance(part, TextContentPart))
    return tuple(parts)


def _function_output_content(
    value: object,
    *,
    param: str,
) -> str | tuple[TextContentPart | ImageContentPart, ...]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise invalid_request(
            "unsupported_function_output",
            "Function output must be a string or a non-empty text/image content-part array.",
            param,
        )

    parts: list[TextContentPart | ImageContentPart] = []
    has_image = False
    allowed_text_types = {"text", "input_text", "output_text"}
    for index, raw_part in enumerate(value):
        part_param = f"{param}[{index}]"
        part = _as_dict(raw_part, code="unsupported_function_output", param=part_param)
        part_type = part.get("type")
        if part_type in allowed_text_types:
            text = part.get("text")
            if not isinstance(text, str):
                raise invalid_request(
                    "unsupported_function_output",
                    "Function-output text content part must contain a string.",
                    f"{part_param}.text",
                )
            parts.append(TextContentPart(text))
            continue
        if part_type == "input_image":
            source = part.get("image_url")
            detail = part.get("detail")
            if not isinstance(source, str) or not source.strip():
                if "file_id" in part:
                    raise invalid_request(
                        "unsupported_image_file",
                        "input_image.file_id is not supported; provide image_url instead.",
                        f"{part_param}.file_id",
                    )
                raise invalid_request(
                    "invalid_image_url",
                    "input_image.image_url must be a non-empty string.",
                    f"{part_param}.image_url",
                )
            if detail is not None and detail not in {"auto", "low", "high"}:
                raise invalid_request(
                    "invalid_image_detail",
                    "input_image.detail must be auto, low, or high.",
                    f"{part_param}.detail",
                )
            parts.append(ImageContentPart(source, detail if isinstance(detail, str) else None))
            has_image = True
            continue
        raise invalid_request(
            "unsupported_function_output",
            "Only text and input_image function-output content parts are supported.",
            f"{part_param}.type",
        )

    if not has_image:
        return "".join(part.text for part in parts if isinstance(part, TextContentPart))
    return tuple(parts)


def _reasoning_text(value: object, *, param: str) -> str:
    if not isinstance(value, list):
        raise invalid_request("invalid_reasoning_item", "Reasoning content must be an array.", param)
    parts: list[str] = []
    for index, raw_part in enumerate(value):
        part = _as_dict(raw_part, code="invalid_reasoning_item", param=f"{param}[{index}]")
        if part.get("type") != "reasoning_text":
            raise invalid_request(
                "unsupported_content_part",
                "Only reasoning_text content is supported for reasoning history.",
                f"{param}[{index}].type",
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise invalid_request(
                "invalid_reasoning_item",
                "Reasoning text must be a string.",
                f"{param}[{index}].text",
            )
        parts.append(text)
    return "".join(parts)


def _parse_instructions(value: object) -> tuple[CanonicalItem, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise invalid_request("invalid_instructions", "instructions must be a string.", "instructions")
    return (MessageItem(MessageRole.DEVELOPER, value),)


def _parse_input(value: object) -> tuple[CanonicalItem, ...]:
    items: list[CanonicalItem] = []
    if isinstance(value, str):
        items.append(MessageItem(MessageRole.USER, value))
        return tuple(items)
    if not isinstance(value, list) or not value:
        raise invalid_request("invalid_input", "input must be a string or non-empty item array.", "input")

    role_map = {
        "system": MessageRole.SYSTEM,
        "developer": MessageRole.DEVELOPER,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }
    call_index = 0
    for index, raw_item in enumerate(value):
        item = _as_dict(raw_item, code="invalid_input", param=f"input[{index}]")
        item_type = item.get("type")
        if item_type in {None, "message"}:
            role = item.get("role")
            if role not in role_map:
                raise invalid_request(
                    "unsupported_message_role",
                    "Unsupported Responses message role.",
                    f"input[{index}].role",
                )
            if role == "user":
                content = _user_message_content(item.get("content"), param=f"input[{index}].content")
                if isinstance(content, str):
                    items.append(MessageItem(MessageRole.USER, content))
                else:
                    items.append(MultimodalMessageItem(MessageRole.USER, content))
            else:
                text = _message_text(item.get("content"), param=f"input[{index}].content")
                items.append(MessageItem(role_map[role], text))
            continue
        if item_type == "reasoning":
            items.append(ReasoningItem(_reasoning_text(item.get("content"), param=f"input[{index}].content")))
            continue
        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or not isinstance(name, str)
                or not name.strip()
                or not isinstance(arguments, str)
            ):
                raise invalid_request("invalid_function_call", "Function call item is malformed.", f"input[{index}]")
            items.append(ToolCallItem(call_id, name, arguments, call_index))
            call_index += 1
            continue
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if not isinstance(call_id, str) or not call_id.strip():
                raise invalid_request(
                    "invalid_function_output",
                    "Function output requires call_id.",
                    f"input[{index}].call_id",
                )
            content = _function_output_content(output, param=f"input[{index}].output")
            if isinstance(content, str):
                items.append(ToolResultItem(call_id, content))
            else:
                items.append(MultimodalToolResultItem(call_id, content))
            call_index = 0
            continue
        raise invalid_request(
            "unsupported_input_item",
            "Unsupported Responses input item type.",
            f"input[{index}].type",
        )
    return tuple(items)


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
                "Only function tools are supported in Responses V1.",
                f"tools[{index}].type",
            )
        name = tool.get("name")
        description = tool.get("description")
        strict = tool.get("strict", False)
        if not isinstance(name, str) or not name.strip():
            raise invalid_request("invalid_tools", "Function tool name is required.", f"tools[{index}].name")
        if description is not None and not isinstance(description, str):
            raise invalid_request("invalid_tools", "Function description must be text.", f"tools[{index}].description")
        if not isinstance(strict, bool):
            raise invalid_request("invalid_tools", "Function strict must be boolean.", f"tools[{index}].strict")
        schema = _schema_from_object(
            tool.get("parameters", {"type": "object", "properties": {}}),
            param=f"tools[{index}].parameters",
        )
        if strict:
            try:
                validate_strict_function_schema(schema)
            except (TypeError, ValueError) as exc:
                raise invalid_request(
                    "invalid_json_schema",
                    f"Strict function schema is invalid: {exc}",
                    f"tools[{index}].parameters",
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
        name = value.get("name")
        if isinstance(name, str):
            return ToolChoice(ToolChoiceMode.NAMED, name)
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
    value = body.get("max_output_tokens", default_value)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise invalid_request("invalid_max_output_tokens", "A positive max_output_tokens value is required.", "max_output_tokens")
    return value


def _parse_reasoning(value: object):  # type: ignore[no-untyped-def]
    if value is None:
        return parse_reasoning_effort(None, param="reasoning.effort")
    reasoning = _as_dict(value, code="invalid_reasoning", param="reasoning")
    return parse_reasoning_effort(reasoning.get("effort"), param="reasoning.effort")


def _parse_structured_output(value: object) -> StructuredOutputSpec | None:
    if value is None:
        return None
    text = _as_dict(value, code="invalid_text_config", param="text")
    format_value = text.get("format")
    if format_value is None:
        return None
    format_object = _as_dict(format_value, code="invalid_text_format", param="text.format")
    format_type = format_object.get("type")
    if format_type in {None, "text"}:
        return None
    if format_type == "json_object":
        return StructuredOutputSpec(_schema_from_object({"type": "object"}, param="text.format"))
    if format_type == "json_schema":
        return StructuredOutputSpec(
            _schema_from_object(format_object.get("schema"), param="text.format.schema")
        )
    raise invalid_request("invalid_text_format", "Unsupported Responses text format.", "text.format.type")


def _parse_previous_response_id(body: dict[str, object]) -> str | None:
    previous = body.get("previous_response_id")
    if previous is not None and (not isinstance(previous, str) or not previous.strip()):
        raise invalid_request(
            "invalid_previous_response_id",
            "previous_response_id must be a non-empty string.",
            "previous_response_id",
        )
    return previous


def _parse_state_options(body: dict[str, object]) -> tuple[str | None, bool]:
    if body.get("conversation") is not None:
        raise invalid_request(
            "unsupported_conversation",
            "Responses conversation state is not supported in V1.",
            "conversation",
        )
    if body.get("background") is True:
        raise invalid_request(
            "unsupported_background",
            "Background Responses are not supported in V1.",
            "background",
        )
    previous = _parse_previous_response_id(body)
    store = body.get("store", True)
    if not isinstance(store, bool):
        raise invalid_request("invalid_store", "store must be boolean.", "store")
    return previous, store


def _parse_count_state_options(body: dict[str, object]) -> str | None:
    if body.get("conversation") is not None:
        raise invalid_request(
            "unsupported_conversation",
            "Responses conversation state is not supported in V1.",
            "conversation",
        )
    return _parse_previous_response_id(body)


@dataclass(frozen=True, slots=True)
class ParsedResponsesRequest:
    serving: ServingRequest
    model: str
    stream: bool
    previous_response_id: str | None
    store: bool
    instruction_items: tuple[CanonicalItem, ...]
    state_input_items: tuple[CanonicalItem, ...]

    @property
    def protocol(self) -> OpenAIProtocol:
        return OpenAIProtocol.RESPONSES

    @property
    def include_usage(self) -> bool:
        return False

    def serving_with_context(self, previous_items: tuple[CanonicalItem, ...]) -> ServingRequest:
        canonical = CanonicalRequest(
            self.serving.input.request_id,
            self.model,
            (*self.instruction_items, *previous_items, *self.state_input_items),
        )
        return ServingRequest(
            canonical,
            self.serving.reasoning,
            self.serving.tools,
            self.serving.max_output_tokens,
            self.serving.structured_output,
            self.serving.seed,
            self.serving.sampling,
        )


class ResponsesRequestAdapter:
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

    def parse(self, body: dict[str, object], *, request_id: str) -> ParsedResponsesRequest:
        if not isinstance(body, dict):
            raise TypeError("body must be a dictionary")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise invalid_request("invalid_model", "model must be a non-empty string.", "model")
        previous_response_id, store = _parse_state_options(body)

        instruction_items = _parse_instructions(body.get("instructions"))
        state_input_items = _parse_input(body.get("input"))
        items = (*instruction_items, *state_input_items)
        policy = _parse_tool_policy(body)
        reasoning = _parse_reasoning(body.get("reasoning"))
        output_limit = _parse_output_limit(body, self._default_output_limit)
        structured = _parse_structured_output(body.get("text"))
        sampling = parse_sampling(body, self._sampling_overrides)
        seed = body.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise invalid_request("invalid_seed", "seed must be an integer.", "seed")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise invalid_request("invalid_stream", "stream must be boolean.", "stream")

        serving = ServingRequest(
            CanonicalRequest(request_id, model, items),
            reasoning,
            policy,
            output_limit,
            structured,
            seed,
            sampling,
        )
        return ParsedResponsesRequest(
            serving,
            model,
            stream,
            previous_response_id,
            store,
            instruction_items,
            state_input_items,
        )

    def parse_count(self, body: dict[str, object], *, request_id: str) -> ParsedResponsesRequest:
        if not isinstance(body, dict):
            raise TypeError("body must be a dictionary")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise invalid_request("invalid_model", "model must be a non-empty string.", "model")
        previous_response_id = _parse_count_state_options(body)

        instruction_items = _parse_instructions(body.get("instructions"))
        state_input_items = _parse_input(body.get("input"))
        items = (*instruction_items, *state_input_items)
        policy = _parse_tool_policy(body)
        reasoning = _parse_reasoning(body.get("reasoning"))
        structured = _parse_structured_output(body.get("text"))

        serving = ServingRequest(
            CanonicalRequest(request_id, model, items),
            reasoning,
            policy,
            1,
            structured,
        )
        return ParsedResponsesRequest(
            serving,
            model,
            False,
            previous_response_id,
            False,
            instruction_items,
            state_input_items,
        )


