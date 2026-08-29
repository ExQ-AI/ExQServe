"""Protocol-neutral contracts shared by model dialect implementations."""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Protocol

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolPolicy
from exqserve.core.events import GenerationEvent
from exqserve.core.request import CanonicalRequest

type TemplateScalar = str | int | float | bool | None


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _validate_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_token_ids(input_ids: tuple[int, ...]) -> None:
    if not isinstance(input_ids, tuple):
        raise TypeError("input_ids must be a tuple")
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in input_ids):
        raise TypeError("input_ids must contain only integers")
    if any(token_id < 0 for token_id in input_ids):
        raise ValueError("input_ids must be non-negative")


def _validate_scalar(value: TemplateScalar) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("template scalar floats must be finite")
    if not isinstance(value, str | int | float | bool | None):
        raise TypeError("template kwargs values must be scalar JSON values")


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    reasoning: bool
    tool_calling: bool
    parallel_tool_calls: bool
    system_role: bool
    developer_role: bool
    reasoning_history: bool
    vision: bool = False

    def __post_init__(self) -> None:
        for name in (
            "reasoning",
            "tool_calling",
            "parallel_tool_calls",
            "system_role",
            "developer_role",
            "reasoning_history",
            "vision",
        ):
            _validate_bool(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class TemplateToolCall:
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        _validate_non_empty("name", self.name)
        if not isinstance(self.arguments_json, str):
            raise TypeError("arguments_json must be a string")


@dataclass(frozen=True, slots=True)
class TemplateTool:
    name: str
    description: str | None
    parameters_json: str

    def __post_init__(self) -> None:
        _validate_non_empty("name", self.name)
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string or None")
        if not isinstance(self.parameters_json, str):
            raise TypeError("parameters_json must be a string")


@dataclass(frozen=True, slots=True)
class TemplateToolResponse:
    name: str
    response_json: str

    def __post_init__(self) -> None:
        _validate_non_empty("name", self.name)
        if not isinstance(self.response_json, str):
            raise TypeError("response_json must be a string")


@dataclass(frozen=True, slots=True)
class TemplateTextPart:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class TemplateImagePart:
    source: str
    detail: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty("source", self.source)
        if self.detail is not None:
            if not isinstance(self.detail, str):
                raise TypeError("detail must be a string or None")
            if self.detail not in {"auto", "low", "high"}:
                raise ValueError("detail must be auto, low, high, or None")


type TemplateContentPart = TemplateTextPart | TemplateImagePart


@dataclass(frozen=True, slots=True)
class TemplateMessage:
    role: str
    content: str | tuple[TemplateContentPart, ...]
    reasoning_content: str | None = None
    tool_calls: tuple[TemplateToolCall, ...] = ()
    tool_responses: tuple[TemplateToolResponse, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty("role", self.role)
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported template role: {self.role!r}")
        if isinstance(self.content, tuple):
            if self.role not in {"user", "tool"}:
                raise ValueError("multimodal template content is supported only for user/tool messages")
            if not self.content:
                raise ValueError("multimodal template content must not be empty")
            if not all(isinstance(part, TemplateTextPart | TemplateImagePart) for part in self.content):
                raise TypeError("multimodal content must contain only template text/image parts")
            if not any(isinstance(part, TemplateImagePart) for part in self.content):
                raise ValueError("multimodal template content must contain an image part")
        elif not isinstance(self.content, str):
            raise TypeError("content must be a string or content-part tuple")
        if self.reasoning_content is not None and not isinstance(self.reasoning_content, str):
            raise TypeError("reasoning_content must be a string or None")
        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple")
        if not all(isinstance(call, TemplateToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must contain only TemplateToolCall values")
        if not isinstance(self.tool_responses, tuple):
            raise TypeError("tool_responses must be a tuple")
        if not all(isinstance(response, TemplateToolResponse) for response in self.tool_responses):
            raise TypeError("tool_responses must contain only TemplateToolResponse values")
        if self.tool_responses and self.role not in {"assistant", "tool"}:
            raise ValueError("tool_responses are supported only for assistant/tool messages")
        for field_name in ("tool_call_id", "name"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_non_empty(field_name, value)


@dataclass(frozen=True, slots=True)
class TemplateRequest:
    messages: tuple[TemplateMessage, ...]
    tools: tuple[TemplateTool, ...]
    template_kwargs: tuple[tuple[str, TemplateScalar], ...]
    add_generation_prompt: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            raise TypeError("messages must be a tuple")
        if not all(isinstance(message, TemplateMessage) for message in self.messages):
            raise TypeError("messages must contain only TemplateMessage values")
        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple")
        if not all(isinstance(tool, TemplateTool) for tool in self.tools):
            raise TypeError("tools must contain only TemplateTool values")
        if not isinstance(self.template_kwargs, tuple):
            raise TypeError("template_kwargs must be a tuple")

        seen_keys: set[str] = set()
        for entry in self.template_kwargs:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("template_kwargs entries must be (name, value) tuples")
            key, value = entry
            _validate_non_empty("template kwarg name", key)
            if key in seen_keys:
                raise ValueError(f"duplicate template kwarg: {key!r}")
            seen_keys.add(key)
            _validate_scalar(value)

        _validate_bool("add_generation_prompt", self.add_generation_prompt)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    input_ids: tuple[int, ...]
    runtime_attachments: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_token_ids(self.input_ids)
        if not isinstance(self.runtime_attachments, tuple):
            raise TypeError("runtime_attachments must be a tuple")


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    text: str
    input_ids: tuple[int, ...]
    prompt_hash: str
    stop_conditions: tuple[str | int, ...]
    template_request: TemplateRequest
    runtime_attachments: tuple[object, ...] = ()
    raw_output_is_text_only: bool = False
    structured_output_trigger: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_token_ids(self.input_ids)
        if (
            not isinstance(self.prompt_hash, str)
            or len(self.prompt_hash) != 64
            or any(char not in string.hexdigits.lower() for char in self.prompt_hash)
            or self.prompt_hash != self.prompt_hash.lower()
        ):
            raise ValueError("prompt_hash must be 64 lowercase hexadecimal characters")
        if not isinstance(self.stop_conditions, tuple):
            raise TypeError("stop_conditions must be a tuple")
        for condition in self.stop_conditions:
            if isinstance(condition, str):
                if condition == "":
                    raise ValueError("string stop conditions must not be empty")
            elif isinstance(condition, int) and not isinstance(condition, bool):
                if condition < 0:
                    raise ValueError("integer stop conditions must be non-negative")
            else:
                raise TypeError("stop conditions must be strings or integers")
        if not isinstance(self.template_request, TemplateRequest):
            raise TypeError("template_request must be a TemplateRequest")
        if not isinstance(self.runtime_attachments, tuple):
            raise TypeError("runtime_attachments must be a tuple")
        _validate_bool("raw_output_is_text_only", self.raw_output_is_text_only)
        if self.structured_output_trigger is not None:
            if not isinstance(self.structured_output_trigger, str):
                raise TypeError("structured_output_trigger must be a string or None")
            if not self.structured_output_trigger.strip():
                raise ValueError("structured_output_trigger must not be empty")


class ChatTemplateAdapter(Protocol):
    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        """Render one deterministic template request using the loaded model assets."""
        ...

    def tokenize_encoded_prompt(self, text: str) -> RenderedPrompt:
        """Tokenize a model-native prompt that already owns its BOS/special-token envelope."""
        ...


class PromptCompilerLike(Protocol):
    def compile(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> CompiledPrompt:
        ...


class ParserFinishLike(Protocol):
    @property
    def events(self) -> tuple[GenerationEvent, ...]:
        ...

    @property
    def incomplete_tool_call(self) -> bool:
        ...


class IncrementalParserLike(Protocol):
    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        ...

    def finish(self) -> ParserFinishLike:
        ...
