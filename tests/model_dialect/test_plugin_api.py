from __future__ import annotations

from dataclasses import dataclass

import pytest

from exqserve.model.registry import (
    ModelDialectPluginError,
    ModelDialectSelectionError,
    default_model_dialect_registry,
    discover_model_dialect_plugins,
)
from exqserve.plugin_api import (
    MODEL_DIALECT_ENTRY_POINT_GROUP,
    MODEL_DIALECT_PLUGIN_API_VERSION,
    CanonicalRequest,
    ChatTemplateAdapter,
    CompiledPrompt,
    GenerationEvent,
    MessageItem,
    MessageRole,
    ModelCapabilities,
    ModelDialectPluginRegistration,
    ReasoningPolicy,
    RenderedPrompt,
    TemplateMessage,
    TemplateRequest,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolChoice,
    ToolChoiceMode,
    ToolPolicy,
)

_CAPABILITIES = ModelCapabilities(
    reasoning=False,
    tool_calling=False,
    parallel_tool_calls=False,
    system_role=True,
    developer_role=False,
    reasoning_history=False,
)
_POLICY = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)


class _Adapter:
    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        text = "|".join(str(message.content) for message in request.messages)
        return RenderedPrompt(text, (1, 2, 3))

    def tokenize_encoded_prompt(self, text: str) -> RenderedPrompt:
        return RenderedPrompt(text, (1, 2, 3))


class _Compiler:
    capabilities = _CAPABILITIES

    def __init__(self, adapter: ChatTemplateAdapter) -> None:
        self._adapter = adapter

    def compile(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> CompiledPrompt:
        del reasoning, tool_policy
        messages = tuple(
            TemplateMessage(item.role.value, item.text)
            for item in request.items
            if isinstance(item, MessageItem)
        )
        template_request = TemplateRequest(messages, (), ())
        rendered = self._adapter.render_and_tokenize(template_request)
        return CompiledPrompt(
            rendered.text,
            rendered.input_ids,
            "a" * 64,
            ("<stop>",),
            template_request,
        )


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool = False


class _Parser:
    def __init__(self, request_id: str) -> None:
        self._request_id = request_id
        self._text = ""

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        self._text += chunk
        return (TextStarted(self._request_id), TextDelta(self._request_id, chunk))

    def finish(self) -> _Finish:
        return _Finish((TextCompleted(self._request_id, self._text),))


@dataclass(frozen=True)
class _Dialect:
    dialect_id: str = "external-test"
    architecture: str = "ExternalForCausalLM"
    capabilities: ModelCapabilities = _CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        return architecture == self.architecture

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> _Compiler:
        return _Compiler(template_adapter)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> _Parser:
        del reasoning, tool_policy
        return _Parser(request_id)


class _EntryPoint:
    def __init__(self, name: str, loaded: object, *, error: Exception | None = None) -> None:
        self.name = name
        self.group = MODEL_DIALECT_ENTRY_POINT_GROUP
        self._loaded = loaded
        self._error = error

    def load(self) -> object:
        if self._error is not None:
            raise self._error
        return self._loaded


def _entry_point(
    dialect: _Dialect,
    *,
    api_version: int = MODEL_DIALECT_PLUGIN_API_VERSION,
    name: str = "external-plugin",
) -> _EntryPoint:
    registration = ModelDialectPluginRegistration(api_version, (dialect,))
    return _EntryPoint(name, registration)


def test_plugin_api_v1_is_protocol_neutral_and_sufficient_for_compiler_parser() -> None:
    assert MODEL_DIALECT_PLUGIN_API_VERSION == 1
    dialect = _Dialect()
    compiler = dialect.create_compiler(_Adapter())
    request = CanonicalRequest(
        "req-plugin",
        "model",
        (MessageItem(MessageRole.USER, "hello"),),
    )
    compiled = compiler.compile(request, ReasoningPolicy(), _POLICY)
    parser = dialect.create_parser(request.request_id, ReasoningPolicy(), _POLICY)
    events = (*parser.feed("answer"), *parser.finish().events)

    assert compiled.text == "hello"
    assert compiled.input_ids == (1, 2, 3)
    assert events == (
        TextStarted("req-plugin"),
        TextDelta("req-plugin", "answer"),
        TextCompleted("req-plugin", "answer"),
    )


def test_plugin_entry_point_discovery_and_explicit_selection() -> None:
    dialect = _Dialect()
    entry = _entry_point(dialect)
    discovered = discover_model_dialect_plugins((entry,))  # type: ignore[arg-type]
    registry = default_model_dialect_registry(entry_points=(entry,))  # type: ignore[arg-type]

    assert discovered == (dialect,)
    assert registry.resolve("ExternalForCausalLM").dialect_id == "external-test"
    assert registry.resolve("AnythingElse", "external-test") is dialect


def test_explicit_plugin_selection_handles_architecture_sharing_fine_tune() -> None:
    dialect = _Dialect("qwen-custom", "Qwen3_5ForConditionalGeneration")
    registry = default_model_dialect_registry(
        entry_points=(_entry_point(dialect),)  # type: ignore[arg-type]
    )

    with pytest.raises(ModelDialectSelectionError, match="ambiguous"):
        registry.resolve("Qwen3_5ForConditionalGeneration")
    assert registry.resolve("Qwen3_5ForConditionalGeneration", "qwen-custom") is dialect


def test_duplicate_plugin_dialect_id_fails_registry_construction() -> None:
    duplicate = _Dialect("qwen", "OtherArchitecture")
    with pytest.raises(ModelDialectSelectionError, match="duplicate model dialect id.*qwen"):
        default_model_dialect_registry(
            entry_points=(_entry_point(duplicate),)  # type: ignore[arg-type]
        )


def test_unknown_explicit_dialect_never_falls_back() -> None:
    registry = default_model_dialect_registry(entry_points=())
    with pytest.raises(ModelDialectSelectionError, match="unknown model dialect"):
        registry.resolve("UnknownArchitecture", "missing-plugin")


def test_plugin_api_version_mismatch_is_stable_startup_error() -> None:
    entry = _entry_point(_Dialect(), api_version=2, name="future-plugin")
    with pytest.raises(ModelDialectPluginError, match="future-plugin.*API version 2"):
        discover_model_dialect_plugins((entry,))  # type: ignore[arg-type]


def test_plugin_load_failure_is_normalized_without_leaking_internal_exception() -> None:
    entry = _EntryPoint(
        "broken-plugin",
        object(),
        error=RuntimeError("private plugin traceback detail"),
    )
    with pytest.raises(ModelDialectPluginError) as exc_info:
        discover_model_dialect_plugins((entry,))  # type: ignore[arg-type]

    assert "broken-plugin" in str(exc_info.value)
    assert "private plugin traceback detail" not in str(exc_info.value)


def test_plugin_wrong_registration_shape_is_rejected() -> None:
    entry = _EntryPoint("wrong-plugin", _Dialect())
    with pytest.raises(ModelDialectPluginError, match="did not return"):
        discover_model_dialect_plugins((entry,))  # type: ignore[arg-type]
