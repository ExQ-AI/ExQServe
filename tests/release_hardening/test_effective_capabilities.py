from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.items import ImageContentPart, MessageRole, MultimodalMessageItem
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import ToolConstraintMode
from exqserve.model.registry import GenericHFDialect, MuseGlimmerDialect, QwenDialect
from exqserve.runtime.contracts import RuntimeModelMetadata
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime
from exqserve.server.capabilities import (
    CapabilityGuardedPromptCompiler,
    resolve_effective_model_snapshot,
    validate_heterogeneous_switch_overrides,
)
from exqserve.server.config import ServerConfig


def test_effective_snapshot_is_frozen_and_context_authoritative(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(12000, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )
    config = ServerConfig(model_directory=tmp_path, cache_tokens=32768, max_total_tokens=20000)
    snapshot = resolve_effective_model_snapshot(config, QwenDialect(), runtime)
    request_control = config.request_control_config_for_context(snapshot.context_window)

    assert snapshot.context_window == 12000
    assert request_control.max_total_tokens == snapshot.context_window
    with pytest.raises(FrozenInstanceError):
        snapshot.context_window = 1  # type: ignore[misc]


@pytest.mark.parametrize("dialect_vision", (False, True))
@pytest.mark.parametrize("runtime_vision", (False, True))
@pytest.mark.parametrize("vision_loaded", (False, True))
@pytest.mark.parametrize("operator_enabled", (False, True))
def test_effective_vision_matrix_is_conservative(
    tmp_path: Path,
    dialect_vision: bool,
    runtime_vision: bool,
    vision_loaded: bool,
    operator_enabled: bool,
) -> None:
    runtime = SimpleNamespace(
        capabilities=replace(ExLlamaV3Runtime.capabilities, vision=runtime_vision),
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=vision_loaded,
    )
    dialect = (
        QwenDialect()
        if dialect_vision
        else replace(
            GenericHFDialect(),
            capabilities=replace(GenericHFDialect().capabilities, vision=False),
        )
    )
    snapshot = resolve_effective_model_snapshot(
        ServerConfig(model_directory=tmp_path, vision_enabled=operator_enabled),
        dialect,
        runtime,
    )

    assert snapshot.vision_available is (
        dialect_vision and runtime_vision and vision_loaded and operator_enabled
    )


@pytest.mark.parametrize("tokenization, expected", ((False, False), (True, True)))
def test_reasoning_control_requires_explicit_runtime_tokenization(
    tmp_path: Path,
    tokenization: bool,
    expected: bool,
) -> None:
    runtime = SimpleNamespace(
        capabilities=replace(ExLlamaV3Runtime.capabilities, tokenization=tokenization),
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )
    snapshot = resolve_effective_model_snapshot(ServerConfig(tmp_path), QwenDialect(), runtime)
    assert snapshot.reasoning_control_available is expected


def test_muse_structural_provenance_requirement_fails_closed(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        capabilities=replace(
            ExLlamaV3Runtime.capabilities,
            structural_token_provenance=False,
        ),
        model_metadata=RuntimeModelMetadata(65536, "MuseGlimmerForConditionalGeneration"),
        vision_loaded=False,
    )

    with pytest.raises(ValueError, match="requires runtime structural provenance"):
        resolve_effective_model_snapshot(ServerConfig(tmp_path), MuseGlimmerDialect(), runtime)


def test_external_dialect_without_structural_descriptor_gets_no_invented_markers(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "UnknownArchitecture"),
        vision_loaded=False,
    )
    snapshot = resolve_effective_model_snapshot(ServerConfig(tmp_path), GenericHFDialect(), runtime)

    assert snapshot.structural_requirements.prompt_markers == ()
    assert snapshot.structural_requirements.output_markers == ()


def test_safe_heterogeneous_switch_is_allowed_without_bound_overrides(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = ServerConfig(first)
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )
    snapshot = resolve_effective_model_snapshot(config, QwenDialect(), runtime)

    validate_heterogeneous_switch_overrides(config, second, QwenDialect(), snapshot)


@pytest.mark.parametrize("override", ("chat_template", "draft_model", "lora"))
def test_model_bound_overrides_reject_heterogeneous_switch(tmp_path: Path, override: str) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    if override == "chat_template":
        template = tmp_path / "template.jinja"
        template.write_text("template body", encoding="utf-8")
        config = ServerConfig(model_directory=first, chat_template=template)
    elif override == "draft_model":
        config = ServerConfig(model_directory=first, draft_model=tmp_path / "draft")
    else:
        config = ServerConfig(model_directory=first, loras=(tmp_path / "lora",))
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )
    snapshot = resolve_effective_model_snapshot(config, QwenDialect(), runtime)

    with pytest.raises(ValueError, match="model-bound"):
        validate_heterogeneous_switch_overrides(config, second, QwenDialect(), snapshot)


def test_forced_dialect_incompatibility_rejects_heterogeneous_switch(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = ServerConfig(model_directory=first, model_dialect="qwen")
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "Gemma4ForCausalLM"),
        vision_loaded=False,
    )
    snapshot = resolve_effective_model_snapshot(config, QwenDialect(), runtime)

    with pytest.raises(ValueError, match="forced model dialect is incompatible"):
        validate_heterogeneous_switch_overrides(config, second, QwenDialect(), snapshot)


def test_target_vision_policy_is_checked_against_target_snapshot(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = ServerConfig(model_directory=first, vision_enabled=True)
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )
    snapshot = resolve_effective_model_snapshot(config, QwenDialect(), runtime)

    with pytest.raises(ValueError, match="vision capability policy"):
        validate_heterogeneous_switch_overrides(config, second, QwenDialect(), snapshot)


def test_target_tool_policy_is_checked_against_target_snapshot(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = ServerConfig(model_directory=first, tool_constraint_mode=ToolConstraintMode.SCHEMA)
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "UnknownArchitecture"),
        vision_loaded=False,
    )
    dialect = GenericHFDialect()
    snapshot = resolve_effective_model_snapshot(config, dialect, runtime)

    assert not snapshot.tool_generation_available
    with pytest.raises(ValueError, match="constrained-tool policy"):
        validate_heterogeneous_switch_overrides(config, second, dialect, snapshot)


def test_dialect_tool_false_cannot_be_upgraded_by_provider_or_runtime(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = ServerConfig(model_directory=first, tool_constraint_mode=ToolConstraintMode.SCHEMA)
    base = QwenDialect()
    dialect = replace(
        base,
        capabilities=replace(
            base.capabilities,
            tool_calling=False,
            parallel_tool_calls=False,
        ),
    )
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )

    snapshot = resolve_effective_model_snapshot(config, dialect, runtime)

    assert snapshot.dialect_capabilities.tool_calling is False
    assert snapshot.tool_generation_available is False
    assert snapshot.strict_tool_generation_available is False
    with pytest.raises(ValueError, match="constrained-tool policy"):
        validate_heterogeneous_switch_overrides(config, second, dialect, snapshot)


def test_tool_calling_true_without_provider_still_fails_closed(tmp_path: Path) -> None:
    base = GenericHFDialect()
    dialect = replace(
        base,
        capabilities=replace(base.capabilities, tool_calling=True),
    )
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "UnknownArchitecture"),
        vision_loaded=False,
    )

    snapshot = resolve_effective_model_snapshot(ServerConfig(tmp_path), dialect, runtime)

    assert snapshot.dialect_capabilities.tool_calling is True
    assert snapshot.tool_generation_available is False
    assert snapshot.strict_tool_generation_available is False


def test_builtin_qwen_tool_capability_remains_available(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )

    snapshot = resolve_effective_model_snapshot(ServerConfig(tmp_path), QwenDialect(), runtime)

    assert snapshot.dialect_capabilities.tool_calling is True
    assert snapshot.tool_generation_available is True
    assert snapshot.strict_tool_generation_available is True


def test_unsupported_image_is_rejected_before_backend_compiler(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        capabilities=ExLlamaV3Runtime.capabilities,
        model_metadata=RuntimeModelMetadata(65536, "Qwen3_5ForConditionalGeneration"),
        vision_loaded=False,
    )
    snapshot = resolve_effective_model_snapshot(
        ServerConfig(model_directory=tmp_path, vision_enabled=True),
        QwenDialect(),
        runtime,
    )

    class NeverCompiler:
        called = False

        def compile(self, request: object, reasoning: object, tool_policy: object) -> object:
            del request, reasoning, tool_policy
            self.called = True
            raise AssertionError("backend compiler must not run")

    compiler = NeverCompiler()
    guarded = CapabilityGuardedPromptCompiler(compiler, snapshot)  # type: ignore[arg-type]
    request = CanonicalRequest(
        "req",
        "model",
        (
            MultimodalMessageItem(
                MessageRole.USER,
                (ImageContentPart("image-source"),),
            ),
        ),
    )
    tools = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), False)

    with pytest.raises(ValueError, match="image input is unsupported"):
        guarded.compile(request, ReasoningPolicy(), tools)
    assert compiler.called is False
