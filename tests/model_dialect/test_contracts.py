from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from exqserve.model.contracts import (
    CompiledPrompt,
    ModelCapabilities,
    RenderedPrompt,
    TemplateMessage,
    TemplateRequest,
    TemplateTool,
    TemplateToolCall,
)


def test_model_capabilities_are_immutable_plain_model_semantics() -> None:
    caps = ModelCapabilities(
        reasoning=True,
        tool_calling=True,
        parallel_tool_calls=True,
        system_role=True,
        developer_role=False,
        reasoning_history=True,
    )

    assert caps.developer_role is False
    with pytest.raises(FrozenInstanceError):
        caps.reasoning = False  # type: ignore[misc]


def test_template_values_are_deeply_stable_string_tuple_contracts() -> None:
    call = TemplateToolCall(name="bash", arguments_json='{"cmd":"pwd"}')
    tool = TemplateTool(name="bash", description=None, parameters_json='{"type":"object"}')
    message = TemplateMessage(role="assistant", content="", tool_calls=(call,))
    request = TemplateRequest(
        messages=(message,),
        tools=(tool,),
        template_kwargs=(("enable_thinking", True),),
    )

    assert request.add_generation_prompt is True
    assert request.messages[0].tool_calls == (call,)
    assert request.tools == (tool,)


def test_template_contracts_reject_invalid_identity_and_container_types() -> None:
    with pytest.raises(ValueError, match="name"):
        TemplateToolCall("", "{}")
    with pytest.raises(ValueError, match="role"):
        TemplateMessage("", "x")
    with pytest.raises(TypeError, match="messages"):
        TemplateRequest(messages=[], tools=(), template_kwargs=())  # type: ignore[arg-type]


def test_rendered_and_compiled_prompt_validate_token_ids() -> None:
    rendered = RenderedPrompt(text="hello", input_ids=(1, 2, 3))
    compiled = CompiledPrompt(
        text=rendered.text,
        input_ids=rendered.input_ids,
        prompt_hash="a" * 64,
        stop_conditions=("<stop>",),
        template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
    )

    assert compiled.input_ids == (1, 2, 3)
    assert compiled.raw_output_is_text_only is False
    assert compiled.structured_output_trigger is None
    with pytest.raises(ValueError, match="input_ids"):
        RenderedPrompt(text="x", input_ids=())
    with pytest.raises(TypeError, match="raw_output_is_text_only"):
        CompiledPrompt("x", (1,), "a" * 64, (), TemplateRequest((), (), ()), (), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="structured_output_trigger"):
        CompiledPrompt(
            "x",
            (1,),
            "a" * 64,
            (),
            TemplateRequest((), (), ()),
            structured_output_trigger="   ",
        )
    with pytest.raises(TypeError, match="structured_output_trigger"):
        CompiledPrompt(
            "x",
            (1,),
            "a" * 64,
            (),
            TemplateRequest((), (), ()),
            structured_output_trigger=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="prompt_hash"):
        CompiledPrompt(
            text="x",
            input_ids=(1,),
            prompt_hash="short",
            stop_conditions=(),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
        )
