from __future__ import annotations

import json

import pytest

import exqserve.tool_wire.controls.qwen as qwen_controls
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode
from exqserve.runtime.contracts import ConstraintInstallation
from exqserve.server.qwen_parser_binding import resolve_qwen_parser_context
from exqserve.tool_wire.controls.qwen import (
    compile_qwen_tool_wire,
    compile_qwen_tool_wire_decode_plan,
)


def _constraint():
    tool = FunctionTool(
        "lookup",
        "lookup",
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"integer"}},'
            '"required":["id"],"additionalProperties":false}'
        ),
        strict=True,
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
    assert bundle.constraint is not None
    assert bundle.grammar_fingerprint is not None
    return bundle.constraint, bundle.grammar_fingerprint


def _nonstrict_policy() -> ToolPolicy:
    tool = FunctionTool(
        "write",
        "write",
        JsonSchema(
            '{"type":"object","properties":{'
            '"text":{"type":"string"},'
            '"count":{"type":"integer"},'
            '"meta":{"type":"object","properties":{"x":{"type":"integer"}},'
            '"required":["x"],"additionalProperties":false}},'
            '"required":["text"],"additionalProperties":false}'
        ),
        strict=False,
    )
    return ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)


def test_parser_binding_requires_matching_installed_identity() -> None:
    constraint, identity = _constraint()
    context = resolve_qwen_parser_context(
        constraint,
        ConstraintInstallation(True, identity, (42,), GenerationGuarantee.SCHEMA),
        ConstraintFallbackPolicy.FAIL_CLOSED,
    )
    assert context is not None
    assert context.hard_constraint_installed is True
    assert context.constraint_identity == identity
    assert context.trigger_token_ids == (42,)
    assert context.tool_region_decoder is not None


def test_parser_binding_confirmed_noninstallation_keeps_legacy_path_for_nonstrict() -> None:
    constraint, _ = _constraint()
    context = resolve_qwen_parser_context(
        constraint,
        ConstraintInstallation(False, None, (), GenerationGuarantee.NONE),
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None
    assert context.hard_constraint_installed is False
    assert context.tool_region_decoder is None


def test_parser_binding_builds_validation_only_decoder_without_constraint_identity() -> None:
    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None
    assert context.hard_constraint_installed is False
    assert context.constraint_identity is None
    assert context.trigger_token_ids == ()
    assert context.tool_region_decoder is None
    assert context.tool_region_decoder_factory is not None

    decoder = context.tool_region_decoder_factory(_nonstrict_policy())
    assert decoder is not None

    compact = (
        '<tool_call><function=write><parameter=text>"hello"</parameter>'
        '<parameter=count>7</parameter>'
        '<parameter=meta>{"x":1}</parameter></function></tool_call>'
    )
    decoder.feed(compact)
    result = decoder.finish()
    assert result.complete is True
    assert [(call.name, call.arguments_json) for call in result.calls] == [
        ("write", '{"text":"hello","count":7,"meta":{"x":1}}')
    ]

    native = decoder.fresh(0)
    native.feed(
        '<tool_call><function=write><parameter=text>\nraw text\n</parameter>\n'
        '<parameter=count>\n8\n</parameter>\n'
        '<parameter=meta>\n{"x":2}\n</parameter>\n</function></tool_call>'
    )
    native_result = native.finish()
    assert native_result.complete is True
    assert [(call.name, call.arguments_json) for call in native_result.calls] == [
        ("write", '{"text":"raw text","count":8,"meta":{"x":2}}')
    ]


def test_validation_only_decoder_remains_available_beyond_generation_tool_branch_cap() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{"x":{"type":"integer"}},'
        '"required":["x"],"additionalProperties":false}'
    )
    tools = tuple(FunctionTool(f"f{index}", None, schema, strict=False) for index in range(256))
    policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
    decode_compilation = compile_qwen_tool_wire_decode_plan(policy)
    assert len(decode_compilation.tool_hints) == 128
    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None and context.tool_region_decoder_factory is not None

    for index in (0, 127, 128, 200, 255):
        decoder = context.tool_region_decoder_factory(policy)
        assert decoder is not None
        decoder.feed(
            f"<tool_call><function=f{index}><parameter=x>{index}</parameter></function></tool_call>"
        )
        result = decoder.finish()
        assert result.complete
        assert [(call.name, call.arguments_json) for call in result.calls] == [
            (f"f{index}", f'{{"x":{index}}}')
        ]


def test_validation_only_lazy_hints_preserve_string_semantics_beyond_tool_cap() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{"x":{"type":"string"}},'
        '"required":["x"],"additionalProperties":false}'
    )
    tools = tuple(FunctionTool(f"f{index}", None, schema, strict=False) for index in range(256))
    policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
    compilation = compile_qwen_tool_wire_decode_plan(policy)
    assert len(compilation.tool_hints) == 128
    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None and context.tool_region_decoder_factory is not None

    for index in (128, 255):
        for raw in ("true", "123", "{}", "[]"):
            decoder = context.tool_region_decoder_factory(policy)
            assert decoder is not None
            decoder.feed(
                f"<tool_call><function=f{index}><parameter=x>{raw}</parameter></function></tool_call>"
            )
            result = decoder.finish()
            assert result.complete
            assert [(call.name, call.arguments_json) for call in result.calls] == [
                (f"f{index}", json.dumps({"x": raw}, separators=(",", ":")))
            ]


def test_validation_only_lazy_hints_preserve_shallow_string_for_oversized_schema() -> None:
    schema = JsonSchema(
        json.dumps(
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": False,
                "description": "z" * 1_050_000,
            },
            separators=(",", ":"),
        )
    )
    tool = FunctionTool("big", None, schema, strict=False)
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
    compilation = compile_qwen_tool_wire_decode_plan(policy)
    assert compilation.tool_hints == ()

    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None and context.tool_region_decoder_factory is not None
    decoder = context.tool_region_decoder_factory(policy)
    assert decoder is not None
    decoder.feed("<tool_call><function=big><parameter=x>true</parameter></function></tool_call>")
    result = decoder.finish()
    assert result.complete
    assert [(call.name, call.arguments_json) for call in result.calls] == [("big", '{"x":"true"}')]


def test_validation_only_lazy_hints_preserve_shallow_string_for_wide_schema() -> None:
    properties = {f"p{index}": {"type": "string"} for index in range(1025)}
    properties["x"] = {"type": "string"}
    schema = JsonSchema(
        json.dumps(
            {
                "type": "object",
                "properties": properties,
                "required": ["x"],
                "additionalProperties": False,
            },
            separators=(",", ":"),
        )
    )
    tool = FunctionTool("wide", None, schema, strict=False)
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
    compilation = compile_qwen_tool_wire_decode_plan(policy)
    assert compilation.tool_hints == ()

    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None and context.tool_region_decoder_factory is not None
    decoder = context.tool_region_decoder_factory(policy)
    assert decoder is not None
    decoder.feed("<tool_call><function=wide><parameter=x>{}</parameter></function></tool_call>")
    result = decoder.finish()
    assert result.complete
    assert [(call.name, call.arguments_json) for call in result.calls] == [("wide", '{"x":"{}"}')]


def test_validation_only_lazy_hints_preserve_raw_close_semantics_beyond_tool_cap() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{'
        '"content":{"type":"string"},"file_path":{"type":"string"}},'
        '"required":["content","file_path"],"additionalProperties":false}'
    )
    tools = tuple(FunctionTool(f"f{index}", None, schema, strict=False) for index in range(256))
    policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None and context.tool_region_decoder_factory is not None

    for index in (128, 255):
        decoder = context.tool_region_decoder_factory(policy)
        assert decoder is not None
        decoder.feed(
            f"<tool_call><function=f{index}><parameter=content>"
            "before </parameter></function></tool_call> after"
            "</parameter><parameter=file_path>/tmp/x</parameter></function></tool_call>"
        )
        result = decoder.finish()
        assert result.complete
        assert [(call.name, call.arguments_json) for call in result.calls] == [
            (
                f"f{index}",
                '{"content":"before </parameter></function></tool_call> after","file_path":"/tmp/x"}',
            )
        ]


def test_validation_only_semantic_arbitration_reuses_one_request_local_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{"content":{"type":"string","minLength":80}},'
        '"required":["content"],"additionalProperties":false}'
    )
    policy = ToolPolicy(
        (FunctionTool("write", None, schema, strict=False),),
        ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=False,
    )
    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None and context.tool_region_decoder_factory is not None

    original_validator = qwen_controls.Draft202012Validator
    validator_builds = 0

    def counting_validator(schema_value: object):
        nonlocal validator_builds
        validator_builds += 1
        return original_validator(schema_value)

    monkeypatch.setattr(qwen_controls, "Draft202012Validator", counting_validator)
    close = "</parameter></function></tool_call>"
    decoder = context.tool_region_decoder_factory(policy)
    assert decoder is not None
    decoder.feed(
        "<tool_call><function=write><parameter=content>"
        + "short"
        + close
        + ("x" * 100)
        + close
    )
    result = decoder.finish()

    assert result.complete
    assert validator_builds == 1
    assert [(call.name, call.arguments_json) for call in result.calls] == [
        ("write", json.dumps({"content": "short" + close + ("x" * 100)}, separators=(",", ":")))
    ]


def test_validation_only_semantic_arbitration_budget_exhaustion_fails_closed() -> None:
    schema = JsonSchema(
        '{"type":"object","properties":{"content":{"type":"string","minLength":100}},'
        '"required":["content"],"additionalProperties":false}'
    )
    policy = ToolPolicy(
        (FunctionTool("write", None, schema, strict=False),),
        ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=False,
    )
    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None and context.tool_region_decoder_factory is not None

    close = "</parameter></function></tool_call>"
    decoder = context.tool_region_decoder_factory(policy)
    assert decoder is not None
    decoder.feed(
        "<tool_call><function=write><parameter=content>"
        + (("x" + close) * 300)
        + "tail"
        + close
    )
    result = decoder.finish()

    assert not result.complete
    assert result.calls == ()
    assert result.issue_code == "compatibility_semantic_work_exceeded"


def test_parser_binding_does_not_guess_unknown_installation() -> None:
    constraint, _ = _constraint()
    with pytest.raises(ValueError):
        resolve_qwen_parser_context(
            constraint,
            None,
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )


def test_parser_binding_rejects_mismatched_installation_identity() -> None:
    constraint, _ = _constraint()
    with pytest.raises(ValueError):
        resolve_qwen_parser_context(
            constraint,
            ConstraintInstallation(True, "other", (42,), GenerationGuarantee.SCHEMA),
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )


def test_parser_binding_strict_rejects_confirmed_noninstallation() -> None:
    constraint, _ = _constraint()
    with pytest.raises(ValueError):
        resolve_qwen_parser_context(
            constraint,
            ConstraintInstallation(False, None, (), GenerationGuarantee.NONE),
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
