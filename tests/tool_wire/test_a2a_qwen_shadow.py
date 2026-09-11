from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

import exqserve.tool_wire.compiler as tool_wire_compiler
import exqserve.tool_wire.controls.qwen as qwen_control
import exqserve.tool_wire.lark_constraint as tool_wire_lark
import exqserve.tool_wire.semantic_authority as tool_wire_semantic_authority
from exqserve.agent._json import canonical_json_dumps, parse_json_strict
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode
from exqserve.model.qwen import _parameter_value_json
from exqserve.tool_wire import (
    CompileBudget,
    ConstraintValueMode,
    DeterministicToolWireEngine,
    PlanCompileDisposition,
    ToolWireEngineStatus,
    admit_tool_sequence,
    certify_prompt_template_parity,
    encode_lossless_raw_string,
)
from exqserve.tool_wire.controls.qwen import (
    compile_qwen_a2a_shadow,
    qwen_a2a_compiler_capabilities,
    qwen_a2a_prompt_observation,
    qwen_a2a_single_call_spec,
)
from tests.tool_wire._support import policy, tool


def _schema(properties: dict[str, dict[str, object]], *, required: tuple[str, ...] | None = None) -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": properties,
            "required": list(properties) if required is None else list(required),
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )


def _compile(
    properties: dict[str, dict[str, object]],
    *,
    required: tuple[str, ...] | None = None,
    presentation: tuple[str, ...] | None = None,
    strict: bool = True,
    allow_parallel: bool = False,
):
    schema = _schema(properties, required=required)
    fn = tool("write", schema, strict=strict)
    order = tuple(properties) if presentation is None else presentation
    return compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=allow_parallel),
        {"write": order},
    )


def _wire(bundle, occurrences: tuple[tuple[str, str], ...]) -> str:
    spec = bundle.spec
    branch = bundle.plan.tool("write")
    argument_by_name = {argument.name: argument for argument in branch.arguments}
    parts = [spec.tool_open.text, spec.function_open.render("write")]
    for name, value in occurrences:
        argument = argument_by_name[name]
        assert argument.framing_variant_id is not None
        variant = spec.framing_variant(argument.framing_variant_id)
        parts.extend(
            (
                variant.argument_open.render(name),
                value,
                variant.argument_close.canonical.text,
            )
        )
    parts.extend((spec.function_close.canonical.text, spec.tool_close.canonical.text))
    return "".join(parts)


def _finish(bundle, wire: str, chunks: tuple[str, ...]):
    engine = DeterministicToolWireEngine(bundle.spec, bundle.plan)
    for chunk in chunks:
        engine.feed(chunk)
    return engine.finish()


def test_qwen_a2a_static_control_models_shared_raw_and_structured_parameter_opener() -> None:
    spec = qwen_a2a_single_call_spec()
    assert spec.tool_open.text == "<tool_call>"
    assert spec.tool_close.canonical.text == "</tool_call>"
    assert spec.function_open.render("write") == "<function=write>"
    assert spec.function_close.canonical.text == "</function>"
    assert len(spec.argument_framings) == 2
    assert {variant.argument_open.render("value") for variant in spec.argument_framings} == {
        "<parameter=value>"
    }
    assert spec.framing_selector.select("string") == "qwen-raw-string"
    for schema_type in ("integer", "number", "boolean", "object", "array", "null"):
        assert spec.framing_selector.select(schema_type) == "qwen-json-structured"
    raw = spec.framing_variant("qwen-raw-string")
    assert raw.value_framing.codec.value == "raw_string_strip_json_string_or_text"
    assert raw.value_framing.codec.decode_raw_payload("\n  value\t\n") == "value"
    assert raw.value_framing.codec.decode_raw_payload('"value"') == "value"
    assert raw.value_framing.codec.decode_raw_payload('"a\\nb"') == "a\nb"
    assert raw.value_framing.codec.decode_raw_payload("true") == "true"
    assert raw.value_framing.codec.decode_raw_payload("a\nb") == "a\nb"
    assert raw.value_framing.forbidden_close_language is not None
    assert raw.value_framing.forbidden_close_language.forms == raw.argument_close.forms
    assert raw.argument_close.texts == ("</parameter>",)
    assert spec.multiplicity.min_calls_per_sequence == 1
    assert spec.multiplicity.max_calls_per_sequence == 1
    assert not spec.multiplicity.adjacent_tools
    assert tuple(trigger.terminal.text for trigger in spec.activation_triggers) == ("<tool_call>",)


def test_qwen_a2a_static_name_language_matches_production_tag_rule_shape() -> None:
    spec = qwen_a2a_single_call_spec()
    whitespace = {
        value
        for value in spec.function_name_codec.forbidden_sequences
        if len(value) == 1 and value.isspace()
    }
    runtime_whitespace = {chr(codepoint) for codepoint in range(0x110000) if chr(codepoint).isspace()}
    assert whitespace == runtime_whitespace
    for invalid in ("bad name", "bad\tname", "bad<name", "bad>name", "bad\u3000name"):
        assert not spec.function_name_codec.is_losslessly_representable_for_terminal(
            invalid,
            spec.function_open,
        )


def test_qwen_a2a_plain_raw_string_can_retain_schema_guarantee() -> None:
    bundle = _compile({"content": {"type": "string"}})
    assert bundle.constrained
    branch = bundle.plan.tool("write")
    argument = branch.arguments[0]
    assert argument.value_mode is ConstraintValueMode.ANY_SAFE_RAW
    assert argument.proof.guarantee is GenerationGuarantee.SCHEMA
    assert branch.guarantee is GenerationGuarantee.SCHEMA
    assert bundle.constraint is not None
    assert bundle.grammar_fingerprint == bundle.plan.constraint_fingerprint
    assert "VALUE_CHAR" not in bundle.constraint.lark_grammar
    assert "/[^<]/" not in bundle.constraint.lark_grammar


def test_qwen_a2a_format_mode_uses_same_raw_close_exclusion_language() -> None:
    schema = _schema({"content": {"type": "string"}})
    fn = tool("write", schema, strict=False)
    bundle = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
    )
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert argument.value_mode is ConstraintValueMode.ANY_SAFE_RAW
    assert argument.proof.guarantee is GenerationGuarantee.FORMAT
    assert bundle.plan.tool("write").guarantee is GenerationGuarantee.FORMAT
    assert bundle.constraint is not None
    assert "/[^<]/" not in bundle.constraint.lark_grammar


def test_qwen_a2a_raw_codec_normalizes_only_outer_presentation_whitespace() -> None:
    bundle = _compile(
        {"content": {"type": "string"}, "count": {"type": "integer"}},
        presentation=("content", "count"),
    )
    compact = _wire(bundle, (("content", "hello<world"), ("count", "3")))
    template = (
        "<tool_call>\n<function=write>\n"
        "<parameter=content>\nhello<world\n</parameter>\n"
        "<parameter=count>\n3\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    interior = _wire(bundle, (("content", "line1\nline2"), ("count", "3")))

    compact_result = _finish(bundle, compact, (compact,))
    template_result = _finish(bundle, template, tuple(template))
    interior_result = _finish(bundle, interior, (interior,))
    for result in (compact_result, template_result, interior_result):
        assert result.is_complete and result.sequence is not None
        assert admit_tool_sequence(bundle.spec, bundle.plan, result.sequence).is_valid

    compact_values = [parse_json_strict(item.canonical_value_json) for item in compact_result.sequence.calls[0].occurrences]
    template_values = [parse_json_strict(item.canonical_value_json) for item in template_result.sequence.calls[0].occurrences]
    interior_values = [parse_json_strict(item.canonical_value_json) for item in interior_result.sequence.calls[0].occurrences]
    assert compact_values == ["hello<world", 3]
    assert template_values == compact_values
    assert interior_values == ["line1\nline2", 3]


@pytest.mark.parametrize(
    ("wire_value", "semantic"),
    (
        ("foo", "foo"),
        ("123", "123"),
        ("true", "true"),
        ("false", "false"),
        ("null", "null"),
        ('{"x":1}', '{"x":1}'),
        ("[1,2]", "[1,2]"),
        ('"foo"', "foo"),
        ('"a\\nb"', "a\nb"),
        ('"a\\\"b"', 'a"b'),
        ('"a\\\\b"', "a\\b"),
        ('"\\u4f60"', "你"),
        ('""', ""),
        ("\nfoo\n", "foo"),
        ("line1\nline2", "line1\nline2"),
        ('" leading"', " leading"),
        ('"trailing "', "trailing "),
        ('"x\\u003c/parameter>y"', "x</parameter>y"),
    ),
)
def test_qwen_a2a_raw_string_decoder_matches_frozen_production_semantics(
    wire_value: str,
    semantic: str,
) -> None:
    bundle = _compile({"content": {"type": "string"}})
    wire = _wire(bundle, (("content", wire_value),))
    result = _finish(bundle, wire, (wire,))
    assert result.is_complete and result.sequence is not None
    assert admit_tool_sequence(bundle.spec, bundle.plan, result.sequence).is_valid
    occurrence = result.sequence.calls[0].occurrences[0]
    assert occurrence.canonical_value_json == _parameter_value_json(
        wire_value,
        string_parameter=True,
    )
    assert parse_json_strict(occurrence.canonical_value_json) == semantic


@pytest.mark.parametrize("wire_value", ('"a\\nb"', '"\\u4f60"', '"x\\u003c/parameter>y"'))
def test_qwen_a2a_raw_string_decoder_is_chunk_invariant(wire_value: str) -> None:
    bundle = _compile({"content": {"type": "string"}})
    wire = _wire(bundle, (("content", wire_value),))
    baseline = _finish(bundle, wire, (wire,))
    assert baseline.is_complete and baseline.sequence is not None
    for split in range(len(wire) + 1):
        result = _finish(bundle, wire, (wire[:split], wire[split:]))
        assert result == baseline


def test_qwen_a2a_finite_raw_values_use_lossless_encoder_not_identity_only() -> None:
    values = ["safe", " safe", "safe ", "\tbad", '"quoted"', "x</parameter>y"]
    bundle = _compile({"content": {"type": "string", "enum": values}})
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert argument.value_mode is ConstraintValueMode.FINITE_VALUES
    assert argument.admitted_values_json == tuple(canonical_json_dumps(value) for value in values)
    raw = bundle.spec.framing_variant("qwen-raw-string").value_framing
    for semantic_json in argument.admitted_values_json:
        semantic = parse_json_strict(semantic_json)
        assert isinstance(semantic, str)
        wire = encode_lossless_raw_string(semantic, raw)
        assert wire is not None
        assert "</parameter>" not in wire
        assert raw.codec.decode_raw_payload(wire) == semantic


@pytest.mark.parametrize("count", (1, 10, 100))
def test_qwen_a2a_raw_finite_format_work_is_fully_metered(count: int) -> None:
    values = [f"value-{index}" for index in range(count)]
    schema = _schema({"content": {"type": "string", "enum": values}})
    request_policy = policy(tool("write", schema, strict=False), allow_parallel=False)
    roomy = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, 1_000_000),
    )
    assert roomy.constrained
    exact_work = roomy.plan.budget_result.work_units

    exact = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, exact_work),
    )
    assert exact.constrained
    assert exact.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert exact.plan.budget_result.within_budget
    assert exact.plan.budget_result.work_units == exact_work

    rejected = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, exact_work - 1),
    )
    assert rejected.plan.disposition is PlanCompileDisposition.REJECTED
    # V3 records authoritative product complexity, not failed-candidate instruction telemetry.
    assert rejected.constraint is None
    assert rejected.plan.constraint_fingerprint is None
    assert rejected.plan.activation is None


@pytest.mark.parametrize("count", (1, 10, 100, 1000))
def test_qwen_a2a_finite_raw_emission_reuses_semantic_wire_payloads(count: int) -> None:
    # V3's artifact module no longer imports either authority, so emission cannot silently reparse
    # semantic JSON or re-run lossless RAW encoding.
    assert not hasattr(tool_wire_lark, "parse_json_strict")
    assert not hasattr(tool_wire_lark, "encode_lossless_raw_string")
    values = [f"value-{index}" for index in range(count)]
    schema = _schema({"content": {"type": "string", "enum": values}})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=False), allow_parallel=False),
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 1_000_000, 100_000_000, 1_000_000),
    )
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert argument.admitted_values_json is not None
    assert argument.admitted_wire_payloads is not None
    assert len(argument.admitted_wire_payloads) == count


def test_qwen_a2a_hard_source_envelope_rejects_before_schema_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls = 0
    original_parse = tool_wire_compiler.parse_json_strict

    def counted_parse(text: str):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(text)

    monkeypatch.setattr(tool_wire_compiler, "parse_json_strict", counted_parse)
    payload_size = tool_wire_compiler._HARD_MAX_SCHEMA_SOURCE_CHARS_PER_TOOL + 1
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
            "description": "x" * payload_size,
        },
        separators=(",", ":"),
    )
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("x",)},
        budget=CompileBudget(1000, 10_000_000, 100_000_000, 10_000_000),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert parse_calls == 0


def test_v3_hard_artifact_size_caps_candidate_before_authority() -> None:
    ledger = tool_wire_compiler._CompileBudgetLedger(
        CompileBudget(1000, 100_000_000, 100_000_000, 100_000_000)
    )
    oversized = tool_wire_compiler._ConstraintArtifactCandidate(
        cost=tool_wire_compiler._ArtifactProductCost(
            estimated_rules=1,
            estimated_bytes=tool_wire_compiler._HARD_MAX_FINAL_ARTIFACT_BYTES + 1,
            work_units=1,
        ),
        payload=object(),
    )
    order_cost = tool_wire_compiler._OrderProductCost(1, 0, 1, 1)
    assert not tool_wire_compiler._candidate_products_fit(ledger, order_cost, oversized)


def test_qwen_a2a_hard_presentation_name_envelope_rejects_before_schema_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls = 0
    original_parse = tool_wire_compiler.parse_json_strict

    def counted_parse(text: str):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(text)

    monkeypatch.setattr(tool_wire_compiler, "parse_json_strict", counted_parse)
    invalid_name = "z" * (tool_wire_compiler._HARD_MAX_NAME_CHARS + 1)
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", _schema({"x": {"type": "string"}}), strict=True)),
        {"write": (invalid_name,)},
        budget=CompileBudget(1000, 100_000, 10_000_000, 100_000),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert parse_calls == 0


def test_qwen_a2a_hard_required_name_envelope_rejects_before_semantic_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_calls = 0
    original_compile = tool_wire_compiler._compile_tool_semantics

    def counted_compile(*args: object, **kwargs: object):
        nonlocal semantic_calls
        semantic_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(tool_wire_compiler, "_compile_tool_semantics", counted_compile)
    invalid_name = "z" * (tool_wire_compiler._HARD_MAX_NAME_CHARS + 1)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": [invalid_name],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True)),
        {"write": ("x",)},
        budget=CompileBudget(1000, 100_000, 10_000_000, 100_000),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert semantic_calls == 0


@pytest.mark.parametrize("mode", (ToolConstraintMode.SCHEMA, ToolConstraintMode.FORMAT))
def test_qwen_a2a_v31_unselected_optional_finite_only_pays_source_bytes(
    mode: ToolConstraintMode,
) -> None:
    def compile_count(count: int):
        schema = _schema(
            {
                "r": {"type": "integer"},
                "o": {"type": "string", "enum": [f"v{index}" for index in range(count)]},
            },
            required=("r",),
        )
        fn = tool("write", schema, strict=mode is ToolConstraintMode.SCHEMA)
        bundle = compile_qwen_a2a_shadow(
            policy(fn, allow_parallel=False),
            {"write": ("r", "o")},
            mode=mode,
            budget=CompileBudget(1, 1_000_000, 100_000_000, 10_000_000),
        )
        return bundle, len(fn.parameters.canonical_json.encode("utf-8"))

    small, small_source_bytes = compile_count(1)
    large, large_source_bytes = compile_count(1000)

    assert small.constrained and large.constrained
    assert small.plan.tool("write").order_plan.orders == (("r",),)
    assert large.plan.tool("write").order_plan.orders == (("r",),)
    optional = large.plan.tool("write").arguments[1]
    assert not optional.generated
    assert optional.admitted_wire_payloads is None
    assert (
        large.plan.budget_result.estimated_bytes - large_source_bytes
        == small.plan.budget_result.estimated_bytes - small_source_bytes
    )
    assert large.plan.budget_result.estimated_rules == small.plan.budget_result.estimated_rules
    assert large.plan.budget_result.work_units == small.plan.budget_result.work_units


def test_qwen_a2a_v31_selected_optional_finite_commits_wire_product() -> None:
    schema = _schema(
        {
            "r": {"type": "integer"},
            "o": {"type": "string", "enum": ["a", "b"]},
        },
        required=("r",),
    )
    request_policy = policy(tool("write", schema, strict=False), allow_parallel=False)
    orders = {"write": ("r", "o")}
    narrow = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1, 1_000_000, 100_000_000, 10_000_000),
    )
    rich = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(3, 1_000_000, 100_000_000, 10_000_000),
    )

    assert narrow.constrained and rich.constrained
    assert narrow.plan.tool("write").order_plan.orders == (("r",),)
    assert len(rich.plan.tool("write").order_plan.orders) == 3
    assert not narrow.plan.tool("write").arguments[1].generated
    rich_optional = rich.plan.tool("write").arguments[1]
    assert rich_optional.generated
    assert rich_optional.admitted_wire_payloads == ("a", "b")
    assert rich.plan.budget_result.estimated_bytes > narrow.plan.budget_result.estimated_bytes


def test_v31_schema_node_hard_allowance_bounds_child_iteration() -> None:
    class TrackingList(list[object]):
        def __init__(self, values: list[object]) -> None:
            super().__init__(values)
            self.reads = 0

        def __iter__(self) -> Iterator[object]:
            for value in super().__iter__():
                self.reads += 1
                yield value

    children = TrackingList([{} for _ in range(100)])
    context_handle = tool_wire_compiler._SCHEMA_NODE_LIMIT_CONTEXT.set(5)
    try:
        with pytest.raises(tool_wire_compiler._HardComplexityExceeded):
            tool_wire_compiler._validate_schema_hard_envelope({"examples": children})
    finally:
        tool_wire_compiler._SCHEMA_NODE_LIMIT_CONTEXT.reset(context_handle)

    assert children.reads == 4


def test_qwen_a2a_v31_required_finite_wire_hard_allowance_stops_before_second_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = "x</parameter>y"
    framing = qwen_a2a_single_call_spec().framing_variant("qwen-raw-string").value_framing
    first_payload = encode_lossless_raw_string(semantic, framing)
    assert first_payload is not None
    first_payload_bytes = len(first_payload.encode("utf-8"))
    monkeypatch.setattr(
        tool_wire_compiler,
        "_HARD_MAX_FINITE_WIRE_PAYLOAD_BYTES_TOTAL",
        first_payload_bytes + 1,
    )
    materialization_calls = 0
    original_materialize = tool_wire_compiler._boundary_safe_json_string_literal

    def counted_materialize(value: str, close_forms: tuple[str, ...]) -> str:
        nonlocal materialization_calls
        materialization_calls += 1
        return original_materialize(value, close_forms)

    monkeypatch.setattr(
        tool_wire_compiler,
        "_boundary_safe_json_string_literal",
        counted_materialize,
    )
    schema = _schema({"r": {"type": "string", "const": semantic}}, required=("r",))
    bundle = compile_qwen_a2a_shadow(
        policy(
            tool("first", schema, strict=True),
            tool("second", schema, strict=True),
            allow_parallel=False,
        ),
        {"first": ("r",), "second": ("r",)},
        budget=CompileBudget(2, 1_000_000, 100_000_000, 10_000_000),
    )

    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert materialization_calls == 1


@pytest.mark.parametrize(
    "value",
    (
        "plain",
        '"quoted"',
        "x</parameter>y",
        "<leading",
        "line\nfeed",
        "slash\\quote\"",
        "控制🙂",
        "",
    ),
)
def test_v31_bounded_raw_encoder_matches_canonical_codec(value: str) -> None:
    framing = qwen_a2a_single_call_spec().framing_variant("qwen-raw-string").value_framing
    expected = encode_lossless_raw_string(value, framing)
    actual = tool_wire_compiler._encode_lossless_raw_string_bounded(
        value,
        framing,
        10_000_000,
    )
    assert actual == expected


def test_qwen_a2a_artifact_optional_order_enrichment_is_monotonic() -> None:
    schema = _schema(
        {
            "r": {"type": "string"},
            "o1": {"type": "string"},
            "o2": {"type": "string"},
        },
        required=("r",),
    )
    request_policy = policy(tool("write", schema, strict=False), allow_parallel=False)
    orders = {"write": ("r", "o1", "o2")}

    narrow_perm = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        budget=CompileBudget(10, 10_000, 1_000_000, 3_000),
    )
    rich_perm = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        budget=CompileBudget(11, 10_000, 1_000_000, 3_000),
    )
    assert narrow_perm.constrained and rich_perm.constrained
    assert len(narrow_perm.plan.tool("write").order_plan.orders) == 1
    assert len(rich_perm.plan.tool("write").order_plan.orders) == 11

    narrow_rules = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        budget=CompileBudget(20, 74, 1_000_000, 3_000),
    )
    rich_rules = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        budget=CompileBudget(20, 75, 1_000_000, 3_000),
    )
    assert narrow_rules.constrained and rich_rules.constrained
    assert len(narrow_rules.plan.tool("write").order_plan.orders) == 1
    assert len(rich_rules.plan.tool("write").order_plan.orders) == 11


@pytest.mark.parametrize(
    ("dimension", "values"),
    (
        ("max_permutations", (1, 2, 3, 4, 10)),
        ("max_estimated_rules", (1, 5, 10, 15, 16, 17, 18, 50)),
        ("max_estimated_bytes", (100, 1000, 5000, 8000, 10_000, 12_000, 13_000, 13_758, 13_759, 15_000)),
        ("max_work_units", (1, 50, 100, 200, 300, 324, 325, 400, 440, 441, 500)),
    ),
)
def test_qwen_a2a_v3_budget_dimensions_are_monotonic(
    dimension: str,
    values: tuple[int, ...],
) -> None:
    finite = [f"v{index}" for index in range(100)]
    schema = _schema(
        {
            "r": {"type": "string", "enum": finite},
            "o": {"type": "string", "enum": finite},
        },
        required=("r",),
    )
    request_policy = policy(tool("write", schema, strict=False), allow_parallel=False)
    orders = {"write": ("r", "o")}
    last_order_count = 0
    executable_seen = False

    for value in values:
        budget_values = {
            "max_permutations": 1000,
            "max_estimated_rules": 1_000_000,
            "max_estimated_bytes": 100_000_000,
            "max_work_units": 10_000_000,
        }
        budget_values[dimension] = value
        bundle = compile_qwen_a2a_shadow(
            request_policy,
            orders,
            mode=ToolConstraintMode.FORMAT,
            budget=CompileBudget(**budget_values),
        )
        executable = bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        if executable_seen:
            assert executable, f"{dimension} regressed PASS -> REJECT at {value=}"
        if executable:
            executable_seen = True
            order_count = len(bundle.plan.tool("write").order_plan.orders)
            assert order_count >= last_order_count
            last_order_count = order_count

    assert executable_seen


@pytest.mark.parametrize("mode", (ToolConstraintMode.SCHEMA, ToolConstraintMode.FORMAT))
@pytest.mark.parametrize("count", (1, 3, 10, 100, 1000))
def test_qwen_a2a_v3_finite_optional_permutation_budget_never_rejects_narrow_pass(
    mode: ToolConstraintMode,
    count: int,
) -> None:
    finite = [f"v{index}" for index in range(count)]
    schema = _schema(
        {
            "r": {"type": "string", "enum": finite},
            "o": {"type": "string", "enum": finite},
        },
        required=("r",),
    )
    request_policy = policy(tool("write", schema, strict=False), allow_parallel=False)
    orders = {"write": ("r", "o")}
    roomy_narrow = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=mode,
        budget=CompileBudget(2, 1_000_000, 100_000_000, 10_000_000),
    )
    exact_work = roomy_narrow.plan.budget_result.work_units

    narrow = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=mode,
        budget=CompileBudget(2, 1_000_000, 100_000_000, exact_work),
    )
    richer_budget = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=mode,
        budget=CompileBudget(3, 1_000_000, 100_000_000, exact_work),
    )

    assert narrow.constrained and richer_budget.constrained
    assert len(narrow.plan.tool("write").order_plan.orders) == 1
    assert len(richer_budget.plan.tool("write").order_plan.orders) == 1
    assert richer_budget.plan.budget_result.work_units == exact_work


def test_qwen_a2a_v31_multitool_budget_swap_keeps_minimal_baseline_and_is_deterministic() -> None:
    first_schema = _schema(
        {
            "r": {"type": "string"},
            "o1": {"type": "string"},
            "o2": {"type": "string"},
        },
        required=("r",),
    )
    second_schema = _schema(
        {
            "r": {"type": "string"},
            "o": {"type": "string"},
        },
        required=("r",),
    )
    request_policy = policy(
        tool("a", first_schema, strict=False),
        tool("b", second_schema, strict=False),
        allow_parallel=False,
    )
    orders = {"a": ("r", "o1", "o2"), "b": ("r", "o")}

    low = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(4, 1_000_000, 100_000_000, 10_000_000),
    )
    high_budget = CompileBudget(12, 1_000_000, 100_000_000, 10_000_000)
    high = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=ToolConstraintMode.FORMAT,
        budget=high_budget,
    )
    high_repeat = compile_qwen_a2a_shadow(
        request_policy,
        orders,
        mode=ToolConstraintMode.FORMAT,
        budget=high_budget,
    )

    assert low.constrained and high.constrained and high_repeat.constrained
    for bundle in (low, high, high_repeat):
        assert ("r",) in bundle.plan.tool("a").order_plan.orders
        assert ("r",) in bundle.plan.tool("b").order_plan.orders
    assert len(low.plan.tool("b").order_plan.orders) == 3
    assert len(high.plan.tool("a").order_plan.orders) == 11
    assert low.plan.tool("a").order_plan.orders != high.plan.tool("a").order_plan.orders
    assert low.plan.tool("b").order_plan.orders != high.plan.tool("b").order_plan.orders
    assert high.plan == high_repeat.plan
    assert high.constraint == high_repeat.constraint
    assert high.grammar_fingerprint == high_repeat.grammar_fingerprint


def test_qwen_a2a_artifact_is_built_and_fingerprinted_once_for_minimal_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_calls = 0
    finalize_calls = 0
    plan_calls = 0
    fingerprint_calls = 0
    original_build = qwen_control.build_lark_tool_constraint_candidate
    original_finalize = qwen_control.finalize_lark_tool_constraint_candidate
    original_compile_plan = qwen_control.compile_tool_wire_plan
    original_sha256 = tool_wire_lark.sha256

    def counted_build(*args: object, **kwargs: object):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    def counted_finalize(*args: object, **kwargs: object):
        nonlocal finalize_calls
        finalize_calls += 1
        return original_finalize(*args, **kwargs)

    def counted_compile_plan(*args: object, **kwargs: object):
        nonlocal plan_calls
        plan_calls += 1
        return original_compile_plan(*args, **kwargs)

    def counted_sha256(*args: object, **kwargs: object):
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return original_sha256(*args, **kwargs)

    monkeypatch.setattr(qwen_control, "build_lark_tool_constraint_candidate", counted_build)
    monkeypatch.setattr(qwen_control, "finalize_lark_tool_constraint_candidate", counted_finalize)
    monkeypatch.setattr(qwen_control, "compile_tool_wire_plan", counted_compile_plan)
    monkeypatch.setattr(tool_wire_lark, "sha256", counted_sha256)

    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", _schema({"content": {"type": "string"}}), strict=True)),
        {"write": ("content",)},
    )
    assert bundle.constrained
    assert build_calls == 1
    assert finalize_calls == 1
    assert plan_calls == 1
    assert fingerprint_calls == 1
    assert bundle.plan.constraint_fingerprint == bundle.grammar_fingerprint


@pytest.mark.parametrize("count", (1, 10, 100, 1000, 5000))
def test_qwen_a2a_raw_finite_tiny_budget_stops_at_first_unaffordable_work(count: int) -> None:
    values = [f"value-{index}" for index in range(count)]
    schema = _schema({"content": {"type": "string", "enum": values}})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=False), allow_parallel=False),
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=4,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert not bundle.plan.budget_result.within_budget
    assert bundle.plan.budget_result.work_units > 4
    assert bundle.constraint is None
    assert bundle.plan.constraint_fingerprint is None
    assert bundle.plan.activation is None


@pytest.mark.parametrize("count", (1, 10, 100, 1000, 5000))
def test_qwen_a2a_const_enum_budget_counts_domain_scan_and_effective_intersection(count: int) -> None:
    values = ["target", *(f"irrelevant-{index}" for index in range(max(0, count - 1)))]
    schema = _schema(
        {"content": {"type": "string", "const": "target", "enum": values}}
    )
    request_policy = policy(tool("write", schema, strict=False), allow_parallel=False)
    roomy = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, 1_000_000),
    )
    exact_work = roomy.plan.budget_result.work_units
    bundle = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, exact_work),
    )
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.plan.budget_result.within_budget
    assert bundle.plan.budget_result.work_units == exact_work
    assert argument.admitted_values_json == (canonical_json_dumps("target"),)
    assert argument.admitted_wire_payloads == ("target",)

    rejected = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, exact_work - 1),
    )
    assert rejected.plan.disposition is PlanCompileDisposition.REJECTED
    # V3 records authoritative product complexity, not failed-candidate instruction telemetry.
    assert rejected.constraint is None
    assert rejected.plan.constraint_fingerprint is None
    assert rejected.plan.activation is None


def test_qwen_a2a_large_finite_budget_stops_before_validation_or_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_semantic_work(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("candidate validation/encoding must not run after domain scan exhausts budget")

    monkeypatch.setattr(tool_wire_compiler, "schema_value_is_valid", unexpected_semantic_work)
    monkeypatch.setattr(tool_wire_compiler, "encode_lossless_raw_string", unexpected_semantic_work)
    values = [f"value-{index}" for index in range(5000)]
    schema = _schema({"content": {"type": "string", "enum": values}})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=False), allow_parallel=False),
        {"write": ("content",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=4,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert bundle.plan.budget_result.work_units > 4
    assert not bundle.plan.budget_result.within_budget
    assert bundle.constraint is None


def test_qwen_a2a_metered_support_scan_stops_before_wide_schema_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_legacy_scan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("semantic-final SCHEMA must not use the legacy unmetered support scan")

    original_charge = tool_wire_compiler._SemanticBudgetMeter.charge_work
    charge_calls = 0

    def counted_charge(self: object, units: int = 1) -> None:
        nonlocal charge_calls
        charge_calls += 1
        original_charge(self, units)

    monkeypatch.setattr(tool_wire_compiler, "_unsupported_schema_keyword", unexpected_legacy_scan)
    monkeypatch.setattr(tool_wire_compiler._SemanticBudgetMeter, "charge_work", counted_charge)
    children = {f"p{index}": {"type": "integer"} for index in range(1000)}
    schema = _schema(
        {
            "value": {
                "type": "object",
                "properties": children,
                "required": list(children),
                "additionalProperties": False,
            }
        }
    )
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("value",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=2,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert not bundle.plan.budget_result.within_budget
    assert bundle.plan.budget_result.work_units > 2
    assert charge_calls == 0


def test_qwen_a2a_metered_support_walker_precharges_wide_child_container() -> None:
    class CountingDict(dict[str, object]):
        def __init__(self, values: dict[str, object]) -> None:
            super().__init__(values)
            self.values_reads = 0

        def values(self):
            for value in super().values():
                self.values_reads += 1
                yield value

    properties = CountingDict(
        {f"p{index}": {"type": "integer"} for index in range(10_000)}
    )
    root = {
        "type": "object",
        "properties": properties,
        "required": [],
        "additionalProperties": False,
    }
    meter = tool_wire_compiler._SemanticBudgetMeter(max_bytes=10_000_000, max_work_units=4)
    with pytest.raises(tool_wire_compiler._BranchBudgetExceeded):
        tool_wire_compiler._unsupported_schema_keyword_metered(
            root,
            frozenset({"type", "properties", "required", "additionalProperties"}),
            meter,
        )
    assert meter.work_units == 10_004
    assert properties.values_reads == 0


def test_legacy_raw_finite_scalable_work_uses_the_plan_budget_owner() -> None:
    from tests.tool_wire._support import raw_compiler_capabilities, raw_spec, schema_plan

    values = [f"value-{index}" for index in range(10)]
    schema = _schema({"content": {"type": "string", "enum": values}})
    request_policy = policy(tool("write", schema, strict=True))
    roomy = schema_plan(
        raw_spec(),
        request_policy,
        {"write": ("content",)},
        compile_budget=CompileBudget(1000, 100_000, 10_000_000, 1_000_000),
        compiler_capabilities=raw_compiler_capabilities(),
    )
    exact_work = roomy.budget_result.work_units
    plan = schema_plan(
        raw_spec(),
        request_policy,
        {"write": ("content",)},
        compile_budget=CompileBudget(1000, 100_000, 10_000_000, exact_work),
        compiler_capabilities=raw_compiler_capabilities(),
    )
    argument = plan.tool("write").arguments[0]
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert plan.budget_result.within_budget
    assert plan.budget_result.work_units == exact_work
    assert argument.generated
    assert argument.proof.guarantee is GenerationGuarantee.SCHEMA
    assert argument.admitted_values_json is not None
    assert len(argument.admitted_values_json) == 10

    rejected = schema_plan(
        raw_spec(),
        policy(tool("write", schema, strict=True)),
        {"write": ("content",)},
        compile_budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=exact_work - 1,
        ),
        compiler_capabilities=raw_compiler_capabilities(),
    )
    assert rejected.disposition is PlanCompileDisposition.REJECTED


def test_qwen_a2a_structured_format_discards_large_finite_cardinality_from_work_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_candidate_validation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("FORMAT skeleton must not validate discarded enum candidates")

    monkeypatch.setattr(tool_wire_compiler, "schema_value_is_valid", unexpected_candidate_validation)
    schema = _schema({"count": {"type": "integer", "enum": list(range(5000))}})
    request_policy = policy(tool("write", schema, strict=False), allow_parallel=False)
    roomy = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("count",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, 1_000_000),
    )
    exact_work = roomy.plan.budget_result.work_units
    rejected = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("count",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, exact_work - 1),
    )
    assert rejected.plan.disposition is PlanCompileDisposition.REJECTED
    assert rejected.constraint is None
    assert rejected.plan.constraint_fingerprint is None
    assert rejected.plan.activation is None

    bundle = compile_qwen_a2a_shadow(
        request_policy,
        {"write": ("count",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(1000, 100_000, 10_000_000, exact_work),
    )
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.plan.budget_result.within_budget
    assert bundle.plan.budget_result.work_units == exact_work
    assert argument.value_mode is ConstraintValueMode.STRUCTURED_FORMAT
    assert argument.generation_schema_json == (
        '{"maximum":1000000000000000000,"minimum":-1000000000000000000,"type":"integer"}'
    )


def test_qwen_a2a_generation_schema_and_grammar_bytes_share_one_compile_budget() -> None:
    schema = _schema({"count": {"type": "integer"}})
    fn = tool("write", schema, strict=True)
    roomy = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("count",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=100_000,
        ),
    )
    assert roomy.constrained
    assert roomy.constraint is not None
    exact_bytes = roomy.plan.budget_result.estimated_bytes

    rejected = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("count",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=exact_bytes - 1,
            max_work_units=100_000,
        ),
    )
    assert rejected.plan.disposition is PlanCompileDisposition.REJECTED
    assert rejected.constraint is None
    assert rejected.plan.constraint_fingerprint is None
    assert rejected.plan.activation is None

    exact = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("count",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=exact_bytes,
            max_work_units=100_000,
        ),
    )
    argument = exact.plan.tool("write").arguments[0]
    assert exact.constrained
    assert exact.constraint is not None
    assert argument.generation_schema_json is not None
    grammar_bytes = len(exact.constraint.lark_grammar.encode("utf-8"))
    assert exact.plan.budget_result.estimated_bytes == exact_bytes
    assert exact_bytes >= (
        len(schema.encode("utf-8"))
        + len(argument.generation_schema_json.encode("utf-8"))
        + grammar_bytes
    )


def test_qwen_a2a_hard_schema_depth_rejects_before_generation_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_transform(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("generation transform must not run beyond the hard schema-depth envelope")

    monkeypatch.setattr(tool_wire_compiler, "_decoder_safe_generation_schema", unexpected_transform)
    nested: dict[str, object] = {"type": "integer"}
    for _ in range(tool_wire_compiler._HARD_MAX_SCHEMA_DEPTH + 1):
        nested = {"type": "array", "items": nested}
    schema = _schema({"value": nested})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("value",)},
        budget=CompileBudget(1000, 1_000_000, 100_000_000, 1_000_000),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert bundle.plan.tools == ()
    assert bundle.constraint is None


def test_qwen_a2a_generation_witness_work_has_exact_budget_boundary() -> None:
    children = {f"p{index}": {"type": "integer"} for index in range(10)}
    schema = _schema(
        {
            "value": {
                "type": "object",
                "properties": children,
                "required": list(children),
                "additionalProperties": False,
            }
        }
    )
    fn = tool("write", schema, strict=True)
    roomy = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("value",)},
    )
    exact_work = roomy.plan.budget_result.work_units
    assert roomy.constrained
    assert exact_work > 1

    rejected = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("value",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=exact_work - 1,
        ),
    )
    assert rejected.plan.disposition is PlanCompileDisposition.REJECTED
    assert rejected.constraint is None

    exact = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("value",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=exact_work,
        ),
    )
    assert exact.constrained
    assert exact.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert exact.plan.budget_result.within_budget
    assert exact.plan.budget_result.work_units == exact_work


def test_qwen_a2a_hard_finite_cardinality_rejects_before_semantic_nonemptiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_non_empty = tool_wire_compiler._exact_non_emptiness

    def counted_non_empty(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original_non_empty(*args, **kwargs)

    monkeypatch.setattr(tool_wire_compiler, "_exact_non_emptiness", counted_non_empty)
    values = list(range(tool_wire_compiler._HARD_MAX_FINITE_CARDINALITY + 1))
    schema = _schema({"value": {"type": "integer", "enum": values}})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("value",)},
        budget=CompileBudget(1000, 1_000_000, 100_000_000, 1_000_000),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert calls == 0


@pytest.mark.parametrize(
    ("property_schema", "expected_nonempty"),
    (
        ({"type": "integer", "const": 1, "enum": [1.0]}, True),
        ({"type": "number", "const": 1.0, "enum": [1]}, True),
        ({"type": "number", "const": -0.0, "enum": [0]}, True),
        (
            {"type": "array", "items": {"type": "number"}, "const": [1], "enum": [[1.0]]},
            True,
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
                "additionalProperties": False,
                "const": {"x": 1},
                "enum": [{"x": 1.0}],
            },
            True,
        ),
        ({"type": "boolean", "const": True, "enum": [1]}, False),
    ),
)
def test_qwen_a2a_structured_const_enum_uses_draft_json_equality(
    property_schema: dict[str, object],
    expected_nonempty: bool,
) -> None:
    bundle = _compile({"value": property_schema})
    argument = bundle.plan.tool("write").arguments[0]
    assert (argument.proof.non_empty.value == "proven_non_empty") is expected_nonempty
    assert bundle.constrained is expected_nonempty


def test_qwen_a2a_structured_numeric_equality_keeps_exact_budget_boundary() -> None:
    schema = _schema({"value": {"type": "number", "const": 1, "enum": [1.0]}})
    fn = tool("write", schema, strict=True)
    roomy = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("value",)},
    )
    exact_work = roomy.plan.budget_result.work_units
    assert roomy.constrained

    rejected = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("value",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=exact_work - 1,
        ),
    )
    assert rejected.plan.disposition is PlanCompileDisposition.REJECTED
    assert rejected.constraint is None

    exact = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("value",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=exact_work,
        ),
    )
    assert exact.constrained
    assert exact.plan.budget_result.work_units == exact_work


def test_qwen_a2a_structured_const_identity_waits_for_first_semantic_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_equality(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("const equality work must not start before an affordable charge")

    monkeypatch.setattr(tool_wire_compiler, "schema_values_equal", unexpected_equality)
    schema: dict[str, object] = {
        "type": "array",
        "items": {"type": "integer"},
        "const": list(range(10_000)),
        "enum": [list(range(10_000))],
    }
    meter = tool_wire_compiler._SemanticBudgetMeter(max_bytes=10_000_000, max_work_units=0)
    meter.provide_schema_json(canonical_json_dumps(schema))
    with pytest.raises(tool_wire_compiler._BranchBudgetExceeded):
        tool_wire_compiler._exact_non_emptiness(
            qwen_a2a_compiler_capabilities(),
            schema,
            semantic_budget=meter,
        )
    assert meter.work_units == 1


def test_qwen_a2a_hard_property_cardinality_rejects_before_semantic_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_calls = 0
    original_compile = tool_wire_compiler._compile_tool_semantics

    def counted_compile(*args: object, **kwargs: object):
        nonlocal semantic_calls
        semantic_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(tool_wire_compiler, "_compile_tool_semantics", counted_compile)
    width = tool_wire_compiler._HARD_MAX_OBJECT_PROPERTIES + 1
    properties = {f"p{index}": {"type": "integer"} for index in range(width)}
    schema = _schema(properties)
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": tuple(properties)},
        budget=CompileBudget(1000, 10_000_000, 100_000_000, 10_000_000),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert semantic_calls == 0


def test_qwen_a2a_hard_tool_count_rejects_before_schema_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parse = tool_wire_compiler.parse_json_strict
    parse_calls = 0

    def counted_parse(text: str):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(text)

    monkeypatch.setattr(tool_wire_compiler, "parse_json_strict", counted_parse)
    schema = _schema({})
    tool_count = tool_wire_compiler._HARD_MAX_EXPOSED_TOOLS + 1
    functions = tuple(tool(f"f{index}", schema, strict=True) for index in range(tool_count))
    bundle = compile_qwen_a2a_shadow(
        policy(*functions, allow_parallel=False),
        {function.name: () for function in functions},
        budget=CompileBudget(1000, 10_000_000, 100_000_000, 10_000_000),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert parse_calls == 0


def test_v3_exhaustive_permutation_width_has_hard_ceiling() -> None:
    spec = qwen_a2a_single_call_spec()
    within = tuple(f"p{index}" for index in range(tool_wire_compiler._MAX_EXHAUSTIVE_PERMUTABLE_WIDTH))
    over = (*within, "p6")

    within_prepared = tool_wire_compiler._PreparedOrderState(
        presentation_order=within,
        required_names=frozenset(),
        required=(),
        optional=within,
        minimum_order_bytes=0,
    )
    over_prepared = tool_wire_compiler._PreparedOrderState(
        presentation_order=over,
        required_names=frozenset(),
        required=(),
        optional=over,
        minimum_order_bytes=0,
    )

    assert tool_wire_compiler._full_order_candidate_v3(spec, within_prepared, 10_000) is not None
    assert tool_wire_compiler._full_order_candidate_v3(spec, over_prepared, 10_000) is None
    assert tool_wire_compiler._minimal_order_plan(over_prepared, spec).orders == ((),)


@pytest.mark.parametrize("width", (10, 1000, 10_000))
def test_decoder_safe_nested_required_precharges_before_scan(width: int) -> None:
    class TrackingList(list[str]):
        def __init__(self, values: list[str]) -> None:
            super().__init__(values)
            self.iter_reads = 0

        def __iter__(self):
            for value in super().__iter__():
                self.iter_reads += 1
                yield value

    required = TrackingList([f"r{index}" for index in range(width)])
    meter = tool_wire_compiler._SemanticBudgetMeter(
        max_bytes=100_000_000,
        max_work_units=2,
    )
    with pytest.raises(tool_wire_compiler._BranchBudgetExceeded):
        tool_wire_compiler._decoder_safe_generation_schema(
            {
                "type": "object",
                "properties": {},
                "required": required,
                "additionalProperties": False,
            },
            preserve_schema_semantics=False,
            semantic_budget=meter,
        )
    assert required.iter_reads == 0
    assert meter.work_units == width + 1


def test_qwen_a2a_static_tool_count_rejects_before_order_mapping_or_rejection_replay() -> None:
    class CountingOrders(Mapping[str, tuple[str, ...]]):
        def __init__(self, values: dict[str, tuple[str, ...]]) -> None:
            self._values = values
            self.iter_reads = 0
            self.contains_reads = 0
            self.getitem_reads = 0

        def __len__(self) -> int:
            return len(self._values)

        def __iter__(self) -> Iterator[str]:
            for key in self._values:
                self.iter_reads += 1
                yield key

        def __getitem__(self, key: str) -> tuple[str, ...]:
            self.getitem_reads += 1
            return self._values[key]

        def __contains__(self, key: object) -> bool:
            self.contains_reads += 1
            return key in self._values

    schema = _schema({})
    functions = tuple(tool(f"f{index}", schema, strict=True) for index in range(1000))
    orders = CountingOrders({function.name: () for function in functions})
    bundle = compile_qwen_a2a_shadow(
        policy(*functions, allow_parallel=False),
        orders,
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=1_000_000,
            max_estimated_bytes=100_000_000,
            max_work_units=1,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    # V3 hard-tool admission rejects before any order mapping access and does not synthesize
    # instruction-level attempted-work telemetry for an inadmissible request.
    assert (orders.iter_reads, orders.contains_reads, orders.getitem_reads) == (0, 0, 0)
    assert bundle.plan.presentation_orders == ()


def test_qwen_a2a_tool_source_byte_lower_bound_rejects_before_schema_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parse = tool_wire_compiler.parse_json_strict
    parse_calls = 0

    def counted_parse(text: str):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(text)

    monkeypatch.setattr(tool_wire_compiler, "parse_json_strict", counted_parse)
    schema = _schema({"x": {"type": "string", "const": "x" * 10_000}})
    fn = tool("write", schema, strict=True)
    spec = qwen_a2a_single_call_spec()
    source_lower_bound = (
        len(spec.spec_id.encode("utf-8"))
        + len(fn.name.encode("utf-8"))
        + len(fn.parameters.canonical_json.encode("utf-8"))
    )
    bundle = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("x",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=source_lower_bound - 1,
            max_work_units=100_000,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert bundle.plan.budget_result.estimated_bytes == source_lower_bound
    assert parse_calls == 0


def test_qwen_a2a_required_name_byte_overflow_stops_before_next_tool_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parse = tool_wire_compiler.parse_json_strict
    parsed_names: list[str] = []

    def counted_parse(text: str):
        parsed_names.append(text)
        return original_parse(text)

    monkeypatch.setattr(tool_wire_compiler, "parse_json_strict", counted_parse)
    required_name = "p" * 100
    schema = _schema({required_name: {"type": "string"}})
    first = tool("first", schema, strict=True)
    second = tool("second", schema, strict=True)
    spec = qwen_a2a_single_call_spec()
    first_source_bytes = (
        len(spec.spec_id.encode("utf-8"))
        + len(first.name.encode("utf-8"))
        + len(first.parameters.canonical_json.encode("utf-8"))
    )
    bundle = compile_qwen_a2a_shadow(
        policy(first, second, allow_parallel=False),
        {"first": (required_name,), "second": (required_name,)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=first_source_bytes + 1,
            max_work_units=100_000,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert parsed_names == [first.parameters.canonical_json]


def test_qwen_a2a_static_byte_lower_bound_rejects_before_any_schema_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parse = tool_wire_compiler.parse_json_strict
    parse_calls = 0

    def counted_parse(text: str):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(text)

    monkeypatch.setattr(tool_wire_compiler, "parse_json_strict", counted_parse)
    schema = _schema({"x": {"type": "string"}})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("x",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=1,
            max_work_units=100_000,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert bundle.plan.budget_result.estimated_bytes > 1
    assert parse_calls == 0


def test_qwen_a2a_root_support_classification_does_not_walk_wide_root_before_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingDict(dict[str, object]):
        def __init__(self, values: dict[str, object]) -> None:
            super().__init__(values)
            self.iter_reads = 0

        def __iter__(self):
            for key in super().__iter__():
                self.iter_reads += 1
                yield key

    original_parse = tool_wire_compiler.parse_json_strict
    parse_calls = 0
    tracked_root: TrackingDict | None = None

    def tracked_parse(text: str):
        nonlocal parse_calls, tracked_root
        parse_calls += 1
        parsed = original_parse(text)
        if parse_calls == 1 and isinstance(parsed, dict):
            tracked_root = TrackingDict(parsed)
            return tracked_root
        return parsed

    monkeypatch.setattr(tool_wire_compiler, "parse_json_strict", tracked_parse)
    root: dict[str, object] = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    root.update({f"custom_{index}": index for index in range(10_000)})
    schema = json.dumps(root, separators=(",", ":"))
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("x",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=1_000_000,
            max_estimated_bytes=100_000_000,
            max_work_units=2,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    # V3 allows the hard-bounded parse/envelope walk before product-budget rejection; it no longer
    # promises that a tiny semantic score prevents every implementation traversal.
    assert parse_calls == 1
    assert tracked_root is not None
    assert tracked_root.iter_reads == 0


def test_qwen_a2a_metered_root_witness_reuses_parsed_schema_before_first_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls = 0
    original_parse = tool_wire_semantic_authority.parse_json_strict

    def counted_parse(text: str):
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(text)

    monkeypatch.setattr(tool_wire_semantic_authority, "parse_json_strict", counted_parse)
    value = list(range(10_000))
    schema = _schema(
        {
            "x": {
                "type": "array",
                "items": {"type": "integer"},
                "const": value,
            }
        }
    )
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("x",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=100_000_000,
            max_work_units=2,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert parse_calls == 0


@pytest.mark.parametrize(
    "property_schema",
    (
        {"type": "array", "const": list(range(1000)), "enum": [list(range(1000))]},
        {"type": "string", "const": "x" * 10_000},
    ),
)
def test_qwen_a2a_format_source_schema_dump_waits_for_affordable_work(
    monkeypatch: pytest.MonkeyPatch,
    property_schema: dict[str, object],
) -> None:
    original_dump = tool_wire_compiler.canonical_json_dumps
    large_dumps = 0

    def counted_dump(value):
        nonlocal large_dumps
        if isinstance(value, dict):
            const = value.get("const")
            if isinstance(const, (list, str)) and len(const) >= 1000:
                large_dumps += 1
        return original_dump(value)

    monkeypatch.setattr(tool_wire_compiler, "canonical_json_dumps", counted_dump)
    schema = _schema({"x": property_schema})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=False), allow_parallel=False),
        {"write": ("x",)},
        mode=ToolConstraintMode.FORMAT,
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=100_000_000,
            max_work_units=2,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert large_dumps == 0


def test_qwen_a2a_generation_schema_is_in_final_v3_product_score() -> None:
    schema = _schema({"x": {"type": "integer", "enum": list(range(5000))}})
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("x",)},
        budget=CompileBudget(1000, 1_000_000, 100_000_000, 1_000_000),
    )
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert argument.generation_schema_json is not None
    assert bundle.constraint is not None
    assert bundle.plan.budget_result.estimated_bytes >= (
        len(schema.encode("utf-8"))
        + len(argument.generation_schema_json.encode("utf-8"))
        + len(bundle.constraint.lark_grammar.encode("utf-8"))
    )


def test_qwen_a2a_root_exact_witness_stops_before_unaffordable_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validations = 0
    original_validate = tool_wire_semantic_authority.schema_value_is_valid

    def counted_validate(*args: object, **kwargs: object) -> bool:
        nonlocal validations
        validations += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(tool_wire_semantic_authority, "schema_value_is_valid", counted_validate)
    values = [f"value-{index}" for index in range(2000)]
    schema = _schema(
        {"content": {"type": "string", "const": "target", "enum": values}}
    )
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("content",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=10_000_000,
            max_work_units=4,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert not bundle.plan.budget_result.within_budget
    assert bundle.plan.budget_result.work_units > 4
    assert validations == 0


def test_qwen_a2a_close_safe_finite_wire_expansion_counts_toward_byte_budget() -> None:
    semantic = "</parameter>" * 50
    schema = _schema({"content": {"type": "string", "const": semantic}})
    spec = qwen_a2a_single_call_spec()
    minimum_source_bytes = (
        len(spec.spec_id.encode("utf-8"))
        + len(b"write")
        + len(schema.encode("utf-8"))
        + len(b"content")
    )
    max_bytes = minimum_source_bytes + 100
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("content",)},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=100_000,
            max_estimated_bytes=max_bytes,
            max_work_units=100_000,
        ),
    )
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert not bundle.plan.budget_result.within_budget
    assert bundle.plan.budget_result.estimated_bytes > max_bytes
    assert not bundle.constrained


def test_qwen_a2a_structured_format_has_emit_capable_value_mode() -> None:
    schema = _schema({"count": {"type": "integer", "multipleOf": 2}})
    fn = tool("write", schema, strict=False)
    bundle = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("count",)},
        mode=ToolConstraintMode.FORMAT,
    )
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.plan.tool("write").guarantee is GenerationGuarantee.FORMAT
    assert argument.generated
    assert argument.value_mode is ConstraintValueMode.STRUCTURED_FORMAT
    assert argument.proof.guarantee is GenerationGuarantee.FORMAT
    assert argument.generation_schema_json == (
        '{"maximum":1000000000000000000,"minimum":-1000000000000000000,"type":"integer"}'
    )
    assert bundle.constraint is not None
    assert "%json {}" not in bundle.constraint.lark_grammar
    assert f"%json {argument.generation_schema_json}" in bundle.constraint.lark_grammar

    strict_bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("count",)},
        mode=ToolConstraintMode.SCHEMA,
    )
    strict_argument = strict_bundle.plan.tool("write").arguments[0]
    assert not strict_bundle.constrained
    assert not strict_argument.generated
    assert strict_argument.value_mode is ConstraintValueMode.VALIDATION_ONLY
    assert strict_argument.proof.guarantee is GenerationGuarantee.NONE
    assert strict_argument.generation_schema_json is None


def test_qwen_a2a_generated_value_modes_are_all_emitter_owned() -> None:
    raw = _compile({"content": {"type": "string"}})
    finite = _compile({"content": {"type": "string", "const": "safe"}})
    structured_schema = _compile({"count": {"type": "integer", "minimum": 0}})
    format_schema = _schema({"count": {"type": "integer", "multipleOf": 2}})
    structured_format = compile_qwen_a2a_shadow(
        policy(tool("write", format_schema, strict=False), allow_parallel=False),
        {"write": ("count",)},
        mode=ToolConstraintMode.FORMAT,
    )
    modes = {
        argument.value_mode
        for bundle in (raw, finite, structured_schema, structured_format)
        for branch in bundle.plan.tools
        for argument in branch.arguments
        if argument.generated
    }
    assert modes == {
        ConstraintValueMode.ANY_SAFE_RAW,
        ConstraintValueMode.FINITE_VALUES,
        ConstraintValueMode.STRUCTURED_FORMAT,
        ConstraintValueMode.STRUCTURED_SCHEMA,
    }
    assert set(ConstraintValueMode) == modes | {ConstraintValueMode.VALIDATION_ONLY}
    assert all(bundle.constraint is not None for bundle in (raw, finite, structured_schema, structured_format))


def test_qwen_a2a_finite_string_language_preserves_close_semantics_via_safe_alias() -> None:
    values = ["safe", "a<b", "bad</parameter>value"]
    bundle = _compile({"content": {"type": "string", "enum": values}})
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.constrained
    assert argument.value_mode is ConstraintValueMode.FINITE_VALUES
    assert argument.proof.guarantee is GenerationGuarantee.SCHEMA
    assert argument.admitted_values_json == tuple(canonical_json_dumps(value) for value in values)
    framing = bundle.spec.framing_variant("qwen-raw-string").value_framing
    encoded = encode_lossless_raw_string("bad</parameter>value", framing)
    assert encoded == '"bad\\u003c/parameter>value"'
    assert "</parameter>" not in encoded
    assert framing.codec.decode_raw_payload(encoded) == "bad</parameter>value"


def test_qwen_a2a_required_close_text_const_is_executable_via_safe_json_string_wire() -> None:
    semantic = "x</parameter>y"
    bundle = _compile({"content": {"type": "string", "const": semantic}})
    argument = bundle.plan.tool("write").arguments[0]
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert argument.admitted_values_json == (canonical_json_dumps(semantic),)
    assert argument.generated
    assert argument.proof.guarantee is GenerationGuarantee.SCHEMA
    assert bundle.constraint is not None
    framing = bundle.spec.framing_variant("qwen-raw-string").value_framing
    encoded = encode_lossless_raw_string(semantic, framing)
    assert encoded == '"x\\u003c/parameter>y"'
    assert canonical_json_dumps(encoded) in bundle.constraint.lark_grammar


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("minLength", 2),
        ("maxLength", 8),
        ("pattern", "^x+$"),
        ("format", "email"),
        ("oneOf", [{"type": "string"}, {"type": "null"}]),
        ("anyOf", [{"type": "string"}, {"type": "integer"}]),
    ],
)
def test_qwen_a2a_unsupported_string_semantics_never_claim_schema(
    keyword: str,
    value: object,
) -> None:
    property_schema: dict[str, object] = {"type": "string", keyword: value}
    bundle = _compile({"content": property_schema}, strict=True)
    argument = bundle.plan.tool("write").arguments[0]
    assert not bundle.constrained
    assert argument.proof.guarantee is GenerationGuarantee.NONE
    assert not argument.generated


def test_qwen_a2a_refs_are_inventoried_as_unsupported_without_false_schema() -> None:
    schema = json.dumps(
        {
            "$defs": {"value": {"type": "string"}},
            "type": "object",
            "properties": {"content": {"$ref": "#/$defs/value"}},
            "required": ["content"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    bundle = compile_qwen_a2a_shadow(
        policy(tool("write", schema, strict=True), allow_parallel=False),
        {"write": ("content",)},
    )
    argument = bundle.plan.tool("write").arguments[0]
    assert not bundle.constrained
    assert argument.value_mode is ConstraintValueMode.VALIDATION_ONLY
    assert argument.proof.guarantee is GenerationGuarantee.NONE
    assert not argument.generated


def test_qwen_a2a_mixed_raw_and_structured_plan_shares_one_parser_contract() -> None:
    bundle = _compile(
        {
            "content": {"type": "string"},
            "count": {"type": "integer", "minimum": 0},
            "enabled": {"type": "boolean"},
            "meta": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        },
        presentation=("content", "count", "enabled", "meta"),
    )
    wire = _wire(
        bundle,
        (
            ("content", "a<b\n```{x}```"),
            ("count", "7"),
            ("enabled", "true"),
            ("meta", '{"x":2}'),
        ),
    )
    result = _finish(bundle, wire, (wire,))
    assert result.is_complete and result.sequence is not None
    assert admit_tool_sequence(bundle.spec, bundle.plan, result.sequence).is_valid
    call = result.sequence.calls[0]
    assert call.index == 0
    assert [(item.name, item.framing_variant_id) for item in call.occurrences] == [
        ("content", "qwen-raw-string"),
        ("count", "qwen-json-structured"),
        ("enabled", "qwen-json-structured"),
        ("meta", "qwen-json-structured"),
    ]
    assert [parse_json_strict(item.canonical_value_json) for item in call.occurrences] == [
        "a<b\n```{x}```",
        7,
        True,
        {"x": 2},
    ]


def test_qwen_a2a_every_character_split_and_one_character_chunks_are_invariant() -> None:
    bundle = _compile(
        {"content": {"type": "string"}, "count": {"type": "integer"}},
        presentation=("content", "count"),
    )
    wire = _wire(bundle, (("content", "prefix </param <x>"), ("count", "9")))
    baseline = _finish(bundle, wire, (wire,))
    assert baseline.is_complete
    for split in range(len(wire) + 1):
        assert _finish(bundle, wire, (wire[:split], wire[split:])) == baseline
    assert _finish(bundle, wire, tuple(wire)) == baseline


def test_qwen_a2a_raw_close_collision_is_structural_not_source_language_heuristic() -> None:
    bundle = _compile({"content": {"type": "string"}})
    malicious = _wire(bundle, (("content", 'prefix "</parameter>" suffix'),))
    result = _finish(bundle, malicious, (malicious,))
    assert result.status is ToolWireEngineStatus.MALFORMED
    assert result.sequence is None
    assert {issue.code for issue in result.issues} <= {
        "argument_open_malformed",
        "function_open_malformed",
        "function_name_not_representable",
        "trailing_wire",
    }


def test_qwen_a2a_file_path_before_content_presentation_order_is_preserved() -> None:
    properties = {
        "content": {"type": "string"},
        "file_path": {"type": "string"},
    }
    bundle = _compile(
        properties,
        presentation=("file_path", "content"),
        allow_parallel=True,
    )
    branch = bundle.plan.tool("write")
    assert branch.order_plan.orders[0] == ("file_path", "content")
    assert ("content", "file_path") in branch.order_plan.orders
    assert bundle.parallel_generation_narrowed
    assert bundle.spec.multiplicity.max_calls_per_sequence == 1
    wire = _wire(bundle, (("file_path", "/tmp/a.py"), ("content", "print(1)")))
    result = _finish(bundle, wire, tuple(wire))
    assert result.is_complete and result.sequence is not None
    assert [item.name for item in result.sequence.calls[0].occurrences] == [
        "file_path",
        "content",
    ]


def test_qwen_a2a_prompt_template_parity_uses_existing_a0_certifier() -> None:
    bundle = _compile(
        {"content": {"type": "string"}, "count": {"type": "integer"}},
        presentation=("content", "count"),
    )
    observation = qwen_a2a_prompt_observation(bundle.plan)
    result = certify_prompt_template_parity(bundle.spec, observation, bundle.plan)
    assert result.is_valid
    assert observation.tool_open == "<tool_call>"
    assert observation.function_open_prefix == "<function="
    assert {argument.argument_open_prefix for argument in observation.arguments} == {"<parameter="}
    assert {argument.argument_close for argument in observation.arguments} == {"</parameter>"}


def test_qwen_a2a_historical_fixture_inventory_is_carried_forward_without_production_patch() -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "qwen_c1f1e3c_a1a2_regressions.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["production_patch_authorized"] is False
    assert {case["id"] for case in fixture["cases"]} == {
        "unique_valid_late_tool_hidden_by_lexical_pruning",
        "early_tool_commit_before_later_competing_valid_interpretation_resolves",
    }

    bundle = _compile({"content": {"type": "string"}})
    # Both historical defects require a competing interpretation in which an exact parameter
    # close appears inside what would otherwise be raw content.  A2a generation excludes that
    # interpretation before parsing, rather than recovering intent with quote/source heuristics.
    conflict = 'prefix "</parameter></function></tool_call>" suffix'
    wire = _wire(bundle, (("content", conflict),))
    result = _finish(bundle, wire, (wire,))
    assert not result.is_complete


def test_qwen_a2a_incomplete_eos_remains_internal_and_unpublished() -> None:
    bundle = _compile({"content": {"type": "string"}})
    partial = "<tool_call><function=write><parameter=content>unterminated"
    result = _finish(bundle, partial, (partial[:13], partial[13:]))
    assert result.status is ToolWireEngineStatus.INCOMPLETE
    assert result.sequence is None
