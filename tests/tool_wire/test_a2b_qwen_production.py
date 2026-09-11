from __future__ import annotations

import json
from itertools import pairwise

import pytest

from exqserve.agent._json import canonical_json_dumps, parse_json_strict
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.events import (
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.generation_guarantees import (
    ConstraintFallbackPolicy,
    GenerationGuarantee,
)
from exqserve.core.tokens import NativeTokenSpan
from exqserve.model.contracts import ToolConstraintMode, ToolConstraintUnsupported
from exqserve.model.qwen import QwenIncrementalParser
from exqserve.runtime.contracts import ConstraintInstallation
from exqserve.server.qwen_parser_binding import resolve_qwen_parser_context
from exqserve.serving.tool_batch import ToolCallBatchGate
from exqserve.tool_wire.contracts import (
    PlanCompileDisposition,
    ValueCodecKind,
    ValueFramingKind,
    WireArgumentOccurrence,
    WireToolCall,
    WireToolSequence,
    encode_lossless_raw_string,
)
from exqserve.tool_wire.controls import qwen as qwen_control
from exqserve.tool_wire.controls.qwen import compile_qwen_tool_wire
from exqserve.tool_wire.controls.qwen import (
    qwen_production_tool_constraint as qwen_tool_constraint,
)
from exqserve.tool_wire.lark_constraint import ToolWireConstraintLoweringUnsupported
from tests.tool_wire._legacy_admission import admit_tool_sequence


def _tool(name: str, schema: str, *, strict: bool = False) -> FunctionTool:
    return FunctionTool(name, None, JsonSchema(schema), strict)


def _policy(*tools: FunctionTool, parallel: bool = False) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(ToolChoiceMode.AUTO), parallel)


def _integer_schema(name: str = "x") -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {name: {"type": "integer"}},
            "required": [name],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )


class _ByteTokenizer:
    eos_token_id = 256
    bos_token_id = None
    tokens = tuple(bytes([value]) for value in range(256)) + (b"<eos>",)
    special_token_ids = (256,)

    def __call__(self, value: bytes | str) -> list[int]:
        if isinstance(value, str):
            value = value.encode("utf-8")
        return list(value)


def _validation_parser(request_id: str, policy: ToolPolicy) -> QwenIncrementalParser:
    context = resolve_qwen_parser_context(
        None,
        None,
        ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
    )
    assert context is not None
    return QwenIncrementalParser(request_id, tool_policy=policy, parser_context=context)


def _constraint_accepts(constraint, suffix: str) -> bool:
    llguidance = pytest.importorskip("llguidance")
    grammar = llguidance.LLMatcher.grammar_from_lark(constraint.lark_grammar)
    vocabulary = llguidance.LLTokenizer(llguidance.TokenizerWrapper(_ByteTokenizer()))
    matcher = llguidance.LLMatcher(vocabulary, grammar)
    return bool(
        matcher.consume_tokens(list(suffix.encode("utf-8")))
        and matcher.is_accepting()
        and not matcher.is_error()
    )


def _native_spans(text: str) -> tuple[NativeTokenSpan, ...]:
    spans: list[NativeTokenSpan] = []
    for marker, token_id in (("<tool_call>", 248058), ("</tool_call>", 248059)):
        cursor = 0
        while True:
            start = text.find(marker, cursor)
            if start < 0:
                break
            spans.append(NativeTokenSpan(start, start + len(marker), token_id, marker))
            cursor = start + len(marker)
    return tuple(sorted(spans, key=lambda span: span.start))


def test_a2b_production_bundle_binds_one_tool_wire_artifact() -> None:
    fn = _tool("save", _integer_schema(), strict=True)
    bundle = compile_qwen_tool_wire(_policy(fn), ToolConstraintMode.OFF)

    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.constraint is not None
    assert bundle.plan.constraint_fingerprint == bundle.grammar_fingerprint
    assert bundle.plan.activation is not None
    assert bundle.plan.activation.trigger_ids == ("tool-open",)
    assert bundle.constraint.trigger == "<tool_call>"
    assert bundle.spec.multiplicity.adjacent_tools
    assert bundle.spec.multiplicity.max_calls_per_sequence is None
    assert bundle.parallel_generation_narrowed
    assert not bundle.order_generation_narrowed
    assert not bundle.branch_generation_narrowed
    assert bundle.request_policy_narrowed


def test_a2b_bundle_reports_order_narrowing_explicitly() -> None:
    properties = {f"p{index}": {"type": "integer"} for index in range(7)}
    schema = json.dumps(
        {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    bundle = compile_qwen_tool_wire(
        _policy(_tool("wide", schema)),
        ToolConstraintMode.SCHEMA,
    )

    assert bundle.constraint is not None
    assert bundle.plan.tool("wide").order_plan.narrowed
    assert bundle.parallel_generation_narrowed
    assert bundle.order_generation_narrowed
    assert not bundle.branch_generation_narrowed
    assert bundle.request_policy_narrowed


def test_a2b_qwen_provider_delegates_to_one_production_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = compile_qwen_tool_wire
    calls = 0

    def counted(policy: ToolPolicy, mode: ToolConstraintMode, *, max_parallel_calls: int):
        nonlocal calls
        calls += 1
        return original(policy, mode, max_parallel_calls=max_parallel_calls)

    monkeypatch.setattr(
        "exqserve.tool_wire.controls.qwen.compile_qwen_tool_wire",
        counted,
    )
    constraint = qwen_tool_constraint(
        _policy(_tool("save", _integer_schema(), strict=True)),
        ToolConstraintMode.OFF,
    )

    assert constraint is not None
    assert calls == 1


def test_a2b_off_non_strict_stays_unconstrained() -> None:
    fn = _tool("lookup", _integer_schema())
    bundle = compile_qwen_tool_wire(_policy(fn), ToolConstraintMode.OFF)

    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None
    assert bundle.grammar_fingerprint is None
    assert qwen_tool_constraint(_policy(fn), ToolConstraintMode.OFF) is None


def test_a2b_mixed_off_preserves_strict_schema_and_loose_format() -> None:
    strict = _tool("strict_tool", _integer_schema("strict_value"), strict=True)
    loose = _tool("loose_tool", _integer_schema("loose_value"))
    bundle = compile_qwen_tool_wire(_policy(strict, loose), ToolConstraintMode.OFF)

    assert bundle.constraint is not None
    assert bundle.constraint.guarantee_for_tool("strict_tool") is GenerationGuarantee.SCHEMA
    assert bundle.constraint.guarantee_for_tool("loose_tool") is GenerationGuarantee.FORMAT
    assert bundle.plan.tool("strict_tool").guarantee is GenerationGuarantee.SCHEMA
    assert bundle.plan.tool("loose_tool").guarantee is GenerationGuarantee.FORMAT
    assert '"<function=strict_tool>"' in bundle.constraint.lark_grammar
    assert '"<function=loose_tool>"' in bundle.constraint.lark_grammar


def test_a2b_non_strict_unsupported_schema_falls_back_but_strict_fails_closed() -> None:
    schema = (
        '{"type":"object","properties":{"command":'
        '{"type":"string","pattern":"^git .+$"}},"required":["command"],'
        '"additionalProperties":false}'
    )
    loose = _tool("run", schema)
    strict = _tool("run", schema, strict=True)

    schema_bundle = compile_qwen_tool_wire(_policy(loose), ToolConstraintMode.SCHEMA)
    assert schema_bundle.constraint is not None
    assert schema_bundle.constraint.guarantee_for_tool("run") is GenerationGuarantee.FORMAT
    assert schema_bundle.branch_generation_narrowed
    assert schema_bundle.request_policy_narrowed
    format_constraint = qwen_tool_constraint(_policy(loose), ToolConstraintMode.FORMAT)
    assert format_constraint is not None
    assert format_constraint.guarantee_for_tool("run") is GenerationGuarantee.FORMAT
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(_policy(strict), ToolConstraintMode.OFF)


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("minProperties", 1),
        ("maxProperties", 1),
        ("dependentRequired", {"x": ["y"]}),
        ("propertyNames", {"pattern": "^x$"}),
        ("unevaluatedProperties", False),
    ),
)
def test_a2b_root_semantics_reach_non_strict_schema_to_format_fallback(
    keyword: str,
    value: object,
) -> None:
    schema_value: dict[str, object] = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": [],
        "additionalProperties": False,
        keyword: value,
    }
    schema = json.dumps(schema_value, separators=(",", ":"))

    loose = compile_qwen_tool_wire(_policy(_tool("loose", schema)), ToolConstraintMode.SCHEMA)
    assert loose.constraint is not None
    assert loose.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert loose.constraint.guarantee_for_tool("loose") is GenerationGuarantee.FORMAT
    assert loose.branch_generation_narrowed
    assert loose.request_policy_narrowed

    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(_policy(_tool("strict", schema, strict=True)), ToolConstraintMode.SCHEMA)


def test_a2b_root_annotation_does_not_force_format_fallback() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": [],
            "additionalProperties": False,
            "$comment": "annotation only",
        },
        separators=(",", ":"),
    )

    bundle = compile_qwen_tool_wire(_policy(_tool("t", schema)), ToolConstraintMode.SCHEMA)
    assert bundle.constraint is not None
    assert bundle.constraint.guarantee_for_tool("t") is GenerationGuarantee.SCHEMA


def test_a2b_invalid_tag_name_falls_back_or_fails_closed_by_strictness() -> None:
    loose = _tool("bad name", _integer_schema())
    strict = _tool("bad name", _integer_schema(), strict=True)

    assert qwen_tool_constraint(_policy(loose), ToolConstraintMode.FORMAT) is None
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(_policy(strict), ToolConstraintMode.OFF)


def test_production_generation_can_be_explicitly_limited_to_one_tool_envelope() -> None:
    first = _tool("a", _integer_schema())
    second = _tool("b", _integer_schema())
    serial = compile_qwen_tool_wire(
        _policy(first, second, parallel=False),
        ToolConstraintMode.SCHEMA,
    )
    parallel = compile_qwen_tool_wire(
        _policy(first, second, parallel=True),
        ToolConstraintMode.SCHEMA,
        max_parallel_calls=1,
    )
    assert serial.constraint is not None and parallel.constraint is not None

    first_suffix = "<function=a><parameter=x>\n1\n</parameter>\n</function></tool_call>"
    two_suffix = (
        first_suffix
        + "<tool_call><function=b><parameter=x>\n2\n</parameter>\n</function></tool_call>"
    )
    repeat_suffix = (
        first_suffix
        + "<tool_call><function=a><parameter=x>\n1\n</parameter>\n</function></tool_call>"
    )

    assert _constraint_accepts(serial.constraint, first_suffix)
    assert _constraint_accepts(parallel.constraint, first_suffix)
    assert not _constraint_accepts(serial.constraint, two_suffix)
    assert not _constraint_accepts(parallel.constraint, two_suffix)
    assert not _constraint_accepts(parallel.constraint, repeat_suffix)
    assert serial.parallel_generation_narrowed
    assert parallel.parallel_generation_narrowed


def test_a2b_shared_admission_honors_request_parallel_policy() -> None:
    first = _tool("a", _integer_schema())
    second = _tool("b", _integer_schema())
    serial = compile_qwen_tool_wire(_policy(first, second, parallel=False), ToolConstraintMode.SCHEMA)
    parallel = compile_qwen_tool_wire(_policy(first, second, parallel=True), ToolConstraintMode.SCHEMA)
    assert serial.constraint is not None and parallel.constraint is not None

    def call(bundle, name: str, index: int, value: int) -> WireToolCall:
        argument = bundle.plan.tool(name).arguments[0]
        assert argument.framing_variant_id is not None
        return WireToolCall(
            name,
            index,
            (
                WireArgumentOccurrence(
                    "x",
                    canonical_json_dumps(value),
                    argument.framing_variant_id,
                ),
            ),
        )

    serial_two = WireToolSequence((call(serial, "a", 0, 1), call(serial, "b", 1, 2)))
    parallel_two = WireToolSequence((call(parallel, "a", 0, 1), call(parallel, "b", 1, 2)))

    serial_result = admit_tool_sequence(serial.spec, serial.plan, serial_two)
    assert not serial_result.is_valid
    assert "request_parallel_tools_forbidden" in {issue.code for issue in serial_result.issues}
    assert admit_tool_sequence(parallel.spec, parallel.plan, parallel_two).is_valid


@pytest.mark.parametrize("limit", (1, 2, 8))
def test_production_grammar_allows_repeated_calls_only_up_to_configured_limit(limit: int) -> None:
    bundle = compile_qwen_tool_wire(
        _policy(_tool("read", _integer_schema()), parallel=True),
        ToolConstraintMode.SCHEMA,
        max_parallel_calls=limit,
    )
    assert bundle.constraint is not None
    suffix = '<function=read><parameter=x>\n1\n</parameter>\n</function></tool_call>'
    following = '<tool_call>' + suffix
    assert _constraint_accepts(bundle.constraint, suffix)
    assert _constraint_accepts(bundle.constraint, suffix + following * (limit - 1))
    assert not _constraint_accepts(bundle.constraint, suffix + following * limit)
    assert bundle.plan.max_calls_per_sequence == limit


@pytest.mark.parametrize("limit", (0, -1, True, 1.5))
def test_production_grammar_rejects_invalid_call_limit(limit) -> None:
    with pytest.raises(ValueError, match="max_parallel_calls"):
        compile_qwen_tool_wire(
            _policy(_tool("read", _integer_schema()), parallel=True),
            ToolConstraintMode.SCHEMA,
            max_parallel_calls=limit,
        )


def test_a2b_hidden_tools_do_not_consume_exposed_generation_hard_cap() -> None:
    tools = tuple(_tool(f"t{index}", _integer_schema()) for index in range(129))
    none_policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.NONE), False)
    named_policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.NAMED, "t0"), False)
    auto_policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), False)

    none_bundle = compile_qwen_tool_wire(none_policy, ToolConstraintMode.SCHEMA)
    named_bundle = compile_qwen_tool_wire(named_policy, ToolConstraintMode.SCHEMA)
    auto_schema = compile_qwen_tool_wire(auto_policy, ToolConstraintMode.SCHEMA)
    auto_off = compile_qwen_tool_wire(auto_policy, ToolConstraintMode.OFF)

    assert none_bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert none_bundle.constraint is None
    assert named_bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert named_bundle.constraint is not None
    assert tuple(tool.tool_name for tool in named_bundle.plan.tools) == ("t0",)
    assert auto_schema.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert auto_schema.constraint is None
    assert auto_off.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert auto_off.constraint is None


def test_a2b_exposed_hard_cap_short_circuits_before_schema_prevalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = tuple(_tool(f"t{index}", _integer_schema()) for index in range(129))
    policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), False)

    def unexpected_prevalidation(schema: JsonSchema) -> object:
        del schema
        raise AssertionError("generation hard cap must short-circuit before schema prevalidation")

    monkeypatch.setattr(
        "exqserve.tool_wire.controls.qwen.qwen_parameter_envelope_schema",
        unexpected_prevalidation,
    )

    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None
    assert bundle.branch_generation_narrowed


def test_a2b_per_tool_source_cap_short_circuits_before_qwen_schema_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
            "description": "x" * 1_100_000,
        },
        separators=(",", ":"),
    )

    def unexpected_prevalidation(value: JsonSchema) -> object:
        del value
        raise AssertionError("per-Tool source cap must run before Qwen schema parsing")

    monkeypatch.setattr(
        "exqserve.tool_wire.controls.qwen.qwen_parameter_envelope_schema",
        unexpected_prevalidation,
    )
    bundle = compile_qwen_tool_wire(_policy(_tool("t", schema)), ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None
    assert bundle.plan.budget_result.work_units == 0


def test_a2b_aggregate_source_cap_short_circuits_before_qwen_schema_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = tuple(
        _tool(
            f"t{index}",
            json.dumps(
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                    "additionalProperties": False,
                    "description": "x" * 400_000,
                },
                separators=(",", ":"),
            ),
        )
        for index in range(12)
    )

    def unexpected_prevalidation(value: JsonSchema) -> object:
        del value
        raise AssertionError("aggregate source cap must run before Qwen schema parsing")

    monkeypatch.setattr(
        "exqserve.tool_wire.controls.qwen.qwen_parameter_envelope_schema",
        unexpected_prevalidation,
    )
    bundle = compile_qwen_tool_wire(_policy(*tools), ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None
    assert bundle.plan.budget_result.work_units == 0


def test_a2b_exposed_hard_cap_short_circuits_before_effective_mode_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.tool_wire.controls import qwen as qwen_control

    tools = tuple(_tool(f"t{index}", _integer_schema()) for index in range(129))
    policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), False)

    def unexpected_modes(policy: ToolPolicy, mode: ToolConstraintMode) -> object:
        del policy, mode
        raise AssertionError("exposed Tool hard cap must precede per-Tool mode construction")

    monkeypatch.setattr(qwen_control, "_qwen_effective_tool_modes", unexpected_modes)
    bundle = qwen_control.compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None


@pytest.mark.parametrize("name", ("n" * 1025, "n\ud800"))
def test_a2b_tool_name_hard_rejects_before_qwen_schema_parse(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.tool_wire.controls import qwen as qwen_control

    def unexpected_parse(schema: JsonSchema) -> object:
        del schema
        raise AssertionError("Tool-name hard rejection must precede Qwen schema parsing")

    monkeypatch.setattr(qwen_control._tool_constraints, "constraint_schema", unexpected_parse)
    bundle = qwen_control.compile_qwen_tool_wire(
        _policy(_tool(name, _integer_schema(), strict=True)),
        ToolConstraintMode.SCHEMA,
    )

    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert bundle.constraint is None


def test_a2b_object_cardinality_hard_rejects_before_qwen_envelope_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.tool_wire.controls import qwen as qwen_control

    class TrackingDict(dict[str, object]):
        item_reads = 0

        def items(self):
            for item in super().items():
                self.item_reads += 1
                yield item

    class TrackingList(list[str]):
        iter_reads = 0

        def __iter__(self):
            for item in super().__iter__():
                self.iter_reads += 1
                yield item

    count = 1025
    plain_properties = {f"p{index}": {"type": "integer"} for index in range(count)}
    plain_required = [f"p{index}" for index in range(count)]
    schema = json.dumps(
        {
            "type": "object",
            "properties": plain_properties,
            "required": plain_required,
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    tracked_properties = TrackingDict(plain_properties)
    tracked_required = TrackingList(plain_required)
    tracked_root: dict[str, object] = {
        "type": "object",
        "properties": tracked_properties,
        "required": tracked_required,
        "additionalProperties": False,
    }

    monkeypatch.setattr(
        qwen_control._tool_constraints,
        "constraint_schema",
        lambda value: tracked_root,
    )
    bundle = qwen_control.compile_qwen_tool_wire(
        _policy(_tool("t", schema)),
        ToolConstraintMode.SCHEMA,
    )

    assert tracked_properties.item_reads == 0
    assert tracked_required.iter_reads == 0
    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None


def test_a2b_shared_root_ref_optional_products_never_create_negative_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = json.dumps(
        {
            "$defs": {
                "shared": {
                    "type": "object",
                    "properties": {"leaf": {"type": "integer"}},
                    "required": ["leaf"],
                    "additionalProperties": False,
                }
            },
            "type": "object",
            "properties": {
                f"p{index}": {"$ref": "#/$defs/shared"}
                for index in range(10)
            },
            "required": [],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )

    from exqserve.tool_wire.controls import qwen as qwen_control

    resolver_calls = 0
    original_resolver = qwen_control.qwen_property_schema

    def counted_resolver(
        root_schema: dict[str, object],
        property_schema: dict[str, object],
    ) -> dict[str, object]:
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(root_schema, property_schema)

    monkeypatch.setattr(qwen_control, "qwen_property_schema", counted_resolver)
    bundle = qwen_control.compile_qwen_tool_wire(
        _policy(_tool("t", schema)),
        ToolConstraintMode.SCHEMA,
    )

    assert resolver_calls == 1
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.plan.budget_result.work_units >= 0
    assert bundle.plan.tool("t").order_plan.accepts(())
    assert len(bundle.plan.tool("t").order_plan.optional_names) == 10


def test_a2b_required_shared_root_ref_reuses_one_resolver_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.tool_wire.controls import qwen as qwen_control

    names = [f"p{index}" for index in range(20)]
    schema = json.dumps(
        {
            "$defs": {
                "shared": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                }
            },
            "type": "object",
            "properties": {name: {"$ref": "#/$defs/shared"} for name in names},
            "required": names,
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    resolver_calls = 0
    original_resolver = qwen_control.qwen_property_schema

    def counted_resolver(
        root_schema: dict[str, object],
        property_schema: dict[str, object],
    ) -> dict[str, object]:
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(root_schema, property_schema)

    monkeypatch.setattr(qwen_control, "qwen_property_schema", counted_resolver)
    bundle = qwen_control.compile_qwen_tool_wire(
        _policy(_tool("t", schema)),
        ToolConstraintMode.SCHEMA,
    )

    assert resolver_calls == 1
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.constraint is not None


def test_a2b_detached_ref_resolution_stops_at_remaining_aggregate_node_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.tool_wire import compiler

    target: dict[str, object] = {
        "type": "integer",
        "title": "DETACHED_TARGET",
        "examples": [None] * 30,
    }
    root_with_ref: dict[str, object] = {
        "$defs": {"shared": target},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/shared"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    filler_root: dict[str, object] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
        "examples": [None] * 30,
    }
    schema_with_ref = json.dumps(root_with_ref, separators=(",", ":"))
    filler_schema = json.dumps(filler_root, separators=(",", ":"))
    parsed_ref = parse_json_strict(schema_with_ref)
    parsed_filler = parse_json_strict(filler_schema)
    assert isinstance(parsed_ref, dict)
    assert isinstance(parsed_filler, dict)
    source_nodes = compiler._json_node_count(parsed_ref) + compiler._json_node_count(parsed_filler)
    detached_nodes = compiler._json_node_count(target)
    remaining_nodes = 5
    assert detached_nodes > remaining_nodes

    monkeypatch.setattr(
        compiler,
        "_HARD_MAX_SCHEMA_NODES_TOTAL",
        source_nodes + remaining_nodes,
    )
    original_count = compiler._json_node_count
    full_detached_counts: list[int] = []

    def counted(value):
        nodes = original_count(value)
        if isinstance(value, dict) and value.get("title") == "DETACHED_TARGET":
            full_detached_counts.append(nodes)
        return nodes

    monkeypatch.setattr(compiler, "_json_node_count", counted)
    bundle = compile_qwen_tool_wire(
        _policy(
            _tool("a", schema_with_ref, strict=True),
            _tool("b", filler_schema),
        ),
        ToolConstraintMode.SCHEMA,
    )

    assert full_detached_counts == []
    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert bundle.constraint is None


@pytest.mark.parametrize(
    ("right_depth", "expected_disposition"),
    (
        (35, PlanCompileDisposition.CONSTRAINED_EXECUTABLE),
        (36, PlanCompileDisposition.REJECTED),
    ),
)
def test_a2b_detached_ref_resolution_enforces_exact_depth_boundary(
    right_depth: int,
    expected_disposition: PlanCompileDisposition,
) -> None:
    def nested_array(depth: int, leaf: dict[str, object]) -> dict[str, object]:
        result = leaf
        for _ in range(depth):
            result = {"type": "array", "items": result}
        return result

    schema = json.dumps(
        {
            "$defs": {
                "left": nested_array(27, {"$ref": "#/$defs/right"}),
                "right": nested_array(right_depth, {"type": "integer"}),
            },
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/left"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )

    bundle = compile_qwen_tool_wire(
        _policy(_tool("t", schema, strict=True)),
        ToolConstraintMode.SCHEMA,
    )

    assert bundle.plan.disposition is expected_disposition
    assert (bundle.constraint is not None) is (
        expected_disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    )


@pytest.mark.parametrize(
    ("strict_present", "expected_disposition"),
    (
        (False, PlanCompileDisposition.VALIDATION_ONLY),
        (True, PlanCompileDisposition.REJECTED),
    ),
)
def test_a2b_over_exposed_tool_cap_does_not_rescan_strict_flags(
    strict_present: bool,
    expected_disposition: PlanCompileDisposition,
) -> None:
    strict_reads = 0

    class TrackingTool(FunctionTool):
        def __getattribute__(self, name: str):
            nonlocal strict_reads
            if name == "strict":
                strict_reads += 1
            return super().__getattribute__(name)

    schema = JsonSchema(_integer_schema())
    tools = tuple(
        TrackingTool(f"t{index}", None, schema, strict_present and index == 128)
        for index in range(129)
    )
    policy = ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), False)
    strict_reads = 0

    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert strict_reads == 0
    assert bundle.plan.disposition is expected_disposition
    assert bundle.constraint is None


def test_a2b_oversize_tool_name_is_rejected_before_hashing() -> None:
    hash_calls = 0
    hashed_chars = 0

    class TrackingStr(str):
        def __hash__(self) -> int:
            nonlocal hash_calls, hashed_chars
            hash_calls += 1
            hashed_chars += len(self)
            return super().__hash__()

    name = TrackingStr("n" * 1025)
    policy = _policy(FunctionTool(name, None, JsonSchema(_integer_schema()), False))
    hash_calls = 0
    hashed_chars = 0

    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert hash_calls == 0
    assert hashed_chars == 0
    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None


def test_a2b_failed_optional_ref_candidate_does_not_contaminate_later_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.model.tool_constraints import qwen_property_schema
    from exqserve.tool_wire import compiler

    def object_target(width: int) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {f"p{index}": {"type": "integer"} for index in range(width)},
            "required": [f"p{index}" for index in range(width)],
            "additionalProperties": False,
        }

    def tool_schema(definitions: dict[str, object], optional_refs: tuple[str, ...]) -> str:
        properties: dict[str, object] = {"r": {"type": "integer"}}
        properties.update(
            {name: {"$ref": f"#/$defs/{definition}"} for name, definition in zip(optional_refs, definitions, strict=True)}
        )
        return json.dumps(
            {
                "$defs": definitions,
                "type": "object",
                "properties": properties,
                "required": ["r"],
                "additionalProperties": False,
            },
            separators=(",", ":"),
        )

    b = _tool("b", tool_schema({"a": object_target(4), "b": object_target(4)}, ("ob1", "ob2")))
    c = _tool(
        "c",
        tool_schema({"c": {"type": "integer", "minimum": 1, "maximum": 3}}, ("oc",)),
    )
    root_b = parse_json_strict(b.parameters.canonical_json)
    root_c = parse_json_strict(c.parameters.canonical_json)
    assert isinstance(root_b, dict)
    assert isinstance(root_c, dict)
    source_nodes = compiler._json_node_count(root_b) + compiler._json_node_count(root_c)
    prop_b = root_b["properties"]["ob1"]  # type: ignore[index]
    prop_c = root_c["properties"]["oc"]  # type: ignore[index]
    assert isinstance(prop_b, dict)
    assert isinstance(prop_c, dict)
    detached_b_nodes = compiler._json_node_count(qwen_property_schema(root_b, prop_b))
    detached_c_nodes = compiler._json_node_count(qwen_property_schema(root_c, prop_c))
    monkeypatch.setattr(
        compiler,
        "_HARD_MAX_SCHEMA_NODES_TOTAL",
        source_nodes + max(detached_b_nodes, detached_c_nodes),
    )

    for tools in ((b, c), (c, b)):
        bundle = compile_qwen_tool_wire(_policy(*tools), ToolConstraintMode.SCHEMA)
        assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        assert not bundle.plan.tool("b").order_plan.optional_names
        assert bundle.plan.tool("c").order_plan.optional_names == frozenset({"oc"})


@pytest.mark.parametrize("definitions_key", ("$defs", "definitions"))
def test_a2b_local_ref_comment_annotation_keeps_strict_schema(
    definitions_key: str,
) -> None:
    schema = json.dumps(
        {
            definitions_key: {
                "item": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                }
            },
            "type": "object",
            "properties": {
                "x": {
                    "$ref": f"#/{definitions_key}/item",
                    "$comment": "annotation only",
                }
            },
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )

    bundle = compile_qwen_tool_wire(
        _policy(_tool("t", schema, strict=True)),
        ToolConstraintMode.SCHEMA,
    )

    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.constraint is not None
    assert bundle.constraint.guarantee_for_tool("t") is GenerationGuarantee.SCHEMA


def test_a2b_shared_ref_dag_falls_back_without_detached_exponential_expansion() -> None:
    levels = 14
    definitions: dict[str, object] = {f"d{levels}": {"type": "integer"}}
    for index in range(levels - 1, -1, -1):
        ref = {"$ref": f"#/$defs/d{index + 1}"}
        definitions[f"d{index}"] = {"anyOf": [ref, dict(ref)]}
    schema = json.dumps(
        {
            "$defs": definitions,
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/d0"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )

    bundle = compile_qwen_tool_wire(_policy(_tool("t", schema)), ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert bundle.constraint is None


def test_a2b_mixed_schema_downgrades_loose_branch_to_format_without_deleting_it() -> None:
    strict = _tool("strict", _integer_schema(), strict=True)
    loose = _tool(
        "loose",
        '{"type":"object","properties":{"s":{"type":"string","pattern":"^x+$"}},'
        '"required":["s"],"additionalProperties":false}',
    )

    bundle = compile_qwen_tool_wire(_policy(strict, loose, parallel=True), ToolConstraintMode.SCHEMA)

    assert bundle.constraint is not None
    assert bundle.constraint.guarantee_for_tool("strict") is GenerationGuarantee.SCHEMA
    assert bundle.constraint.guarantee_for_tool("loose") is GenerationGuarantee.FORMAT
    assert '"<function=strict>"' in bundle.constraint.lark_grammar
    assert '"<function=loose>"' in bundle.constraint.lark_grammar
    assert bundle.branch_generation_narrowed
    assert bundle.request_policy_narrowed


def test_a2b_mixed_schema_rejects_loose_branch_that_cannot_even_format_frame() -> None:
    strict = _tool("strict", _integer_schema(), strict=True)
    loose = _tool("bad name", _integer_schema())

    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(_policy(strict, loose, parallel=True), ToolConstraintMode.SCHEMA)


@pytest.mark.parametrize(
    "property_schema",
    (
        {"type": "integer", "minimum": 5},
        {"type": "integer", "maximum": -5},
        {"type": "integer", "minimum": 5, "maximum": 10},
        {"type": "number", "minimum": 5},
    ),
)
def test_a2b_schema_numeric_bounds_keep_production_schema_capability(
    property_schema: dict[str, object],
) -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": property_schema},
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )

    loose = compile_qwen_tool_wire(_policy(_tool("t", schema)), ToolConstraintMode.SCHEMA)
    strict = compile_qwen_tool_wire(_policy(_tool("t", schema, strict=True)), ToolConstraintMode.SCHEMA)

    assert loose.constraint is not None
    assert strict.constraint is not None
    assert loose.constraint.guarantee_for_tool("t") is GenerationGuarantee.SCHEMA
    assert strict.constraint.guarantee_for_tool("t") is GenerationGuarantee.SCHEMA


@pytest.mark.parametrize("schema_type", ("integer", "number"))
def test_a2b_huge_numeric_bound_does_not_escape_as_numeric_overflow(schema_type: str) -> None:
    huge_minimum = 10**400
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"x": {"type": schema_type, "minimum": huge_minimum}},
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )

    bundle = compile_qwen_tool_wire(_policy(_tool("t", schema)), ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.constraint is not None
    assert bundle.constraint.guarantee_for_tool("t") is GenerationGuarantee.FORMAT
    assert bundle.branch_generation_narrowed


@pytest.mark.parametrize(
    "schema",
    (
        '{"type":"array","items":{"type":"integer"}}',
        '{"type":"object","properties":{},"required":["missing"]}',
        '{"$defs":{"obj":{"type":"object","properties":{}}},"$ref":"#/$defs/obj"}',
    ),
)
def test_a2b_expected_unsupported_shapes_use_stable_fallback_surface(schema: str) -> None:
    tool = _tool("t", schema)

    format_bundle = compile_qwen_tool_wire(_policy(tool), ToolConstraintMode.FORMAT)
    assert format_bundle.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert format_bundle.constraint is None
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(_policy(tool), ToolConstraintMode.SCHEMA)


def _published_parallel_calls(
    request_policy: ToolPolicy,
    chunks: tuple[str, ...],
) -> tuple[list[object], bool]:
    parser = _validation_parser("a2b-parallel-publication", request_policy)
    gate = ToolCallBatchGate(
        request_policy,
        tool_call_fanout_limit=32,
        atomic_parallel_tools=request_policy.allow_parallel,
        constrained_parallel_tool_call_limit=8,
    )
    immediate = []
    for chunk in chunks:
        for event in parser.feed_with_native_tokens(chunk, _native_spans(chunk)):
            if isinstance(event, ToolCallStarted):
                decision = gate.on_started(event)
            elif isinstance(event, ToolCallArgumentsDelta):
                decision = gate.on_arguments_delta(event)
            elif isinstance(event, ToolCallCompleted):
                decision = gate.on_completed(event)
            else:
                continue
            assert decision.failure is None
            immediate.extend(decision.events)
    finished = parser.finish()
    for event in finished.events:
        if isinstance(event, ToolCallStarted):
            decision = gate.on_started(event)
        elif isinstance(event, ToolCallArgumentsDelta):
            decision = gate.on_arguments_delta(event)
        elif isinstance(event, ToolCallCompleted):
            decision = gate.on_completed(event)
        else:
            continue
        assert decision.failure is None
        immediate.extend(decision.events)
    if finished.incomplete_tool_call:
        gate.abort()
        committed = ()
    else:
        committed = gate.commit_events()
    calls = [
        event.call
        for event in (*immediate, *committed)
        if isinstance(event, ToolCallCompleted)
    ]
    return calls, finished.incomplete_tool_call


def test_a2b_parallel_tool_batch_is_atomic_and_allows_repeated_calls() -> None:
    first = _tool("a", _integer_schema())
    second = _tool("b", _integer_schema())
    request_policy = _policy(first, second, parallel=True)
    first_wire = "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
    second_wire = "<tool_call><function=b><parameter=x>2</parameter></function></tool_call>"

    adjacent, incomplete = _published_parallel_calls(
        request_policy,
        (first_wire, second_wire),
    )
    assert not incomplete
    assert [(call.name, call.arguments_json) for call in adjacent] == [
        ("a", '{"x":1}'),
        ("b", '{"x":2}'),
    ]

    repeated, incomplete = _published_parallel_calls(
        _policy(first, parallel=True),
        (first_wire, first_wire),
    )
    assert not incomplete
    assert [(call.name, call.arguments_json) for call in repeated] == [
        ("a", '{"x":1}'),
        ("a", '{"x":1}'),
    ]

    truncated, incomplete = _published_parallel_calls(
        request_policy,
        (first_wire, "<tool_call><function=b><parameter=x>"),
    )
    assert incomplete
    assert truncated == []


def _production_wire(
    schema: str,
    semantic: dict[str, object],
    *,
    order: tuple[str, ...] | None = None,
) -> tuple[ToolPolicy, object, str, WireToolSequence]:
    fn = _tool("t", schema)
    request_policy = _policy(fn)
    bundle = compile_qwen_tool_wire(request_policy, ToolConstraintMode.SCHEMA)
    assert bundle.constraint is not None
    branch = bundle.plan.tool("t")
    selected_order = order or tuple(name for name in branch.order_plan.orders[0] if name in semantic)
    assert branch.order_plan.accepts(selected_order)
    argument_by_name = {argument.name: argument for argument in branch.arguments}
    occurrences: list[WireArgumentOccurrence] = []
    parts = ["<tool_call><function=t>"]
    for name in selected_order:
        value = semantic[name]
        argument = argument_by_name[name]
        assert argument.framing_variant_id is not None
        variant = bundle.spec.framing_variant(argument.framing_variant_id)
        canonical = canonical_json_dumps(value)
        if variant.value_framing.kind is ValueFramingKind.RAW_UNTIL:
            assert isinstance(value, str)
            payload = encode_lossless_raw_string(value, variant.value_framing)
            assert payload is not None
        else:
            payload = canonical
        occurrences.append(
            WireArgumentOccurrence(name, canonical, argument.framing_variant_id)
        )
        parts.append(variant.argument_open.render(name))
        parts.append(payload)
        parts.append(variant.argument_close.canonical.text)
    parts.append("</function></tool_call>")
    wire = "".join(parts)
    sequence = WireToolSequence((WireToolCall("t", 0, tuple(occurrences)),))
    assert admit_tool_sequence(bundle.spec, bundle.plan, sequence).is_valid
    return request_policy, bundle, wire, sequence


@pytest.mark.parametrize(
    ("schema", "semantic", "order"),
    (
        (
            '{"type":"object","properties":{"s":{"type":"string"}},"required":["s"],"additionalProperties":false}',
            {"s": "foo"},
            None,
        ),
        (
            '{"type":"object","properties":{"s":{"type":"string","enum":["foo"," leading","中文🙂"]}},"required":["s"],"additionalProperties":false}',
            {"s": "中文🙂"},
            None,
        ),
        (
            '{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}',
            {"x": -7},
            None,
        ),
        (
            '{"type":"object","properties":{"x":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}},"required":["x"],"additionalProperties":false}',
            {"x": {"a": 3}},
            None,
        ),
        (
            '{"type":"object","properties":{"x":{"type":"array","items":{"type":"integer"}}},"required":["x"],"additionalProperties":false}',
            {"x": [1, 2, 3]},
            None,
        ),
        (
            '{"type":"object","properties":{"r":{"type":"integer"},"o":{"type":"string"}},"required":["r"],"additionalProperties":false}',
            {"r": 2},
            None,
        ),
        (
            '{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},"required":["a","b"],"additionalProperties":false}',
            {"a": 1, "b": False},
            ("a", "b"),
        ),
        (
            '{"type":"object","properties":{"s":{"type":"string"}},"required":["s"],"additionalProperties":false}',
            {"s": "prefix </parameter></function></tool_call> suffix"},
            None,
        ),
        (
            '{"type":"object","properties":{"s":{"type":"string"}},"required":["s"],"additionalProperties":false}',
            {"s": "中文🙂𐐷"},
            None,
        ),
    ),
)
def test_a2b_tool_wire_wires_match_production_parser_across_chunk_splits(
    schema: str,
    semantic: dict[str, object],
    order: tuple[str, ...] | None,
) -> None:
    request_policy, bundle, wire, _ = _production_wire(schema, semantic, order=order)
    assert bundle.constraint is not None
    assert bundle.grammar_fingerprint is not None

    def parse(chunks: tuple[str, ...]) -> tuple[list[object], object]:
        context = resolve_qwen_parser_context(
            bundle.constraint,
            ConstraintInstallation(
                True,
                bundle.grammar_fingerprint,
                (248058,),
                GenerationGuarantee.SCHEMA,
            ),
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
        parser = QwenIncrementalParser(
            "a2b-production-matrix",
            tool_policy=request_policy,
            parser_context=context,
        )
        events = []
        for chunk in chunks:
            events.extend(parser.feed_with_native_tokens(chunk, _native_spans(chunk)))
        finished = parser.finish()
        events.extend(finished.events)
        calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
        return calls, finished

    expected = semantic
    whole_calls, whole_finished = parse((wire,))
    assert not whole_finished.incomplete_tool_call
    assert whole_finished.terminal_issue is None
    assert len(whole_calls) == 1
    assert parse_json_strict(whole_calls[0].arguments_json) == expected

    structural_points = sorted(
        {
            0,
            len(wire),
            wire.find("<function="),
            wire.find("<parameter="),
            wire.rfind("</function>"),
            wire.rfind("</tool_call>"),
        }
    )
    structural_points = [point for point in structural_points if 0 <= point <= len(wire)]
    structural_chunks = tuple(
        wire[start:end]
        for start, end in pairwise(structural_points)
        if start != end
    )
    structural_calls, structural_finished = parse(structural_chunks)
    assert not structural_finished.incomplete_tool_call
    assert structural_finished.terminal_issue is None
    assert len(structural_calls) == 1
    assert parse_json_strict(structural_calls[0].arguments_json) == expected

    for point in {
        max(len("<tool_call>"), len(wire) // 3),
        max(len("<tool_call>"), len(wire) // 2),
        max(len("<tool_call>"), 2 * len(wire) // 3),
        len(wire) - 1,
    }:
        split_calls, split_finished = parse((wire[:point], wire[point:]))
        assert not split_finished.incomplete_tool_call
        assert split_finished.terminal_issue is None
        assert len(split_calls) == 1
        assert parse_json_strict(split_calls[0].arguments_json) == expected


def test_a2b_production_finite_raw_values_use_direct_native_spelling_only() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "s": {
                    "type": "string",
                    "enum": ["foo", " leading", "中文🙂", '"quoted"'],
                }
            },
            "required": ["s"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    bundle = compile_qwen_tool_wire(
        _policy(_tool("write", schema, strict=True)),
        ToolConstraintMode.SCHEMA,
    )
    argument = bundle.plan.tool("write").arguments[0]

    assert bundle.constraint is not None
    assert argument.admitted_wire_payloads == ("foo", "中文🙂")


def test_a2b_production_raw_codec_preserves_legacy_canonical_string_semantics() -> None:
    schema = (
        '{"type":"object","properties":{"s":{"type":"string"}},'
        '"required":["s"],"additionalProperties":false}'
    )
    bundle = compile_qwen_tool_wire(
        _policy(_tool("write", schema, strict=True)),
        ToolConstraintMode.SCHEMA,
    )
    raw = bundle.spec.framing_variant("qwen-raw-string").value_framing

    assert raw.codec is ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT
    assert raw.codec.decode_raw_payload("\nfoo\n") == "foo"
    assert raw.codec.decode_raw_payload('"foo"') == "foo"
    assert raw.codec.decode_raw_payload('"a\\nb"') == "a\nb"
    assert raw.codec.decode_raw_payload("true") == "true"
    assert raw.codec.decode_raw_payload("123") == "123"
    assert raw.codec.decode_raw_payload('{"a":1}') == '{"a":1}'
    with pytest.raises(ValueError, match="non-UTF8 surrogate scalar"):
        raw.codec.decode_raw_payload('"\\ud800"')
    with pytest.raises(ValueError, match="non-UTF8 surrogate scalar"):
        raw.codec.decode_raw_payload("\ud800")


def test_a2b_production_strict_raw_const_rejects_reserved_native_close() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"s": {"type": "string", "const": "x\n</parameter>\ny"}},
            "required": ["s"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    policy = _policy(_tool("write", schema, strict=True))
    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is PlanCompileDisposition.REJECTED
    assert bundle.constraint is None
    with pytest.raises(ToolConstraintUnsupported):
        qwen_tool_constraint(policy, ToolConstraintMode.SCHEMA)


def test_a2b_raw_close_safe_wire_matches_shared_admission_and_production_parser() -> None:
    schema = (
        '{"type":"object","properties":{"s":{"type":"string"}},'
        '"required":["s"],"additionalProperties":false}'
    )
    fn = _tool("write", schema, strict=True)
    request_policy = _policy(fn)
    bundle = compile_qwen_tool_wire(request_policy, ToolConstraintMode.SCHEMA)
    assert bundle.constraint is not None
    branch = bundle.plan.tool("write")
    argument = branch.arguments[0]
    assert argument.framing_variant_id is not None
    variant = bundle.spec.framing_variant(argument.framing_variant_id)
    assert variant.value_framing.kind is ValueFramingKind.RAW_UNTIL

    semantic = "prefix </parameter></function></tool_call> suffix"
    payload = encode_lossless_raw_string(semantic, variant.value_framing)
    assert payload is not None
    assert "\n</parameter>\n" not in payload
    canonical = canonical_json_dumps(semantic)
    wire = (
        "<tool_call><function=write>"
        + variant.argument_open.render("s")
        + payload
        + variant.argument_close.canonical.text
        + "</function></tool_call>"
    )
    assert _constraint_accepts(bundle.constraint, wire[len("<tool_call>") :])
    sequence = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (WireArgumentOccurrence("s", canonical, argument.framing_variant_id),),
            ),
        )
    )
    assert admit_tool_sequence(bundle.spec, bundle.plan, sequence).is_valid

    assert bundle.grammar_fingerprint is not None
    context = resolve_qwen_parser_context(
        bundle.constraint,
        ConstraintInstallation(
            True,
            bundle.grammar_fingerprint,
            (248058,),
            GenerationGuarantee.SCHEMA,
        ),
        ConstraintFallbackPolicy.FAIL_CLOSED,
    )
    parser = QwenIncrementalParser(
        "a2b-production-parity",
        tool_policy=request_policy,
        parser_context=context,
    )
    events = list(parser.feed_with_native_tokens(wire, _native_spans(wire)))
    finished = parser.finish()
    events.extend(finished.events)
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert not finished.incomplete_tool_call
    assert finished.terminal_issue is None
    assert len(calls) == 1
    assert calls[0].name == "write"
    assert parse_json_strict(calls[0].arguments_json) == {"s": semantic}


def _ref_string_chain_schema(definitions_key: str, terminal_index: int) -> str:
    definitions: dict[str, object] = {
        f"d{index}": {"$ref": f"#/{definitions_key}/d{index + 1}"}
        for index in range(terminal_index)
    }
    definitions[f"d{terminal_index}"] = {"type": "string"}
    return json.dumps(
        {
            definitions_key: definitions,
            "type": "object",
            "properties": {"x": {"$ref": f"#/{definitions_key}/d0"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )


def _parse_ref_string_wire(policy: ToolPolicy, raw_value: str) -> object:
    wire = (
        "<tool_call><function=f><parameter=x>"
        + raw_value
        + "</parameter></function></tool_call>"
    )
    parser = _validation_parser("a2b-ref-boundary", policy)
    events = list(parser.feed_with_native_tokens(wire, _native_spans(wire)))
    finished = parser.finish()
    events.extend(finished.events)
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert not finished.incomplete_tool_call
    assert finished.terminal_issue is None
    assert len(calls) == 1
    return parse_json_strict(calls[0].arguments_json)


@pytest.mark.parametrize("definitions_key", ("$defs", "definitions"))
@pytest.mark.parametrize(
    ("terminal_index", "expected_disposition"),
    (
        (63, PlanCompileDisposition.CONSTRAINED_EXECUTABLE),
        (64, PlanCompileDisposition.CONSTRAINED_EXECUTABLE),
        (65, PlanCompileDisposition.REJECTED),
    ),
)
def test_a2b_ref_string_parser_matches_generation_at_ref_expansion_boundary(
    definitions_key: str,
    terminal_index: int,
    expected_disposition: PlanCompileDisposition,
) -> None:
    tool = _tool("f", _ref_string_chain_schema(definitions_key, terminal_index), strict=True)
    policy = _policy(tool)
    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)

    assert bundle.plan.disposition is expected_disposition
    if expected_disposition is PlanCompileDisposition.REJECTED:
        assert bundle.constraint is None
        return

    assert bundle.constraint is not None
    assert bundle.constraint.guarantee_for_tool("f") is GenerationGuarantee.SCHEMA
    parsed = _parse_ref_string_wire(policy, "123")
    assert parsed == {"x": "123"}


@pytest.mark.parametrize("definitions_key", ("$defs", "definitions"))
def test_a2b_ref64_parser_preserves_json_looking_raw_string_semantics(
    definitions_key: str,
) -> None:
    tool = _tool("f", _ref_string_chain_schema(definitions_key, 64), strict=True)
    policy = _policy(tool)
    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert bundle.constraint is not None

    for raw_value in ("true", "123", "null", '{"a":1}', "[1,2]"):
        parsed = _parse_ref_string_wire(policy, raw_value)
        assert parsed == {"x": raw_value}


@pytest.mark.parametrize("definitions_key", ("$defs", "definitions"))
def test_a2b_ref64_parser_is_invariant_across_all_two_chunk_splits(
    definitions_key: str,
) -> None:
    tool = _tool("f", _ref_string_chain_schema(definitions_key, 64), strict=True)
    policy = _policy(tool)
    bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
    assert bundle.plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    wire = "<tool_call><function=f><parameter=x>123</parameter></function></tool_call>"

    for split in range(len(wire) + 1):
        parser = _validation_parser(f"a2b-ref64-split-{split}", policy)
        events = list(parser.feed(wire[:split]))
        events.extend(parser.feed(wire[split:]))
        finished = parser.finish()
        events.extend(finished.events)
        calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
        assert not finished.incomplete_tool_call, split
        assert finished.terminal_issue is None, split
        assert len(calls) == 1, split
        assert parse_json_strict(calls[0].arguments_json) == {"x": "123"}, split


def test_a2b_native_suffix_lowering_constructs_raw_variants_in_llguidance() -> None:
    llguidance = pytest.importorskip("llguidance")
    vocabulary = llguidance.LLTokenizer(llguidance.TokenizerWrapper(_ByteTokenizer()))
    cases = (
        (
            "one_raw",
            {"a_content": {"type": "string"}},
            ["a_content"],
            "<function=f><parameter=a_content>\nhello\n</parameter>\n</function></tool_call>",
            1,
        ),
        (
            "raw_plus_finite",
            {
                "a_content": {"type": "string"},
                "b_path": {"type": "string", "const": "note.txt"},
            },
            ["a_content", "b_path"],
            (
                "<function=f><parameter=a_content>\nhello\n</parameter>\n"
                "<parameter=b_path>\nnote.txt\n</parameter>\n</function></tool_call>"
            ),
            1,
        ),
        (
            "two_raw",
            {"a_path": {"type": "string"}, "b_content": {"type": "string"}},
            ["a_path", "b_content"],
            (
                "<function=f><parameter=a_path>\nnote.txt\n</parameter>\n"
                "<parameter=b_content>\nhello\n</parameter>\n</function></tool_call>"
            ),
            2,
        ),
    )

    for label, properties, required, suffix, expected_suffix_rules in cases:
        schema = json.dumps(
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            separators=(",", ":"),
        )
        bundle = compile_qwen_tool_wire(
            _policy(_tool("f", schema, strict=True)),
            ToolConstraintMode.SCHEMA,
        )
        assert bundle.constraint is not None, label
        grammar = bundle.constraint.lark_grammar
        assert grammar.count('suffix="\\n</parameter>\\n"') == expected_suffix_rules, label
        assert "_raw_0:" not in grammar, label
        assert len(grammar.encode("utf-8")) < 2_000, label
        parsed = llguidance.LLMatcher.grammar_from_lark(grammar)
        matcher = llguidance.LLMatcher(vocabulary, parsed)
        assert matcher.consume_tokens(list(suffix.encode("utf-8"))), label
        assert matcher.is_accepting(), label
        assert not matcher.is_error(), label


def test_a2b_native_suffix_lowering_preserves_first_close_semantics() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
    bundle = compile_qwen_tool_wire(
        _policy(_tool("write", schema, strict=True)),
        ToolConstraintMode.SCHEMA,
    )
    assert bundle.constraint is not None
    assert _constraint_accepts(
        bundle.constraint,
        "<function=write><parameter=content>\nx</parameterX>y\n</parameter>\n</function></tool_call>",
    )
    assert not _constraint_accepts(
        bundle.constraint,
        (
            "<function=write><parameter=content>\nx\n</parameter>\ny"
            "\n</parameter>\n</function></tool_call>"
        ),
    )


def test_a2b_backend_lowering_failure_follows_strict_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_lowering(*_args, **_kwargs):
        raise ToolWireConstraintLoweringUnsupported("unsupported backend lowering")

    monkeypatch.setattr(qwen_control, "build_lark_tool_constraint_candidate", unsupported_lowering)
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )

    loose = compile_qwen_tool_wire(
        _policy(_tool("write", schema)),
        ToolConstraintMode.SCHEMA,
    )
    assert loose.plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert loose.constraint is None

    with pytest.raises(ToolConstraintUnsupported):
        compile_qwen_tool_wire(
            _policy(_tool("write", schema, strict=True)),
            ToolConstraintMode.SCHEMA,
        )
