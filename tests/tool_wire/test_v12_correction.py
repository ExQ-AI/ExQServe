from __future__ import annotations

import json
from dataclasses import replace

import pytest

from exqserve.core.errors import SemanticCommitClass
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire import (
    ActivationTriggerSpec,
    ArgumentFramingSelector,
    ArgumentFramingSelectorRule,
    ArgumentFramingVariant,
    ArgumentOccurrenceCapabilities,
    ArgumentOrderingMode,
    CloseLanguage,
    CompileBudget,
    ConstraintActivationEvidence,
    ConstraintActivationProof,
    ConstraintCompilerCapabilities,
    IrreversiblePublication,
    LiteralTerminal,
    NameCodec,
    NamedTerminal,
    NonEmptinessStatus,
    PlanCompileDisposition,
    PromptArgumentWireObservation,
    PromptWireObservation,
    SchemaSemanticAuthority,
    SemanticTransducerCase,
    ServingToolPolicySnapshot,
    ToolMultiplicity,
    ToolWireExecutionMode,
    ToolWireFailureClass,
    ToolWireFailureEvidence,
    ToolWireSemanticFailure,
    ToolWireSpec,
    ToolWireStopCause,
    ValueCodecKind,
    ValueFraming,
    ValueFramingKind,
    WireArgumentOccurrence,
    WireChannel,
    WireToolCall,
    WireToolSequence,
    certify_constraint_activation,
    certify_prompt_template_parity,
    certify_semantic_transducer,
    certify_tool_sequence_publication,
    classify_tool_wire_failure,
    compile_tool_wire_plan,
)
from tests.tool_wire._support import policy, schema_plan, structured_spec, tool


def _mixed_capabilities() -> ConstraintCompilerCapabilities:
    return ConstraintCompilerCapabilities(
        "mixed-a0-v12",
        ("type", "properties", "required", "additionalProperties"),
        ("type", "enum", "const", "minimum", "maximum", "properties", "required", "items"),
        SchemaSemanticAuthority.DRAFT_2020_12,
    )


def _mixed_spec(*, deepseek_attributes: bool = False) -> ToolWireSpec:
    close = CloseLanguage((LiteralTerminal("</parameter>", (500,)),))
    name_terminator = " " if deepseek_attributes else ">"
    raw_remainder = 'string="true">' if deepseek_attributes else ""
    json_remainder = 'string="false">' if deepseek_attributes else ""
    raw = ArgumentFramingVariant(
        "raw-string",
        NamedTerminal("<parameter=", name_terminator, raw_remainder),
        close,
        ValueFraming(ValueFramingKind.RAW_UNTIL, ValueCodecKind.RAW_STRING, close),
    )
    structured = ArgumentFramingVariant(
        "json-value",
        NamedTerminal("<parameter=", name_terminator, json_remainder),
        close,
        ValueFraming(ValueFramingKind.STRUCTURED_ESCAPED, ValueCodecKind.JSON),
    )
    tool_open = LiteralTerminal("<tool_call>", (248058,))
    return ToolWireSpec(
        spec_id="mixed-deepseek-v12" if deepseek_attributes else "mixed-qwen-v12",
        tool_open=tool_open,
        tool_close=CloseLanguage((LiteralTerminal("</tool_call>", (248059,)),)),
        function_open=NamedTerminal("<function=", ">"),
        function_close=CloseLanguage((LiteralTerminal("</function>"),)),
        function_name_codec=NameCodec("identity-function", (">",)),
        argument_name_codec=NameCodec("identity-argument", (" ", "=", '"', ">")),
        argument_framings=(raw, structured),
        framing_selector=ArgumentFramingSelector(
            "schema-type-v12",
            (
                ArgumentFramingSelectorRule("raw-string", ("string",)),
                ArgumentFramingSelectorRule(
                    "json-value", ("integer", "number", "boolean", "object", "array", "null")
                ),
            ),
        ),
        occurrence=ArgumentOccurrenceCapabilities(1, False, True),
        ordering=ArgumentOrderingMode.DECLARATION_ORDER,
        multiplicity=ToolMultiplicity(None, True),
        tool_entry_channels=(WireChannel.TEXT, WireChannel.REASONING),
        tool_exit_channel=WireChannel.TEXT,
        activation_triggers=(ActivationTriggerSpec("tool-open", tool_open),),
    )


def _mixed_plan(spec: ToolWireSpec, fn, order: tuple[str, ...] | None = None):
    if order is None:
        order = tuple(json.loads(fn.parameters.canonical_json)["properties"])
    return schema_plan(
        spec,
        policy(fn),
        {fn.name: order},
        compiler_capabilities=_mixed_capabilities(),
    )


def test_qwen_mixed_string_and_integer_framing_is_one_constrained_plan() -> None:
    fn = tool(
        "read",
        '{"type":"object","properties":{"file_path":{"type":"string"},'
        '"offset":{"type":"integer","minimum":0}},'
        '"required":["file_path","offset"],"additionalProperties":false}',
        strict=True,
    )
    spec = _mixed_spec()
    plan = _mixed_plan(spec, fn, ("file_path", "offset"))
    branch = plan.tool("read")

    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert branch.guarantee is GenerationGuarantee.SCHEMA
    assert [argument.framing_variant_id for argument in branch.arguments] == [
        "raw-string",
        "json-value",
    ]
    assert all(argument.generated for argument in branch.arguments)


def test_deepseek_per_argument_string_attribute_and_prompt_parity_are_resolved_from_plan() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"},'
        '"offset":{"type":"integer","minimum":0}},"required":["content","offset"],'
        '"additionalProperties":false}',
        strict=True,
    )
    spec = _mixed_spec(deepseek_attributes=True)
    plan = _mixed_plan(spec, fn, ("content", "offset"))
    content, offset = plan.tool("write").arguments
    assert spec.framing_variant(content.framing_variant_id or "").argument_open.suffix == ' string="true">'
    assert spec.framing_variant(offset.framing_variant_id or "").argument_open.suffix == ' string="false">'

    observation = PromptWireObservation(
        source_id="deepseek-template",
        tool_open="<tool_call>",
        tool_close="</tool_call>",
        function_open_prefix="<function=",
        function_open_suffix=">",
        function_close="</function>",
        encoded_tool_names=(("write", "write"),),
        arguments=(
            PromptArgumentWireObservation(
                "write", "content", "raw-string", "<parameter=", ' string="true">',
                "</parameter>", "content"
            ),
            PromptArgumentWireObservation(
                "write", "offset", "json-value", "<parameter=", ' string="false">',
                "</parameter>", "offset"
            ),
        ),
    )
    assert certify_prompt_template_parity(spec, observation, plan).is_valid

    wrong = replace(
        observation,
        arguments=(
            observation.arguments[0],
            replace(observation.arguments[1], argument_open_suffix=' string="true">'),
        ),
    )
    assert {issue.code for issue in certify_prompt_template_parity(spec, wrong, plan).issues} == {
        "argument_open_mismatch"
    }

    incomplete = replace(observation, arguments=(observation.arguments[0],))
    incomplete_codes = {
        issue.code for issue in certify_prompt_template_parity(spec, incomplete, plan).issues
    }
    assert "argument_framing_observation_missing" in incomplete_codes


def test_structural_name_terminators_make_tool_or_argument_non_authoritative() -> None:
    bad_tool = tool(
        "bad>name",
        '{"type":"object","properties":{"value":{"type":"string"}},'
        '"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = _mixed_spec()
    assert spec.function_name_codec.is_losslessly_representable("valid_tool")
    assert spec.function_name_codec.decode(spec.function_name_codec.encode("valid_tool")) == "valid_tool"
    assert spec.argument_name_codec.is_losslessly_representable("valid_arg")
    assert spec.argument_name_codec.decode(spec.argument_name_codec.encode("valid_arg")) == "valid_arg"
    tool_plan = _mixed_plan(spec, bad_tool)
    assert tool_plan.tool("bad>name").name_representable is False
    assert tool_plan.disposition is PlanCompileDisposition.REJECTED
    assert tool_plan.activation is None

    bad_arg = tool(
        "ok",
        '{"type":"object","properties":{"bad>arg":{"type":"string"}},'
        '"required":["bad>arg"],"additionalProperties":false}',
        strict=True,
    )
    arg_plan = _mixed_plan(spec, bad_arg)
    assert arg_plan.tool("ok").arguments[0].name_representable is False
    assert arg_plan.disposition is PlanCompileDisposition.REJECTED
    assert arg_plan.activation is None


def test_unsatisfiable_supported_integer_schema_is_proven_empty_not_schema_safe_nonempty() -> None:
    fn = tool(
        "measure",
        '{"type":"object","properties":{"count":{"type":"integer","minimum":2,"maximum":1}},'
        '"required":["count"],"additionalProperties":false}',
        strict=True,
    )
    plan = schema_plan(structured_spec(), policy(fn), {"measure": ("count",)})
    argument = plan.tool("measure").arguments[0]

    assert argument.proof.non_empty is not NonEmptinessStatus.PROVEN_NON_EMPTY
    assert argument.generated is False
    assert plan.disposition is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert plan.activation is None


def test_compiled_finite_raw_language_is_publication_authority_before_toolpolicy() -> None:
    fn = tool(
        "choose",
        '{"type":"object","properties":{"value":{"type":"string","enum":['
        '"safe","bad</parameter>"]}},"required":["value"],"additionalProperties":false}',
    )
    from tests.tool_wire._support import raw_spec

    spec = raw_spec()
    tool_policy = policy(fn)
    plan = schema_plan(spec, tool_policy, {"choose": ("value",)})
    sequence = WireToolSequence(
        (WireToolCall("choose", 0, (WireArgumentOccurrence("value", '"bad</parameter>"'),)),)
    )
    decision = certify_tool_sequence_publication(
        spec, plan, sequence, tool_policy, ServingToolPolicySnapshot(8, 8)
    )

    assert decision.published_events == ()
    publication_codes = {issue.code for issue in decision.structural_issues}
    assert publication_codes >= {
        "value_not_admitted_by_plan",
        "forbidden_close_in_value",
    }
    certification = certify_semantic_transducer(
        spec,
        plan,
        (SemanticTransducerCase("wire", sequence),),
        lambda _wire, _plan: sequence,
    )[0]
    assert {issue.code for issue in certification.result.issues} == publication_codes


def test_non_generated_optional_occurrence_is_rejected_before_object_collapse() -> None:
    fn = tool(
        "search",
        '{"type":"object","properties":{"query":{"type":"string"},'
        '"hint":{"type":"string","pattern":"^x"}},"required":["query"],'
        '"additionalProperties":false}',
    )
    from tests.tool_wire._support import raw_spec

    spec = replace(raw_spec(), ordering=ArgumentOrderingMode.DECLARATION_ORDER)
    tool_policy = policy(fn)
    plan = schema_plan(spec, tool_policy, {"search": ("query", "hint")})
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert plan.tool("search").arguments[1].generated is False
    sequence = WireToolSequence(
        (
            WireToolCall(
                "search",
                0,
                (
                    WireArgumentOccurrence("query", '"x"'),
                    WireArgumentOccurrence("hint", '"x"'),
                ),
            ),
        )
    )
    decision = certify_tool_sequence_publication(
        spec, plan, sequence, tool_policy, ServingToolPolicySnapshot(8, 8)
    )
    assert decision.published_events == ()
    assert "argument_not_generated_by_plan" in {issue.code for issue in decision.structural_issues}


def test_rejected_and_validation_only_plans_have_no_activation_or_constraint_integrity_authority() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
    )
    spec = _mixed_spec()
    tiny_budget = CompileBudget(1, 1, 1, 1)
    rejected = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=_mixed_capabilities(),
        presentation_orders={"write": ("content",)},
        budget=tiny_budget,
        parser_branch_id="tiny",
        constraint_fingerprint="constraint-v1",
        activation_trigger_ids=("tool-open",),
    )
    assert rejected.disposition is PlanCompileDisposition.REJECTED
    assert rejected.budget_result.within_budget is False
    assert rejected.activation is None and rejected.constraint_fingerprint is None

    proof, activation_result = certify_constraint_activation(
        rejected,
        ConstraintActivationEvidence(
            rejected.fingerprint,
            rejected.spec_fingerprint,
            "constraint-v1",
            "tiny",
            True,
            True,
            "tool-open",
        ),
    )
    assert proof is None
    assert "plan_not_constrained_executable" in {issue.code for issue in activation_result.issues}

    unsupported = tool(
        "search",
        '{"type":"object","properties":{"query":{"type":"string","pattern":"^x"}},'
        '"required":["query"],"additionalProperties":false}',
        strict=False,
    )
    validation_only = _mixed_plan(spec, unsupported)
    assert validation_only.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert validation_only.activation is None and validation_only.constraint_fingerprint is None

    with pytest.raises(TypeError, match="certifier-owned"):
        ConstraintActivationProof(
            validation_only.fingerprint,
            validation_only.spec_fingerprint,
            "fabricated",
            validation_only.parser_branch_id,
            "tool-open",
        )
    disposition = classify_tool_wire_failure(
        validation_only,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.MALFORMED,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
            observed_wire_outside_enforced_language=True,
        ),
    )
    assert disposition.failure_class is not ToolWireFailureClass.CONSTRAINT_INTEGRITY


def test_constrained_executable_plan_constructor_requires_activation_authority() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    plan = _mixed_plan(_mixed_spec(), fn, ("content",))
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE

    with pytest.raises(ValueError, match="constraint identity and activation"):
        replace(plan, activation=None)


def test_global_compile_work_budget_is_reserved_across_multiple_tools() -> None:
    properties = ",".join(f'"p{i}":{{"type":"string"}}' for i in range(6))
    required = ",".join(f'"p{i}"' for i in range(6))
    schema = (
        '{"type":"object","properties":{'
        + properties
        + '},"required":['
        + required
        + '],"additionalProperties":false}'
    )
    first = tool("first", schema)
    second = tool("second", schema)
    from tests.tool_wire._support import raw_spec

    spec = raw_spec(ordering=ArgumentOrderingMode.PERMUTABLE)
    order = tuple(f"p{i}" for i in range(6))
    plan = compile_tool_wire_plan(
        spec,
        policy(first, second),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=_mixed_capabilities(),
        presentation_orders={"first": order, "second": order},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=5000,
            max_estimated_bytes=10_000_000,
            max_work_units=800,
        ),
        parser_branch_id="global-budget",
        constraint_fingerprint="constraint-v1",
        activation_trigger_ids=("tool-open",),
    )

    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert plan.budget_result.within_budget is True
    assert plan.budget_result.work_units <= 800
    assert plan.budget_result.narrowed_permutations is True
    assert plan.tool("first").order_plan.narrowed is True
    assert plan.tool("first").order_plan.orders == (order,)
    assert plan.tool("second").order_plan.narrowed is True
    assert plan.tool("second").order_plan.orders == (order,)

    richer = compile_tool_wire_plan(
        spec,
        policy(first, second),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=_mixed_capabilities(),
        presentation_orders={"first": order, "second": order},
        budget=CompileBudget(
            max_permutations=1000,
            max_estimated_rules=5000,
            max_estimated_bytes=10_000_000,
            max_work_units=6000,
        ),
        parser_branch_id="global-budget-richer",
        constraint_fingerprint="constraint-v1",
        activation_trigger_ids=("tool-open",),
    )
    assert richer.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert richer.budget_result.within_budget is True
    assert richer.budget_result.work_units <= 6000
    assert richer.tool("first").order_plan.full_order_count == 720
    assert richer.tool("first").order_plan.narrowed is False
    assert richer.tool("second").order_plan.narrowed is True
    assert richer.tool("second").order_plan.orders == (order,)


def test_optional_order_expansion_cannot_starve_later_mandatory_order_work() -> None:
    from tests.tool_wire._support import raw_spec

    schema = (
        '{"type":"object","properties":{"r":{"type":"string"},'
        '"o1":{"type":"string"},"o2":{"type":"string"}},'
        '"required":["r"],"additionalProperties":false}'
    )
    functions = (tool("a", schema, strict=False), tool("b", schema, strict=False))
    orders = {"a": ("r", "o1", "o2"), "b": ("r", "o1", "o2")}
    spec = raw_spec(ordering=ArgumentOrderingMode.PERMUTABLE)
    request_policy = policy(*functions, allow_parallel=False)

    def compile_with(
        *,
        permutations: int = 20,
        rules: int = 10_000,
        byte_count: int = 100_000,
        work: int = 120,
    ):
        return compile_tool_wire_plan(
            spec,
            request_policy,
            ToolConstraintMode.SCHEMA,
            compiler_capabilities=_mixed_capabilities(),
            presentation_orders=orders,
            budget=CompileBudget(permutations, rules, byte_count, work),
            parser_branch_id="order-expansion-monotonicity",
            constraint_fingerprint="constraint-v1",
            activation_trigger_ids=("tool-open",),
        )

    seen_executable = False
    for work in range(80, 111):
        plan = compile_with(work=work)
        if seen_executable:
            assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        seen_executable = seen_executable or plan.constrained_executable
    assert seen_executable

    for low, high in (
        (compile_with(permutations=11), compile_with(permutations=12)),
        (compile_with(rules=28), compile_with(rules=29)),
        (compile_with(byte_count=350), compile_with(byte_count=375)),
    ):
        assert low.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        assert high.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        assert high.budget_result.narrowed_permutations is True


@pytest.mark.parametrize("tool_count", (1, 2, 3, 10))
def test_plan_level_permutation_budget_is_consumed_across_tools(tool_count: int) -> None:
    from tests.tool_wire._support import raw_spec

    schema = (
        '{"type":"object","properties":{"r":{"type":"string"},"o":{"type":"string"}},'
        '"required":["r"],"additionalProperties":false}'
    )
    functions = tuple(tool(f"f{index}", schema, strict=False) for index in range(tool_count))
    orders = {function.name: ("r", "o") for function in functions}
    spec = raw_spec(ordering=ArgumentOrderingMode.PERMUTABLE)
    full_permutations = 3 * tool_count

    exact = compile_tool_wire_plan(
        spec,
        policy(*functions, allow_parallel=False),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=_mixed_capabilities(),
        presentation_orders=orders,
        budget=CompileBudget(full_permutations, 1_000_000, 100_000_000, 1_000_000),
        parser_branch_id="aggregate-permutations-exact",
        constraint_fingerprint="constraint-v1",
        activation_trigger_ids=("tool-open",),
    )
    assert exact.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert exact.budget_result.within_budget is True
    assert exact.budget_result.permutations_reserved == full_permutations
    assert exact.budget_result.narrowed_permutations is False
    assert [len(branch.order_plan.orders) for branch in exact.tools] == [3] * tool_count

    under = compile_tool_wire_plan(
        spec,
        policy(*functions, allow_parallel=False),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=_mixed_capabilities(),
        presentation_orders=orders,
        budget=CompileBudget(full_permutations - 1, 1_000_000, 100_000_000, 1_000_000),
        parser_branch_id="aggregate-permutations-under",
        constraint_fingerprint="constraint-v1",
        activation_trigger_ids=("tool-open",),
    )
    assert under.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    assert under.budget_result.within_budget is True
    assert under.budget_result.permutations_reserved == sum(
        len(branch.order_plan.orders) for branch in under.tools
    )
    assert under.budget_result.permutations_reserved <= full_permutations - 1
    assert under.budget_result.narrowed_permutations is True
    assert any(branch.order_plan.narrowed for branch in under.tools)


def test_buffered_internal_commit_is_distinct_from_streaming_irreversible_publication() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
    )
    plan = _mixed_plan(_mixed_spec(), fn)
    buffered = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.AMBIGUOUS,
            SemanticCommitClass.PARTIAL_TOOL_COMMITTED,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
        ),
    )
    streaming = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.AMBIGUOUS,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.CONTENT,
            execution_mode=ToolWireExecutionMode.STREAMING,
        ),
    )
    validated_tool = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.AMBIGUOUS,
            SemanticCommitClass.TOOL_COMPLETED,
            irreversible_publication=IrreversiblePublication.VALIDATED_TOOL,
            execution_mode=ToolWireExecutionMode.STREAMING,
        ),
    )
    assert buffered.recovery_precondition_no_publication is True
    assert streaming.recovery_precondition_no_publication is False
    assert validated_tool.recovery_precondition_no_publication is False
