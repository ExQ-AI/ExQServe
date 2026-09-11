"""Qwen constrained-path Tool-Wire control for A2a shadow certification only.

The module intentionally does not import the production Qwen parser or runtime constraint
builder. It contributes static Qwen framing facts plus one bounded shadow compilation session
that binds the request plan to the exact emitted Lark grammar fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from exqserve.agent.tools import ToolPolicy
from exqserve.model.contracts import ToolConstraintMode, ToolGenerationConstraint
from exqserve.tool_wire.certification import (
    PromptArgumentWireObservation,
    PromptWireObservation,
)
from exqserve.tool_wire.compiler import _ConstraintArtifactCandidate, compile_tool_wire_plan
from exqserve.tool_wire.contracts import (
    ActivationTriggerSpec,
    ArgumentFramingSelector,
    ArgumentFramingSelectorRule,
    ArgumentFramingVariant,
    ArgumentOccurrenceCapabilities,
    ArgumentOrderingMode,
    CloseLanguage,
    CompileBudget,
    CompiledToolWirePlan,
    ConstraintCompilerCapabilities,
    LiteralTerminal,
    NameCodec,
    NamedTerminal,
    SchemaSemanticAuthority,
    ToolBranchPlan,
    ToolMultiplicity,
    ToolWireSpec,
    ValueCodecKind,
    ValueFraming,
    ValueFramingKind,
    WireChannel,
)
from exqserve.tool_wire.lark_constraint import (
    build_lark_tool_constraint_candidate,
    finalize_lark_tool_constraint_candidate,
)

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_FUNCTION_OPEN_PREFIX = "<function="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_OPEN_PREFIX = "<parameter="
_PARAMETER_CLOSE = "</parameter>"
_STRUCTURAL_WS_MAX = 8

_QWEN_NAME_WHITESPACE = (
    "\t",
    "\n",
    "\x0b",
    "\x0c",
    "\r",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    " ",
    "\x85",
    "\xa0",
    "\u1680",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
)
_QWEN_NAME_FORBIDDEN = ("<", ">") + _QWEN_NAME_WHITESPACE


@dataclass(frozen=True, slots=True)
class QwenA2AShadowCompilation:
    spec: ToolWireSpec
    plan: CompiledToolWirePlan
    constraint: ToolGenerationConstraint | None
    grammar_fingerprint: str | None
    parallel_generation_narrowed: bool

    @property
    def constrained(self) -> bool:
        return self.plan.constrained_executable and self.constraint is not None


def qwen_a2a_single_call_spec() -> ToolWireSpec:
    """Return the isolated A2a Qwen branch selected for prevention-first proof.

    A2a deliberately narrows generation to exactly one Tool envelope.  Production parallel
    multi-envelope migration remains an A2b concern, so one textual Tool-entry trigger covers
    every generated Tool path in this shadow slice.
    """

    parameter_close = CloseLanguage((LiteralTerminal(_PARAMETER_CLOSE),))
    common_parameter_open = NamedTerminal(_PARAMETER_OPEN_PREFIX, ">")
    raw_variant = ArgumentFramingVariant(
        "qwen-raw-string",
        common_parameter_open,
        parameter_close,
        ValueFraming(
            ValueFramingKind.RAW_UNTIL,
            ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
            parameter_close,
        ),
    )
    structured_variant = ArgumentFramingVariant(
        "qwen-json-structured",
        common_parameter_open,
        parameter_close,
        ValueFraming(ValueFramingKind.STRUCTURED_ESCAPED, ValueCodecKind.JSON),
    )
    # Existing Qwen runtime/tokenizer regression corpus identifies the model-native Tool
    # opener/closer tokens as 248058/248059.  A2a records that identity but does not change
    # runtime installation; A2b must re-verify it against the selected production tokenizer.
    tool_open = LiteralTerminal(_TOOL_OPEN, (248058,))
    name_codec = NameCodec("qwen-tag-name-identity", _QWEN_NAME_FORBIDDEN)
    return ToolWireSpec(
        spec_id="qwen-a2a-constrained-shadow-single-call-v1",
        tool_open=tool_open,
        tool_close=CloseLanguage((LiteralTerminal(_TOOL_CLOSE, (248059,)),)),
        function_open=NamedTerminal(_FUNCTION_OPEN_PREFIX, ">"),
        function_close=CloseLanguage((LiteralTerminal(_FUNCTION_CLOSE),)),
        function_name_codec=name_codec,
        argument_name_codec=name_codec,
        argument_framings=(raw_variant, structured_variant),
        framing_selector=ArgumentFramingSelector(
            "qwen-a2a-by-schema-type",
            (
                ArgumentFramingSelectorRule(raw_variant.variant_id, ("string",)),
                ArgumentFramingSelectorRule(
                    structured_variant.variant_id,
                    ("integer", "number", "boolean", "object", "array", "null"),
                ),
            ),
        ),
        occurrence=ArgumentOccurrenceCapabilities(1, False, True),
        ordering=ArgumentOrderingMode.PERMUTABLE,
        multiplicity=ToolMultiplicity(1, False, min_calls_per_sequence=1),
        tool_entry_channels=(WireChannel.TEXT, WireChannel.REASONING),
        tool_exit_channel=WireChannel.TEXT,
        activation_triggers=(ActivationTriggerSpec("tool-open", tool_open),),
    )


def qwen_a2a_compiler_capabilities() -> ConstraintCompilerCapabilities:
    """Capabilities actually emitted by the A2a Lark compiler.

    String keywords beyond ``type``/``enum``/``const`` intentionally remain unsupported in
    A2a, so framing safety alone cannot become a false SCHEMA guarantee.
    """

    return ConstraintCompilerCapabilities(
        "qwen-a2a-lark-v1",
        ("type", "properties", "required", "additionalProperties"),
        (
            "type",
            "enum",
            "const",
            "minimum",
            "maximum",
            "properties",
            "required",
            "additionalProperties",
            "items",
        ),
        SchemaSemanticAuthority.DRAFT_2020_12,
        decoder_safe_generation_schema=True,
    )


def qwen_a2a_default_budget() -> CompileBudget:
    return CompileBudget(
        max_permutations=1000,
        max_estimated_rules=100_000,
        max_estimated_bytes=10_000_000,
        max_work_units=100_000,
    )


def compile_qwen_a2a_shadow(
    policy: ToolPolicy,
    presentation_orders: Mapping[str, tuple[str, ...]],
    *,
    mode: ToolConstraintMode = ToolConstraintMode.SCHEMA,
    budget: CompileBudget | None = None,
) -> QwenA2AShadowCompilation:
    """Compile one Qwen A2a plan and, when authorized, its exact shadow grammar."""

    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")
    if mode not in {ToolConstraintMode.SCHEMA, ToolConstraintMode.FORMAT}:
        raise ValueError("Qwen A2a vertical slice certifies SCHEMA or FORMAT mode only")
    spec = qwen_a2a_single_call_spec()
    capabilities = qwen_a2a_compiler_capabilities()
    compile_budget = qwen_a2a_default_budget() if budget is None else budget

    constraint: ToolGenerationConstraint | None = None
    grammar_fingerprint: str | None = None

    def build_constraint_artifact(
        session_spec: ToolWireSpec,
        tools: tuple[ToolBranchPlan, ...],
        trigger_ids: tuple[str, ...],
    ) -> _ConstraintArtifactCandidate:
        return build_lark_tool_constraint_candidate(
            session_spec,
            tools,
            trigger_ids,
            structural_ws_max=_STRUCTURAL_WS_MAX,
        )

    def finalize_constraint_artifact(candidate: _ConstraintArtifactCandidate) -> str:
        nonlocal constraint, grammar_fingerprint
        constraint, grammar_fingerprint = finalize_lark_tool_constraint_candidate(candidate)
        return grammar_fingerprint

    plan = compile_tool_wire_plan(
        spec,
        policy,
        mode,
        compiler_capabilities=capabilities,
        presentation_orders=presentation_orders,
        budget=compile_budget,
        parser_branch_id="qwen-a2a-shadow-single-call",
        activation_trigger_ids=("tool-open",),
        constraint_artifact_builder=build_constraint_artifact,
        constraint_artifact_finalizer=finalize_constraint_artifact,
    )
    if not plan.constrained_executable:
        constraint = None
        grammar_fingerprint = None
    elif (
        constraint is None
        or grammar_fingerprint is None
        or plan.constraint_fingerprint != grammar_fingerprint
    ):
        raise RuntimeError("Qwen A2a artifact session did not bind one stable grammar fingerprint")

    return QwenA2AShadowCompilation(
        spec,
        plan,
        constraint,
        grammar_fingerprint,
        policy.allow_parallel,
    )


def qwen_a2a_prompt_observation(plan: CompiledToolWirePlan) -> PromptWireObservation:
    """Return canonical model-facing Qwen tag evidence for existing A0 parity checks."""

    spec = qwen_a2a_single_call_spec()
    if plan.spec_fingerprint != spec.fingerprint:
        raise ValueError("plan belongs to another Qwen A2a static spec")
    arguments: list[PromptArgumentWireObservation] = []
    for tool in plan.tools:
        for argument in tool.arguments:
            if not argument.generated or argument.framing_variant_id is None:
                continue
            variant = spec.framing_variant(argument.framing_variant_id)
            arguments.append(
                PromptArgumentWireObservation(
                    tool_name=tool.tool_name,
                    argument_name=argument.name,
                    framing_variant_id=argument.framing_variant_id,
                    argument_open_prefix=variant.argument_open.prefix,
                    argument_open_suffix=variant.argument_open.suffix,
                    argument_close=variant.argument_close.canonical.text,
                    encoded_argument_name=spec.argument_name_codec.encode(argument.name),
                )
            )
    return PromptWireObservation(
        source_id="qwen-a2a-canonical-tool-template-v1",
        tool_open=spec.tool_open.text,
        tool_close=spec.tool_close.canonical.text,
        function_open_prefix=spec.function_open.prefix,
        function_open_suffix=spec.function_open.suffix,
        function_close=spec.function_close.canonical.text,
        encoded_tool_names=tuple(
            (tool.tool_name, spec.function_name_codec.encode(tool.tool_name))
            for tool in plan.tools
        ),
        arguments=tuple(arguments),
    )
