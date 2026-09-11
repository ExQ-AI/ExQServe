from __future__ import annotations

from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
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
    ConstraintCompilerCapabilities,
    LiteralTerminal,
    NameCodec,
    NamedTerminal,
    SchemaSemanticAuthority,
    ToolMultiplicity,
    ToolWireSpec,
    ValueCodecKind,
    ValueFraming,
    ValueFramingKind,
    WireChannel,
    compile_tool_wire_plan,
)


def raw_spec(
    *,
    ordering: ArgumentOrderingMode = ArgumentOrderingMode.PERMUTABLE,
    max_calls: int | None = None,
) -> ToolWireSpec:
    argument_close = CloseLanguage(
        (
            LiteralTerminal("</parameter>", (500,)),
            LiteralTerminal("</parameter >", (501,)),
        )
    )
    tool_open = LiteralTerminal("<tool_call>", (248058,))
    raw_variant = ArgumentFramingVariant(
        "raw-string",
        NamedTerminal("<parameter=", ">"),
        argument_close,
        ValueFraming(
            ValueFramingKind.RAW_UNTIL,
            ValueCodecKind.RAW_STRING,
            argument_close,
        ),
    )
    return ToolWireSpec(
        spec_id="synthetic-tagged-raw-v1",
        tool_open=tool_open,
        tool_close=CloseLanguage((LiteralTerminal("</tool_call>", (248059,)),)),
        function_open=NamedTerminal("<function=", ">"),
        function_close=CloseLanguage((LiteralTerminal("</function>"),)),
        function_name_codec=NameCodec("identity-function", (">",)),
        argument_name_codec=NameCodec("identity-argument", (">",)),
        argument_framings=(raw_variant,),
        framing_selector=ArgumentFramingSelector(
            "raw-string-by-schema-type",
            (ArgumentFramingSelectorRule("raw-string", ("string",)),),
        ),
        occurrence=ArgumentOccurrenceCapabilities(1, False, True),
        ordering=ordering,
        multiplicity=ToolMultiplicity(max_calls, True),
        tool_entry_channels=(WireChannel.TEXT, WireChannel.REASONING),
        tool_exit_channel=WireChannel.TEXT,
        activation_triggers=(ActivationTriggerSpec("tool-open", tool_open),),
    )


def single_call_raw_spec() -> ToolWireSpec:
    return raw_spec(max_calls=1)


def structured_spec() -> ToolWireSpec:
    tool_open = LiteralTerminal("<tool>", (700,))
    argument_close = CloseLanguage((LiteralTerminal("</arg>"),))
    structured_variant = ArgumentFramingVariant(
        "json",
        NamedTerminal("<arg=", ">"),
        argument_close,
        ValueFraming(ValueFramingKind.STRUCTURED_ESCAPED, ValueCodecKind.JSON),
    )
    return ToolWireSpec(
        spec_id="synthetic-structured-json-v1",
        tool_open=tool_open,
        tool_close=CloseLanguage((LiteralTerminal("</tool>", (701,)),)),
        function_open=NamedTerminal("<fn=", ">"),
        function_close=CloseLanguage((LiteralTerminal("</fn>"),)),
        function_name_codec=NameCodec("identity-function", (">",)),
        argument_name_codec=NameCodec("identity-argument", (">",)),
        argument_framings=(structured_variant,),
        framing_selector=ArgumentFramingSelector(
            "json-by-schema-type",
            (
                ArgumentFramingSelectorRule(
                    "json",
                    ("string", "integer", "number", "boolean", "object", "array", "null"),
                ),
            ),
        ),
        occurrence=ArgumentOccurrenceCapabilities(1, False, True),
        ordering=ArgumentOrderingMode.DECLARATION_ORDER,
        multiplicity=ToolMultiplicity(None, True),
        tool_entry_channels=(WireChannel.TEXT,),
        tool_exit_channel=WireChannel.TEXT,
        activation_triggers=(ActivationTriggerSpec("tool-open", tool_open),),
    )


def raw_compiler_capabilities() -> ConstraintCompilerCapabilities:
    return ConstraintCompilerCapabilities(
        "synthetic-raw-compiler-v1",
        ("type", "properties", "required", "additionalProperties"),
        ("type", "enum", "const"),
        SchemaSemanticAuthority.DRAFT_2020_12,
    )


def structured_compiler_capabilities() -> ConstraintCompilerCapabilities:
    return ConstraintCompilerCapabilities(
        "synthetic-json-compiler-v1",
        ("type", "properties", "required", "additionalProperties"),
        ("type", "enum", "const", "minimum", "maximum", "properties", "required", "items"),
        SchemaSemanticAuthority.DRAFT_2020_12,
    )


def tool(
    name: str,
    schema_json: str,
    *,
    strict: bool = False,
) -> FunctionTool:
    return FunctionTool(name, None, JsonSchema(schema_json), strict)


def policy(*tools: FunctionTool, allow_parallel: bool = True) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(ToolChoiceMode.AUTO), allow_parallel)


def budget(max_permutations: int = 1000) -> CompileBudget:
    return CompileBudget(
        max_permutations=max_permutations,
        max_estimated_rules=100_000,
        max_estimated_bytes=10_000_000,
        max_work_units=100_000,
    )


def schema_plan(
    spec: ToolWireSpec,
    tool_policy: ToolPolicy,
    presentation_orders: dict[str, tuple[str, ...]],
    *,
    compile_budget: CompileBudget | None = None,
    compiler_capabilities: ConstraintCompilerCapabilities | None = None,
):
    if compiler_capabilities is None:
        codecs = {variant.value_framing.codec for variant in spec.argument_framings}
        compiler_capabilities = (
            structured_compiler_capabilities()
            if codecs == {ValueCodecKind.JSON}
            else raw_compiler_capabilities()
        )
    return compile_tool_wire_plan(
        spec,
        tool_policy,
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=compiler_capabilities,
        presentation_orders=presentation_orders,
        budget=budget() if compile_budget is None else compile_budget,
        parser_branch_id="synthetic-constrained-branch",
        constraint_fingerprint="constraint-v1",
        activation_trigger_ids=("tool-open",),
    )
