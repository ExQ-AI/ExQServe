"""Test-only DeepSeek-V4 structured DSML control data for Tool-Wire A1 shadow certification.

This module is intentionally isolated from the production DeepSeek parser.  It contributes
static DSML framing data only; the shared engine contains no DeepSeek/model-name dispatch.
"""

from __future__ import annotations

from exqserve.tool_wire.contracts import (
    ActivationTriggerSpec,
    ArgumentFramingSelector,
    ArgumentFramingSelectorRule,
    ArgumentFramingVariant,
    ArgumentOccurrenceCapabilities,
    ArgumentOrderingMode,
    CloseLanguage,
    LiteralTerminal,
    NameCodec,
    NamedTerminal,
    ToolMultiplicity,
    ToolWireSpec,
    ValueCodecKind,
    ValueFraming,
    ValueFramingKind,
    WireChannel,
)

_DSML = "｜DSML｜"
_TOOL_OPEN = f"<{_DSML}tool_calls>"
_TOOL_CLOSE = f"</{_DSML}tool_calls>"
_INVOKE_OPEN_PREFIX = f'<{_DSML}invoke name="'
_INVOKE_CLOSE = f"</{_DSML}invoke>"
_PARAMETER_OPEN_PREFIX = f'<{_DSML}parameter name="'
_PARAMETER_CLOSE = f"</{_DSML}parameter>"
_DEEPSEEK_NAME_WHITESPACE = (
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
_DEEPSEEK_FUNCTION_NAME_FORBIDDEN = ('"', "<", ">") + _DEEPSEEK_NAME_WHITESPACE


def deepseek_v4_structured_dsml_spec() -> ToolWireSpec:
    """Return the A1 structured ``string=false`` DSML control spec.

    The control intentionally models only self-delimiting JSON argument values.  Native
    ``string=true`` raw-value behavior remains outside A1 and is not represented here.
    """

    argument_close = CloseLanguage((LiteralTerminal(_PARAMETER_CLOSE),))
    structured_variant = ArgumentFramingVariant(
        "deepseek-v4-json-string-false",
        NamedTerminal(
            _PARAMETER_OPEN_PREFIX,
            '"',
            ' string="false">',
        ),
        argument_close,
        ValueFraming(ValueFramingKind.STRUCTURED_ESCAPED, ValueCodecKind.JSON),
    )
    tool_open = LiteralTerminal(_TOOL_OPEN)
    return ToolWireSpec(
        spec_id="deepseek-v4-dsml-structured-shadow-v1",
        tool_open=tool_open,
        tool_close=CloseLanguage((LiteralTerminal(_TOOL_CLOSE),)),
        function_open=NamedTerminal(_INVOKE_OPEN_PREFIX, '"', ">"),
        function_close=CloseLanguage((LiteralTerminal(_INVOKE_CLOSE),)),
        function_name_codec=NameCodec(
            "deepseek-v4-dsml-function-identity",
            _DEEPSEEK_FUNCTION_NAME_FORBIDDEN,
        ),
        argument_name_codec=NameCodec("deepseek-v4-dsml-argument-identity", ('"',)),
        argument_framings=(structured_variant,),
        framing_selector=ArgumentFramingSelector(
            "deepseek-v4-structured-json-all-types",
            (
                ArgumentFramingSelectorRule(
                    structured_variant.variant_id,
                    ("string", "integer", "number", "boolean", "object", "array", "null"),
                ),
            ),
        ),
        occurrence=ArgumentOccurrenceCapabilities(1, False, True),
        ordering=ArgumentOrderingMode.PERMUTABLE,
        multiplicity=ToolMultiplicity(None, True, min_calls_per_sequence=1),
        tool_entry_channels=(WireChannel.TEXT, WireChannel.REASONING),
        tool_exit_channel=WireChannel.TEXT,
        activation_triggers=(ActivationTriggerSpec("tool-open", tool_open),),
    )
