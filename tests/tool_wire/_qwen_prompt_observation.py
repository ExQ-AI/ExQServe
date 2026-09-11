"""Historical Qwen prompt observation for template regression tests only."""

from exqserve.tool_wire.contracts import CompiledToolWirePlan
from tests.tool_wire._legacy_certification import (
    PromptArgumentWireObservation,
    PromptWireObservation,
)
from tests.tool_wire._legacy_qwen_a2a import qwen_a2a_single_call_spec


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
