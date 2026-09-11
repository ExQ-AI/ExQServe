"""Qwen-specific Tool-Wire decoder binding at the composition boundary."""

from __future__ import annotations

from exqserve.core.generation_guarantees import ConstraintFallbackPolicy
from exqserve.model.contracts import ParserCreationContext, ToolGenerationConstraint
from exqserve.runtime.contracts import ConstraintInstallation
from exqserve.tool_wire.controls.qwen import (
    QwenToolWireDecodeAuthority,
    build_qwen_compatibility_tool_region_decoder,
    build_qwen_constrained_tool_region_decoder,
)


def resolve_qwen_parser_context(
    tool_constraint: ToolGenerationConstraint | None,
    installation: ConstraintInstallation | None,
    fallback_policy: ConstraintFallbackPolicy,
) -> ParserCreationContext | None:
    if tool_constraint is None or tool_constraint.decode_authority is None:
        return ParserCreationContext(
            hard_constraint_installed=False,
            tool_region_decoder_factory=build_qwen_compatibility_tool_region_decoder,
        )
    identity = tool_constraint.constraint_fingerprint
    if identity is None:
        raise ValueError("missing constraint identity")
    authority = tool_constraint.decode_authority
    if not isinstance(authority, QwenToolWireDecodeAuthority):
        raise TypeError("unsupported Qwen Tool decode authority")
    if installation is None:
        raise ValueError("unknown constraint installation")
    if not installation.installed:
        if fallback_policy is ConstraintFallbackPolicy.FAIL_CLOSED:
            raise ValueError("required constraint not installed")
        return ParserCreationContext(
            hard_constraint_installed=False,
            tool_region_decoder_factory=build_qwen_compatibility_tool_region_decoder,
        )
    if installation.constraint_fingerprint != identity:
        raise ValueError("constraint installation mismatch")
    return ParserCreationContext(
        hard_constraint_installed=True,
        constraint_identity=identity,
        trigger_token_ids=installation.trigger_token_ids,
        generation_guarantee=installation.guarantee,
        tool_region_decoder=build_qwen_constrained_tool_region_decoder(authority),
    )
