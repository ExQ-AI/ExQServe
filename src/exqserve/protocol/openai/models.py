"""OpenAI model-discovery serialization over protocol-neutral served-model metadata."""

from __future__ import annotations

from exqserve.core.model import ServedModelInfo
from exqserve.protocol.openai.common import OpenAIProtocolError

# Compatibility alias; the canonical model metadata remains protocol-neutral.
OpenAIModelInfo = ServedModelInfo


def model_to_wire(info: ServedModelInfo) -> dict[str, object]:
    if not isinstance(info, ServedModelInfo):
        raise TypeError("info must be ServedModelInfo")
    return {
        "id": info.id,
        "object": "model",
        "created": info.created,
        "owned_by": "exqserve",
        "context_length": info.context_length,
    }


def model_not_found(model_id: str) -> OpenAIProtocolError:
    return OpenAIProtocolError(
        404,
        "invalid_request_error",
        "model_not_found",
        f"The model '{model_id}' is not served by this ExQServe process.",
        "model",
    )


def require_served_model(model_id: str, served: ServedModelInfo) -> None:
    if model_id != served.id:
        raise model_not_found(model_id)
