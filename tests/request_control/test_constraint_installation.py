from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exqserve.control.request import ControlledSession
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.runtime.contracts import ConstraintInstallation, RuntimeEvent


class _LegacyRuntimeSession:
    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        raise StopAsyncIteration

    def inject_text(self, text: str) -> None:
        del text

    async def cancel(self) -> None:
        return None


class _CredentialRuntimeSession(_LegacyRuntimeSession):
    constraint_installation = ConstraintInstallation(
        True,
        "runtime-fingerprint",
        (248058,),
        GenerationGuarantee.SCHEMA,
    )


async def _release(_: ControlledSession) -> None:
    return None


def test_legacy_runtime_missing_installation_capability_remains_unknown() -> None:
    async def scenario() -> None:
        controlled = ControlledSession(
            _LegacyRuntimeSession(),
            request_id="legacy-runtime",
            injection_allowed=True,
            timeout_seconds=None,
            release=_release,
        )
        assert controlled.constraint_installation is None

    asyncio.run(scenario())


def test_control_layer_passes_runtime_installation_without_reinterpreting_it() -> None:
    async def scenario() -> None:
        controlled = ControlledSession(
            _CredentialRuntimeSession(),
            request_id="credential-runtime",
            injection_allowed=False,
            timeout_seconds=None,
            release=_release,
        )
        assert controlled.constraint_installation == _CredentialRuntimeSession.constraint_installation

    asyncio.run(scenario())
