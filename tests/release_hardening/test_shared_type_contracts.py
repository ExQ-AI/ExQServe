from __future__ import annotations

from typing import get_type_hints

from exqserve.control.request import RuntimeSubmitter
from exqserve.runtime.contracts import RuntimeSessionLike
from exqserve.server.app import ServerRuntimeLike


def test_runtime_submitters_share_one_runtime_session_contract() -> None:
    control_hints = get_type_hints(RuntimeSubmitter.submit)
    server_hints = get_type_hints(ServerRuntimeLike.submit)

    assert control_hints["return"] is RuntimeSessionLike
    assert server_hints["return"] is RuntimeSessionLike
