from __future__ import annotations

import exqserve
import exqserve.core


def test_package_namespace_imports() -> None:
    assert exqserve.__name__ == "exqserve"
    assert exqserve.core.__name__ == "exqserve.core"
