import os

import pytest

# Everything under qy2_wmma/ targets the QY2 (capability 22) architecture.
# On any other capability, skip the whole folder instead of running the tests
# against the wrong target.
try:
    import torch
    _c = torch.musa.get_device_capability()
    _CAP_CODE = int(_c[0]) * 10 + int(_c[1])
except Exception:  # pragma: no cover
    _CAP_CODE = None

_HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_collection_modifyitems(config, items):
    if _CAP_CODE == 22:
        return
    skip_marker = pytest.mark.skip(
        reason=f"QY2 (cap-22) WMMA tests require a capability-22 device; got {_CAP_CODE}"
    )
    for item in items:
        if str(item.fspath).startswith(_HERE):
            item.add_marker(skip_marker)
