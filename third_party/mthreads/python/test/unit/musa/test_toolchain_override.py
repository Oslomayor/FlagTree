import os

import pytest
from triton.backends.mthreads import compiler


@pytest.mark.parametrize("binary", ["llc", "ld.lld"])
def test_environment_toolchain_override_precedes_bundle(tmp_path, monkeypatch, binary):
    override = tmp_path / binary
    override.write_text("#!/bin/sh\necho 'LLVM version 14.0.0'\n")
    override.chmod(0o755)
    bundled = str(tmp_path / "bundled" / binary)
    monkeypatch.setattr(compiler, "_local_backend_tool_path", lambda name: bundled)
    assert compiler._select_tool_path(binary, str(override), lambda: None) == str(override)


@pytest.mark.parametrize("binary", ["llc", "ld.lld"])
def test_default_toolchain_still_uses_bundle(tmp_path, monkeypatch, binary):
    bundled = str(tmp_path / binary)
    monkeypatch.setattr(compiler, "_local_backend_tool_path", lambda name: bundled)
    assert compiler._select_tool_path(binary, None, lambda: None) == bundled


def test_fresh_knobs_preserves_musa_toolchain(tmp_path, monkeypatch, request):
    overrides = {
        "llc_path": ("TRITON_MUSA_LLC_PATH", str(tmp_path / "llc")),
        "llc_asm_path": ("TRITON_MUSA_LLC_ASM_PATH", str(tmp_path / "llc-asm")),
        "lld_path": ("TRITON_MUSA_LLD_PATH", str(tmp_path / "ld.lld")),
    }
    for env_name, path in overrides.values():
        monkeypatch.setenv(env_name, path)
    # Request after setting the overrides to detect fixture-induced removal.
    fresh = request.getfixturevalue("fresh_knobs")
    for knob, (env_name, path) in overrides.items():
        assert os.environ[env_name] == path
        assert getattr(fresh.musa, knob) == path
