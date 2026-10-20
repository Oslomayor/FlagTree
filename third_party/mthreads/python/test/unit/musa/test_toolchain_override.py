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
