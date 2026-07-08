from pathlib import Path

from triton.backends.compiler import GPUTarget
from triton.backends.mthreads import compiler as musa_compiler
from triton.backends.mthreads.compiler import MUSABackend, MUSAOptions, _llc_extra_options, _resolve_toolchain_paths


def _write_tool(path: Path, version: str) -> None:
    path.write_text(f"#!/bin/sh\necho '{version}'\n")
    path.chmod(0o755)


def test_musa_toolchain_defaults_to_usr_local_musa(tmp_path, monkeypatch):
    default_home = tmp_path / "default-musa"

    monkeypatch.setattr(musa_compiler, "_DEFAULT_MUSA_PREFIX", str(default_home))
    monkeypatch.delenv("TRITON_MUSA_LLC_PATH", raising=False)
    monkeypatch.delenv("TRITON_MUSA_LLD_PATH", raising=False)

    llc_path, lld_path, llc_asm_path = _resolve_toolchain_paths(MUSAOptions())

    assert llc_path == str(default_home / "bin" / "llc")
    assert lld_path == str(default_home / "bin" / "ld.lld")
    assert llc_asm_path is None


def test_parse_options_bad_env_path_falls_back_to_default_tool(tmp_path, monkeypatch):
    default_home = tmp_path / "default-musa"
    default_bin = default_home / "bin"
    default_bin.mkdir(parents=True)
    llc = default_bin / "llc"
    lld = default_bin / "ld.lld"
    _write_tool(llc, version="LLVM version 19.1.0")
    _write_tool(lld, version="LLD 19.1.0")

    monkeypatch.setattr(musa_compiler.knobs, "_DEFAULT_MUSA_PREFIX", str(default_home))
    monkeypatch.setenv("TRITON_MUSA_LLC_PATH", str(tmp_path / "missing-llc"))
    monkeypatch.setenv("TRITON_MUSA_LLD_PATH", str(tmp_path / "missing-lld"))
    del musa_compiler.knobs.musa.llc
    del musa_compiler.knobs.musa.lld

    backend = MUSABackend(GPUTarget("musa", "ph1", 32))
    options = backend.parse_options({})

    assert options.llc_path == str(llc)
    assert options.lld_path == str(lld)


def test_llc_extra_options_sqmma_accepts_user_options():
    opts = _llc_extra_options(
        {"uses_sqmma": True},
        MUSAOptions(llc_options="-foo=1 --bar"),
    )

    assert opts == [
        "-mtgpu-enable-const-calc=1",
        "-mtgpu-alloc-shared-memory-from-zero=1",
        "-foo=1",
        "--bar",
    ]
