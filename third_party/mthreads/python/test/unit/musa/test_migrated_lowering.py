"""Run migrated MUSA FileCheck fixtures through FlagTree's Python pass bindings."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from triton._C import libtriton
from triton._C.libtriton import ir, passes


@pytest.fixture
def filecheck():
    candidates = [shutil.which("FileCheck")]
    if os.environ.get("LLVM_SYSPATH"):
        candidates.append(str(Path(os.environ["LLVM_SYSPATH"]) / "bin/FileCheck"))
    candidates.append(str(Path.home() / ".flagtree/mthreads/mthreads-llvm22/bin/FileCheck"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("FileCheck is required for migrated lowering regressions")


def _parse(name, tmp_path, dtype="f16"):
    context = ir.context()
    ir.load_dialects(context)
    libtriton.mthreads.load_dialects(context)
    fixture = Path(__file__).parent / "ir" / name
    text = fixture.read_text()
    if dtype == "bf16":
        text = text.replace("f16", "bf16")
    source = tmp_path / name
    source.write_text(text)
    return context, ir.parse_mlir_module(str(source), context), fixture


def _check(filecheck, fixture, module, prefix="CHECK", extra=()):
    result = subprocess.run(
        [filecheck, str(fixture), f"--check-prefix={prefix}", *extra],
        input=module.str_nodebug(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_masked_load_carriers(tmp_path, filecheck):
    context, module, fixture = _parse("masked_integer_load_default_value.mlir", tmp_path)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
    pm.run(module, "migrated_musa_lowering")
    _check(filecheck, fixture, module)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_sqmma_mixed_source_order(tmp_path, filecheck, dtype, pipeline):
    context, module, fixture = _parse("sqmma_mixed_source_order_pipeline.mlir", tmp_path, dtype)
    pm = ir.pass_manager(context)
    libtriton.mthreads.passes.ttgpuir.add_accelerate_matmul(pm)
    if pipeline:
        libtriton.mthreads.passes.ttgpuir.add_pipeline(pm, 2, False)
        passes.common.add_canonicalizer(pm)
    pm.run(module, "migrated_musa_lowering")
    prefix = ("BF16-" if dtype == "bf16" else "") + ("PIPE" if pipeline else "ACCEL")
    _check(filecheck, fixture, module, prefix, ("--implicit-check-not=tt.load", ) if pipeline else ())
