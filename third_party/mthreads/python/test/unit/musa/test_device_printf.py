import re

import pytest
import triton
import triton.language as tl
from triton._C import libtriton
from pathlib import Path

if not hasattr(libtriton, "mthreads"):
    pytest.skip("musa backend not built in libtriton", allow_module_level=True)

from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir


def _get_musa_backend():
    if "mthreads" not in backends:
        pytest.skip("musa backend not discovered")
    target = GPUTarget("musa", "ph1", 32)
    return backends["mthreads"].compiler(target)


def _compile_to_llir(fn, signature, constexprs=None):
    target = GPUTarget("musa", "ph1", 32)
    backend = _get_musa_backend()

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    options = backend.parse_options({})
    module_map = backend.get_module_map()
    codegen_fns = backend.get_codegen_implementation(options)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})

    ttir = src.make_ir(target, options, codegen_fns, module_map, context)
    stages = {}
    backend.add_stages(stages, options, src.language)
    meta = {}
    ttir = stages["ttir"](ttir, meta)
    ttgir = stages["ttgir"](ttir, meta)
    llir = stages["llir"](ttgir, meta)
    return llir, meta


def test_default_libprintf_path(fresh_knobs):
    backend = _get_musa_backend()
    from triton.backends.mthreads import compiler as musa_compiler

    with fresh_knobs.musa.scope():
        del fresh_knobs.musa.libprintf_path
        options = backend.parse_options({})

    expected = Path(musa_compiler.__file__).resolve().parent / "lib" / "libprintf.bc"
    assert Path(dict(options.extern_libs)["printf"]).resolve() == expected


def test_non_mp31_does_not_add_libprintf():
    backend = backends["mthreads"].compiler(GPUTarget("musa", "qy2", 128))

    options = backend.parse_options({})

    assert "printf" not in dict(options.extern_libs)


def test_libprintf_path_precedence(fresh_knobs, tmp_path):
    backend = _get_musa_backend()
    environment_override = tmp_path / "libprintf.environment.bc"
    explicit_override = tmp_path / "libprintf.explicit.bc"
    environment_override.write_bytes(b"environment")
    explicit_override.write_bytes(b"explicit")

    with fresh_knobs.musa.scope():
        fresh_knobs.musa.libprintf_path = str(environment_override)
        environment_options = backend.parse_options({})
        explicit_options = backend.parse_options({"extern_libs": {"printf": str(explicit_override)}})

    assert dict(environment_options.extern_libs)["printf"] == str(environment_override)
    assert dict(explicit_options.extern_libs)["printf"] == str(explicit_override)


def test_libprintf_content_changes_options_hash(tmp_path):
    runtime = tmp_path / "libprintf.bc"
    runtime.write_bytes(b"first runtime")
    options = backends["mthreads"].compiler(GPUTarget("musa", "ph1", 32)).parse_options({
        "extern_libs": {
            "printf": str(runtime),
        },
    })

    first_hash = options.hash()
    runtime.write_bytes(b"second runtime")
    assert options.hash() != first_hash


def test_device_print_uses_linked_vprintf_runtime():

    @triton.jit
    def kernel_device_print(i8_ptr, i32_ptr, i64_ptr, f32_ptr):
        offs = tl.arange(0, 1)
        signed_i8 = tl.load(i8_ptr + offs)
        unsigned_i8 = signed_i8.to(tl.uint8)
        i32 = tl.load(i32_ptr + offs)
        i64 = tl.load(i64_ptr + offs)
        f32 = tl.load(f32_ptr + offs)
        tl.device_print("values: ", signed_i8, unsigned_i8, i32, i64, f32, i8_ptr)
        tl.device_print("no args")

    llir, _ = _compile_to_llir(
        kernel_device_print,
        {
            "i8_ptr": "*i8",
            "i32_ptr": "*i32",
            "i64_ptr": "*i64",
            "f32_ptr": "*fp32",
        },
    )

    assert "@printf(" not in llir
    assert '"has_printf"' in llir
    for function in ("vprintf", "vprintf_internal"):
        definition = re.search(rf"^define internal .* @{function}\([^#\n]+#(\d+)", llir, re.MULTILINE)
        assert definition is not None, "\n".join(line for line in llir.splitlines() if function in line)
        attributes = re.search(rf"^attributes #{definition.group(1)} = \{{([^}}]+)\}}$", llir, re.MULTILINE)
        assert attributes is not None
        assert "noinline" in attributes.group(1)
        assert "alwaysinline" not in attributes.group(1)
