import re

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _integer_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    a = tl.load(A + m[:, None] * K + k[None, :])
    b = tl.load(B + k[:, None] * N + n[None, :])
    c = tl.dot(a, b)
    tl.store(C + m[:, None] * N + n[None, :], c)


@pytest.mark.parametrize("arch,warp_size,m,n,num_warps,native", [
    (31, 32, 32, 32, 16, False),
    (31, 32, 64, 64, 4, True),
    (22, 128, 32, 32, 4, False),
    (22, 128, 64, 64, 4, True),
])
def test_integer_dot_exact_warp_coverage(arch, warp_size, m, n, num_warps, native):
    # Compile both architectures without requiring a QY2 device at runtime.
    target = GPUTarget("musa", arch, warp_size)
    backend = backends["mthreads"].compiler(target)
    options = backend.parse_options({"num_warps": num_warps})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(_integer_dot, {"A": "*i8", "B": "*i8", "C": "*i32"}, constexprs={"M": m, "N": n, "K": 128})
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    stages = {}
    backend.add_stages(stages, options, source.language)
    metadata = {}
    for stage in ("ttir", "ttgir", "llir"):
        module = stages[stage](module, metadata)
    has_native = bool(re.search(r"llvm\.musa\.(sqmma\.smma|imma)\.m\d+n\d+k\d+\.mma", module))
    assert has_native == native
    assert "fmuladd" not in module
    assert not re.search(r"\b[su]itofp\b", module)
