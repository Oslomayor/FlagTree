import pytest
import torch

import triton
import triton.language as tl

INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
# torch_musa cannot materialize uint16/uint32 device tensors.
UINT_DTYPES = [torch.uint8, torch.uint64]

# mtcc mis-lowers fptosi/fptoui bfloat -> 64-bit int as a saturating 32-bit
# conversion followed by extension; fptosi float -> i64 is correct. Until the
# toolchain legalization is fixed, Triton cannot produce correct code for these
# two combinations through any lowering (InstCombine folds the f32 two-step
# into the broken direct form), so they are expected to fail.
MTCC_BROKEN_BF16_TO_I64 = "mtcc legalizes fptosi/fptoui bfloat->i64 as s32_sat + extend"
# mtcc cannot select fp_round f64 -> bf16 (llc aborts); bfloat -> double fpext
# works. Pre-existing toolchain gap, identical IR under the old lowering.
MTCC_NO_F64_TO_BF16 = "mtcc cannot select fptrunc double->bfloat"

SIZE = 8192
BLOCK = 1024


@triton.jit
def _cast_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


def _rand_int(dtype, size, device):
    info = torch.iinfo(dtype)
    if dtype == torch.uint64:
        # torch.randint cannot cover the full uint64 range; int64 bits reinterpreted
        # still exercises every representable bit pattern uniformly enough.
        return torch.randint(-2**63, 2**63 - 1, (size, ), dtype=torch.int64, device=device).view(torch.uint64)
    return torch.randint(info.min, info.max, (size, ), dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", INT_DTYPES + UINT_DTYPES)
def test_int_to_bf16(device, dtype):
    if device != "musa":
        pytest.skip("requires MUSA")
    torch.manual_seed(0)
    x = _rand_int(dtype, SIZE, device)
    # Dense small values hit RNE tie points around the bf16 mantissa boundary.
    small = torch.arange(-2048, 2048, dtype=torch.int64, device=device).to(dtype)
    x = torch.cat([x, small])

    out = torch.empty(x.numel(), dtype=torch.bfloat16, device=device)
    grid = (triton.cdiv(x.numel(), BLOCK), )
    _cast_kernel[grid](x, out, x.numel(), BLOCK=BLOCK)

    ref = x.to(torch.bfloat16)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("dtype", INT_DTYPES + UINT_DTYPES)
def test_bf16_to_int(device, dtype, request):
    if device != "musa":
        pytest.skip("requires MUSA")
    if dtype in (torch.int64, torch.uint64):
        # Non-strict so a fixed mtcc surfaces as XPASS.
        request.node.add_marker(pytest.mark.xfail(reason=MTCC_BROKEN_BF16_TO_I64, strict=False))
    torch.manual_seed(0)
    info = torch.iinfo(dtype)
    # Stay well inside the destination range: out-of-range fptosi is UB on both
    # the hardware and the torch reference, so it cannot be part of the contract.
    lo, hi = max(info.min // 4, -2**62), min(info.max // 4, 2**62)
    floats = torch.empty(SIZE, dtype=torch.float32, device=device).uniform_(lo, hi)
    specials = torch.tensor([0.0, 0.5, -0.5, 0.999, -0.999, 1.5, -1.5, 127.5, 255.5], dtype=torch.float32,
                            device=device)
    in_range = (specials >= info.min) & (specials <= info.max)
    x = torch.cat([floats, specials[in_range]]).to(torch.bfloat16)

    out = torch.empty(x.numel(), dtype=dtype, device=device)
    grid = (triton.cdiv(x.numel(), BLOCK), )
    _cast_kernel[grid](x, out, x.numel(), BLOCK=BLOCK)

    ref = x.to(dtype)
    out_bits, ref_bits = out, ref
    if dtype == torch.uint64:
        # muDNN has no UINT64 EQ kernel; compare the bit patterns instead.
        out_bits, ref_bits = out.view(torch.int64), ref.view(torch.int64)
    mismatches = (out_bits != ref_bits).nonzero().flatten()
    assert mismatches.numel() == 0, (f"{mismatches.numel()} mismatches, first ones: "
                                     f"in={x[mismatches[:5]].float().tolist()} "
                                     f"out={out[mismatches[:5]].tolist()} ref={ref[mismatches[:5]].tolist()}")


# Same-width (bf16<->f16) and wide (bf16<->f64) float casts have no coverage in
# test_cast/test_conversions; the former goes through our ExtF(bf16->f32) +
# TruncF path, the latter through fpext/fptrunc bfloat<->double.
FLOAT_PAIRS = [
    (torch.bfloat16, torch.float16),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.float64),
    (torch.float64, torch.bfloat16),
]


@pytest.mark.parametrize("src_dtype,dst_dtype", FLOAT_PAIRS)
def test_bf16_float_cast(device, src_dtype, dst_dtype, request):
    if device != "musa":
        pytest.skip("requires MUSA")
    if (src_dtype, dst_dtype) == (torch.float64, torch.bfloat16):
        request.node.add_marker(pytest.mark.xfail(reason=MTCC_NO_F64_TO_BF16, strict=False))
    torch.manual_seed(0)
    sample_dtype = torch.float64 if src_dtype == torch.float64 else torch.float32
    floats = torch.empty(SIZE, dtype=sample_dtype, device=device).uniform_(-1e4, 1e4)
    specials = torch.tensor(
        [0.0, 1.0, -1.0, 0.1, 1.0 + 2**-9, 65504.0, -65504.0, 65520.0, 6e-5, 1e30, -1e30,
         float("inf"),
         float("-inf")], dtype=sample_dtype, device=device)
    x = torch.cat([floats, specials]).to(src_dtype)

    out = torch.empty(x.numel(), dtype=dst_dtype, device=device)
    grid = (triton.cdiv(x.numel(), BLOCK), )
    _cast_kernel[grid](x, out, x.numel(), BLOCK=BLOCK)

    ref = x.to(dst_dtype)
    mismatches = (out != ref).nonzero().flatten()
    assert mismatches.numel() == 0, (f"{mismatches.numel()} mismatches, first ones: "
                                     f"in={x[mismatches[:5]].float().tolist()} "
                                     f"out={out[mismatches[:5]].float().tolist()} "
                                     f"ref={ref[mismatches[:5]].float().tolist()}")
