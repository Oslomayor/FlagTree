import pytest
import torch

import triton
import triton.language as tl

SIZE = 8192
BLOCK = 1024


@triton.jit
def _binary_kernel(a_ptr, b_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    if OP == 0:
        out = a + b
    elif OP == 1:
        out = a - b
    elif OP == 2:
        out = a * b
    elif OP == 3:
        out = a / b
    else:
        out = tl.fdiv(a, b)
    tl.store(out_ptr + offs, out, mask=mask)


# bf16 elementwise arithmetic lowers natively (no f32 promote/round trip);
# a single correctly-rounded op must stay bit-identical to the torch reference.
# OP=3 uses `/` (frontend promotes to f32); OP=4 uses tl.fdiv, which reaches
# the backend as arith.divf bfloat and takes the MUSA f32-promote path.
@pytest.mark.parametrize("op", [0, 1, 2, 3, 4], ids=["add", "sub", "mul", "div", "tl.fdiv"])
def test_bf16_binary_arith(device, op):
    if device != "musa":
        pytest.skip("requires MUSA")
    torch.manual_seed(0)
    a = torch.randn(SIZE, dtype=torch.float32, device=device) * 100
    b = torch.randn(SIZE, dtype=torch.float32, device=device) * 100
    specials_a = torch.tensor([0.0, 1.0, -1.0, 0.1, 65504.0, -65504.0, 1e30, -1e30],
                              dtype=torch.float32, device=device)
    specials_b = torch.tensor([1.0, -1.0, 2.0, 0.5, 1.0, -1.0, 1e-30, 1e30],
                              dtype=torch.float32, device=device)
    a = torch.cat([a, specials_a]).to(torch.bfloat16)
    b = torch.cat([b, specials_b]).to(torch.bfloat16)

    out = torch.empty(a.numel(), dtype=torch.bfloat16, device=device)
    grid = (triton.cdiv(a.numel(), BLOCK), )
    _binary_kernel[grid](a, b, out, a.numel(), OP=op, BLOCK=BLOCK)

    # Reference on CPU: torch_musa's device bf16 div is broken (returns the
    # dividend); CPU bf16 arithmetic is well-defined (float opmath + RTNE).
    ref = getattr(torch, ["add", "sub", "mul", "div", "div"][op])(a.cpu(), b.cpu()).to(device)
    mismatches = ((out != ref) & ~(torch.isnan(out) & torch.isnan(ref))).nonzero().flatten()
    assert mismatches.numel() == 0, (
        f"{mismatches.numel()} mismatches, first ones: "
        f"a={a[mismatches[:5]].float().tolist()} b={b[mismatches[:5]].float().tolist()} "
        f"out={out[mismatches[:5]].float().tolist()} ref={ref[mismatches[:5]].float().tolist()}")
