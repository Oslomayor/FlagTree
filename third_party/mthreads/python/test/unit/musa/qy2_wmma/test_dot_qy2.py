"""QY2 WMMA dot test — validates tl.dot on capability-22 (QY2) architecture.

QY2 WMMA uses canonical instruction shape M32×N32×K16 on wave128.
This test is only collected when running on a QY2 device (capability 22).
"""

import pytest
import torch
import triton
import triton.language as tl


def get_resolution(dtype):
    atol_map = {torch.float16: 2.1e-3, torch.bfloat16: 1.4e-3}
    rtol_map = {torch.float16: 1e-3, torch.bfloat16: 7.9e-3}
    return atol_map.get(dtype, None), rtol_map.get(dtype, None)


@triton.jit
def _qy2_dot_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _run_qy2_dot(dtype, M=32, N=32, K=16, num_warps=1):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=dtype, device="musa")
    b = torch.randn((K, N), dtype=dtype, device="musa")
    c = torch.empty((M, N), dtype=torch.float32, device="musa")
    grid = (1, 1)

    _qy2_dot_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=16,
        num_warps=num_warps,
    )
    torch.musa.synchronize()

    ref = torch.matmul(a.float(), b.float())
    atol, rtol = get_resolution(dtype)
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qy2_dot_m32n32k16(dtype):
    """QY2 canonical shape: M32×N32×K16 on wave128."""
    _run_qy2_dot(dtype, M=32, N=32, K=16)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qy2_dot_m64n64k16(dtype):
    """Larger tile: M64xN64xK16."""
    _run_qy2_dot(dtype, M=64, N=64, K=16)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qy2_dot_m32n32k32_two_tiles(dtype):
    """K=32 with BLOCK_K=16 exercises 2 K-tile recurrence (Bug-1 regression)."""
    _run_qy2_dot(dtype, M=32, N=32, K=32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qy2_dot_m32n32k16_4warps(dtype):
    """num_warps=4 exercises multi-warp A/B operand distribution (Bug-2 regression)."""
    _run_qy2_dot(dtype, M=32, N=32, K=16, num_warps=4)


@triton.jit
def _qy2_dot_batched_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qy2_dot_batched(dtype):
    """Rank-3 batched dot exercises rank-3 B operand mapping (Bug-3 regression)."""
    torch.manual_seed(0)
    batch, M, N, K = 2, 32, 32, 16
    a = torch.randn((batch, M, K), dtype=dtype, device="musa")
    b = torch.randn((batch, K, N), dtype=dtype, device="musa")
    c = torch.empty((batch, M, N), dtype=torch.float32, device="musa")
    grid = (batch, )

    _qy2_dot_batched_kernel[grid](
        a,
        b,
        c,
        batch,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=K,
        num_warps=1,
    )
    torch.musa.synchronize()

    ref = torch.matmul(a.float(), b.float())
    atol, rtol = get_resolution(dtype)
    torch.testing.assert_close(c, ref, atol=atol, rtol=rtol)
