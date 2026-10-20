"""QY2 WMMA matmul test — end-to-end matmul on capability-22 (QY2) architecture.
"""

import itertools
import pytest
import torch
import triton
import triton.language as tl


def get_resolution(dtype):
    atol_map = {torch.float16: 2.1e-3, torch.bfloat16: 1.4e-3}
    rtol_map = {torch.float16: 1e-3, torch.bfloat16: 7.9e-3}
    return atol_map.get(dtype, None), rtol_map.get(dtype, None)


@triton.jit
def _qy2_matmul_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(k + offs_k)[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k)[:, None] < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def triton_matmul(a, b, num_warps=1):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _qy2_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=16, num_warps=num_warps,
    )
    return c


@pytest.mark.parametrize(
    "dtype, M, N, K",
    [
        (torch.float16, 32, 32, 16),
        (torch.float16, 64, 64, 16),
        (torch.bfloat16, 32, 32, 16),
        (torch.bfloat16, 64, 64, 16),
    ],
)
def test_qy2_matmul(dtype, M, N, K):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=dtype, device="musa")
    b = torch.randn((K, N), dtype=dtype, device="musa")
    c_triton = triton_matmul(a, b)
    torch.musa.synchronize()
    ref = torch.matmul(a.float(), b.float())
    atol, rtol = get_resolution(dtype)
    torch.testing.assert_close(c_triton, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype, M, N, K",
    [
        (torch.float16, 32, 32, 32),
        (torch.bfloat16, 32, 32, 32),
    ],
)
def test_qy2_matmul_k32_two_tiles(dtype, M, N, K):
    """K=32 with BLOCK_K=16 exercises 2 K-tile recurrence (Bug-1 regression)."""
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=dtype, device="musa")
    b = torch.randn((K, N), dtype=dtype, device="musa")
    c_triton = triton_matmul(a, b)
    torch.musa.synchronize()
    ref = torch.matmul(a.float(), b.float())
    atol, rtol = get_resolution(dtype)
    torch.testing.assert_close(c_triton, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype, M, N, K",
    [
        (torch.float16, 32, 32, 16),
        (torch.bfloat16, 32, 32, 16),
    ],
)
def test_qy2_matmul_4warps(dtype, M, N, K):
    """num_warps=4 exercises multi-warp A/B operand distribution (Bug-2 regression)."""
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=dtype, device="musa")
    b = torch.randn((K, N), dtype=dtype, device="musa")
    c_triton = triton_matmul(a, b, num_warps=4)
    torch.musa.synchronize()
    ref = torch.matmul(a.float(), b.float())
    atol, rtol = get_resolution(dtype)
    torch.testing.assert_close(c_triton, ref, atol=atol, rtol=rtol)
