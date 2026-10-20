import itertools

import pytest
import torch

import triton
import triton.language as tl


def get_backend():
    return triton.runtime.driver.active.get_current_target().backend


# ---- Inline matmul kernel (triton.ops was removed in 3.1+/3.2 upstream) ----
_ordered_dtypes = [torch.int8, torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a
    assert a in _ordered_dtypes
    assert b in _ordered_dtypes
    for d in _ordered_dtypes:
        if a is d:
            return b
        if b is d:
            return a


def _init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _matmul_kernel(A, B, C, M, N, K,  #
                   stride_am, stride_ak,  #
                   stride_bk, stride_bn,  #
                   stride_cm, stride_cn,  #
                   acc_dtype: tl.constexpr,  #
                   input_precision: tl.constexpr,  #
                   fp8_fast_accum: tl.constexpr,  #
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                   GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr  #
                   ):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=input_precision)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def _matmul(a, b, acc_dtype, input_precision, fp8_fast_accum, output_dtype):
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    ab_dtype = get_higher_dtype(a.dtype, b.dtype)
    if output_dtype is None:
        output_dtype = ab_dtype

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)
    supported_acc_dtypes = {
        torch.float16: (torch.float32, torch.float16),
        torch.bfloat16: (torch.float32, torch.bfloat16),
        torch.float32: (torch.float32, ),
    }
    if acc_dtype is None:
        acc_dtype = supported_acc_dtypes[ab_dtype][0]
    else:
        assert isinstance(acc_dtype, torch.dtype), "acc_dtype must be a torch.dtype"
        assert acc_dtype in supported_acc_dtypes[a.dtype], "acc_dtype not compatible with the type of a"
        assert acc_dtype in supported_acc_dtypes[b.dtype], "acc_dtype not compatible with the type of b"

    def to_tl_type(ty):
        return getattr(tl, str(ty).split(".")[-1])

    acc_dtype = to_tl_type(acc_dtype)
    ab_dtype = to_tl_type(ab_dtype)
    output_dtype = to_tl_type(output_dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    _matmul_kernel[grid](
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
        acc_dtype=acc_dtype,
        input_precision=input_precision,
        fp8_fast_accum=fp8_fast_accum,
        GROUP_M=8,
        AB_DTYPE=ab_dtype,
    )
    return c


@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, INPUT_PRECISION, F8_FASTACCUM, ACC_DTYPE, OUTPUT_DTYPE",
    itertools.chain(
        *[[
            # 1 warp
            (32, 32, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 32, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (32, 64, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 64, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # 2 warp
            (64, 32, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (32, 64, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (128, 32, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (32, 128, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # 4 warp
            (128, 64, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 128, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # variable input
            (64, 64, 16, 1, 2, 2, 128, 128, 64, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (128, 128, 16, 1, 8, 2, 256, 256, 128, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            # Test input precision and fast accumulation flags (even though they won't affect FP16/BF16 MMA)
            (64, 64, 16, 1, 2, 2, 128, 128, 64, AT, BT, DTYPE, DTYPE, "tf32x3", False, "float32", "float32"),
            (64, 64, 16, 1, 2, 2, 128, 128, 64, AT, BT, DTYPE, DTYPE, None, False, "float32", "float32"),
        ] for DTYPE in ["float16", "bfloat16"] for AT in [False, True] for BT in [False, True]],
        # n-stage
        *[[
            (32, 32, 16, 1, 1, STAGES, 64, 64, 32, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (64, 64, 16, 1, 4, STAGES, 128, 128, 64, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
            (128, 128, 16, 1, 8, STAGES, 256, 256, 128, AT, BT, DTYPE, DTYPE, None, True, "float32", "float32"),
        ] for DTYPE in ["float16", "bfloat16"] for AT in [False, True] for BT in [False, True] for STAGES in [4]],
    ),
)
def test_op(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, INPUT_PRECISION,
            F8_FASTACCUM, ACC_DTYPE, OUTPUT_DTYPE):
    capability = torch.musa.get_device_capability()
    assert (capability[0] == 2 and capability[1] == 2) and "Only used for QY2."
    torch.manual_seed(0)

    def init_input(m, n, dtype, acc_dtype):
        if dtype == "int8":
            return torch.randint(-128, 127, (m, n), device="musa", dtype=torch.int8)
        # Use small range of values to prevent numerical issues
        min_exp = -4 if acc_dtype == "float16" else -10
        exponents = torch.randint(min_exp, 0, size=(m, n), device="musa")
        ret = (2.**exponents).to(getattr(torch, dtype))
        # Randomly flip signs
        signs = torch.randint(0, 2, (m, n), device="musa") * 2 - 1
        return ret * signs

    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K, 'SPLIT_K': SPLIT_K}
    pre_hook = None if SPLIT_K == 1 else lambda nargs: nargs['C'].zero_()
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE, pre_hook=pre_hook)]
    kernel = _matmul_kernel
    kernel.configs = configs

    # get matrix shape
    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K * SPLIT_K if K is None else K

    # allocate/transpose inputs
    a = init_input(M, K, ADTYPE, ACC_DTYPE)
    b = init_input(K, N, BDTYPE, ACC_DTYPE)
    a = a if not AT else a.T.contiguous().T
    b = b if not BT else b.T.contiguous().T

    # run test
    th_a = a.float() if a.dtype in [torch.float16, torch.bfloat16] else a
    th_b = b.float() if b.dtype in [torch.float16, torch.bfloat16] else b
    th_c = torch.matmul(th_a, th_b)  # FP32 output
    acc_dtype = torch.float32 if ACC_DTYPE in ["float32"] else torch.float32
    output_dtype = acc_dtype

    try:
        tt_c = _matmul(a, b, acc_dtype, INPUT_PRECISION, F8_FASTACCUM, output_dtype)

        # Adjust tolerances based on input types
        rtol = 1e-2 if ADTYPE == "float16" or BDTYPE == "float16" else 5e-2
        atol = 1e-2 if ADTYPE == "float16" or BDTYPE == "float16" else 5e-2
        torch.testing.assert_close(th_c, tt_c, rtol=rtol, atol=atol)
    except triton.OutOfResources as e:
        pytest.skip(str(e))
