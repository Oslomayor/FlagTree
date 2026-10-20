import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.driver import driver

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="requires torch.musa runtime",
)


@triton.jit
def _atomic_rmw_kernel(dst, value, old_values, op: tl.constexpr):
    pid = tl.program_id(0)
    value = tl.load(value)
    if op == "add":
        old = tl.atomic_add(dst, value, sem="relaxed", scope="gpu")
    elif op == "min":
        old = tl.atomic_min(dst, value, sem="relaxed", scope="gpu")
    elif op == "max":
        old = tl.atomic_max(dst, value, sem="relaxed", scope="gpu")
    elif op == "and":
        old = tl.atomic_and(dst, value, sem="relaxed", scope="gpu")
    elif op == "or":
        old = tl.atomic_or(dst, value, sem="relaxed", scope="gpu")
    elif op == "xor":
        old = tl.atomic_xor(dst, value, sem="relaxed", scope="gpu")
    elif op == "xchg":
        old = tl.atomic_xchg(dst, value, sem="relaxed", scope="gpu")
    tl.store(old_values + pid, old)


@triton.jit
def _atomic_cas_kernel(dst, compare, value, old_values):
    pid = tl.program_id(0)
    old = tl.atomic_cas(
        dst,
        tl.load(compare),
        tl.load(value),
        sem="relaxed",
        scope="gpu",
    )
    tl.store(old_values + pid, old)


_ATOMIC_CASES = (
    # atomic_add supports all floating-point and 32/64-bit integer types.
    [("add", dtype) for dtype in ("float16", "bfloat16", "float32", "float64", "int32", "uint32", "int64", "uint64")]
    # atomic_min/max support integer types and f32/f64. Float min/max are
    # implemented in the frontend with integer atomics on the bit pattern.
    + [(op, dtype) for op in ("min", "max") for dtype in ("float32", "float64", "int32", "uint32", "int64", "uint64")]
    # Bitwise atomics only support 32/64-bit integer types.
    + [(op, dtype) for op in ("and", "or", "xor") for dtype in ("int32", "uint32", "int64", "uint64")]
    # The frontend restricts atomic_xchg to f32/f64 and 32/64-bit integers;
    # f16/bf16 are currently only valid for atomic_add.
    + [("xchg", dtype) for dtype in ("float32", "float64", "int32", "uint32", "int64", "uint64")])

_ATOMIC_CAS_DTYPES = (
    "int16",
    "uint16",
    "float16",
    "bfloat16",
    "int32",
    "uint32",
    "float32",
    "int64",
    "uint64",
    "float64",
)


def _make_tensor(dtype_name, shape, fill_value, device):
    """Use signed storage for MUSA dtypes that torch_musa cannot allocate."""
    if dtype_name.startswith("uint"):
        storage_dtype = getattr(torch, "int" + dtype_name[4:])
        tensor = torch.full(shape, fill_value, device=device, dtype=storage_dtype)
        return triton.reinterpret(tensor, getattr(tl, dtype_name)), tensor
    dtype = getattr(torch, dtype_name)
    tensor = torch.full(shape, fill_value, device=device, dtype=dtype)
    return tensor, tensor


def _make_atomic_inputs(op, dtype_name, num_updates, device):
    is_float = dtype_name.startswith("float") or dtype_name == "bfloat16"
    is_unsigned = dtype_name.startswith("uint")

    if op == "add":
        initial, value, expected = 0, 1, num_updates
    elif op == "min":
        initial, value, expected = (100, 7, 7) if is_unsigned else (100, -7, -7)
    elif op == "max":
        initial, value, expected = (0, 7, 7) if is_unsigned else (-100, 7, 7)
    elif op == "and":
        initial, value, expected = (0xFFFF, 0x00F0, 0x00F0)
    elif op == "or":
        initial, value, expected = (0, 0x00F0, 0x00F0)
    elif op == "xor":
        initial, value, expected = (0, 0x00F0, 0x00F0)
    elif op == "xchg":
        initial, value, expected = (0, 7, 7)
    else:
        raise AssertionError(f"unknown atomic op: {op}")

    dst, dst_base = _make_tensor(dtype_name, (1, ), initial, device)
    value, _ = _make_tensor(dtype_name, (1, ), value, device)
    old_values, _ = _make_tensor(dtype_name, (num_updates, ), 0, device)
    expected = torch.full_like(dst_base, expected)
    return dst, value, old_values, expected, is_float


def _make_atomic_cas_inputs(dtype_name, num_updates, device):
    dst, dst_base = _make_tensor(dtype_name, (1, ), 0, device)
    compare, _ = _make_tensor(dtype_name, (1, ), 0, device)
    value, _ = _make_tensor(dtype_name, (1, ), 7, device)
    old_values, _ = _make_tensor(dtype_name, (num_updates, ), 0, device)
    expected = torch.full_like(dst_base, 7)
    return dst, compare, value, old_values, expected


@pytest.mark.parametrize("op, dtype_name", _ATOMIC_CASES)
def test_atomic_rmw(op, dtype_name):
    capability = tuple(torch.musa.get_device_capability())
    if capability not in ((3, 1), (3, 2)):
        pytest.skip("MUSA native floating atomics are currently implemented for PH1/PH1S")

    num_updates = 33
    dst, value, old_values, expected, is_float = _make_atomic_inputs(op, dtype_name, num_updates, "musa")

    _atomic_rmw_kernel.device_caches.clear()
    _atomic_rmw_kernel[(num_updates, )](dst, value, old_values, op, num_warps=1)
    torch.musa.synchronize()

    device = driver.active.get_current_device()
    compiled = list(_atomic_rmw_kernel.device_caches[device][0].values())
    assert len(compiled) == 1
    llir = compiled[0].asm["llir"]
    if op == "add":
        expected_llir_op = "fadd" if is_float else "add"
    elif op in ("min", "max") and dtype_name.startswith("uint"):
        expected_llir_op = "u" + op
    else:
        expected_llir_op = op
    assert f"atomicrmw {expected_llir_op}" in llir
    # Only floating-point atomic_add is currently assigned the explicit MUSA
    # sync scope. Other RMW operations use LLVM's default device scope.
    if op == "add" and is_float:
        assert 'syncscope("musa_device")' in llir
    assert "__mt_atomicAdd_" not in llir

    dst_base = getattr(dst, "base", dst)
    if is_float:
        torch.testing.assert_close(dst_base.cpu(), expected.cpu(), rtol=0.01, atol=0.01)
    else:
        torch.testing.assert_close(dst_base.cpu(), expected.cpu())


@pytest.mark.parametrize("dtype_name", _ATOMIC_CAS_DTYPES)
def test_atomic_cas(dtype_name):
    capability = tuple(torch.musa.get_device_capability())
    if capability not in ((3, 1), (3, 2)):
        pytest.skip("MUSA atomic tests are currently enabled for PH1/PH1S")

    num_updates = 33
    dst, compare, value, old_values, expected = _make_atomic_cas_inputs(dtype_name, num_updates, "musa")

    _atomic_cas_kernel.device_caches.clear()
    _atomic_cas_kernel[(num_updates, )](dst, compare, value, old_values, num_warps=1)
    torch.musa.synchronize()

    device = driver.active.get_current_device()
    compiled = list(_atomic_cas_kernel.device_caches[device][0].values())
    assert len(compiled) == 1
    llir = compiled[0].asm["llir"]
    assert "cmpxchg" in llir

    dst_base = getattr(dst, "base", dst)
    torch.testing.assert_close(dst_base.cpu(), expected.cpu())
