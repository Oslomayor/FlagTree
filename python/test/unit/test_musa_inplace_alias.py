from types import SimpleNamespace

from triton.backends.compiler import GPUTarget
from triton.runtime.jit import _make_pointer_alias_spec, _musa_target_capability


class _Param:

    def __init__(self, is_constexpr=False):
        self.is_constexpr = is_constexpr


class _Ptr:

    def __init__(self, value):
        self.value = value

    def data_ptr(self):
        return self.value


def test_make_pointer_alias_spec_skips_constexprs_and_groups_equal_pointers():
    params = [_Param(), _Param(is_constexpr=True), _Param(), _Param()]
    values = [_Ptr(100), SimpleNamespace(), _Ptr(200), _Ptr(100)]

    assert _make_pointer_alias_spec(params, values) == "0:2"


def test_musa_target_capability_accepts_numeric_and_ph1_arches():
    assert _musa_target_capability(GPUTarget("musa", "ph1", 32)) == 31
    assert _musa_target_capability(GPUTarget("musa", "31", 32)) == 31
    assert _musa_target_capability(GPUTarget("musa", 31, 32)) == 31
    assert _musa_target_capability(GPUTarget("cuda", 90, 32)) is None
