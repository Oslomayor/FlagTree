from pathlib import Path

from triton.backends.mthreads import driver as musa_driver


def _clear_musa_driver_path_caches():
    musa_driver._musa_prefix_dirs.cache_clear()
    musa_driver._musa_include_dirs.cache_clear()
    musa_driver._libmusa_dirs.cache_clear()


def _make_sdk(root: Path) -> tuple[Path, Path]:
    include = root / "include"
    lib = root / "lib"
    include.mkdir(parents=True)
    lib.mkdir(parents=True)
    (include / "musa.h").write_text("")
    (lib / "libmusa.so").write_text("")
    return include, lib


def test_musa_driver_paths_default_to_usr_local_musa(tmp_path, monkeypatch):
    default_include, default_lib = _make_sdk(tmp_path / "default-musa")

    monkeypatch.setattr(musa_driver, "_DEFAULT_MUSA_PREFIX", str(default_include.parent))
    monkeypatch.delenv("TRITON_MUSA_INCLUDE_PATH", raising=False)
    monkeypatch.delenv("TRITON_LIBMUSA_PATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(musa_driver.subprocess, "check_output", lambda *args, **kwargs: b"")
    _clear_musa_driver_path_caches()

    try:
        assert str(default_include) in musa_driver._musa_include_dirs()
        assert musa_driver._libmusa_dirs() == [str(default_lib)]
    finally:
        _clear_musa_driver_path_caches()


def test_musa_driver_paths_prefer_explicit_triton_env(tmp_path, monkeypatch):
    default_include, _ = _make_sdk(tmp_path / "default-musa")
    explicit_include, explicit_lib = _make_sdk(tmp_path / "explicit-musa")

    monkeypatch.setattr(musa_driver, "_DEFAULT_MUSA_PREFIX", str(default_include.parent))
    monkeypatch.setenv("TRITON_MUSA_INCLUDE_PATH", str(explicit_include))
    monkeypatch.setenv("TRITON_LIBMUSA_PATH", str(explicit_lib))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(musa_driver.subprocess, "check_output", lambda *args, **kwargs: b"")
    _clear_musa_driver_path_caches()

    try:
        include_dirs = musa_driver._musa_include_dirs()
        assert include_dirs.index(str(explicit_include)) < include_dirs.index(str(default_include))
        assert musa_driver._libmusa_dirs() == [str(explicit_lib)]
    finally:
        _clear_musa_driver_path_caches()


def test_musa_driver_explicit_lib_path_is_exclusive(tmp_path, monkeypatch):
    _, default_lib = _make_sdk(tmp_path / "default-musa")
    _, explicit_lib = _make_sdk(tmp_path / "explicit-musa")

    monkeypatch.setattr(musa_driver, "_DEFAULT_MUSA_PREFIX", str(default_lib.parent))
    monkeypatch.setenv("TRITON_LIBMUSA_PATH", str(explicit_lib))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(default_lib))
    monkeypatch.setattr(
        musa_driver.subprocess,
        "check_output",
        lambda *args, **kwargs: f"libmusa.so.1 => {default_lib / 'libmusa.so'}\n".encode(),
    )
    _clear_musa_driver_path_caches()

    try:
        assert musa_driver._libmusa_dirs() == [str(explicit_lib)]
    finally:
        _clear_musa_driver_path_caches()
