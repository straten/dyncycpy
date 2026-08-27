"""
CuPy-by-Claude Stage 1 (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
pycyc.backend's numpy/CuPy array-backend selection.
"""

import numpy as np
import pytest

import pycyc


def test_get_xp_false_returns_numpy():
    assert pycyc.get_xp(False) is np


def test_get_fft_numpy_matches_scipy_fft():
    import scipy.fft

    assert pycyc.get_fft(np) is scipy.fft


def test_to_device_to_host_roundtrip_numpy():
    arr = np.array([1.0, 2.0, 3.0])
    on_device = pycyc.to_device(arr, np)
    back = pycyc.to_host(on_device)
    np.testing.assert_array_equal(back, arr)
    assert isinstance(back, np.ndarray)


def test_get_xp_true_without_cupy_raises_clear_error(monkeypatch):
    """Simulates a machine with no cupy installed: get_xp(True) must raise
    ImportError with an install hint, not some other/unclear error --
    pycyc itself has no hard dependency on cupy."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("No module named 'cupy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="cupy"):
        pycyc.get_xp(True)


def test_get_xp_true_with_cupy_returns_cupy():
    cp = pytest.importorskip("cupy")
    assert pycyc.get_xp(True) is cp


def test_get_fft_cupy_matches_cupyx_scipy_fft():
    cp = pytest.importorskip("cupy")
    cupyx_scipy_fft = pytest.importorskip("cupyx.scipy.fft")
    assert pycyc.get_fft(cp) is cupyx_scipy_fft
