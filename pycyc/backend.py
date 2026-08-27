"""
pycyc.backend - thin numpy/CuPy array-backend selection.

CuPy-by-Claude Stage 1 (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
lets the batched functions in pycyc.transforms/pycyc.objective run on either
backend from explicit parameters, matching this package's existing style of
pure functions rather than reaching into ambient/global state. pycyc itself
must keep importing and working with no GPU/cupy present -- cupy is only
ever imported here, lazily, when a caller actually asks for it.
"""

from __future__ import annotations

__all__ = ["get_xp", "get_fft", "to_device", "to_host"]

import numpy as np
import scipy.fft as _scipy_fft


def get_xp(use_gpu: bool):
    """
    Return the array module to compute with: cupy if use_gpu is True, else
    numpy. Raises ImportError (with an install hint) if use_gpu is True but
    cupy isn't installed -- pycyc has no hard dependency on cupy, so this
    failure only ever surfaces when a caller explicitly opts into the GPU
    path.
    """
    if not use_gpu:
        return np
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "use_gpu=True requires cupy, which is not installed. Install "
            "the package matching your CUDA version, e.g. `pip install "
            "cupy-cuda12x` -- see https://docs.cupy.dev/en/stable/install.html."
        ) from exc
    return cp


def get_fft(xp):
    """
    Return the fft submodule matching the given array module: cupyx.scipy.fft
    for cupy (mirrors scipy.fft's API, including norm="ortho"), else
    scipy.fft.
    """
    if xp is np:
        return _scipy_fft
    import cupyx.scipy.fft as _cupy_fft

    return _cupy_fft


def to_device(arr, xp):
    """asarray onto xp's backend -- a no-op copy-free cast for numpy->numpy."""
    return xp.asarray(arr)


def to_host(arr):
    """
    Bring an array back to a plain numpy array on the host, regardless of
    which backend produced it -- a no-op for arrays that are already numpy
    (including scalars/Python numbers, which have no .get()).
    """
    get = getattr(arr, "get", None)
    if get is not None:
        return get()
    return np.asarray(arr)
