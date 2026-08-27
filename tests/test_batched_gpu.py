"""
CuPy-by-Claude Stage 3 (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
CyclicSolver.compute_gradient_batched with self.use_gpu=True, on the real
GPU when one's available (importorskip-gated, so the suite still passes
on a machine with no cupy/GPU). Confirms the device path reaches the same
merit/gradient as the CPU path, and specifically validates the accuracy
cost of the default complex64 downcast against complex128 -- not just
assumed acceptable.
"""

import numpy as np
import pytest

import pycyc

from .test_batched_solver import _build_gradient_test_solver
from .test_outer_loop import _build_synthetic_cyclic_solver

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("use_jitter_profiles", [False, True])
def test_gpu_matches_cpu_complex128(use_jitter_profiles):
    """use_gpu=True with gpu_dtype=None (no downcast) should match the CPU
    path to close to full float64 precision -- isolates "does the device
    path compute the right thing" from "how much does complex64 cost"."""
    rng = np.random.default_rng(31)
    CS_cpu = _build_gradient_test_solver(rng, use_jitter_profiles=use_jitter_profiles)
    rng = np.random.default_rng(31)
    CS_gpu = _build_gradient_test_solver(rng, use_jitter_profiles=use_jitter_profiles)
    CS_gpu.use_gpu = True
    CS_gpu.gpu_dtype = None

    CS_cpu.merit = 0
    CS_cpu.nterm_merit = 0
    CS_cpu.compute_gradient()

    CS_gpu.merit = 0
    CS_gpu.nterm_merit = 0
    CS_gpu.compute_gradient_batched()

    np.testing.assert_allclose(CS_gpu.h_time_delay_grad, CS_cpu.h_time_delay_grad, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(CS_gpu.merit, CS_cpu.merit, rtol=1e-8, atol=1e-8)
    assert CS_gpu.nterm_merit == CS_cpu.nterm_merit


def test_gpu_complex64_close_to_cpu_complex128():
    """The default gpu_dtype=complex64 downcast must stay close to the
    complex128 CPU result -- not exact (that's the point of downcasting),
    but within a documented, checked tolerance rather than an unverified
    assumption."""
    rng = np.random.default_rng(37)
    CS_cpu = _build_gradient_test_solver(rng)
    rng = np.random.default_rng(37)
    CS_gpu = _build_gradient_test_solver(rng)
    CS_gpu.use_gpu = True
    CS_gpu.gpu_dtype = np.complex64  # the default, spelled out explicitly here

    CS_cpu.merit = 0
    CS_cpu.nterm_merit = 0
    CS_cpu.compute_gradient()

    CS_gpu.merit = 0
    CS_gpu.nterm_merit = 0
    CS_gpu.compute_gradient_batched()

    # relative tolerance appropriate to complex64's ~7 decimal digits, not
    # complex128's ~15-16 -- loose by float64 standards, deliberately so
    np.testing.assert_allclose(CS_gpu.h_time_delay_grad, CS_cpu.h_time_delay_grad, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(CS_gpu.merit, CS_cpu.merit, rtol=1e-3, atol=1e-3)


def test_gpu_cyclic_spectra_cache_reused_across_calls():
    """self.cyclic_spectra should be transferred to device once and reused,
    not re-transferred every compute_gradient_batched() call -- confirms
    the cache actually caches (same array object returned) rather than
    silently re-uploading every time."""
    rng = np.random.default_rng(41)
    CS = _build_gradient_test_solver(rng)
    CS.use_gpu = True

    CS.merit = 0
    CS.nterm_merit = 0
    CS.compute_gradient_batched()
    cached_after_first = CS._gpu_cyclic_spectra
    assert cached_after_first is not None

    CS.merit = 0
    CS.nterm_merit = 0
    CS.compute_gradient_batched()
    assert CS._gpu_cyclic_spectra is cached_after_first  # same device array, not re-uploaded

    # reassigning self.cyclic_spectra (e.g. a reload) must invalidate the cache
    CS.cyclic_spectra = CS.cyclic_spectra.copy()
    CS.merit = 0
    CS.nterm_merit = 0
    CS.compute_gradient_batched()
    assert CS._gpu_cyclic_spectra is not cached_after_first


def test_gpu_chunking_matches_single_batch():
    """gpu_chunk_size must still give the same answer under use_gpu=True,
    the same invariant test_compute_gradient_batched_chunking_matches_single_batch
    checks on xp=numpy -- confirms chunking composes correctly with the
    device path and its caching, not just with plain numpy slicing."""
    rng = np.random.default_rng(43)
    CS_whole = _build_gradient_test_solver(rng, nsubint=9)
    CS_whole.use_gpu = True
    rng = np.random.default_rng(43)
    CS_chunked = _build_gradient_test_solver(rng, nsubint=9)
    CS_chunked.use_gpu = True
    CS_chunked.gpu_chunk_size = 4

    CS_whole.merit = 0
    CS_whole.nterm_merit = 0
    CS_whole.compute_gradient_batched()

    CS_chunked.merit = 0
    CS_chunked.nterm_merit = 0
    CS_chunked.compute_gradient_batched()

    np.testing.assert_allclose(CS_chunked.h_time_delay_grad, CS_whole.h_time_delay_grad, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(CS_chunked.merit, CS_whole.merit, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("model_gain_variations", [False, True])
def test_update_profile_batched_gpu_complex64_close_to_cpu(model_gain_variations):
    """Stage 5's real-device counterpart to test_gpu_complex64_close_to_cpu_complex128:
    updateProfile_batched under a real self.use_gpu=True (complex64,
    genuine cupy backend) must stay close to the threaded/complex128 CPU
    path -- same rtol=1e-3 rationale (complex64's ~7 decimal digits)."""
    rng = np.random.default_rng(113)
    CS_threaded = _build_synthetic_cyclic_solver(rng, nsubint=6, inject_jitter=False)
    rng = np.random.default_rng(113)
    CS_gpu = _build_synthetic_cyclic_solver(rng, nsubint=6, inject_jitter=False)

    for CS in (CS_threaded, CS_gpu):
        CS.model_gain_variations = model_gain_variations
        if model_gain_variations:
            CS.optimal_gains = np.ones(6)
        CS.updateProfile()  # prime self.pp_scattered via the ordinary path first

    CS_gpu.use_gpu = True  # real cupy backend, gpu_dtype=complex64 (the default)

    CS_threaded.updateProfile()
    CS_gpu.updateProfile()

    np.testing.assert_allclose(CS_gpu.pp_intrinsic, CS_threaded.pp_intrinsic, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(CS_gpu.intrinsic_profiles, CS_threaded.intrinsic_profiles, rtol=1e-3, atol=1e-3)
    if model_gain_variations:
        np.testing.assert_allclose(CS_gpu.optimal_gains, CS_threaded.optimal_gains, rtol=1e-3, atol=1e-3)
