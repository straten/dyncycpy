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
