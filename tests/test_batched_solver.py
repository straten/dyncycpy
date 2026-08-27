"""
CuPy-by-Claude Stage 2 (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
CyclicSolver.compute_gradient_batched, compared end-to-end against the
existing ThreadPoolExecutor-based compute_gradient on synthetic
multi-subint data, still xp=numpy (self.use_gpu=False) throughout -- the
GPU device path starts Stage 3.
"""

import numpy as np
import pytest

import pycyc


def _build_gradient_test_solver(rng, nsubint=6, use_jitter_profiles=False, use_integrated_profile=True):
    nchan, nphase = 16, 10
    nharm = nphase // 2 + 1
    nlag = nchan

    CS = object.__new__(pycyc.CyclicSolver)
    CS.bw = 1.0
    CS.ref_freq = 1e5
    CS.pad_cyclic_spectra = False
    CS.include_Nyquist = True
    CS.maxharm = None
    CS.exclude_DC = 0
    CS.nlag = nlag
    CS.nchan = nchan
    CS.nharm = nharm
    CS.nphase = nphase
    CS.nspec = nsubint
    CS.nsubint = nsubint
    CS.npol = 1
    CS.nthread = 1
    CS.shear_phasors = pycyc.create_shear_phasors(nchan, nharm, CS.bw, CS.ref_freq)
    CS.dump_residual = False
    CS.iprint = 0
    CS.rindex = 0
    CS.save_cyclic_spectra = True
    CS.use_integrated_profile = use_integrated_profile
    CS.use_gpu = False
    CS.gpu_chunk_size = None

    ht_true = rng.standard_normal(nlag) + 1j * rng.standard_normal(nlag)
    hf_true = pycyc.time2freq(ht_true)
    s0_true = rng.standard_normal(nharm) + 1j * rng.standard_normal(nharm)

    if use_jitter_profiles:
        jitter_basis_true = (rng.standard_normal((2, nharm)) + 1j * rng.standard_normal((2, nharm))) * 0.3
        jitter_weights_true = (rng.standard_normal((nsubint, 2)) + 1j * rng.standard_normal((nsubint, 2))) * 0.5
        s_t_true = s0_true[None, :] + jitter_weights_true @ jitter_basis_true
        CS.jitter_profiles = s_t_true
    else:
        s_t_true = np.tile(s0_true[None, :], (nsubint, 1))
        CS.jitter_profiles = None

    params = CS._model_params()
    cyclic_spectra = np.zeros((nsubint, 1, nchan, nharm), dtype=np.complex128)
    for isub in range(nsubint):
        cs_model, _hp, _hm = pycyc.make_model_cs(params, hf_true, s_t_true[isub])
        noise = 0.02 * (rng.standard_normal((nchan, nharm)) + 1j * rng.standard_normal((nchan, nharm)))
        cyclic_spectra[isub, 0] = cs_model + noise
    CS.cyclic_spectra = cyclic_spectra

    CS.intrinsic_ph = np.tile(s0_true[None, :], (CS.npol, 1))
    CS.ph_ref = CS.intrinsic_ph[0]
    CS.ph_numer = np.zeros((CS.npol, nsubint, nharm), dtype=np.complex128)
    CS.ph_denom = np.ones((CS.npol, nsubint, nharm), dtype=np.complex128)

    # a perturbed (not exact) wavefield guess, same for every subint --
    # what's under test is the gradient there, not convergence
    ht_guess = ht_true + 0.3 * (rng.standard_normal(nlag) + 1j * rng.standard_normal(nlag))
    CS.h_time_delay = np.tile(ht_guess, (nsubint, 1))
    CS.optimal_gains = rng.uniform(0.7, 1.3, size=nsubint)
    CS.h_time_delay_grad = np.zeros((nsubint, nchan), dtype=np.complex128)

    return CS


@pytest.mark.parametrize("use_jitter_profiles", [False, True])
def test_compute_gradient_batched_matches_threaded(use_jitter_profiles):
    rng = np.random.default_rng(11)
    CS_threaded = _build_gradient_test_solver(rng, use_jitter_profiles=use_jitter_profiles)
    rng = np.random.default_rng(11)
    CS_batched = _build_gradient_test_solver(rng, use_jitter_profiles=use_jitter_profiles)

    CS_threaded.merit = 0
    CS_threaded.nterm_merit = 0
    CS_threaded.compute_gradient()

    CS_batched.merit = 0
    CS_batched.nterm_merit = 0
    CS_batched.compute_gradient_batched()

    np.testing.assert_allclose(CS_batched.h_time_delay_grad, CS_threaded.h_time_delay_grad, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(CS_batched.merit, CS_threaded.merit, rtol=1e-9, atol=1e-9)
    assert CS_batched.nterm_merit == CS_threaded.nterm_merit


def test_compute_gradient_batched_matches_threaded_without_integrated_profile():
    """The self.use_integrated_profile=False fallback (per-subint
    ph_numer/ph_denom ratio) is a third s0 source distinct from the
    shared-profile and jitter_profiles cases covered above -- also
    (nsubint, nharm)-shaped like jitter_profiles, but taken from a
    different pair of attributes."""
    rng = np.random.default_rng(5)
    CS_threaded = _build_gradient_test_solver(rng, use_integrated_profile=False)
    rng = np.random.default_rng(5)
    CS_batched = _build_gradient_test_solver(rng, use_integrated_profile=False)

    # give ph_numer/ph_denom a nontrivial per-subint ratio to fit against
    nharm = CS_threaded.nharm
    rng2 = np.random.default_rng(6)
    ratio = rng2.standard_normal((CS_threaded.nsubint, nharm)) + 1j * rng2.standard_normal(
        (CS_threaded.nsubint, nharm)
    )
    CS_threaded.ph_numer[0] = ratio
    CS_batched.ph_numer[0] = ratio

    CS_threaded.merit = 0
    CS_threaded.nterm_merit = 0
    CS_threaded.compute_gradient()

    CS_batched.merit = 0
    CS_batched.nterm_merit = 0
    CS_batched.compute_gradient_batched()

    np.testing.assert_allclose(CS_batched.h_time_delay_grad, CS_threaded.h_time_delay_grad, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(CS_batched.merit, CS_threaded.merit, rtol=1e-9, atol=1e-9)


def test_compute_gradient_batched_chunking_matches_single_batch():
    """gpu_chunk_size < nspec must give the same result as one big batch --
    Stage 3's GPU memory-budget chunking mustn't change the answer, only
    how many calls it takes to get there."""
    rng = np.random.default_rng(21)
    CS_whole = _build_gradient_test_solver(rng, nsubint=9)
    rng = np.random.default_rng(21)
    CS_chunked = _build_gradient_test_solver(rng, nsubint=9)
    CS_chunked.gpu_chunk_size = 4  # doesn't divide nsubint=9 evenly

    CS_whole.merit = 0
    CS_whole.nterm_merit = 0
    CS_whole.compute_gradient_batched()

    CS_chunked.merit = 0
    CS_chunked.nterm_merit = 0
    CS_chunked.compute_gradient_batched()

    np.testing.assert_allclose(CS_chunked.h_time_delay_grad, CS_whole.h_time_delay_grad, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(CS_chunked.merit, CS_whole.merit, rtol=1e-9, atol=1e-9)
    assert CS_chunked.nterm_merit == CS_whole.nterm_merit


def test_compute_gradient_batched_requires_save_cyclic_spectra():
    rng = np.random.default_rng(1)
    CS = _build_gradient_test_solver(rng)
    CS.save_cyclic_spectra = False
    with pytest.raises(NotImplementedError):
        CS.compute_gradient_batched()
