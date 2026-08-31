"""
Tests for the dynamic_frequency_response_refactor branch (see
/home/wvanstra/.claude/plans/lovely-forging-mountain.md): makes H(nu,T) --
CyclicSolver.h_time_freq, the per-channel frequency response with no
Fourier transform applied to the subintegration axis -- the persistent
state FISTA steps and stores between iterations, transforming to
h(tau;omega)/h(tau;t) only where a regularizer specifically needs that
domain. Grouped here (rather than spread across the existing per-module
test files) so every test this refactor adds is easy to find in one place.
"""

import numpy as np
import pytest

import pycyc

from .helpers import random_complex, wirtinger_finite_diff_grad
from .test_batched_solver import _build_gradient_test_solver


def _minimal_cs(rng, nspec=4, nchan=8):
    """Bare CyclicSolver with just enough state for the h_time_delay <->
    h_time_freq / h_doppler_delay transform-wiring methods -- no cyclic
    spectra, profile, or gradient machinery needed for these."""
    CS = object.__new__(pycyc.CyclicSolver)
    CS.nspec = nspec
    CS.nchan = nchan
    CS.h_time_delay = random_complex(rng, (nspec, nchan))
    return CS


def test_sync_time_freq_from_time_delay_matches_direct_transform():
    rng = np.random.default_rng(101)
    CS = _minimal_cs(rng)
    expected = pycyc.time2freq(CS.h_time_delay, axis=1)

    CS._sync_time_freq_from_time_delay()

    np.testing.assert_array_equal(CS.h_time_freq, expected)


def test_sync_time_freq_round_trips_to_time_delay():
    rng = np.random.default_rng(102)
    CS = _minimal_cs(rng)
    original = CS.h_time_delay.copy()

    CS._sync_time_freq_from_time_delay()
    recovered = pycyc.freq2time(CS.h_time_freq, axis=1)

    np.testing.assert_allclose(recovered, original, rtol=1e-9, atol=1e-9)


def test_doppler_delay_view_matches_direct_transform():
    rng = np.random.default_rng(103)
    CS = _minimal_cs(rng)
    expected = pycyc.time2freq(CS.h_time_delay, axis=0)

    view = CS._doppler_delay_view()

    np.testing.assert_array_equal(view, expected)


def test_doppler_delay_view_round_trips_to_time_delay():
    rng = np.random.default_rng(104)
    CS = _minimal_cs(rng)
    original = CS.h_time_delay.copy()

    view = CS._doppler_delay_view()
    recovered = pycyc.freq2time(view, axis=0)

    np.testing.assert_allclose(recovered, original, rtol=1e-9, atol=1e-9)


def test_doppler_delay_view_is_not_persisted():
    """_doppler_delay_view() is documented as on-demand, not cached --
    mutating its return value must not silently change h_time_delay/
    h_time_freq unless the caller explicitly folds it back."""
    rng = np.random.default_rng(105)
    CS = _minimal_cs(rng)
    original = CS.h_time_delay.copy()

    view = CS._doppler_delay_view()
    view[0, 0] = 12345.0 + 6789.0j

    np.testing.assert_array_equal(CS.h_time_delay, original)


def test_h_doppler_delay_property_matches_doppler_delay_view():
    rng = np.random.default_rng(106)
    CS = _minimal_cs(rng)

    np.testing.assert_array_equal(CS.h_doppler_delay, CS._doppler_delay_view())


def test_h_doppler_delay_property_tracks_h_time_delay_live():
    """Stage 6: h_doppler_delay is a read-only property computed on demand
    from h_time_delay, not persistent state -- it must reflect whatever
    h_time_delay currently holds, even if that changes after the CS is
    built, with no separate sync step."""
    rng = np.random.default_rng(107)
    CS = _minimal_cs(rng)
    CS.h_time_delay = random_complex(rng, CS.h_time_delay.shape)

    np.testing.assert_array_equal(CS.h_doppler_delay, pycyc.time2freq(CS.h_time_delay, axis=0))


def test_h_doppler_delay_property_is_read_only():
    rng = np.random.default_rng(108)
    CS = _minimal_cs(rng)

    with pytest.raises(AttributeError):
        CS.h_doppler_delay = np.zeros_like(CS.h_time_delay)


def test_compute_first_wavefield_from_best_harmonic_writes_h_time_delay_causally():
    """Stage 6: compute_first_wavefield_from_best_harmonic now builds its
    result in a local h_doppler_delay and only ever writes self.h_time_delay
    (h_doppler_delay is a read-only property, not settable) -- check the
    Doppler-delay-domain invariants the function is documented to enforce
    (acausal half zeroed, total power rescaled back to the pre-call value,
    Parseval-invariant so checkable via either domain) survive that change,
    since a domain mixup in the rewrite (e.g. zeroing the wrong axis) would
    silently violate one of these without necessarily crashing."""
    rng = np.random.default_rng(109)
    CS = _build_gradient_test_solver(rng)
    CS.first_wavefield_from_best_harmonic = CS.nharm
    CS.delay_noise_shrinkage_threshold = None
    CS.noise_smoothing_kernel = None
    initial_total_power = np.sum(np.abs(CS.h_time_delay) ** 2)

    CS.compute_first_wavefield_from_best_harmonic()

    h_doppler_delay = CS.h_doppler_delay
    np.testing.assert_array_equal(h_doppler_delay[:, CS.nchan // 2 :], 0.0)
    np.testing.assert_allclose(np.sum(np.abs(h_doppler_delay) ** 2), initial_total_power, rtol=1e-9)
    np.testing.assert_allclose(
        pycyc.freq2time(h_doppler_delay, axis=0), CS.h_time_delay, rtol=1e-9, atol=1e-9
    )


def test_perturb_initial_wavefield_guess_preserves_causality_and_power():
    """Same rationale as the best-harmonic test above: _perturb_initial_
    wavefield_guess now reads/writes h_time_delay only, via a local
    _doppler_delay_view() -- check its documented invariants (acausal half
    stays zero, total power restored to its pre-perturbation value) still
    hold."""
    rng = np.random.default_rng(110)
    CS = _build_gradient_test_solver(rng)
    CS.h_time_delay[:, CS.nchan // 2 :] = 0.0  # establish the causal precondition the method assumes
    CS.initial_guess_noise_perturbation_rms = 0.5
    initial_total_power = np.sum(np.abs(CS.h_time_delay) ** 2)

    CS._perturb_initial_wavefield_guess()

    h_doppler_delay = CS.h_doppler_delay
    np.testing.assert_array_equal(h_doppler_delay[:, CS.nchan // 2 :], 0.0)
    np.testing.assert_allclose(np.sum(np.abs(h_doppler_delay) ** 2), initial_total_power, rtol=1e-9)


def _build_updatewavefield_test_solver(rng, nsubint=3):
    """_build_gradient_test_solver (test_batched_solver.py) sets up cyclic
    spectra/profile state but deliberately not the regularization flags
    updateWavefield reads (its own tests call compute_gradient(_batched)
    directly, bypassing updateWavefield's orchestration) -- add the
    CyclicSolver.__init__ "off" defaults for all of them, so updateWavefield
    can be exercised end-to-end without a real (PSRFITS-backed) CyclicSolver."""
    CS = _build_gradient_test_solver(rng, nsubint=nsubint)
    # _build_gradient_test_solver hardcodes include_Nyquist=True (matching
    # cycsolve.py's real GPU pipeline, which requires it -- see
    # updateProfile_batched's own guard). Overridden to False here: while
    # investigating this stage's own finite-difference check, discovered
    # that pycyc.cyclic_merit_and_grad's analytic gradient does NOT match a
    # finite-difference of its own merit when include_Nyquist=True (~36%
    # relative error, isolated to the Nyquist harmonic specifically --
    # zeroing just that one harmonic in both cs_data and the profile makes
    # the mismatch vanish entirely). That's a pre-existing bug in the
    # shear/gradient math, confirmed with zero involvement of anything this
    # refactor touches (reproduces identically calling cyclic_merit_and_grad
    # directly, bypassing updateWavefield entirely) -- out of scope for this
    # refactor, but real and worth its own investigation: it affects every
    # real run through cycsolve.py --gpu, which requires include_Nyquist=True.
    # include_Nyquist=False here isolates this test from that unrelated bug
    # so it actually checks what it's meant to (the entry/exit domain
    # transforms), not a confound; updateWavefield's own correctness
    # (matching cyclic_merit_and_grad exactly, whatever it returns) was
    # separately confirmed with include_Nyquist=True too.
    CS.include_Nyquist = False
    CS.noise_threshold = None
    CS.noise_shrinkage_threshold = None
    CS.delay_noise_shrinkage_threshold = None
    CS.delay_noise_selection_threshold = None
    CS.delay_taper = None
    CS.doppler_taper = None
    CS.noise_smoothing_kernel = None
    CS.subtract_degenerate_projections = False
    CS.enforce_causality = False
    CS.reduce_temporal_phase_noise_grad = False
    CS.dump_gradient = False
    CS.zap_gradient_harmonics = 0
    CS.low_pass_filter_Doppler = 1.0
    CS.h_time_freq = pycyc.time2freq(CS.h_time_delay, axis=1)
    return CS


def test_updatewavefield_entry_transform_round_trips_h_time_delay():
    """updateWavefield(h_time_freq) must derive exactly the h_time_delay
    that produced h_time_freq (Stage 2: h_time_freq is now the entry
    point, replacing h_doppler_delay)."""
    rng = np.random.default_rng(201)
    CS = _build_updatewavefield_test_solver(rng)
    original_h_time_delay = CS.h_time_delay.copy()

    CS.updateWavefield(CS.h_time_freq)

    np.testing.assert_allclose(CS.h_time_delay, original_h_time_delay, rtol=1e-9, atol=1e-9)


def test_updatewavefield_mutates_input_array_to_regularized_value():
    """updateWavefield(h_time_freq) must write its regularized result back
    into the caller's h_time_freq array in place, matching pre-refactor
    updateWavefield(h_doppler_delay)'s behavior of operating directly on
    (and np.copyto-mutating) its input parameter with no intervening copy.
    fista.take_fista_step's y_n/x_n rely on this: the *next* FISTA step's
    y_n - alpha*y_grad must see the regularized wavefield evaluate() just
    computed the gradient from, not the pre-regularization one -- without
    this, delay_noise_shrinkage_threshold's noise estimate (re-applied every
    iteration, already fragile to any difference between the array it last
    shrank and the array it's asked to shrink again) was found to diverge
    the whole trajectory to a ~27% worse converged demerit on real P2067
    data within about 10 iterations."""
    rng = np.random.default_rng(203)
    # nsubint=32 (not the default 3): delay_noise_power_wavefield's Doppler-
    # extrema noise strip is 10 samples wide, meaningless for nsubint < ~20.
    # A per-subint-independent (not tiled) wavefield gives the Doppler axis
    # genuine structure to estimate a noise floor from -- _build_gradient_
    # test_solver's own default wavefield is identical across subints
    # (np.tile), which is a Doppler-domain delta function with ~zero power
    # at the extrema strip, so shrinkage never actually zeros anything.
    CS = _build_updatewavefield_test_solver(rng, nsubint=32)
    nsubint, nchan = CS.h_time_delay.shape
    # Sparse structure (mostly small "background" with a few larger
    # "signal" bins), like a real shrinkage-thresholded wavefield -- iid
    # Gaussian noise everywhere (no signal/background contrast) leaves
    # nothing for the shrinkage threshold to actually zero.
    h_doppler_delay = 0.05 * random_complex(rng, (nsubint, nchan))
    signal_mask = rng.random((nsubint, nchan)) < 0.1
    nsignal = np.count_nonzero(signal_mask)
    h_doppler_delay[signal_mask] = 10.0 * random_complex(rng, (nsignal,))
    CS.h_time_delay = pycyc.freq2time(h_doppler_delay, axis=0)
    CS.h_time_freq = pycyc.time2freq(CS.h_time_delay, axis=1)
    CS.delay_noise_shrinkage_threshold = 1.0
    CS.delay_noise_selection_threshold = 2.0

    caller_array = CS.h_time_freq.copy()
    original = caller_array.copy()
    CS.updateWavefield(caller_array)

    # the regularizer must actually have changed something, or this test
    # wouldn't distinguish "mutated in place" from "left untouched" -- check
    # against CS.h_time_freq (the internal, definitely-regularized state),
    # not caller_array: without the fix, caller_array trivially never
    # changes at all, which would make that comparison pass for the wrong
    # reason instead of catching a weak fixture.
    assert not np.array_equal(CS.h_time_freq, original), "shrinkage didn't change anything -- test fixture too weak"
    np.testing.assert_array_equal(caller_array, CS.h_time_freq)


def test_updatewavefield_gradient_matches_finite_difference():
    """Stage 4: updateWavefield now returns dM/dH*(nu,T) (h_time_freq_grad)
    instead of dM/dh*(tau,omega) -- check it end-to-end (through the real
    updateWavefield/compute_gradient_batched/subtract_degenerate_dof/
    causality pipeline, not a reimplementation) against a numerical
    Wirtinger derivative of merit(h_time_freq)."""
    rng = np.random.default_rng(202)
    CS = _build_updatewavefield_test_solver(rng)

    shape = CS.h_time_freq.shape

    def merit_fn(flat_h_time_freq):
        CS.updateWavefield(flat_h_time_freq.reshape(shape))
        return CS.merit

    h0 = CS.h_time_freq.flatten()
    numeric_grad = wirtinger_finite_diff_grad(merit_fn, h0, eps=1e-6)

    CS.updateWavefield(h0.reshape(shape))
    analytic_grad = CS.h_time_freq_grad.flatten()

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-4, atol=1e-4)
