"""
Regression test for a fragility found while verifying
dynamic_frequency_response_refactor's behavior-preservation on real data
(see /home/wvanstra/.claude/plans/lovely-forging-mountain.md, Stage 7):
apply_delay_shrinkage_threshold's noise estimate (delay_noise_power_wavefield)
counted "informative" samples via np.count_nonzero's exact bit-equality test.
An array that is mathematically zero in some bins (permanently causally-
zeroed delay columns, or bins already zeroed by a prior shrinkage pass) can
pick up ~1e-14-relative floating-point noise from an unrelated Fourier
round-trip elsewhere in the pipeline (a domain change that is a mathematical
identity but not bit-exact) without becoming physically nonzero -- the exact
count then miscounts such bins as informative, which was found to move the
estimated noise floor (and hence how much of a real wavefield gets shrunk to
zero) by tens of percent on real P2067 data. Fixed by comparing against a
relative-epsilon floor instead of exact zero.
"""

import numpy as np

import pycyc
from pycyc.regularization import apply_delay_shrinkage_threshold, subtract_degenerate_phase
from pycyc.transforms import time2freq

from .helpers import make_cs, random_complex


def _synthetic_wavefield(rng, ndoppler=64, ndelay=32, zero_frac=0.6):
    """A synthetic h(tau;omega)-like array: mostly small values (simulating
    an already-shrinkage-zeroed background) with a sparse set of larger
    "signal" bins survived from a previous shrinkage pass -- exactly the
    kind of array apply_delay_shrinkage_threshold sees on its second
    (updateWavefield) call in the real pipeline."""
    x = 0.05 * (rng.standard_normal((ndoppler, ndelay)) + 1j * rng.standard_normal((ndoppler, ndelay)))
    zero_mask = rng.random((ndoppler, ndelay)) < zero_frac
    x[zero_mask] = 0.0
    signal_mask = (rng.random((ndoppler, ndelay)) < 0.05) & ~zero_mask
    nsignal = np.count_nonzero(signal_mask)
    x[signal_mask] = 10.0 * (rng.standard_normal(nsignal) + 1j * rng.standard_normal(nsignal))
    return x


def test_delay_shrinkage_threshold_robust_to_sub_epsilon_perturbation_of_zeros():
    rng = np.random.default_rng(42)
    x = _synthetic_wavefield(rng)
    zero_mask = x == 0.0
    assert np.count_nonzero(zero_mask) > 0, "test fixture must contain exact zeros"

    # Perturb only the exact zeros, by an amount representative of FFT
    # round-off noise relative to the array's own scale (~1e-14 relative) --
    # everything that was already nonzero is left bit-identical.
    scale = np.max(np.abs(x))
    tiny = 1e-14 * scale
    perturbation = tiny * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    x_perturbed = x.copy()
    x_perturbed[zero_mask] += perturbation[zero_mask]
    assert np.all(x_perturbed[zero_mask] != 0.0), "perturbation must actually break exact-zero equality"

    out = apply_delay_shrinkage_threshold(x.copy(), threshold=1.0, baseline_threshold=2.0)
    out_perturbed = apply_delay_shrinkage_threshold(x_perturbed.copy(), threshold=1.0, baseline_threshold=2.0)

    zero_frac = np.mean(out == 0.0)
    zero_frac_perturbed = np.mean(out_perturbed == 0.0)
    assert abs(zero_frac - zero_frac_perturbed) < 1e-3, (
        f"sub-epsilon perturbation of exact zeros changed the shrinkage zero-fraction "
        f"from {zero_frac} to {zero_frac_perturbed} -- the noise estimate is not robust "
        f"to floating-point-level noise in already-zeroed bins"
    )
    np.testing.assert_allclose(
        np.sum(np.abs(out) ** 2), np.sum(np.abs(out_perturbed) ** 2), rtol=1e-6
    )


# --- subtract_degenerate_phase ----------------------------------------------
#
# Regression tests for two bugs found investigating why cycsolve.py's
# subtract_degenerate_projections made real-data convergence measurably
# worse. The function used to be called subtract_degenerate_delay_and_phase
# and also removed a "rigid delay shift" direction, under the assumption
# (stated in cycsolve.py's own comment) that it was, like phase, an exact
# null direction of the per-subint merit. It is not (confirmed by finite
# difference: dM/dtau != 0 -- a delay shift of H is degenerate with a
# corresponding phase-vs-harmonic rotation of the *profile*, not with
# holding the profile fixed, which is what this function's caller does) --
# so removing any part of the gradient along that direction discards real
# information, however carefully the removal is done. Separately, even for
# the genuinely-degenerate phase direction, the pre-fix code removed the
# *full complex* projection coefficient instead of only its real part,
# which additionally discards a genuine, non-degenerate amplitude-direction
# gradient component. See subtract_degenerate_phase's docstring for the
# full derivation of both.


def _old_buggy_subtract_degenerate_delay_and_phase(h_delay_grad, h_delay):
    """The pre-fix implementation, reproduced here (not imported -- the
    package version is now fixed, and no longer touches delay at all)
    purely so the tests below can demonstrate the fix changes behavior, and
    how."""
    from pycyc.transforms import freq2time

    h_freq_grad = time2freq(h_delay_grad)
    h_freq = time2freq(h_delay)
    nchan = h_freq.shape[0]

    v_phase = 1.0j * h_freq
    v_phase /= np.sqrt(np.sum(np.abs(v_phase) ** 2))
    v_delay = 1.0j * np.fft.fftfreq(nchan) * h_freq
    v_delay /= np.sqrt(np.sum(np.abs(v_delay) ** 2))

    a_phase = np.sum(np.conj(v_phase) * h_freq_grad)
    h_freq_grad = h_freq_grad - a_phase * v_phase

    a_delay = np.sum(np.conj(v_delay) * h_freq_grad)
    h_freq_grad = h_freq_grad - a_delay * v_delay

    return freq2time(h_freq_grad)


def _phase_basis_vector(h_delay):
    h_freq = time2freq(h_delay)
    v_phase = 1.0j * h_freq
    v_phase /= np.sqrt(np.sum(np.abs(v_phase) ** 2))
    return v_phase


def _delay_basis_vector(h_delay, nchan):
    h_freq = time2freq(h_delay)
    v_delay = 1.0j * np.fft.fftfreq(nchan) * h_freq
    v_delay /= np.sqrt(np.sum(np.abs(v_delay) ** 2))
    return v_delay


def test_subtract_degenerate_phase_removes_real_projection_completely():
    """The result must have exactly zero *real* projection onto v_phase --
    the one genuinely degenerate direction."""
    rng = np.random.default_rng(1)
    nchan = 32
    h_delay = random_complex(rng, (nchan,))
    h_delay_grad = random_complex(rng, (nchan,))
    v_phase = _phase_basis_vector(h_delay)

    result_freq = time2freq(subtract_degenerate_phase(h_delay_grad.copy(), h_delay))

    np.testing.assert_allclose(np.real(np.sum(np.conj(v_phase) * result_freq)), 0.0, atol=1e-9)


def test_subtract_degenerate_phase_preserves_genuine_amplitude_gradient_component():
    """The pre-fix code removed a component along i*v_phase = -h_freq (an
    amplitude-rescaling direction) that is *not* degenerate -- confirmed
    separately by finite difference (dM/dt != 0 along it, unlike the true
    real-phase direction). The fixed version must leave Im(<v_phase, g>)
    (the part corresponding to that direction) untouched."""
    rng = np.random.default_rng(2)
    nchan = 32
    h_delay = random_complex(rng, (nchan,))
    h_delay_grad = random_complex(rng, (nchan,))
    v_phase = _phase_basis_vector(h_delay)

    grad_freq_before = time2freq(h_delay_grad)
    im_before = np.imag(np.sum(np.conj(v_phase) * grad_freq_before))

    result_freq = time2freq(subtract_degenerate_phase(h_delay_grad.copy(), h_delay))
    im_after = np.imag(np.sum(np.conj(v_phase) * result_freq))

    assert abs(im_before) > 1e-3, "test fixture too weak: no imaginary-projection component to preserve"
    np.testing.assert_allclose(im_after, im_before, rtol=1e-6, atol=1e-9)

    # the pre-fix version does *not* preserve this: it removes the full
    # complex a_phase*v_phase, zeroing <v_phase,g> exactly right after that
    # step -- and while its second (v_delay-removal) step can partially
    # reintroduce a component there (non-orthogonality, see the "removes
    # both real projections" bug this module's docstring describes), it
    # does not reconstruct the original im_before, so the two results
    # still differ measurably.
    old_result_freq = time2freq(_old_buggy_subtract_degenerate_delay_and_phase(h_delay_grad.copy(), h_delay))
    im_after_old = np.imag(np.sum(np.conj(v_phase) * old_result_freq))
    assert abs(im_after_old - im_before) > 1e-3, (
        "test fixture too weak: the pre-fix bug didn't measurably change "
        "Im(<v_phase,g>), so this comparison isn't actually distinguishing "
        "the two implementations"
    )


def test_subtract_degenerate_phase_delay_projection_shifts_only_via_nonorthogonality():
    """v_phase and v_delay are not orthogonal (their real inner product is
    the wavefield's power-weighted mean delay), so removing v_phase's real
    projection necessarily shifts v_delay's own projection too -- by
    exactly a_phase*dot12 (both real), the unavoidable, purely real
    consequence of an orthogonal-in-neither-direction basis. That's not a
    bug: subtract_degenerate_phase must not apply any *further*, deliberate
    correction targeting v_delay specifically (the pre-fix code's mistake)
    on top of that unavoidable shift."""
    rng = np.random.default_rng(5)
    nchan = 32
    h_delay = random_complex(rng, (nchan,))
    h_delay_grad = random_complex(rng, (nchan,))
    v_phase = _phase_basis_vector(h_delay)
    v_delay = _delay_basis_vector(h_delay, nchan)
    dot12 = np.sum(np.conj(v_phase) * v_delay)
    np.testing.assert_allclose(np.imag(dot12), 0.0, atol=1e-9)  # this inner product is always real

    grad_freq_before = time2freq(h_delay_grad)
    proj_before = np.sum(np.conj(v_delay) * grad_freq_before)
    a_phase = np.real(np.sum(np.conj(v_phase) * grad_freq_before))

    result_freq = time2freq(subtract_degenerate_phase(h_delay_grad.copy(), h_delay))
    proj_after = np.sum(np.conj(v_delay) * result_freq)

    assert abs(a_phase * np.real(dot12)) > 1e-3, "test fixture too weak: no real shift to check"
    np.testing.assert_allclose(proj_after, proj_before - a_phase * dot12, rtol=1e-6, atol=1e-9)


def test_subtract_degenerate_phase_true_gradient_projection_already_near_zero():
    """Ground the phase fix in the real merit: the *true* Wirtinger
    gradient (from cyclic_merit_and_grad, not hand-constructed) must
    already have ~zero real projection onto v_phase -- it's an exact null
    direction of the merit -- so subtract_degenerate_phase should be close
    to a no-op on it."""
    rng = np.random.default_rng(3)
    nchan, nharm, nlag = 32, 8, 32
    bw, ref_freq = 1.0, 1e5
    params = make_cs(nchan, nharm, nlag, bw, ref_freq, include_Nyquist=False)
    s0 = random_complex(rng, (nharm,))
    cs_data = random_complex(rng, (nchan, nharm))
    ht0 = random_complex(rng, (nlag,))

    _merit, grad, _nonzero = pycyc.cyclic_merit_and_grad(ht0, params, s0, cs_data)
    v_phase = _phase_basis_vector(ht0)
    grad_freq = time2freq(grad)

    np.testing.assert_allclose(np.real(np.sum(np.conj(v_phase) * grad_freq)), 0.0, atol=1e-9)

    fixed_result = time2freq(subtract_degenerate_phase(grad.copy(), ht0))
    np.testing.assert_allclose(fixed_result, grad_freq, rtol=1e-6, atol=1e-9)


def test_delay_shift_is_not_actually_degenerate():
    """Direct evidence for the reason v_delay is no longer touched: a rigid
    delay shift of H (real tau) must change the true merit (dM/dtau != 0),
    unlike a real phase rotation (dM/dphi == 0 exactly) -- confirmed by
    finite difference against the real cyclic_merit_and_grad, not assumed."""
    rng = np.random.default_rng(6)
    nchan, nharm, nlag = 24, 6, 24
    bw, ref_freq = 1.0, 1e5
    params = make_cs(nchan, nharm, nlag, bw, ref_freq, include_Nyquist=False)
    s0 = random_complex(rng, (nharm,))
    cs_data = random_complex(rng, (nchan, nharm))
    ht0 = random_complex(rng, (nlag,))

    def merit_at(ht):
        m, _g, _n = pycyc.cyclic_merit_and_grad(ht, params, s0, cs_data)
        return m

    merit0 = merit_at(ht0)
    eps = 1e-6

    hf0 = pycyc.time2freq(ht0)
    nu = np.fft.fftfreq(nchan)
    ht_delay_pert = pycyc.freq2time(hf0 * np.exp(1j * 2 * np.pi * nu * eps))
    dM_dtau = (merit_at(ht_delay_pert) - merit0) / eps

    ht_phase_pert = ht0 * np.exp(1j * eps)
    dM_dphi = (merit_at(ht_phase_pert) - merit0) / eps

    assert abs(dM_dphi) < 1e-3, f"phase should be an exact null direction, got dM/dphi={dM_dphi}"
    assert abs(dM_dtau) > 1.0, f"delay should NOT be a null direction, got dM/dtau={dM_dtau}"
