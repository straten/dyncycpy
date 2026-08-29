"""
Stage 1 of the jitter/degeneracy plan (see
/home/willem/.claude/plans/stateful-dreaming-wall.md): profile-domain
degenerate-direction removal.

Covers the generalization of spectral_shift/spectral_distance
(pycyc/regularization.py) to an explicit `index` array, and the new
epsilon/gain profile template-matching driver, fit_profile_shift
(pycyc/profile.py).
"""

import numpy as np
import pytest

import pycyc

from .helpers import random_complex, real_finite_diff_grad


def test_spectral_shift_default_index_matches_fftfreq():
    """index=None must still mean np.fft.fftfreq(hf.size) -- the existing
    invariant align_to_neighbour's call sites rely on."""
    rng = np.random.default_rng(0)
    hf = random_complex(rng, (32,))
    theta = [0.3, 1.7]

    shifted_default, index_default = pycyc.spectral_shift(theta, hf)
    shifted_explicit, index_explicit = pycyc.spectral_shift(theta, hf, index=np.fft.fftfreq(hf.size))

    np.testing.assert_array_equal(index_default, np.fft.fftfreq(hf.size))
    np.testing.assert_allclose(shifted_default, shifted_explicit)


def test_spectral_distance_gradient_matches_finite_difference_default_index():
    rng = np.random.default_rng(1)
    hf_ref = random_complex(rng, (24,))
    hf = random_complex(rng, (24,))
    theta0 = np.array([0.4, -1.2])

    def merit_only(theta):
        diff, _grad = pycyc.spectral_distance(theta, hf_ref, hf)
        return diff

    _diff, analytic_grad = pycyc.spectral_distance(theta0, hf_ref, hf)
    numeric_grad = real_finite_diff_grad(merit_only, theta0)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-5, atol=1e-5)


def test_spectral_distance_gradient_matches_finite_difference_custom_index():
    """The index this plan actually needs: a harmonic index 0..nharm-1,
    not a wrapped FFT bin frequency."""
    rng = np.random.default_rng(2)
    nharm = 9
    hf_ref = random_complex(rng, (nharm,))
    hf = random_complex(rng, (nharm,))
    index = np.arange(nharm)
    theta0 = np.array([-0.6, 0.25])

    def merit_only(theta):
        diff, _grad = pycyc.spectral_distance(theta, hf_ref, hf, index=index)
        return diff

    _diff, analytic_grad = pycyc.spectral_distance(theta0, hf_ref, hf, index=index)
    numeric_grad = real_finite_diff_grad(merit_only, theta0)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("true_epsilon", [0.0, 0.37, -1.1, 2.4])
@pytest.mark.parametrize("true_gain", [1.0, 2.5])
@pytest.mark.parametrize("fit_gain", [False, True])
def test_fit_profile_shift_recovers_injected_shift_noiseless(true_epsilon, true_gain, fit_gain):
    """Noiseless recovery: s_t built by construction from s_ref with a known
    (epsilon, gain); fit_profile_shift must recover epsilon (and gain, if
    fit_gain) to numerical precision."""
    rng = np.random.default_rng(3)
    nharm = 12
    s_ref = random_complex(rng, (nharm,))
    alpha = np.arange(nharm)

    injected_gain = true_gain if fit_gain else 1.0
    s_t = injected_gain * s_ref * np.exp(1j * alpha * true_epsilon)

    # Noiseless synthetic data: unlike real per-subint fits (see
    # fit_profile_shift's gtol docstring), the gradient at the true optimum
    # really is ~0 here, so it's meaningful -- and necessary, given this
    # test's atol -- to ask BFGS for tighter-than-default precision.
    epsilon, gain, aligned, residual = pycyc.fit_profile_shift(s_ref, s_t, fit_gain=fit_gain, gtol=1e-10)

    np.testing.assert_allclose(epsilon, true_epsilon, atol=1e-6)
    if fit_gain:
        np.testing.assert_allclose(gain, injected_gain, atol=1e-6)
    np.testing.assert_allclose(aligned, s_t, atol=1e-6)
    np.testing.assert_allclose(residual, 0.0, atol=1e-6)


def test_fit_profile_shift_recovers_injected_shift_with_noise():
    """With noise added, the recovered epsilon should still be close to the
    injected value (loose statistical bound, not exact), and the residual's
    power should be close to the injected noise power rather than close to
    zero or close to the full signal power."""
    rng = np.random.default_rng(4)
    nharm = 16
    s_ref = random_complex(rng, (nharm,)) * 5.0  # decent S/N
    alpha = np.arange(nharm)
    true_epsilon = 0.8

    noise_sigma = 0.05
    noise = noise_sigma * random_complex(rng, (nharm,))
    s_t = s_ref * np.exp(1j * alpha * true_epsilon) + noise

    epsilon, gain, aligned, residual = pycyc.fit_profile_shift(s_ref, s_t, fit_gain=False)

    assert abs(epsilon - true_epsilon) < 0.05
    residual_power = np.sum(np.abs(residual) ** 2)
    expected_noise_power = nharm * 2 * noise_sigma**2  # re+im variance
    assert residual_power < 3 * expected_noise_power


def test_fit_profile_shift_matches_independent_grid_search_for_constant_phase_mismatch():
    """A profile that differs from the reference by ONLY a constant
    (alpha-independent) phase is not a valid real-profile shift (see
    fit_profile_shift's docstring) -- with non-uniform |s_ref(alpha)|, the
    weighted-least-squares-optimal linear-ramp approximation to a pure
    constant-phase offset is generically some *small but nonzero* epsilon
    (not exactly 0; there's no reason a random-weighted fit lands exactly on
    the unweighted answer), so this checks fit_profile_shift's result
    against an independent fine grid search rather than asserting epsilon=0
    outright, and separately sanity-bounds it as small relative to a
    radian."""
    rng = np.random.default_rng(5)
    nharm = 10
    s_ref = random_complex(rng, (nharm,))
    s_t = s_ref * np.exp(1j * 0.9)  # constant phase, no alpha dependence
    index = np.arange(nharm)

    epsilon, gain, aligned, residual = pycyc.fit_profile_shift(s_ref, s_t, fit_gain=False)

    grid = np.linspace(-np.pi, np.pi, 20001)
    grid_values = [pycyc.spectral_distance([0.0, e], s_t, s_ref, index=index)[0] for e in grid]
    grid_epsilon = grid[int(np.argmin(grid_values))]

    np.testing.assert_allclose(epsilon, grid_epsilon, atol=2 * (grid[1] - grid[0]))
    assert abs(epsilon) < 0.5  # small-ramp approximation of a pure phase offset, not a real large shift
