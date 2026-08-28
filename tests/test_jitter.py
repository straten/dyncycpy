"""
Stage 3 of the jitter/degeneracy plan (see
/home/willem/.claude/plans/stateful-dreaming-wall.md): reduced-rank (PCA)
jitter model, pycyc/jitter.py.

Monte-Carlo recovery tests -- a different style from the finite-difference
checks used elsewhere in this suite, appropriate for genuinely statistical/
modeling code rather than deterministic gradient math: construct synthetic
data with known injected structure (a rigid epsilon/gain shift, a known
low-rank jitter basis and weights, known-variance noise) and confirm the
pipeline recovers it to the expected statistical accuracy.
"""

import numpy as np
import pytest

import pycyc

from .helpers import random_complex


def _orthonormal_basis(rng, rank, nharm):
    """rank random orthonormal complex row vectors, via QR of a random matrix."""
    m = random_complex(rng, (nharm, rank))
    q, _r = np.linalg.qr(m)
    return q.T  # (rank, nharm)


def test_compute_residuals_recovers_injected_epsilon_and_gain():
    rng = np.random.default_rng(0)
    nsubint, nharm = 20, 14
    s0 = random_complex(rng, (nharm,)) * 3.0
    alpha = np.arange(nharm)

    true_epsilon = rng.uniform(-1.0, 1.0, nsubint)
    true_gain = rng.uniform(0.5, 2.0, nsubint)
    s_t_all = true_gain[:, None] * s0[None, :] * np.exp(1j * np.outer(true_epsilon, alpha))

    epsilon_t, gain_t, residuals = pycyc.compute_residuals(s0, s_t_all, fit_gain=True)

    np.testing.assert_allclose(epsilon_t, true_epsilon, atol=1e-5)
    np.testing.assert_allclose(gain_t, true_gain, atol=1e-5)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-5)


def test_fit_jitter_basis_finds_no_significant_rank_in_pure_noise():
    """With no injected low-rank structure -- residuals are pure i.i.d.
    noise at the assumed variance -- the Marchenko-Pastur threshold should
    (with high probability, this is a statistical test) find rank 0."""
    rng = np.random.default_rng(1)
    nsubint, nharm = 60, 10
    noise_sigma = 0.3
    # random_complex = randn + i*randn -> E[|z|^2] = 2 per unit noise_sigma;
    # noise_variance_per_harmonic is defined (see fit_jitter_basis) as the
    # total complex power E[|noise|^2], matching CyclicSolver.cyclic_variance's
    # convention -- not the per-real/imaginary-component variance.
    noise_variance_per_harmonic = np.full(nharm, 2 * noise_sigma**2)
    residuals = noise_sigma * random_complex(rng, (nsubint, nharm))

    rank, weights, basis, eigenvalues, threshold, full_basis = pycyc.fit_jitter_basis(
        residuals, noise_variance_per_harmonic
    )

    assert rank == 0
    assert weights.shape == (nsubint, 0)
    assert basis.shape == (0, nharm)
    assert eigenvalues.max() < threshold * 1.5  # comfortably below, not just barely
    assert full_basis.shape == (min(nsubint, nharm), nharm)


def test_fit_jitter_basis_recovers_injected_low_rank_structure():
    rng = np.random.default_rng(2)
    nsubint, nharm, true_rank = 80, 16, 2
    noise_sigma = 0.05
    noise_variance_per_harmonic = np.full(nharm, 2 * noise_sigma**2)  # see convention note above

    true_basis = _orthonormal_basis(rng, true_rank, nharm) * 5.0  # well above the noise floor
    true_weights = random_complex(rng, (nsubint, true_rank))
    true_jitter = true_weights @ true_basis
    noise = noise_sigma * random_complex(rng, (nsubint, nharm))
    residuals = true_jitter + noise

    rank, weights, basis, eigenvalues, threshold, full_basis = pycyc.fit_jitter_basis(
        residuals, noise_variance_per_harmonic
    )

    assert rank == true_rank
    assert full_basis.shape == (min(nsubint, nharm), nharm)
    np.testing.assert_array_equal(full_basis[:rank], basis)

    # basis vectors are only defined up to a unitary rotation within the
    # retained subspace -- compare subspaces (projector), not raw vectors
    recovered_jitter = weights @ basis
    # the reconstructed jitter should explain most of the injected signal power
    residual_after = true_jitter - recovered_jitter
    assert np.sum(np.abs(residual_after) ** 2) < 0.1 * np.sum(np.abs(true_jitter) ** 2)


def test_reconstruct_jittered_profile_matches_injected_noiseless_model():
    rng = np.random.default_rng(3)
    nsubint, nharm, true_rank = 50, 12, 2
    s0 = random_complex(rng, (nharm,)) * 4.0
    alpha = np.arange(nharm)

    true_epsilon = rng.uniform(-0.5, 0.5, nsubint)
    true_gain = rng.uniform(0.8, 1.2, nsubint)
    true_basis = _orthonormal_basis(rng, true_rank, nharm) * 4.0
    true_weights = random_complex(rng, (nsubint, true_rank)) * 0.3

    aligned = true_gain[:, None] * s0[None, :] * np.exp(1j * np.outer(true_epsilon, alpha))
    s_t_all = aligned + true_weights @ true_basis  # noiseless

    epsilon_t, gain_t, residuals = pycyc.compute_residuals(s0, s_t_all, fit_gain=True)

    # near-noiseless (not exactly 0, to keep the whitening well-conditioned)
    # -> any real structure is "significant"
    noise_variance_per_harmonic = np.full(nharm, 1e-4)
    rank, weights, basis, _eig, _thr, _full_basis = pycyc.fit_jitter_basis(residuals, noise_variance_per_harmonic)
    assert rank >= true_rank

    reconstructed = pycyc.reconstruct_jittered_profile(s0, epsilon_t, gain_t, weights, basis)

    # Not machine precision: fit_profile_shift's epsilon/gain fit runs
    # against jitter-contaminated data (it has no way to know the jitter
    # component is there), so the fitted (epsilon_t, gain_t) carry a small
    # bias relative to the injected values; the PCA step then absorbs most
    # but not quite all of the resulting small residual mismatch. A ~1%
    # relative reconstruction error from that chain of two approximate fits
    # is the expected, correct outcome here, not evidence of a bug --
    # test_compute_residuals_recovers_injected_epsilon_and_gain (jitter-free
    # data) already separately confirms the epsilon/gain fit alone is
    # accurate to 1e-5.
    np.testing.assert_allclose(reconstructed, s_t_all, atol=2e-2, rtol=2e-2)


def test_estimate_significant_harmonic_cutoff_finds_injected_signal_extent():
    """A real pulsar profile's harmonic power is concentrated at low
    harmonics; estimate_significant_harmonic_cutoff should find roughly
    where it ends, given a ph_ref with real signal only in [1, true_max]
    and pure noise (at the assumed variance) everywhere else."""
    rng = np.random.default_rng(4)
    nharm, nsubint = 200, 236
    true_signal_max = 30
    noise_variance_per_harmonic = np.full(nharm, 2.0)
    pooled_noise_sigma = np.sqrt(noise_variance_per_harmonic / nsubint / 2)

    ph_ref = pooled_noise_sigma * random_complex(rng, nharm)  # pure noise everywhere
    # inject strong signal (20 pooled-sigma, unambiguously significant) at low harmonics
    ph_ref[1 : true_signal_max + 1] += 20 * pooled_noise_sigma[1 : true_signal_max + 1]

    guard_band = 5
    h_noise_start = pycyc.estimate_significant_harmonic_cutoff(
        ph_ref, noise_variance_per_harmonic, nsubint, guard_band=guard_band
    )

    # should land just past true_signal_max + guard_band, with a little
    # slack for the rare spurious noise harmonic exceeding the (strict,
    # Bonferroni-corrected) significance threshold by chance
    assert true_signal_max < h_noise_start <= true_signal_max + guard_band + 5


def test_estimate_significant_harmonic_cutoff_pure_noise_stays_small():
    """With no real signal anywhere, the whole spectrum is a valid noise
    band -- h_noise_start should sit right at guard_band (no significant
    harmonic found at all), not grow to cover most of the spectrum."""
    rng = np.random.default_rng(5)
    nharm, nsubint = 100, 200
    noise_variance_per_harmonic = np.full(nharm, 1.0)
    pooled_noise_sigma = np.sqrt(noise_variance_per_harmonic / nsubint / 2)
    ph_ref = pooled_noise_sigma * random_complex(rng, nharm)

    guard_band = 5
    h_noise_start = pycyc.estimate_significant_harmonic_cutoff(
        ph_ref, noise_variance_per_harmonic, nsubint, guard_band=guard_band
    )

    assert h_noise_start <= guard_band + 3


def test_calibrate_noise_variance_recovers_injected_scale_factor():
    """calibrate_noise_variance should recover a known injected
    mis-calibration factor from pure-noise residuals in the noise-only
    band, to within normal statistical accuracy for the sample size."""
    rng = np.random.default_rng(6)
    nsubint, nharm = 236, 200
    h_noise_start = 50
    assumed_noise_variance_per_harmonic = np.full(nharm, 1.0)
    true_kappa = 2.5

    sigma = np.sqrt(true_kappa * assumed_noise_variance_per_harmonic / 2)
    residuals = sigma[np.newaxis, :] * random_complex(rng, (nsubint, nharm))

    kappa, corrected = pycyc.calibrate_noise_variance(
        residuals, assumed_noise_variance_per_harmonic, h_noise_start
    )

    assert kappa == pytest.approx(true_kappa, rel=0.15)
    np.testing.assert_allclose(corrected, kappa * assumed_noise_variance_per_harmonic)


def test_calibrate_noise_variance_empty_band_returns_unmodified():
    """If h_noise_start leaves no noise-only band (e.g. the profile's
    real signal fills the whole harmonic range), there's nothing to
    calibrate against -- return the input noise model unchanged."""
    nharm = 50
    noise_variance_per_harmonic = np.full(nharm, 3.3)
    residuals = np.zeros((10, nharm), dtype=complex)

    kappa, corrected = pycyc.calibrate_noise_variance(residuals, noise_variance_per_harmonic, nharm)

    assert kappa == 1.0
    np.testing.assert_array_equal(corrected, noise_variance_per_harmonic)


def test_fit_jitter_basis_calibrated_suppresses_rank_from_underestimated_noise():
    """The central claim behind fit_jitter_basis_calibrated: if the
    assumed noise_variance_per_harmonic underestimates the true noise
    (plausible in practice -- CyclicSolver._refresh_jitter_model's
    sigma_D^2 comes from a single representative subint, not an average),
    the uncalibrated Marchenko-Pastur threshold spuriously retains rank
    from pure noise; calibrating against a known-noise-only harmonic band
    (identified from ph_ref) should correct for it."""
    rng = np.random.default_rng(7)
    nsubint, nharm = 236, 200
    true_signal_max = 20
    true_kappa = 2.0

    assumed_noise_variance_per_harmonic = np.full(nharm, 1.0)
    true_noise_variance_per_harmonic = true_kappa * assumed_noise_variance_per_harmonic

    # pure noise residuals (no real jitter structure) at the *true* variance
    sigma = np.sqrt(true_noise_variance_per_harmonic / 2)
    residuals = sigma[np.newaxis, :] * random_complex(rng, (nsubint, nharm))

    # ph_ref: real signal in [1, true_signal_max], pure noise elsewhere,
    # needed to identify the noise-only band for calibration
    pooled_noise_sigma = np.sqrt(true_noise_variance_per_harmonic / nsubint / 2)
    ph_ref = pooled_noise_sigma * random_complex(rng, nharm)
    ph_ref[1 : true_signal_max + 1] += 20 * pooled_noise_sigma[1 : true_signal_max + 1]

    # uncalibrated: whitening trusts the (wrong, too-small) assumed
    # variance, so the whitened residuals are too "loud" relative to the
    # fixed MP threshold -- spurious rank
    rank_uncalibrated, *_ = pycyc.fit_jitter_basis(residuals, assumed_noise_variance_per_harmonic)
    assert rank_uncalibrated > 10  # clearly, substantially spurious

    (
        rank_calibrated,
        _weights,
        _basis,
        _eigenvalues,
        _threshold,
        _full_basis,
        h_noise_start,
        kappa,
    ) = pycyc.fit_jitter_basis_calibrated(ph_ref, residuals, assumed_noise_variance_per_harmonic, nsubint)

    assert true_signal_max < h_noise_start < nharm
    assert kappa == pytest.approx(true_kappa, rel=0.2)
    assert rank_calibrated < rank_uncalibrated
    assert rank_calibrated <= 3  # small allowance for finite-sample false positives, not exactly 0


def test_fit_jitter_basis_calibrated_recovers_injected_rank_when_noise_correct():
    """When the assumed noise model is already correct (kappa ~= 1), the
    calibrated fit should behave the same as the uncalibrated one --
    calibration shouldn't distort a good result, only correct a bad one.

    The injected jitter structure is deliberately confined to the same
    low-harmonic region as the injected profile signal (true_signal_max),
    matching the physical premise this calibration relies on: genuine
    jitter, like the profile itself, is a real pulse-shape feature and so
    should also be concentrated at low harmonics, not broadband. A basis
    with broadband support (e.g. _orthonormal_basis's generic random
    directions across all nharm) would leak into the nominally
    noise-only calibration band and bias kappa upward -- that's a
    limitation of the method worth knowing about, not tested here.
    """
    rng = np.random.default_rng(8)
    nsubint, nharm, true_rank = 236, 200, 3
    noise_variance_per_harmonic = np.full(nharm, 1.0)
    true_signal_max = 20

    true_basis_low_harm = _orthonormal_basis(rng, true_rank, true_signal_max) * 5.0
    true_basis = np.zeros((true_rank, nharm), dtype=complex)
    true_basis[:, :true_signal_max] = true_basis_low_harm
    true_weights = random_complex(rng, (nsubint, true_rank))
    noise = np.sqrt(noise_variance_per_harmonic / 2) * random_complex(rng, (nsubint, nharm))
    residuals = true_weights @ true_basis + noise

    pooled_noise_sigma = np.sqrt(noise_variance_per_harmonic / nsubint / 2)
    ph_ref = pooled_noise_sigma * random_complex(rng, nharm)
    ph_ref[1 : true_signal_max + 1] += 20 * pooled_noise_sigma[1 : true_signal_max + 1]

    rank_uncalibrated, *_ = pycyc.fit_jitter_basis(residuals, noise_variance_per_harmonic)

    rank_calibrated, _weights, _basis, _eig, _thr, _full_basis, _h, kappa = pycyc.fit_jitter_basis_calibrated(
        ph_ref, residuals, noise_variance_per_harmonic, nsubint
    )

    assert kappa == pytest.approx(1.0, abs=0.25)
    assert rank_calibrated == rank_uncalibrated == true_rank
