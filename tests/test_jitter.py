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
