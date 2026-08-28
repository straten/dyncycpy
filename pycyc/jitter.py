"""
pycyc.jitter - reduced-rank (PCA) model of genuine pulse-to-pulse jitter in
the per-subint intrinsic profile, once the epsilon_t/gain_t degenerate
direction (pycyc.profile.fit_profile_shift) has been removed.

Stage 3 of the jitter/degeneracy plan (see
/home/willem/.claude/plans/stateful-dreaming-wall.md): models
S(alpha;T) = g[T] * S0(alpha) * exp(i*alpha*epsilon[T])
             + sum_k w_k[T] * P_k(alpha)
with the rank chosen by comparing the residual covariance's eigenvalues
against the known per-harmonic profile-estimation noise floor (a
Marchenko-Pastur threshold), rather than an arbitrary variance-explained
cutoff. All pure functions of explicit arrays -- no CyclicSolver
dependency -- following the same design as pycyc.objective/pycyc.profile.
"""

from __future__ import annotations

__all__ = [
    "compute_residuals",
    "fit_jitter_basis",
    "estimate_significant_harmonic_cutoff",
    "calibrate_noise_variance",
    "fit_jitter_basis_calibrated",
    "reconstruct_jittered_profile",
    "subspace_principal_angle",
]

import logging

import numpy as np

from .profile import fit_profile_shift

logger = logging.getLogger(__name__)


def compute_residuals(
    s0: np.ndarray, s_t_all: np.ndarray, fit_gain: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit and remove the epsilon_t/gain_t degenerate direction (pycyc.profile.
    fit_profile_shift) from every subint's profile independently.

    Args:
    s0: (nharm,) reference profile.
    s_t_all: (nsubint, nharm) per-subint profile estimates S_t(alpha).
    fit_gain: passed through to fit_profile_shift.

    Returns (epsilon_t, gain_t, residuals): (nsubint,), (nsubint,), and
    (nsubint, nharm) -- residuals is R(alpha;T) in the notation used
    throughout the design discussion.
    """
    nsubint = s_t_all.shape[0]
    epsilon_t = np.zeros(nsubint)
    gain_t = np.ones(nsubint)
    residuals = np.zeros_like(s_t_all)
    for isub in range(nsubint):
        epsilon, gain, _aligned, residual = fit_profile_shift(s0, s_t_all[isub], fit_gain=fit_gain)
        epsilon_t[isub] = epsilon
        gain_t[isub] = gain
        residuals[isub] = residual
    return epsilon_t, gain_t, residuals


def fit_jitter_basis(
    residuals: np.ndarray, noise_variance_per_harmonic: np.ndarray, max_rank: int | None = None
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Fit a reduced-rank basis for the genuine jitter left in `residuals`
    (nsubint, nharm), after whitening by the known per-harmonic noise
    variance and choosing the rank via a Marchenko-Pastur threshold: under
    the null hypothesis that residuals are pure i.i.d. noise at the given
    per-harmonic variance, the eigenvalues of the whitened sample
    covariance matrix (1/nsubint) * whitened^H @ whitened follow the
    Marchenko-Pastur distribution with aspect ratio gamma = nharm/nsubint,
    bounded above by (1 + sqrt(gamma))**2; any eigenvalue exceeding that is
    statistically significant structure, not noise.

    noise_variance_per_harmonic: (nharm,) Var(S_t(alpha)) for a single
    subint's profile estimate at harmonic alpha -- for pycyc's ML profile
    estimator (Eq. A7, a per-harmonic ordinary-least-squares fit against a
    known per-channel design coefficient with i.i.d. per-channel noise
    variance sigma_D^2), this is sigma_D^2 / ph_denom(alpha), with sigma_D^2
    from CyclicSolver.cyclic_variance and ph_denom from
    CyclicSolver.optimize_profile/solve_profile_and_gain -- computing that
    is the caller's job (typically CyclicSolver), not this pure module's.

    Returns (rank, weights, basis, eigenvalues, threshold, full_basis):
    - rank: number of components retained (eigenvalue > threshold).
    - weights: (nsubint, rank) real... complex weights w_k(T).
    - basis: (rank, nharm) complex basis vectors P_k(alpha), in the
      original (non-whitened) profile units, s.t.
      residuals ~= weights @ basis.
    - eigenvalues: (nharm,) or (nsubint,) (whichever is smaller) full
      eigenvalue spectrum of the whitened covariance, largest first -- for
      tracking convergence across outer-loop passes (Stage 4).
    - threshold: the Marchenko-Pastur eigenvalue threshold used.
    - full_basis: (min(nsubint,nharm), nharm) -- every basis vector from
      the SVD, not just the rank-retained subset (i.e. `basis` is
      `full_basis[:rank]`), same physical (non-whitened) units. For
      inspecting the whole eigenspectrum/eigenvectors, e.g. to see what
      the jitter model is doing near a rank boundary or before enough
      significant structure has accumulated to be retained.
    """
    nsubint, nharm = residuals.shape
    sigma = np.sqrt(noise_variance_per_harmonic)
    whitened = residuals / sigma[np.newaxis, :]

    # SVD of the whitened residual matrix directly (numerically preferable
    # to forming the covariance matrix explicitly): whitened = U @ diag(s) @ Vh
    U, s, Vh = np.linalg.svd(whitened, full_matrices=False)
    eigenvalues = (s**2) / nsubint

    gamma = nharm / nsubint
    threshold = (1.0 + np.sqrt(gamma)) ** 2

    rank = int(np.sum(eigenvalues > threshold))
    if max_rank is not None:
        rank = min(rank, max_rank)

    logger.info("fit_jitter_basis: rank = %d", rank)

    weights = U[:, :rank] * s[np.newaxis, :rank]  # whitened-domain weights
    basis = Vh[:rank, :] * sigma[np.newaxis, :]  # un-whitened back to physical profile units
    full_basis = Vh * sigma[np.newaxis, :]  # every SVD component, same un-whitened units

    return rank, weights, basis, eigenvalues, threshold, full_basis


def estimate_significant_harmonic_cutoff(
    ph_ref: np.ndarray,
    noise_variance_per_harmonic: np.ndarray,
    nsubint: int,
    false_alarm_rate: float = 0.01,
    guard_band: int = 5,
) -> int:
    """
    Find the first harmonic beyond which the pooled reference profile
    ph_ref carries no statistically significant power -- i.e. where the
    real pulsar signal (which, like any physically plausible profile
    shape, is concentrated at low harmonics) has decayed into the noise
    floor. Everything from the returned index onward is "non-pulsar"
    harmonic content: real measurement noise, uncontaminated by genuine
    signal, usable as an independent calibration band (see
    calibrate_noise_variance).

    ph_ref: (nharm,) pooled/reference profile in the harmonic domain
    (e.g. phase2harm(pp_intrinsic), DC-zeroed if exclude_DC).
    noise_variance_per_harmonic: (nharm,) Var(S_t(alpha)) for a *single*
    subint's profile estimate (the same quantity fit_jitter_basis takes) --
    ph_ref pools nsubint of these, so its own noise variance is this
    divided by nsubint (exact for equal per-subint weights, the same
    simplifying approximation used elsewhere in this module).
    nsubint: number of subints pooled into ph_ref.
    false_alarm_rate: family-wise false-alarm probability for declaring a
    harmonic significant, Bonferroni-corrected across all nharm harmonics
    tested (i.e. the per-harmonic test uses false_alarm_rate/nharm). For
    a circularly-symmetric complex Gaussian, |z|^2/Var(z) is
    Exponential(mean=1) under the null of no signal, so the per-harmonic
    threshold is -log(per_harmonic_false_alarm_rate).
    guard_band: extra harmonics added past the last significant one,
    to stay clear of any smoothing/leakage into nominally-noise harmonics.

    Returns h_noise_start (int, clipped to nharm): index of the first
    harmonic in the noise-only band. If every harmonic looks significant
    (h_noise_start would exceed nharm), returns nharm -- i.e. an empty
    band, signalling that no calibration is possible from this profile.
    """
    nharm = ph_ref.shape[0]
    pooled_noise_variance = noise_variance_per_harmonic / nsubint
    snr = np.abs(ph_ref) ** 2 / np.maximum(pooled_noise_variance, 1e-300)

    per_harmonic_false_alarm_rate = false_alarm_rate / nharm
    threshold = -np.log(per_harmonic_false_alarm_rate)

    significant = np.flatnonzero(snr > threshold)
    h_signal_max = int(significant.max()) if significant.size > 0 else -1

    return min(h_signal_max + 1 + guard_band, nharm)


def calibrate_noise_variance(
    residuals: np.ndarray,
    noise_variance_per_harmonic: np.ndarray,
    h_noise_start: int,
) -> tuple[float, np.ndarray]:
    """
    Empirically calibrate noise_variance_per_harmonic against the
    observed residual power in a known-signal-free harmonic band (see
    estimate_significant_harmonic_cutoff), correcting for any systematic
    mis-specification of the assumed noise model -- e.g.
    CyclicSolver._refresh_jitter_model's sigma_D^2 comes from a single
    representative subint's cyclic_variance, not an average across all
    subints, so it can be biased if that one subint happens to be
    quieter or noisier than typical.

    residuals: (nsubint, nharm), from compute_residuals.
    noise_variance_per_harmonic: (nharm,), the assumed per-subint noise
    model (same convention as fit_jitter_basis).
    h_noise_start: index of the first harmonic in the noise-only band
    (estimate_significant_harmonic_cutoff's return value).

    Returns (kappa, noise_variance_per_harmonic_corrected):
    - kappa: median ratio of observed to assumed variance over the
      noise-only band -- 1.0 if the assumed model was already correct,
      >1 if it underestimated the true noise (the case that inflates
      fit_jitter_basis's retained rank), <1 if it overestimated it.
      Median rather than mean, to resist any one harmonic's outlier
      fluctuation dominating the estimate.
    - noise_variance_per_harmonic_corrected: kappa * noise_variance_per_
      harmonic (uniformly rescaled -- the noise-only band only
      constrains the overall scale, not per-harmonic shape). Equal to
      the uncorrected input if h_noise_start leaves no noise-only band
      (kappa=1.0 in that case; nothing to calibrate against).
    """
    nharm = residuals.shape[1]
    if h_noise_start >= nharm:
        return 1.0, noise_variance_per_harmonic

    observed_variance = np.mean(np.abs(residuals[:, h_noise_start:]) ** 2, axis=0)
    ratio = observed_variance / np.maximum(noise_variance_per_harmonic[h_noise_start:], 1e-300)
    kappa = float(np.median(ratio))

    return kappa, kappa * noise_variance_per_harmonic


def fit_jitter_basis_calibrated(
    ph_ref: np.ndarray,
    residuals: np.ndarray,
    noise_variance_per_harmonic: np.ndarray,
    nsubint: int,
    max_rank: int | None = None,
    false_alarm_rate: float = 0.01,
    guard_band: int = 5,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int, float]:
    """
    fit_jitter_basis, but with noise_variance_per_harmonic empirically
    recalibrated first against a "non-pulsar" harmonic band -- high
    harmonics where the real pulsar signal (concentrated at low
    harmonics, like any physically plausible profile shape) has decayed
    into the noise floor, identified from the pooled reference profile
    ph_ref (see estimate_significant_harmonic_cutoff). This directly
    measures how big residual-covariance eigenvalues get from noise
    alone in *this* dataset, rather than trusting the Marchenko-Pastur
    formula's idealized-noise assumption outright -- correcting for
    known simplifications elsewhere in the noise model (see
    calibrate_noise_variance).

    Args as fit_jitter_basis, plus ph_ref (pooled reference profile,
    harmonic domain), nsubint (subints pooled into ph_ref),
    false_alarm_rate/guard_band (passed to
    estimate_significant_harmonic_cutoff).

    Returns (rank, weights, basis, eigenvalues, threshold, full_basis,
    h_noise_start, kappa) -- the first six as fit_jitter_basis (rank/
    eigenvalues/threshold are all relative to the *corrected* noise
    model); h_noise_start and kappa are the calibration diagnostics from
    estimate_significant_harmonic_cutoff/calibrate_noise_variance.
    """
    h_noise_start = estimate_significant_harmonic_cutoff(
        ph_ref, noise_variance_per_harmonic, nsubint, false_alarm_rate, guard_band
    )
    kappa, corrected_noise_variance = calibrate_noise_variance(
        residuals, noise_variance_per_harmonic, h_noise_start
    )

    rank, weights, basis, eigenvalues, threshold, full_basis = fit_jitter_basis(
        residuals, corrected_noise_variance, max_rank=max_rank
    )

    logger.info(
        "fit_jitter_basis_calibrated: h_noise_start=%d kappa=%.4g rank=%d",
        h_noise_start,
        kappa,
        rank,
    )

    return rank, weights, basis, eigenvalues, threshold, full_basis, h_noise_start, kappa


def reconstruct_jittered_profile(
    s0: np.ndarray, epsilon_t: np.ndarray, gain_t: np.ndarray, weights: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """
    Rebuild the per-subint jitter-aware profile
    S(alpha;T) = g[T] * S0(alpha) * exp(i*alpha*epsilon[T]) + sum_k w_k[T] * P_k(alpha)
    for use as `s0` in pycyc.objective.cyclic_merit_and_grad -- no changes
    needed there, since it already takes s0 as a plain array parameter.

    Args:
    s0: (nharm,) reference profile.
    epsilon_t, gain_t: (nsubint,) each, from compute_residuals.
    weights: (nsubint, rank), basis: (rank, nharm), from fit_jitter_basis.
    If rank is 0 (no significant jitter found), pass weights/basis with a
    zero-length second/first axis and the jitter term is simply zero.

    Returns: (nsubint, nharm) reconstructed per-subint profile.
    """
    nharm = s0.shape[0]
    alpha = np.arange(nharm)
    aligned = gain_t[:, np.newaxis] * s0[np.newaxis, :] * np.exp(1j * np.outer(epsilon_t, alpha))
    jitter = weights @ basis
    return aligned + jitter


def subspace_principal_angle(basis_a: np.ndarray, basis_b: np.ndarray) -> float | None:
    """
    Largest principal angle (radians) between the subspaces spanned by the
    rows of basis_a and basis_b (each (rank, nharm), not necessarily
    orthonormal or of equal rank -- pycyc.jitter.fit_jitter_basis's `basis`
    is neither, since it's un-whitened back to physical profile units).
    Orthonormalizes each via QR first, then takes arccos of the smallest
    singular value of their inner product.

    Intended as the outer loop's (Stage 4) convergence diagnostic: track
    this between successive passes' jitter bases -- it should shrink
    towards 0 as the outer loop converges to a stationary jitter subspace.

    Returns None if either basis has rank 0 (no significant jitter found),
    since there is no subspace to compare.
    """
    if basis_a.shape[0] == 0 or basis_b.shape[0] == 0:
        return None
    qa, _ = np.linalg.qr(basis_a.T)
    qb, _ = np.linalg.qr(basis_b.T)
    singular_values = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return float(np.arccos(np.min(singular_values)))
