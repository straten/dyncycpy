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
