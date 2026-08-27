"""
pycyc.profile - maximum-likelihood intrinsic-profile and gain fitting
(pycyc.tex Appendix, Equations A5 and A7), as a pure function of explicit
parameters rather than a CyclicSolver instance.

Stage 2 of the pycyc.py refactor plan: replaces
CyclicSolver.optimize_profile's body. CyclicSolver.optimize_profile keeps
its existing signature and now just builds a CyclicModelParams and delegates
here.
"""

from __future__ import annotations

__all__ = ["solve_profile_and_gain", "solve_profile_and_gain_batched", "fit_profile_shift"]

import logging

import numpy as np
from scipy.optimize import minimize

from .model import CyclicModelParams, fscrunch_cs
from .regularization import spectral_distance, spectral_shift
from .transforms import shear_spectra, shear_spectra_batched

logger = logging.getLogger(__name__)


def solve_profile_and_gain(
    cs: np.ndarray,
    hf: np.ndarray,
    params: CyclicModelParams,
    update_gain: bool,
    intrinsic_ph_sum: np.ndarray | None = None,
    intrinsic_ph_sumsq: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Least-squares solve for the intrinsic harmonic profile S_mu(alpha)
    (pycyc.tex Equation A7) and, optionally, the per-subint receiver gain
    g(t) (Equation A5).
    """
    hfplus, hfminus = shear_spectra(hf, params.shear_phasors)

    # cs H(-)H(+)*
    cshmhp = cs * hfminus * np.conj(hfplus)
    # |H(-)|^2 |H(+)|^2
    maghmhp = (np.abs(hfminus) * np.abs(hfplus)) ** 2

    if update_gain and intrinsic_ph_sum is not None:
        # Equation A5 numerator
        tmp = fscrunch_cs(
            np.conj(cshmhp) * intrinsic_ph_sum,
            bw=params.bw,
            ref_freq=params.ref_freq,
            padding=params.pad_cyclic_spectra,
        )
        gain_numer = tmp[1:].sum()  # sum over all harmonics
        # Equation A5 denominator
        tmp = fscrunch_cs(
            maghmhp * intrinsic_ph_sumsq,
            bw=params.bw,
            ref_freq=params.ref_freq,
            padding=params.pad_cyclic_spectra,
        )
        gain_denom = tmp[1:].sum()  # sum over all harmonics
        gain = np.real(gain_numer) / np.real(gain_denom)
        logger.debug("solve_profile_and_gain gain=%s", gain)
    else:
        gain = 1

    # Equation A7 numerator
    ph_numer = (
        fscrunch_cs(cshmhp, bw=params.bw, ref_freq=params.ref_freq, padding=params.pad_cyclic_spectra) * gain
    )
    # Equation A7 denominator
    ph_denom = (
        fscrunch_cs(maghmhp, bw=params.bw, ref_freq=params.ref_freq, padding=params.pad_cyclic_spectra)
        * gain**2
    )

    # When the denominator is zero, set the intrinsic profile to zero
    ph = ph_numer / ph_denom
    ph[np.real(ph_denom) <= 0.0] = 0

    return ph, gain, ph_numer, ph_denom


def solve_profile_and_gain_batched(
    cs_batch: np.ndarray,
    hf_batch: np.ndarray,
    params: CyclicModelParams,
    update_gain: bool,
    intrinsic_ph_sum: np.ndarray | None = None,
    intrinsic_ph_sumsq: np.ndarray | None = None,
    xp=np,
    fft_module=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched sibling of solve_profile_and_gain: fits a whole batch of
    subints' intrinsic profiles (and, optionally, gains) in a single call.
    CuPy-by-Claude Stage 5 (see
    /home/willem/.claude/plans/stateful-dreaming-wall.md) -- intended
    caller is CyclicSolver.updateProfile_batched.

    cs_batch: (batch, nchan, nharm). hf_batch: (batch, nchan).
    intrinsic_ph_sum/intrinsic_ph_sumsq: (nharm,), shared across the whole
    batch (CyclicSolver.intrinsic_ph_sum/.intrinsic_ph_sumsq are running
    sums over the *previous* pass's subints, not per-subint) -- broadcast
    against (batch, nchan, nharm) with no reshaping needed.

    Returns (ph, gain, ph_numer, ph_denom) as (batch, nharm), (batch,),
    (batch, nharm), (batch, nharm) arrays -- gain is always an array here
    (1.0-filled when update_gain is False), unlike solve_profile_and_gain's
    plain float 1, so a caller can always index/broadcast it uniformly.

    pad_cyclic_spectra=True is not supported here (raises
    NotImplementedError), matching make_model_cs_batched's scope note.
    """
    if params.pad_cyclic_spectra:
        raise NotImplementedError(
            "solve_profile_and_gain_batched does not support pad_cyclic_spectra=True "
            "yet -- see the CuPy port plan's Stage 1 scope note (fscrunch_cs's "
            "per-harmonic Python loop isn't batched/GPU-friendly)."
        )

    if fft_module is None:
        from .backend import get_fft

        fft_module = get_fft(xp)

    hfplus, hfminus = shear_spectra_batched(hf_batch, params.shear_phasors, xp=xp, fft_module=fft_module)

    # cs H(-)H(+)*
    cshmhp = cs_batch * hfminus * xp.conj(hfplus)
    # |H(-)|^2 |H(+)|^2
    maghmhp = (xp.abs(hfminus) * xp.abs(hfplus)) ** 2

    if update_gain and intrinsic_ph_sum is not None:
        # Equation A5 numerator -- fscrunch (sum over the radio-frequency
        # axis, axis=1 here vs axis=0 for a single subint) without padding
        tmp = (xp.conj(cshmhp) * intrinsic_ph_sum).sum(axis=1)  # (batch, nharm)
        gain_numer = tmp[:, 1:].sum(axis=1)  # sum over all harmonics -> (batch,)
        # Equation A5 denominator
        tmp = (maghmhp * intrinsic_ph_sumsq).sum(axis=1)
        gain_denom = tmp[:, 1:].sum(axis=1)
        gain = xp.real(gain_numer) / xp.real(gain_denom)  # (batch,)
    else:
        gain = xp.ones(cs_batch.shape[0])

    gain_col = gain[..., xp.newaxis]  # (batch, 1), broadcasts against (batch, nharm)

    # Equation A7 numerator/denominator
    ph_numer = cshmhp.sum(axis=1) * gain_col
    ph_denom = maghmhp.sum(axis=1) * (gain_col**2)

    # When the denominator is zero, set the intrinsic profile to zero
    ph = ph_numer / ph_denom
    ph = xp.where(xp.real(ph_denom) <= 0.0, 0, ph)

    return ph, gain, ph_numer, ph_denom


def fit_profile_shift(
    s_ref: np.ndarray,
    s_t: np.ndarray,
    fit_gain: bool = False,
    search_range: float = np.pi,
    search_points: int = 361,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Fit and remove the "degenerate phase" rigid pulse-shift freedom
    (pycyc.tex, "Degenerate phase") between a per-subint profile s_t(alpha)
    and a reference s_ref(alpha):

        s_t(alpha) ~ gain * s_ref(alpha) * exp(i * alpha * epsilon)

    Only `epsilon` -- not a free overall phase -- is fit. A rigid shift of a
    *real* pulse profile by delay dt corresponds to exactly this alpha-linear
    ramp with zero intercept (the standard Fourier shift theorem for a real
    signal); there is no additional free "constant phase across all
    harmonics" degree of freedom for a real profile shift, so including one
    in the fit (as pycyc.regularization.spectral_distance's full 2-parameter
    (phase, slope) form does, for the unrelated wavefield-alignment use case
    in align_to_neighbour) would just fit noise and inflate the variance of
    the epsilon estimate for no physical reason. This reuses
    spectral_distance's math (harmonic-indexed via index=np.arange(nharm))
    but only ever varies its slope component, holding phase fixed at 0.

    The (epsilon-only) objective is genuinely multi-modal -- once
    |epsilon * (nharm-1)| exceeds ~pi, exp(i*alpha*epsilon) has wrapped and
    BFGS from a fixed epsilon=0 start can converge to the wrong local
    minimum (confirmed empirically: it does, routinely, for nharm~12 and
    epsilon of a few tenths of a radian). So this first does a coarse grid
    search over epsilon in [-search_range, search_range] (analogous to
    minimize_difference's FFT cross-correlation coarse search for the
    unrelated wavefield-alignment case -- multiplying S(alpha) by
    exp(i*alpha*epsilon) is exactly a delay/shift theorem pair with a
    cross-correlation search over the conjugate pulse-phase domain, just
    done directly as a grid search here rather than via an FFT), and uses
    the best grid point as BFGS's starting point. epsilon values genuinely
    larger than ~pi are not identifiable from this fit alone (the ramp has
    aliased) regardless of search_range; widen search_range if a larger true
    epsilon range is physically expected and can be resolved some other way
    (e.g. from a wavefield-domain estimate).

    fit_gain: if True, additionally fits a real amplitude scale by closed-
    form least squares after the epsilon fit (i.e. holding gain fixed at 1
    during the nonlinear epsilon fit, then solving for gain in closed form
    given the fitted epsilon) -- a standard, good approximation for a
    nuisance amplitude parameter, not an exact joint MLE of (epsilon, gain)
    together.

    Returns (epsilon, gain, aligned, residual), where
    aligned = gain * s_ref * exp(i * alpha * epsilon) and
    residual = s_t - aligned -- this residual is R(alpha;T) from the outer
    jitter model (pycyc.jitter).
    """
    nharm = s_ref.shape[0]
    index = np.arange(nharm)

    # spectral_distance's objective value and gradient scale with the
    # square of s_t/s_ref's absolute flux units. On bright real data those
    # can be ~1e4-1e6, which makes scipy's BFGS default (absolute) gtol
    # meaningless relative to the objective's actual curvature scale and
    # triggers spurious "precision loss" warnings even though the fit has
    # already converged. epsilon and gain are both exactly invariant to a
    # positive real rescaling of s_ref and s_t together (the scale cancels:
    # spectral_distance's gradient direction is unchanged, and the gain
    # closed-form below is a ratio of same-degree terms), so fit in
    # normalized units and only rescale back for the returned gain.
    norm = np.sqrt(np.mean(np.abs(s_t) ** 2))
    if norm <= 0:
        norm = 1.0
    s_ref_n = s_ref / norm
    s_t_n = s_t / norm

    def _epsilon_objective(x):
        diff, grad = spectral_distance([0.0, x[0]], s_t_n, s_ref_n, index=index)
        return diff, np.array([grad[1]])

    grid = np.linspace(-search_range, search_range, search_points)
    grid_values = [_epsilon_objective(np.array([e]))[0] for e in grid]
    x0 = [grid[int(np.argmin(grid_values))]]

    logger.info("fit_profile_shift: initial phase shift = %f", x0[0])

    result = minimize(_epsilon_objective, x0=x0, method="BFGS", jac=True)
    if not result.success:
        logger.warning("fit_profile_shift: BFGS did not converge (%s)", result.message)
    epsilon = float(result.x[0])

    aligned, _ = spectral_shift([0.0, epsilon], s_ref, index=index)

    gain = 1.0
    if fit_gain:
        denom = np.sum(np.abs(aligned) ** 2)
        if denom > 0:
            gain = float(np.real(np.sum(np.conj(aligned) * s_t)) / denom)
        aligned = aligned * gain

    residual = s_t - aligned
    return epsilon, gain, aligned, residual
