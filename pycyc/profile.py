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

__all__ = ["solve_profile_and_gain", "fit_profile_shift"]

import logging

import numpy as np
from scipy.optimize import minimize

from .model import CyclicModelParams, fscrunch_cs
from .regularization import spectral_distance, spectral_shift
from .transforms import shear_spectra

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
