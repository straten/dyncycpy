"""
pycyc.profile - maximum-likelihood intrinsic-profile and gain fitting
(pycyc.tex Appendix, Equations A5 and A7), as a pure function of explicit
parameters rather than a CyclicSolver instance.

Stage 2 of the pycyc.py refactor plan: replaces
CyclicSolver.optimize_profile's body. CyclicSolver.optimize_profile keeps
its existing signature and now just builds a CyclicModelParams and delegates
here.
"""

__all__ = ["solve_profile_and_gain"]

import numpy as np

from .model import fscrunch_cs
from .transforms import shear_spectra


def solve_profile_and_gain(cs, hf, params, update_gain, intrinsic_ph_sum=None, intrinsic_ph_sumsq=None):
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
            maghmhp * intrinsic_ph_sumsq, bw=params.bw, ref_freq=params.ref_freq, padding=params.pad_cyclic_spectra
        )
        gain_denom = tmp[1:].sum()  # sum over all harmonics
        gain = np.real(gain_numer) / np.real(gain_denom)
        print(f"solve_profile_and_gain gain={gain}")
    else:
        gain = 1

    # Equation A7 numerator
    ph_numer = fscrunch_cs(cshmhp, bw=params.bw, ref_freq=params.ref_freq, padding=params.pad_cyclic_spectra) * gain
    # Equation A7 denominator
    ph_denom = (
        fscrunch_cs(maghmhp, bw=params.bw, ref_freq=params.ref_freq, padding=params.pad_cyclic_spectra) * gain**2
    )

    # When the denominator is zero, set the intrinsic profile to zero
    ph = ph_numer / ph_denom
    ph[np.real(ph_denom) <= 0.0] = 0

    return ph, gain, ph_numer, ph_denom
