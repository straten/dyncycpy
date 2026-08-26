"""
pycyc.model - cyclic-spectrum shape helpers.

Kept separate from pycyc.transforms: these operate on the specific
(nchan, nharm)-shaped cyclic-spectrum/profile arrays and know about the
band-edge geometry implied by (bw, ref_freq), rather than being generic FFT
conventions.
"""

from __future__ import annotations

__all__ = [
    "CyclicModelParams",
    "cyclic_padding",
    "chan_limits_cs",
    "fscrunch_cs",
    "total_cyclic_power",
    "normalize_profile",
    "normalize_pp",
]

from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft

from .transforms import phase2harm


@dataclass(frozen=True)
class CyclicModelParams:
    """
    The subset of CyclicSolver's configuration that defines the forward
    cyclic-spectrum model, needed by make_model_cs/cyclic_merit_and_grad
    (pycyc.objective) and solve_profile_and_gain (pycyc.profile) -- pulled
    out so those functions take explicit parameters instead of reaching into
    a CyclicSolver instance.

    Per-call diagnostic/debug knobs (which subint, verbosity, whether to
    dump intermediate arrays) are deliberately *not* part of this: they vary
    per call, not per model, and are passed as ordinary keyword arguments
    instead.
    """

    bw: float
    ref_freq: float
    shear_phasors: np.ndarray
    pad_cyclic_spectra: bool
    include_Nyquist: bool
    maxharm: int | None
    exclude_DC: int
    nlag: int


def cyclic_padding(cs: np.ndarray, bw: float, ref_freq: float) -> np.ndarray:
    """
    Zero cs (an (nchan, nharm) cyclic spectrum) outside the band of radio
    frequency channels each harmonic's shifted response H(nu +/- alpha/2)
    can validly occupy, per chan_limits_cs. Mutates and returns cs.
    """
    nharm = cs.shape[1]
    nchan = cs.shape[0]
    for ih in range(nharm):
        imin, imax = chan_limits_cs(ih, nchan, bw, ref_freq)
        cs[:imin, ih] = 0
        cs[imax:, ih] = 0
    return cs


def chan_limits_cs(iharm: int, nchan: int, bw: float, ref_freq: float) -> tuple[int, int]:
    """
    (min, max) radio-frequency channel indices within which harmonic
    `iharm`'s shifted response H(nu +/- iharm*ref_freq/2) fits inside a band
    of `nchan` channels and width `bw` (MHz); channels outside this range
    would require frequencies beyond the edge of the observed band.
    """
    chanbw_Hz = bw * 1e6 / nchan  # width of FFT bins in radio frequency Hz
    shift_Hz = iharm * ref_freq / 2
    ichan = round(shift_Hz / chanbw_Hz)
    if ichan > nchan / 2:
        ichan = int(nchan / 2)
    return (ichan, nchan - ichan)  # min,max


def fscrunch_cs(cs: np.ndarray, bw: float, ref_freq: float, padding: bool) -> np.ndarray:
    """Sum an (nchan, nharm) cyclic spectrum over radio frequency, optionally
    band-edge-truncating it (cyclic_padding) first."""
    cstmp = cs[:]
    if padding:
        cstmp = cyclic_padding(cstmp, bw, ref_freq)
    #    rm = np.abs(cs-cstmp).sum()
    #    print "fscrunch saved:",rm
    return cstmp.sum(0)


def total_cyclic_power(cs: np.ndarray) -> float:
    """
    returns the sum of the power in all radio frequencies and cycle frequencies,
    excluding the DC cycle frequency (mean of periodic spectrum)
    """
    return np.sum(np.abs(cs[:, 1:]) ** 2)


def normalize_profile(ph: np.ndarray) -> np.ndarray:
    """
    Normalize harmonic profile such that first harmonic has magnitude 1
    """
    return ph / np.abs(ph[1])


def normalize_pp(pp: np.ndarray) -> np.ndarray:
    """
    Normalize a profile but keep it in phase rather than harmonics
    """
    ph = phase2harm(pp)
    ph = normalize_profile(ph)
    ph[0] = 0
    return irfft(ph, norm="ortho")
