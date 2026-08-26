"""
pycyc.model - cyclic-spectrum shape helpers.

Kept separate from pycyc.transforms: these operate on the specific
(nchan, nharm)-shaped cyclic-spectrum/profile arrays and know about the
band-edge geometry implied by (bw, ref_freq), rather than being generic FFT
conventions.
"""

__all__ = ["cyclic_padding", "chan_limits_cs", "fscrunch_cs", "total_cyclic_power", "normalize_profile", "normalize_pp"]

import numpy as np
from scipy.fft import irfft

from .transforms import phase2harm

def cyclic_padding(cs, bw, ref_freq):
    nharm = cs.shape[1]
    nchan = cs.shape[0]
    for ih in range(nharm):
        imin, imax = chan_limits_cs(ih, nchan, bw, ref_freq)
        cs[:imin, ih] = 0
        cs[imax:, ih] = 0
    return cs



def chan_limits_cs(iharm, nchan, bw, ref_freq):
    chanbw_Hz = bw * 1e6 / nchan  # width of FFT bins in radio frequency Hz
    shift_Hz = iharm * ref_freq / 2
    ichan = round(shift_Hz / chanbw_Hz)
    if ichan > nchan / 2:
        ichan = int(nchan / 2)
    return (ichan, nchan - ichan)  # min,max



def fscrunch_cs(cs, bw, ref_freq, padding):
    cstmp = cs[:]
    if padding:
        cstmp = cyclic_padding(cstmp, bw, ref_freq)
    #    rm = np.abs(cs-cstmp).sum()
    #    print "fscrunch saved:",rm
    return cstmp.sum(0)



def total_cyclic_power(cs):
    """
    returns the sum of the power in all radio frequencies and cycle frequencies,
    excluding the DC cycle frequency (mean of periodic spectrum)
    """
    return np.sum(np.abs(cs[:, 1:]) ** 2)



def normalize_profile(ph):
    """
    Normalize harmonic profile such that first harmonic has magnitude 1
    """
    return ph / np.abs(ph[1])



def normalize_pp(pp):
    """
    Normalize a profile but keep it in phase rather than harmonics
    """
    ph = phase2harm(pp)
    ph = normalize_profile(ph)
    ph[0] = 0
    return irfft(ph, norm="ortho")

