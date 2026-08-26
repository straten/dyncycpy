"""
pycyc.transforms - pure FFT-convention helpers for cyclic spectroscopy.

These implement the orthonormal DFT conventions defined in pycyc.tex
("Dynamic impulse response function"): time2freq/freq2time are a matched
pair of orthonormal transforms (both scaled by Nchan**-1/2) so that
Parseval's theorem holds, unlike Walker, Demorest & van Straten (2013)
which normalizes only the forward transform. create_shear_phasors +
shear_spectra implement the corresponding frequency-shift theorem.

No function in this module takes a CyclicSolver/CS instance -- everything
here is a pure array-in, array-out function, safe to reuse independently of
CyclicSolver.
"""

__all__ = [
    "ps2cs",
    "cc2cs",
    "cs2cc",
    "cc2pc",
    "pc2cc",
    "time2freq",
    "freq2time",
    "phase2harm",
    "fold",
    "minphase",
    "pad_wavefield",
    "shifted",
    "match_two_filters",
    "create_shear_phasors",
    "shear_spectra",
]

import numpy as np
from scipy.fft import fft, ifft, rfft


def ps2cs(ps, workers=2):
    """
    real-valued periodic spectrum to complex-valued cyclic spectrum
    real-to-complex forward FFT transforms pulse phase to cycle frequency
    the Nyquist harmonic is removed because it typically contains little information,
    and its removal facilitates comparison with simulated cyclic spectra
    """
    return rfft(ps, axis=1, workers=workers, norm="ortho")


def cc2cs(cs, workers=2, axis=0):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex forward FFT transforms lag to radio frequency
    """
    return fft(cs, axis=axis, workers=workers, norm="ortho")


def cs2cc(cc, workers=2, axis=0):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex backward/inverse FFT transforms radio frequency to lag
    """
    return ifft(cc, axis=axis, workers=workers, norm="ortho")


def cc2pc(cc, workers=2, axis=1):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex backward/inverse FFT transforms cycle frequency to pulse phase
    """
    return ifft(cc, axis=axis, workers=workers, norm="ortho")


def pc2cc(pc, workers=2, axis=1):
    """
    complex-valued periodic correlation to cyclic correlation
    complex-to-complex forward FFT transforms pulse phase to cycle frequency
    """
    return fft(pc, axis=axis, workers=workers, norm="ortho")


def time2freq(ht, workers=1, axis=0):
    return fft(ht, axis=axis, workers=workers, norm="ortho")


def freq2time(hf, workers=1, axis=0):
    return ifft(hf, axis=axis, workers=workers, norm="ortho")


def phase2harm(pp, workers=1):
    return rfft(pp, workers=workers, norm="ortho")


def fold(v):
    """
    Fold negative response onto positive time for minimum phase calculation
    """
    n = v.shape[0]
    nt = int(n / 2)
    rf = np.zeros_like(v[:nt])
    rf[:-1] = v[1:nt]
    rf += np.conj(v[: nt - 1 : -1])
    rw = np.zeros_like(v)
    rw[0] = v[0]
    rw[1 : nt + 1] = rf
    return rw


def minphase(v, workers=1):
    clipped = v.copy()
    thresh = 1e-5
    clipped[np.abs(v) < thresh] = thresh
    return np.exp(fft(fold(ifft(np.log(clipped), workers=workers)), workers=workers))


def pad_wavefield(h_time_freq, new_shape):
    old_shape = h_time_freq.shape
    h_time_delay = freq2time(h_time_freq, axis=1)
    h_doppler_delay = time2freq(h_time_delay, axis=0)

    padded_spectrum = np.zeros(new_shape, dtype=np.complex128)

    first_half_time = old_shape[0] // 2
    last_half_time = old_shape[0] - first_half_time

    padded_spectrum[:first_half_time, : old_shape[1]] = h_doppler_delay[:first_half_time, :]
    padded_spectrum[-last_half_time:, : old_shape[1]] = h_doppler_delay[-last_half_time:, :]

    h_time_delay = freq2time(padded_spectrum, axis=0)
    return time2freq(h_time_delay, axis=1)


def shifted(input_array, fraction_of_bin, axis=0):
    # NOTE: appears unused anywhere in this repository (including notebooks,
    # as of the pycyc.py package refactor) -- flagging rather than removing,
    # in case it's used from somewhere outside this checkout.
    # Get the shape of the input array
    shape = input_array.shape

    # Create an array of sample frequencies
    frequency = np.fft.fftfreq(shape[axis])

    # Calculate the shift in radians based on the fraction of a bin
    phase_shift = 2 * np.pi * fraction_of_bin

    if axis == 0:
        return input_array * np.exp(1j * phase_shift * frequency[:, np.newaxis])
    elif axis == 1:
        return input_array * np.exp(1j * phase_shift * frequency[np.newaxis, :])
    else:
        raise ValueError("Invalid axis value. Must be 0 or 1.")


def match_two_filters(hf1, hf2):
    z = (hf1 * np.conj(hf2)).sum()
    z2 = (hf2 * np.conj(hf2)).sum()  # = np.abs(hf2)**2.sum()
    z /= np.abs(z)
    z *= np.sqrt(1.0 * hf1.shape[0] / np.real(z2))
    return hf2 * z


def create_shear_phasors(nchan, nharm, bw_MHz, freq_Hz):
    """Construct the two-dimensional array of phasors used to shift the spectral response function

    Specifically, returns exp(i pi \tau_j \alpha_k)

    Parameters
    ----------
    nchan : number of frequency channels in spectra to be shifted
    nharm : number of shifts to perform
    bw_MHz : bandwidth in MHz
    freq_Hz : modulation frequency in Hz

    Returns
    ------
    .. math::
        exp(i pi \tau_j \alpha_k)

    where i=sqrt(-1), j = row index, and k = column index
    """

    # tau[j] in seconds
    tau = np.fft.fftfreq(nchan) * (nchan * 1e-6 / bw_MHz)
    # tau = np.linspace(0.0, 1.0 - 1.0 / nchan, nchan) * (nchan * 1e-6 / bw_MHz)
    # alpha[k] in Hz
    alpha = freq_Hz * np.arange(nharm)

    # shift by 2 pi alpha / 2
    return np.exp(1j * np.pi * np.outer(tau, alpha))


def shear_spectra(spectrum, phasors):
    """Shifts the spectrum for each column of phasors

    Parameters
    ----------
    spectra : one-dimensional spectrum to be shifted
    phasors : the phase gradients for each shift to be applied

    """

    tmp = ifft(spectrum)
    nharm = phasors.shape[1]

    # copy the spectrum for each shift for each harmonic
    spectra = np.repeat(tmp[:, np.newaxis], nharm, axis=1)
    return fft(spectra * np.conj(phasors), axis=0), fft(spectra * phasors, axis=0)
