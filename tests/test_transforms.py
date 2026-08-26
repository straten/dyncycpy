"""
Transform-convention regression tests.

pycyc.tex ("Dynamic impulse response function") defines time2freq/freq2time
as a matched pair of orthonormal DFTs (both scaled by Nchan^-1/2) so that
Parseval's theorem holds -- unlike Walker, Demorest & van Straten (2013),
which normalizes only the forward transform. create_shear_phasors +
shear_spectra then use a deliberately mixed-normalization ifft-then-fft
trick (see review notes) to implement the frequency-shift theorem cheaply;
that trick is exactly the kind of thing that's easy to get subtly wrong, so
it's checked here against an independent, direct evaluation of the shift
theorem.
"""

import numpy as np

import pycyc

from .helpers import random_complex


def test_time_freq_round_trip():
    rng = np.random.default_rng(5)
    ht = random_complex(rng, (32,))
    ht2 = pycyc.freq2time(pycyc.time2freq(ht))
    np.testing.assert_allclose(ht2, ht, rtol=1e-9, atol=1e-9)


def test_time_freq_parseval():
    """Sec. "Dynamic impulse response function": these definitions conserve
    energy, so sum|h(tau)|^2 == sum|H(nu)|^2."""
    rng = np.random.default_rng(4)
    ht = random_complex(rng, (32,))
    hf = pycyc.time2freq(ht)
    np.testing.assert_allclose(
        np.sum(np.abs(hf) ** 2), np.sum(np.abs(ht) ** 2), rtol=1e-9
    )


def test_shear_spectra_matches_direct_shift_theorem():
    """create_shear_phasors returns exp(i*pi*tau_j*alpha_k). By the shift
    theorem for the forward DFT H(nu) = N^-1/2 sum_tau h(tau) exp(-2 pi i tau
    nu):

        H(nu + alpha/2) = FFT_ortho[ h(tau) * exp(-i*pi*tau*alpha) ]
        H(nu - alpha/2) = FFT_ortho[ h(tau) * exp(+i*pi*tau*alpha) ]

    shear_spectra must reproduce exactly this for every harmonic column,
    independent of its own ifft/fft-without-normalization implementation.
    """
    nchan = 16
    nharm = 5
    bw = 1.0
    ref_freq = 1e5
    rng = np.random.default_rng(3)

    ht = random_complex(rng, (nchan,))
    hf = pycyc.time2freq(ht)

    phasors = pycyc.create_shear_phasors(nchan, nharm, bw, ref_freq)
    hfplus, hfminus = pycyc.shear_spectra(hf, phasors)

    tau = np.fft.fftfreq(nchan) * (nchan * 1e-6 / bw)
    alpha = ref_freq * np.arange(nharm)

    for k in range(nharm):
        expected_plus = pycyc.time2freq(ht * np.exp(-1j * np.pi * tau * alpha[k]))
        expected_minus = pycyc.time2freq(ht * np.exp(1j * np.pi * tau * alpha[k]))

        np.testing.assert_allclose(hfplus[:, k], expected_plus, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hfminus[:, k], expected_minus, rtol=1e-9, atol=1e-9)
