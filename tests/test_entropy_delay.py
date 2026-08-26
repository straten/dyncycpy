"""
Stage 2 of the jitter/degeneracy plan (see
/home/willem/.claude/plans/stateful-dreaming-wall.md): wavefield-domain
epsilon_t via joint (phi, epsilon) spectral-entropy minimization, adapted
from notebooks/Minimum_Spectral_Entropy_Test.ipynb.
"""

import numpy as np

import pycyc

from .helpers import random_complex


def test_spectral_entropy_grad_with_delay_matches_finite_difference():
    rng = np.random.default_rng(0)
    ntime, nchan = 6, 10
    h_time_freq = random_complex(rng, (ntime, nchan))
    params0 = rng.standard_normal(2 * (ntime - 1)) * 0.3

    def entropy_only(p):
        entropy, _grad = pycyc.spectral_entropy_grad_with_delay(p, h_time_freq)
        return entropy

    _entropy, analytic_grad = pycyc.spectral_entropy_grad_with_delay(params0, h_time_freq)

    eps = 1e-6
    n = params0.shape[0]
    numeric_grad = np.zeros(n)
    for k in range(n):
        dp = np.zeros(n)
        dp[k] = eps
        numeric_grad[k] = (entropy_only(params0 + dp) - entropy_only(params0 - dp)) / (2 * eps)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-4, atol=1e-6)


def test_spectral_entropy_grad_with_delay_reduces_to_phi_only_when_epsilon_is_zero():
    """With epsilon held at 0, the joint (phi, epsilon) entropy/gradient
    must exactly match the shipped phi-only spectral_entropy_grad applied to
    the corresponding delay-domain array (h_time_freq -> h_time_delay via
    ifft along the frequency axis) -- these are two independently-coded
    formulations (this one adapted from the notebook; the shipped one
    pre-existing) that should agree exactly in this overlapping case."""
    rng = np.random.default_rng(1)
    ntime, nchan = 5, 8
    h_time_freq = random_complex(rng, (ntime, nchan))
    h_time_delay = pycyc.freq2time(h_time_freq, axis=1)

    phi0 = rng.standard_normal(ntime - 1) * 0.4
    params0 = np.concatenate([phi0, np.zeros(ntime - 1)])

    entropy_joint, grad_joint = pycyc.spectral_entropy_grad_with_delay(params0, h_time_freq)
    entropy_phi, grad_phi = pycyc.spectral_entropy_grad(phi0, h_time_delay)

    np.testing.assert_allclose(entropy_joint, entropy_phi, rtol=1e-10)
    np.testing.assert_allclose(grad_joint[: ntime - 1], grad_phi, rtol=1e-8, atol=1e-10)


def test_minimize_spectral_entropy_with_delay_lowers_entropy_and_returns_fit():
    rng = np.random.default_rng(2)
    ntime, nchan = 6, 12
    h_time_freq = random_complex(rng, (ntime, nchan))

    S_before = pycyc.spectral_entropy_with_delay(h_time_freq)
    phi, epsilon = pycyc.minimize_spectral_entropy_with_delay(h_time_freq, maxiter=200)
    S_after = pycyc.spectral_entropy_with_delay(h_time_freq)

    assert S_after <= S_before + 1e-8
    assert phi.shape == (ntime - 1,)
    assert epsilon.shape == (ntime - 1,)


def test_minimize_spectral_entropy_with_delay_recovers_injected_delay_on_synthetic_scattering_screen():
    """A small version of the notebook's own validated synthetic case:
    build a coherent (low-entropy) wavefield, scramble it with known random
    per-subint phase+delay, and confirm the joint minimizer recovers a
    wavefield close to the original coherent one (entropy comes back down
    near its pre-scramble value)."""
    rng = np.random.default_rng(3)
    ntime, nchan = 8, 16

    # a compact/coherent synthetic delay response: a few narrow spikes in
    # delay, constant across subints (t) before scrambling -- low entropy
    # in h(tau,omega) since it's t-independent (all power at omega=0)
    tau_spikes = [1, 3]
    h_time_delay = np.zeros((ntime, nchan), dtype=np.complex128)
    for tau in tau_spikes:
        h_time_delay[:, tau] = 1.0 + 0.0j
    h_time_freq = pycyc.time2freq(h_time_delay, axis=1)

    S_coherent = pycyc.spectral_entropy_with_delay(h_time_freq)

    true_phi = rng.uniform(-np.pi, np.pi, ntime - 1)
    true_eps = rng.uniform(-1.0, 1.0, ntime - 1)
    nus = np.fft.fftfreq(nchan)
    phs_full = np.concatenate([[0.0], true_phi])
    eps_full = np.concatenate([[0.0], true_eps])
    scrambled = h_time_freq * np.exp(1j * np.outer(eps_full, nus)) * np.exp(1j * phs_full)[:, np.newaxis]

    S_scrambled = pycyc.spectral_entropy_with_delay(scrambled)
    assert S_scrambled > S_coherent + 0.5  # scrambling should have visibly raised the entropy

    recovered = scrambled.copy()
    pycyc.minimize_spectral_entropy_with_delay(recovered, maxiter=500)
    S_recovered = pycyc.spectral_entropy_with_delay(recovered)

    assert S_recovered < S_coherent + 0.5  # back down close to the coherent baseline
