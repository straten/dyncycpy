"""
Shared helpers for the pycyc regression-test suite (Stage 0 of the refactor,
see /home/willem/.claude/plans/stateful-dreaming-wall.md).

These build a lightweight stand-in for a CyclicSolver ("CS") instance using
types.SimpleNamespace, since complex_cyclic_merit_lag/cyclic_merit_lag only
ever read a handful of attributes off it and a full CyclicSolver requires a
PSRFITS file to construct.
"""

import types

import numpy as np

import pycyc


def make_cs(
    nchan,
    nharm,
    nlag,
    bw,
    ref_freq,
    *,
    maxharm=None,
    exclude_DC=0,
    pad_cyclic_spectra=False,
    include_Nyquist=False,
    dump_residual=False,
    iprint=False,
    rindex=0,
):
    """Build a minimal CS namespace sufficient for make_model_cs /
    complex_cyclic_merit_lag / cyclic_merit_lag."""
    cs = types.SimpleNamespace()
    cs.bw = bw
    cs.ref_freq = ref_freq
    cs.pad_cyclic_spectra = pad_cyclic_spectra
    cs.include_Nyquist = include_Nyquist
    cs.maxharm = maxharm
    cs.exclude_DC = exclude_DC
    cs.dump_residual = dump_residual
    cs.iprint = iprint
    cs.rindex = rindex
    cs.nlag = nlag
    cs.shear_phasors = pycyc.create_shear_phasors(nchan, nharm, bw, ref_freq)
    cs.objval = []
    return cs


def random_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def truncate_like_get_cs(cs_data, cs):
    """Apply the same maxharm/pad_cyclic_spectra truncation that
    CyclicSolver.get_cs() applies to real data, so synthetic cs_data obeys
    the invariant complex_cyclic_merit_lag currently relies on (see the
    "maxharm fragility" finding in the refactor plan, Stage 2)."""
    cs_data = cs_data.copy()
    if cs.pad_cyclic_spectra:
        cs_data = pycyc.cyclic_padding(cs_data, cs.bw, cs.ref_freq)
    if cs.maxharm is not None:
        cs_data[:, cs.maxharm + 1 :] = 0.0
    return cs_data


def wirtinger_finite_diff_grad(merit_fn, ht0, eps=1e-6):
    """Numerically estimate dM/dh* (Wirtinger derivative) of a real-valued
    merit_fn(ht) -> float, at each element of the complex array ht0, via
    central finite differences of the real and imaginary parts.

    dM/dh* = 0.5 * (dM/dRe(h) + i * dM/dIm(h))
    """
    nlag = ht0.shape[0]
    grad = np.zeros(nlag, dtype=np.complex128)
    for k in range(nlag):
        dht = np.zeros(nlag, dtype=np.complex128)

        dht[k] = eps
        dMdre = (merit_fn(ht0 + dht) - merit_fn(ht0 - dht)) / (2 * eps)

        dht[k] = 1j * eps
        dMdim = (merit_fn(ht0 + dht) - merit_fn(ht0 - dht)) / (2 * eps)

        grad[k] = 0.5 * (dMdre + 1j * dMdim)
    return grad


class RealMultiSubintFunc:
    """func stand-in whose evaluate() computes the *real* per-subint merit
    and Wirtinger gradient (pycyc.cyclic_merit_and_grad, one independent
    synthetic problem per subint) in the time-delay domain and transforms
    to/from Doppler-delay exactly as CyclicSolver.evaluate/updateWavefield
    do -- unlike a stub with pre-scripted returns, this actually computes a
    gradient from whatever wavefield it's given, so it can drive a genuine,
    many-iteration FISTA trajectory through the real fista.take_fista_step
    rather than a single hand-constructed step. Originally test_fista.py's
    _RealMultiSubintFunc; promoted here so golden-trajectory regression
    tests (see test_golden_regression.py) can share it without duplicating
    the gradient-harness code."""

    def __init__(self, params, s0, cs_data_per_subint):
        self.params = params
        self.s0 = s0
        self.cs_data_per_subint = cs_data_per_subint

    def normalize(self, h_dopp):
        return h_dopp  # identity, matching conserve_wavefield_energy=False

    def evaluate(self, h_dopp):
        h_time = pycyc.freq2time(h_dopp, axis=0)
        merit_total = 0.0
        grad_time = np.zeros_like(h_time)
        for t in range(h_time.shape[0]):
            m, g, _ = pycyc.cyclic_merit_and_grad(h_time[t], self.params, self.s0, self.cs_data_per_subint[t])
            merit_total += m
            grad_time[t] = g
        return merit_total, pycyc.time2freq(grad_time, axis=0)


def real_finite_diff_grad(merit_fn, x0, eps=1e-6):
    """Numerically estimate the gradient of a real-valued merit_fn(x) -> float
    with respect to a real-valued parameter vector x0, via central
    differences."""
    n = x0.shape[0]
    grad = np.zeros(n)
    for k in range(n):
        dx = np.zeros(n)
        dx[k] = eps
        grad[k] = (merit_fn(x0 + dx) - merit_fn(x0 - dx)) / (2 * eps)
    return grad
