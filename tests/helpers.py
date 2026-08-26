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
