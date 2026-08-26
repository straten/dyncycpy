"""
Gradient regression tests for pycyc.objective's cyclic_merit_and_grad /
cyclic_merit_lag_x.

These formalize the finite-difference check that found the missing-`gain`
gradient bug (fixed on master in commit 6b2d8c4, then carried into the pure
pycyc.objective.cyclic_merit_and_grad in Stage 2 of the refactor): the
analytic gradient must match a numerical (Wirtinger) finite-difference
gradient of the same function's own merit output, for every combination of
`gain`, `maxharm`, `exclude_DC`, and `pad_cyclic_spectra` in real use.

Parametrization keeps cs_data consistent with the truncation implied by
`maxharm`/`pad_cyclic_spectra` (via truncate_like_get_cs), matching what
every current caller does via CyclicSolver.get_cs(). A separate test below
(test_gradient_correct_even_without_get_cs_invariant) checks the case where
that invariant does *not* hold -- the "maxharm fragility" fix that is the
other half of Stage 2's objective.py rewrite.
"""

import numpy as np
import pytest

import pycyc

from .helpers import make_cs, random_complex, truncate_like_get_cs, wirtinger_finite_diff_grad, real_finite_diff_grad

NCHAN = 16
NHARM = 6
NLAG = NCHAN
BW_MHZ = 1.0
REF_FREQ_HZ = 1e5  # chosen so pad_cyclic_spectra actually truncates something


@pytest.mark.parametrize("gain", [1.0, 2.5, 0.3])
@pytest.mark.parametrize("maxharm", [None, 3])
@pytest.mark.parametrize("exclude_DC", [0, 1])
@pytest.mark.parametrize("pad_cyclic_spectra", [False, True])
def test_cyclic_merit_and_grad_matches_finite_difference(gain, maxharm, exclude_DC, pad_cyclic_spectra):
    rng = np.random.default_rng(0)

    params = make_cs(
        NCHAN,
        NHARM,
        NLAG,
        BW_MHZ,
        REF_FREQ_HZ,
        maxharm=maxharm,
        exclude_DC=exclude_DC,
        pad_cyclic_spectra=pad_cyclic_spectra,
    )

    s0 = random_complex(rng, (NHARM,))
    cs_data = truncate_like_get_cs(random_complex(rng, (NCHAN, NHARM)), params)
    ht0 = random_complex(rng, (NLAG,))

    def merit_only(ht):
        merit, _grad, _nonzero = pycyc.cyclic_merit_and_grad(ht, params, s0, cs_data, gain=gain)
        return merit

    analytic_merit, analytic_grad, _ = pycyc.cyclic_merit_and_grad(ht0, params, s0, cs_data, gain=gain)
    numeric_grad = wirtinger_finite_diff_grad(merit_only, ht0)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("maxharm, pad_cyclic_spectra", [(3, False), (None, True), (3, True)])
def test_gradient_correct_even_without_get_cs_invariant(maxharm, pad_cyclic_spectra):
    """The 'maxharm fragility' fix: cs_data here is nonzero everywhere
    (violating the invariant every real caller upholds via get_cs, which
    pre-zeros cs_data wherever the model is forced to a constant). Before
    Stage 2, the gradient would have picked up a spurious contribution from
    those positions; cyclic_merit_and_grad now masks them out internally, so
    the gradient matches finite differences regardless of what's in
    cs_data."""
    rng = np.random.default_rng(7)

    params = make_cs(
        NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ, maxharm=maxharm, pad_cyclic_spectra=pad_cyclic_spectra
    )

    s0 = random_complex(rng, (NHARM,))
    cs_data = random_complex(rng, (NCHAN, NHARM))  # deliberately NOT truncated
    ht0 = random_complex(rng, (NLAG,))

    def merit_only(ht):
        merit, _grad, _nonzero = pycyc.cyclic_merit_and_grad(ht, params, s0, cs_data)
        return merit

    analytic_merit, analytic_grad, _ = pycyc.cyclic_merit_and_grad(ht0, params, s0, cs_data)
    numeric_grad = wirtinger_finite_diff_grad(merit_only, ht0)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rindex", [0, 3, NLAG - 1])
def test_cyclic_merit_lag_x_real_parameter_gradient_matches_finite_difference(rindex):
    """cyclic_merit_lag_x operates on the real-valued (pack_real_params-
    packed) parameterization used by scipy.optimize.fmin_l_bfgs_b; check its
    gradient end-to-end including the pack_real_params/unpack_real_params
    round trip and the factor-of-2 Wirtinger-to-real-gradient conversion."""
    rng = np.random.default_rng(1)

    params = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ)
    ph_ref = random_complex(rng, (NHARM,))
    cs_data = truncate_like_get_cs(random_complex(rng, (NCHAN, NHARM)), params)

    ht0 = random_complex(rng, (NLAG,))
    ht0[rindex] = ht0[rindex].real  # cyclic_merit_lag_x's parameterization requires this
    x0 = pycyc.pack_real_params(ht0, rindex)

    def merit_only(x):
        merit, _grad = pycyc.cyclic_merit_lag_x(x, params, rindex, ph_ref, cs_data)
        return merit

    analytic_merit, analytic_grad = pycyc.cyclic_merit_lag_x(x0, params, rindex, ph_ref, cs_data)
    numeric_grad = real_finite_diff_grad(merit_only, x0)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-5, atol=1e-5)


def test_gain_bug_regression():
    """Direct regression test for the specific bug fixed in commit 6b2d8c4,
    without relying on finite differences. With cs_data == 0, residual =
    gain * raw_model is exactly homogeneous in `gain` (no affine "-cs_data"
    term), so the correct gradient -- gain * (grad_sum1 + grad_sum2), with
    grad_sum1/grad_sum2 themselves linear in residual and hence in gain --
    must scale as gain**2. The pre-fix code (missing the outer `gain`
    factor) would have scaled as gain**1 instead."""
    rng = np.random.default_rng(2)
    params = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ)
    s0 = random_complex(rng, (NHARM,))
    cs_data = np.zeros((NCHAN, NHARM), dtype=np.complex128)
    ht0 = random_complex(rng, (NLAG,))

    _, grad_1, _ = pycyc.cyclic_merit_and_grad(ht0, params, s0, cs_data, gain=1.0)
    _, grad_2, _ = pycyc.cyclic_merit_and_grad(ht0, params, s0, cs_data, gain=2.5)

    np.testing.assert_allclose(grad_2, 2.5**2 * grad_1, rtol=1e-10)
