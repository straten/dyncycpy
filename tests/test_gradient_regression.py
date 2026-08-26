"""
Gradient regression tests for complex_cyclic_merit_lag / cyclic_merit_lag.

These formalize the finite-difference check that found the missing-`gain`
gradient bug (fixed on master in commit 6b2d8c4): the analytic gradient
returned by complex_cyclic_merit_lag must match a numerical (Wirtinger)
finite-difference gradient of the same function's own merit output, for
every combination of `gain`, `maxharm`, `exclude_DC`, and
`pad_cyclic_spectra` in real use.

Parametrization intentionally keeps cs_data consistent with the truncation
implied by `maxharm`/`pad_cyclic_spectra` (via truncate_like_get_cs), matching
what every current caller does via CyclicSolver.get_cs(). Stage 2 of the
refactor closes this as an internal invariant of the objective function
itself rather than a caller obligation; a dedicated test for that will be
added at that point.
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
def test_complex_cyclic_merit_lag_gradient_matches_finite_difference(
    gain, maxharm, exclude_DC, pad_cyclic_spectra
):
    rng = np.random.default_rng(0)

    cs = make_cs(
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
    cs_data = truncate_like_get_cs(random_complex(rng, (NCHAN, NHARM)), cs)
    ht0 = random_complex(rng, (NLAG,))

    def merit_only(ht):
        merit, _grad, _nonzero = pycyc.complex_cyclic_merit_lag(ht, cs, s0, cs_data, gain)
        return merit

    analytic_merit, analytic_grad, _ = pycyc.complex_cyclic_merit_lag(ht0, cs, s0, cs_data, gain)
    numeric_grad = wirtinger_finite_diff_grad(merit_only, ht0)

    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rindex", [0, 3, NLAG - 1])
def test_cyclic_merit_lag_real_parameter_gradient_matches_finite_difference(rindex):
    """cyclic_merit_lag operates on the real-valued (get_params-packed)
    parameterization used by scipy.optimize.fmin_l_bfgs_b; check its gradient
    end-to-end including the get_params/get_ht round trip and the factor-of-2
    Wirtinger-to-real-gradient conversion."""
    rng = np.random.default_rng(1)

    cs = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ, rindex=rindex)
    cs.ph_ref = random_complex(rng, (NHARM,))
    cs.cs = truncate_like_get_cs(random_complex(rng, (NCHAN, NHARM)), cs)

    ht0 = random_complex(rng, (NLAG,))
    ht0[rindex] = ht0[rindex].real  # cyclic_merit_lag's parameterization requires this
    x0 = pycyc.get_params(ht0, rindex)

    def merit_only(x):
        cs.objval = []
        merit, _grad = pycyc.cyclic_merit_lag(x, cs)
        return merit

    analytic_merit, analytic_grad = pycyc.cyclic_merit_lag(x0, cs)
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
    cs = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ)
    s0 = random_complex(rng, (NHARM,))
    cs_data = np.zeros((NCHAN, NHARM), dtype=np.complex128)
    ht0 = random_complex(rng, (NLAG,))

    _, grad_1, _ = pycyc.complex_cyclic_merit_lag(ht0, cs, s0, cs_data, 1.0)
    _, grad_2, _ = pycyc.complex_cyclic_merit_lag(ht0, cs, s0, cs_data, 2.5)

    np.testing.assert_allclose(grad_2, 2.5**2 * grad_1, rtol=1e-10)
