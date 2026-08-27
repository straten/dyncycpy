"""
Regression test for the L_min gauge-consistency fix in fista.take_fista_step
(see the "Estimate minimum L using equation (12) from ow23" block).

x_np1/y_n live in a space with an exact degenerate freedom: the cyclic-
spectrum merit function is invariant under a uniform global phase rotation
h -> h*e^{i phi} (verified numerically below against the real
pycyc.cyclic_merit_and_grad, since cs_model = H(+)*conj(H(-))*S cancels
the phase between the two H factors exactly). take_fista_step already
detects and removes this before computing the *position* difference
(`diff = z*x_np1 - y_n`, where z is the unit-modulus phasor aligning
x_np1 to y_n), but previously did *not* apply the same z rotation before
computing the *gradient* difference (`gdiff`) -- even though
grad(h*e^{i phi}) == e^{i phi}*grad(h) exactly for this merit function, so
the same correction is required there too. Without it, gdiff picks up a
spurious contribution from exactly the degenerate phase drift that
var_diff (correctly) already removed, and L_min can blow up by orders of
magnitude whenever an update happens to be dominated by phase drift
rather than genuine wavefield movement.
"""

import numpy as np

import fista
import pycyc


def test_gradient_transforms_with_global_phase():
    """Grounds the fix: confirms grad(h*e^{i phi}) == e^{i phi}*grad(h)
    exactly for the real merit function, not just assumed."""
    rng = np.random.default_rng(0)
    nchan, nharm, nlag = 16, 6, 16
    bw, ref_freq = 1.0, 1e5
    params = pycyc.CyclicModelParams(
        bw=bw,
        ref_freq=ref_freq,
        shear_phasors=pycyc.create_shear_phasors(nchan, nharm, bw, ref_freq),
        pad_cyclic_spectra=False,
        include_Nyquist=True,
        maxharm=None,
        exclude_DC=0,
        nlag=nlag,
    )
    s0 = rng.standard_normal(nharm) + 1j * rng.standard_normal(nharm)
    cs_data = rng.standard_normal((nchan, nharm)) + 1j * rng.standard_normal((nchan, nharm))
    ht = rng.standard_normal(nlag) + 1j * rng.standard_normal(nlag)

    merit0, grad0, _ = pycyc.cyclic_merit_and_grad(ht, params, s0, cs_data)
    phi = 0.7
    merit1, grad1, _ = pycyc.cyclic_merit_and_grad(ht * np.exp(1j * phi), params, s0, cs_data)

    assert np.isclose(merit0, merit1)
    np.testing.assert_allclose(grad1, grad0 * np.exp(1j * phi), rtol=1e-10, atol=1e-10)


class _StubFunc:
    """Minimal stand-in for CyclicSolver, providing exactly what
    take_fista_step(backtrack=False) calls: normalize (identity, matching
    conserve_wavefield_energy=False) and evaluate (returns pre-scripted
    (value, grad) pairs in call order -- first call is always with y_n,
    second with x_np1 -- so the test can engineer x_np1 to be a
    phase-drift-dominated update of y_n without reimplementing
    take_fista_step's own x_np1 derivation)."""

    def __init__(self, results):
        self._results = list(results)
        self._calls = 0

    def normalize(self, h):
        return h

    def evaluate(self, w):
        val, grad = self._results[self._calls]
        self._calls += 1
        return val, grad


def test_L_min_bounded_for_phase_drift_dominated_step():
    """The scenario that exposed the bug: x_np1 differs from y_n mostly by
    a small global phase rotation plus a tiny genuine step. var_diff
    (position difference, phase-corrected) correctly shrinks to ~0; L_min
    must not blow up as a result -- it should reflect the tiny genuine
    step's real local curvature, not the degenerate phase drift.

    take_fista_step's wavefield arrays are 2-D (nsubint, nchan) in real use
    (construct_lambda_matrix assumes this, e.g. shape[1] for the delay
    axis) -- the global phase this test targets applies uniformly across
    the whole 2-D array, matching how x_np1/y_n are actually used."""
    rng = np.random.default_rng(1)
    nsub, nchan = 3, 8
    shape = (nsub, nchan)
    y_n = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    alpha = 1.0
    w = np.exp(1j * 0.05)  # small pure global-phase drift
    tiny_step = 1e-6 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    target_x_np1 = w * y_n + tiny_step

    # with normalize=identity and _lambda=None/delay_for_inf covering the
    # whole array (no-op prox), take_fista_step computes
    # x_np1 = y_n - alpha * y_grad -- solve backwards for the y_grad that
    # lands exactly on target_x_np1
    y_grad = (y_n - target_x_np1) / alpha
    func_grad = w * y_grad  # grad(h*e^{i phi}) == e^{i phi}*grad(h), see above

    func = _StubFunc([(0.0, y_grad), (0.0, func_grad)])

    x_n, y_np1, L_min, t_np1, demerits = fista.take_fista_step(
        iter=0,
        func=func,
        backtrack=False,
        alpha=alpha,
        eta=5,
        y_n=y_n,
        _lambda=None,
        delay_for_inf=-nchan,  # keep the whole array active (no infinite penalty)
        zero_penalty_coords=np.array([]),
        fix_phase_value=None,
        fix_phase_coords=None,
        fix_support=np.array([]),
        t_n=1.0,
        x_n=y_n,
        demerits=np.array([1.0]),  # nonzero so `func_val - demerits[-2]` isn't touched at iter=0
        eps=None,
    )

    np.testing.assert_allclose(x_n, target_x_np1, rtol=1e-10, atol=1e-10)
    # a phase-drift-dominated step must not produce a wildly inflated L --
    # before the fix this was ~5e5 for a comparable scenario; the true
    # local curvature here (dominated by the tiny genuine step, not the
    # phase drift) is a modest, finite number
    assert np.isfinite(L_min)
    assert L_min < 100
