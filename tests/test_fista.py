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

from .helpers import RealMultiSubintFunc


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


def _l_min_for_independent_subint_drift(rng_seed, phi_spread, nsub=10, nchan=8, alpha=1.0):
    """Builds a scenario in the time-delay domain (each row an
    independent subint, with its own independent phase drift phi_t plus a
    tiny genuine step), transforms to Doppler-delay (what take_fista_step
    actually operates on, exactly as CyclicSolver.evaluate returns), and
    runs it through the real take_fista_step. Returns (x_n in Doppler-delay,
    L_min, target_x_np1_time) so callers can check both correctness and
    the stability of L_min across different drift magnitudes."""
    rng = np.random.default_rng(rng_seed)
    shape = (nsub, nchan)
    y_n_time = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    phi_t = rng.uniform(-phi_spread, phi_spread, nsub)
    tiny_step_time = 1e-6 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    target_x_np1_time = np.exp(1j * phi_t)[:, None] * y_n_time + tiny_step_time
    y_grad_time = (y_n_time - target_x_np1_time) / alpha
    func_grad_time = np.exp(1j * phi_t)[:, None] * y_grad_time  # per-subint grad(h*e^{iphi})==e^{iphi}*grad(h)

    y_n = pycyc.time2freq(y_n_time, axis=0)
    y_grad = pycyc.time2freq(y_grad_time, axis=0)
    func_grad = pycyc.time2freq(func_grad_time, axis=0)

    func = _StubFunc([(0.0, y_grad), (0.0, func_grad)])
    x_n, y_np1, L_min, t_np1, demerits = fista.take_fista_step(
        iter=0,
        func=func,
        backtrack=False,
        alpha=alpha,
        eta=5,
        y_n=y_n,
        _lambda=None,
        delay_for_inf=-nchan,
        zero_penalty_coords=np.array([]),
        fix_phase_value=None,
        fix_phase_coords=None,
        fix_support=np.array([]),
        t_n=1.0,
        x_n=y_n,
        demerits=np.array([1.0]),
        eps=None,
    )
    return x_n, L_min, target_x_np1_time


def _per_subint_L_min_formula(x_np1_time, y_n_time, y_grad_time, func_grad_time):
    """Direct implementation of the per-subint formula
    test_L_min_matches_reference_per_subint_formula (below) already
    confirms take_fista_step's real code computes exactly -- used here to
    probe the formula's *properties* against independently-constructed
    (not reverse-engineered through take_fista_step's x_np1 = y_n -
    alpha*y_grad constraint) inputs. That constraint ties y_grad's value
    to whatever x_np1 is chosen, which contaminates a same-y_grad,
    different-x_np1 stability comparison -- driving genuinely independent
    inputs through take_fista_step itself isn't possible without hitting
    that same coupling, so this tests the formula on its own, with
    take_fista_step's equivalence already established separately."""
    z_t = np.sum(np.conj(x_np1_time) * y_n_time, axis=1)
    z_t /= np.abs(z_t)
    diff_t = z_t[:, None] * x_np1_time - y_n_time
    var_diff_t = np.vdot(diff_t, diff_t)
    gdiff_t = y_grad_time - z_t[:, None] * func_grad_time
    return np.sqrt(np.real(np.vdot(gdiff_t, gdiff_t) / var_diff_t))


def _l_min_stability_case(rng_seed, phi_spread, nsub=10, nchan=16, nharm=6):
    """y_n_time and its real gradient (y_grad_time) are independent of
    phi_spread -- only x_np1_time's per-subint rotation (and the gradient
    evaluated there) varies, giving a genuine like-for-like comparison
    across drift magnitudes."""
    rng = np.random.default_rng(rng_seed)
    bw, ref_freq, nlag = 1.0, 1e5, nchan
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
    cs_data_per_subint = [
        rng.standard_normal((nchan, nharm)) + 1j * rng.standard_normal((nchan, nharm)) for _ in range(nsub)
    ]
    y_n_time = rng.standard_normal((nsub, nlag)) + 1j * rng.standard_normal((nsub, nlag))
    phi_t = rng.uniform(-phi_spread, phi_spread, nsub)
    tiny_step_time = 1e-6 * (rng.standard_normal((nsub, nlag)) + 1j * rng.standard_normal((nsub, nlag)))
    x_np1_time = np.exp(1j * phi_t)[:, None] * y_n_time + tiny_step_time

    y_grad_time = np.zeros_like(y_n_time)
    func_grad_time = np.zeros_like(y_n_time)
    for t in range(nsub):
        _, y_grad_time[t], _ = pycyc.cyclic_merit_and_grad(y_n_time[t], params, s0, cs_data_per_subint[t])
        _, func_grad_time[t], _ = pycyc.cyclic_merit_and_grad(x_np1_time[t], params, s0, cs_data_per_subint[t])

    return _per_subint_L_min_formula(x_np1_time, y_n_time, y_grad_time, func_grad_time)


def test_L_min_stable_under_independent_per_subint_phase_drift():
    """The per-subint generalization: with independent (not shared) phase
    drift per subint, a single global phasor can only remove the mean
    component -- L_min must stay close to the same value regardless of
    how large the independent per-subint drift is, since none of it is
    genuine wavefield movement. y_n and its real gradient are identical
    (same seed) between the two calls; only the injected drift differs.

    This per-subint formula's own var_diff/gdiff are provably exact
    regardless of drift magnitude (each row's contribution is fully
    removed by construction, not just approximately cancelled) -- unlike
    a single shared phasor, whose var_diff/gdiff can each individually be
    inflated by many orders of magnitude by independent per-subint drift
    (confirmed separately during development). Whether that contamination
    also visibly distorts the final L *ratio* turned out to depend on the
    specific noise realization in testing -- sometimes it stayed nearly
    as stable as this formula's, sometimes not -- so this test asserts
    the property this formula is actually guaranteed to have (stability),
    not a specific side-by-side margin over the single-phasor version."""
    seed = 101
    L_small_spread = _l_min_stability_case(seed, phi_spread=0.1)
    L_large_spread = _l_min_stability_case(seed, phi_spread=0.5)

    assert np.isfinite(L_small_spread) and np.isfinite(L_large_spread)
    np.testing.assert_allclose(L_large_spread, L_small_spread, rtol=0.1)


def test_L_min_matches_reference_per_subint_formula():
    """Direct check that take_fista_step's actual L_min matches the
    per-subint formula it's documented to implement (time-delay domain,
    per-row phase alignment applied to both position and gradient
    differences) -- not just "doesn't blow up," but the specific,
    derived value."""
    seed = 7
    x_n, L_min, target_x_np1_time = _l_min_for_independent_subint_drift(seed, phi_spread=0.3)

    rng = np.random.default_rng(seed)
    shape = (10, 8)
    y_n_time = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    phi_t = rng.uniform(-0.3, 0.3, 10)
    tiny_step_time = 1e-6 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    target_x_np1_time_ref = np.exp(1j * phi_t)[:, None] * y_n_time + tiny_step_time
    y_grad_time = y_n_time - target_x_np1_time_ref
    func_grad_time = np.exp(1j * phi_t)[:, None] * y_grad_time

    z_t = np.sum(np.conj(target_x_np1_time_ref) * y_n_time, axis=1)
    z_t /= np.abs(z_t)
    diff_t = z_t[:, None] * target_x_np1_time_ref - y_n_time
    var_diff_t = np.vdot(diff_t, diff_t)
    gdiff_t = y_grad_time - z_t[:, None] * func_grad_time
    L_ref = np.sqrt(np.real(np.vdot(gdiff_t, gdiff_t) / var_diff_t))

    np.testing.assert_allclose(L_min, L_ref, rtol=1e-8)


def test_no_gauge_drift_over_long_fista_run():
    """Regression test for the reasoning documented in pycyc.tex
    (degenerate-phase section, "Stability of the gauge alignment under
    FISTA/momentum dynamics"): the per-subint gauge-alignment angle used
    by L_min must not accumulate drift over a long FISTA trajectory --
    including through the Nesterov momentum term, whose exactness is not
    obvious a priori (it's a linear combination of two different
    iterations' gradient-descended points, not a single step) -- because
    the merit function's phase invariance is exact (not a small-angle
    approximation) at every point visited along the trajectory, not just
    the starting point.

    Drives the real fista.take_fista_step (not a reimplementation) for
    300 iterations of a genuine multi-subint gradient-descent-with-
    momentum trajectory (real pycyc.cyclic_merit_and_grad gradients, no
    prox/normalize to keep the check focused on the gradient+momentum
    dynamics specifically), independently recomputing the global and
    per-subint alignment angles after each call from that call's own
    inputs/outputs (x_np1, y_n) -- not from any value take_fista_step
    itself returns, so this doesn't just re-check the formula, it checks
    the actual trajectory's gauge stability."""
    rng = np.random.default_rng(9)
    nsub, nchan, nharm = 5, 16, 6
    bw, ref_freq, nlag = 1.0, 1e5, nchan
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
    cs_data_per_subint = [
        rng.standard_normal((nchan, nharm)) + 1j * rng.standard_normal((nchan, nharm)) for _ in range(nsub)
    ]
    func = RealMultiSubintFunc(params, s0, cs_data_per_subint)

    y_n_time = rng.standard_normal((nsub, nlag)) + 1j * rng.standard_normal((nsub, nlag))
    y_n = pycyc.time2freq(y_n_time, axis=0)
    x_n = y_n.copy()
    t_n = 1.0
    alpha = 1e-6
    demerits = np.array([1.0])

    max_angle_global_deg = 0.0
    max_angle_subint_deg = 0.0
    n_iter = 300

    for i in range(n_iter):
        x_n_new, y_n_new, L, t_n, demerits = fista.take_fista_step(
            iter=i,
            func=func,
            backtrack=False,
            alpha=alpha,
            eta=5,
            y_n=y_n,
            _lambda=None,
            delay_for_inf=-nchan,  # no-op prox, isolates the gradient+momentum dynamics
            zero_penalty_coords=np.array([]),
            fix_phase_value=None,
            fix_phase_coords=None,
            fix_support=np.array([]),
            t_n=t_n,
            x_n=x_n,
            demerits=demerits,
            eps=None,
        )
        assert np.isfinite(L)

        # independently recompute both alignment angles from this call's
        # own inputs (y_n) and outputs (x_n_new == x_np1) -- not derived
        # from L or from take_fista_step's internals
        z = np.vdot(x_n_new, y_n)
        z /= np.abs(z)
        max_angle_global_deg = max(max_angle_global_deg, abs(np.degrees(np.angle(z))))

        x_time = pycyc.freq2time(x_n_new, axis=0)
        y_time = pycyc.freq2time(y_n, axis=0)
        z_t = np.sum(np.conj(x_time) * y_time, axis=1)
        z_t /= np.abs(z_t)
        max_angle_subint_deg = max(max_angle_subint_deg, float(np.max(np.abs(np.degrees(np.angle(z_t))))))

        x_n, y_n = x_n_new, y_n_new
        alpha = 1.0 / max(L, 1.0)  # adapt like a real FISTA loop would

    # Both angles must stay at floating-point noise level (~1e-14 deg
    # observed) over the whole run, many orders of magnitude below
    # anything that would indicate real accumulation -- 1e-6 deg leaves
    # generous headroom above the noise floor while still catching any
    # genuine systematic drift.
    assert max_angle_global_deg < 1e-6, f"global alignment angle drifted: {max_angle_global_deg:.3e} deg"
    assert max_angle_subint_deg < 1e-6, f"per-subint alignment angle drifted: {max_angle_subint_deg:.3e} deg"
