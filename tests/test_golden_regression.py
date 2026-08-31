"""
Small synthetic end-to-end golden regression test.

Freezes the current numerical behavior of make_model_cs +
complex_cyclic_merit_lag across a few plain gradient-descent steps (not
scipy's L-BFGS-B, to keep this deterministic and dependency-free). The exact
step rule is not meant to be a "correct" optimizer -- it's just a fixed,
reproducible sequence of calls whose output is pinned here so that later
refactor stages (which are meant to reorganize, not change, the math) can be
verified against it. Regenerate the golden values with the snippet in the
refactor plan's Stage 0 notes only if a stage *intentionally* changes
behavior (e.g. Stage 2's maxharm-fragility fix).
"""

import numpy as np

import fista
import pycyc

from .helpers import RealMultiSubintFunc, make_cs, random_complex, truncate_like_get_cs

GOLDEN_MERITS = [
    77.80795610652605,
    77.77286168847351,
    77.7374314968219,
    77.70165054143055,
    77.66550364338947,
    77.62897542888709,
]


def test_golden_merit_trajectory():
    rng = np.random.default_rng(42)
    nchan, nharm, nlag = 16, 6, 16
    bw, ref_freq = 1.0, 1e5
    cs = make_cs(nchan, nharm, nlag, bw, ref_freq, maxharm=4, pad_cyclic_spectra=True)

    s0 = random_complex(rng, (nharm,))
    ht_true = random_complex(rng, (nlag,))
    hf_true = pycyc.time2freq(ht_true)
    cs_model_true, _, _ = pycyc.make_model_cs(cs, hf_true, s0)
    noise = 0.05 * random_complex(rng, cs_model_true.shape)
    cs_data = truncate_like_get_cs(cs_model_true + noise, cs)

    ht = np.zeros(nlag, dtype=np.complex128)
    ht[0] = 1.0 + 0j

    step = 0.001
    merits = []
    for _ in range(5):
        merit, grad, _ = pycyc.cyclic_merit_and_grad(ht, cs, s0, cs_data, gain=1.0)
        merits.append(merit)
        ht = ht - step * grad
    merit_final, _, _ = pycyc.cyclic_merit_and_grad(ht, cs, s0, cs_data, gain=1.0)
    merits.append(merit_final)

    np.testing.assert_allclose(merits, GOLDEN_MERITS, rtol=1e-8)


# Stage 0 fixture for the dynamic_frequency_response_refactor branch (see
# /home/wvanstra/.claude/plans/lovely-forging-mountain.md): unlike
# test_golden_merit_trajectory above (single subint, plain gradient descent,
# no domain-transform bookkeeping to speak of), this drives a genuine
# multi-subint FISTA trajectory -- including the Nesterov momentum
# recursion and the CyclicSolver.evaluate-style h_doppler_delay <->
# h_time_delay transform boundary (RealMultiSubintFunc.evaluate) -- through
# the real fista.take_fista_step. That boundary is exactly what the refactor
# relocates (Stages 2-5), so this is the trajectory most likely to notice an
# unintended behavior change as those stages land. Regenerate GOLDEN_MERITS_FISTA
# only if a stage *intentionally* changes behavior; every structural stage
# (2-6) should leave it unchanged, since none of them are meant to touch the
# math, only which array is treated as persistent state.
#
# Regenerated once, intentionally: this fixture uses include_Nyquist=True
# with fully random complex cs_data (no constraint on the Nyquist harmonic
# being real), which is exactly the case pycyc.objective.make_model_cs's
# Nyquist-harmonic gradient fix (.abs() -> .real(), plus the matching
# cyclic_merit_and_grad correction -- see tests/test_gradient_regression.py)
# changes; these values reflect the corrected gradient.
GOLDEN_MERITS_FISTA = [
    4222.114088567756,
    4221.422400594005,
    4220.536133858493,
    4219.460459675667,
    4218.19867223329,
    4216.753140473786,
    4215.12570982791,
    4213.317904521486,
    4211.331041533142,
    4209.1663000166545,
    4206.824766123768,
    4204.307463272442,
    4201.615373327308,
    4198.749451853565,
    4195.710639364232,
    4192.4998697753035,
    4189.11807686388,
    4185.566199265303,
    4181.845184380006,
    4177.9559914522015,
]
GOLDEN_FINAL_POWER_FISTA = 138.02613929412604


def test_golden_fista_trajectory_multi_subint():
    rng = np.random.default_rng(21)
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
    s0 = random_complex(rng, (nharm,))
    cs_data_per_subint = [random_complex(rng, (nchan, nharm)) for _ in range(nsub)]
    func = RealMultiSubintFunc(params, s0, cs_data_per_subint)

    y_n_time = random_complex(rng, (nsub, nlag))
    y_n = pycyc.time2freq(y_n_time, axis=0)
    x_n = y_n.copy()
    t_n = 1.0
    alpha = 1e-6
    demerits = np.array([1.0])

    n_iter = 20
    for i in range(n_iter):
        x_n, y_n, L, t_n, demerits = fista.take_fista_step(
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

    np.testing.assert_allclose(demerits[1:], GOLDEN_MERITS_FISTA, rtol=1e-8)
    np.testing.assert_allclose(np.sum(np.abs(x_n) ** 2), GOLDEN_FINAL_POWER_FISTA, rtol=1e-8)
