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

import pycyc

from .helpers import make_cs, random_complex, truncate_like_get_cs

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
        merit, grad, _ = pycyc.complex_cyclic_merit_lag(ht, cs, s0, cs_data, 1.0)
        merits.append(merit)
        ht = ht - step * grad
    merit_final, _, _ = pycyc.complex_cyclic_merit_lag(ht, cs, s0, cs_data, 1.0)
    merits.append(merit_final)

    np.testing.assert_allclose(merits, GOLDEN_MERITS, rtol=1e-8)
