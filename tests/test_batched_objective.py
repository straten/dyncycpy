"""
CuPy-by-Claude Stage 1 (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
verifies the new batched siblings of shear_spectra/make_model_cs/
cyclic_merit_and_grad (pycyc.transforms/pycyc.objective) against the
existing, already-verified per-subint functions -- xp=numpy throughout,
no cupy/GPU involved yet (that starts Stage 3). Two independent checks per
function, matching this suite's established discipline: (1) the batched
result must match stacking the per-subint function's results one subint at
a time, and (2) the batched merit's gradient must independently match a
finite-difference check, not just transitively trust the per-subint
version's own finite-difference tests.
"""

import numpy as np
import pytest

import pycyc

from .helpers import make_cs, random_complex, truncate_like_get_cs, wirtinger_finite_diff_grad

NCHAN = 16
NHARM = 6
NLAG = NCHAN
BW_MHZ = 1.0
REF_FREQ_HZ = 1e5
BATCH = 5


def test_shear_spectra_batched_matches_per_subint():
    rng = np.random.default_rng(0)
    phasors = pycyc.create_shear_phasors(NCHAN, NHARM, BW_MHZ, REF_FREQ_HZ)
    spectrum_batch = random_complex(rng, (BATCH, NCHAN))

    hp_batch, hm_batch = pycyc.shear_spectra_batched(spectrum_batch, phasors)

    for i in range(BATCH):
        hp_i, hm_i = pycyc.shear_spectra(spectrum_batch[i], phasors)
        np.testing.assert_allclose(hp_batch[i], hp_i, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(hm_batch[i], hm_i, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("include_Nyquist", [False, True])
@pytest.mark.parametrize("shared_profile", [True, False])
def test_make_model_cs_batched_matches_per_subint(include_Nyquist, shared_profile):
    rng = np.random.default_rng(1)
    params = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ, include_Nyquist=include_Nyquist)

    hf_batch = random_complex(rng, (BATCH, NCHAN))
    s0_batch = random_complex(rng, (NHARM,)) if shared_profile else random_complex(rng, (BATCH, NHARM))

    cs_batch, hp_batch, hm_batch = pycyc.make_model_cs_batched(params, hf_batch, s0_batch)

    for i in range(BATCH):
        s0_i = s0_batch if shared_profile else s0_batch[i]
        cs_i, hp_i, hm_i = pycyc.make_model_cs(params, hf_batch[i], s0_i)
        np.testing.assert_allclose(cs_batch[i], cs_i, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(hp_batch[i], hp_i, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(hm_batch[i], hm_i, rtol=1e-10, atol=1e-10)


def test_make_model_cs_batched_rejects_padding():
    params = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ, pad_cyclic_spectra=True)
    hf_batch = np.zeros((BATCH, NCHAN), dtype=np.complex128)
    s0_batch = np.zeros((NHARM,), dtype=np.complex128)
    with pytest.raises(NotImplementedError):
        pycyc.make_model_cs_batched(params, hf_batch, s0_batch)


@pytest.mark.parametrize("gain_shared", [True, False])
@pytest.mark.parametrize("maxharm", [None, 3])
@pytest.mark.parametrize("exclude_DC", [0, 1])
@pytest.mark.parametrize("shared_profile", [True, False])
def test_cyclic_merit_and_grad_batched_matches_per_subint(gain_shared, maxharm, exclude_DC, shared_profile):
    rng = np.random.default_rng(2)
    params = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ, maxharm=maxharm, exclude_DC=exclude_DC)

    ht_batch = random_complex(rng, (BATCH, NLAG))
    s0_batch = random_complex(rng, (NHARM,)) if shared_profile else random_complex(rng, (BATCH, NHARM))
    cs_data_batch = np.stack(
        [truncate_like_get_cs(random_complex(rng, (NCHAN, NHARM)), params) for _ in range(BATCH)]
    )
    gain_batch = 1.7 if gain_shared else rng.uniform(0.5, 2.0, size=BATCH)

    merit_batch, grad_batch, nonzero_batch = pycyc.cyclic_merit_and_grad_batched(
        ht_batch, params, s0_batch, cs_data_batch, gain_batch=gain_batch
    )

    for i in range(BATCH):
        s0_i = s0_batch if shared_profile else s0_batch[i]
        gain_i = gain_batch if gain_shared else gain_batch[i]
        merit_i, grad_i, nonzero_i = pycyc.cyclic_merit_and_grad(
            ht_batch[i], params, s0_i, cs_data_batch[i], gain=gain_i
        )
        np.testing.assert_allclose(merit_batch[i], merit_i, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(grad_batch[i], grad_i, rtol=1e-9, atol=1e-9)
        assert nonzero_batch[i] == nonzero_i


@pytest.mark.parametrize("shared_profile", [True, False])
def test_cyclic_merit_and_grad_batched_matches_finite_difference(shared_profile):
    """Independent check that the batched gradient is correct, not just
    consistent with the per-subint function -- perturbs one batch element's
    ht while holding the rest fixed, same style as
    test_gradient_regression.py's per-subint check."""
    rng = np.random.default_rng(3)
    params = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ)

    ht_batch = random_complex(rng, (BATCH, NLAG))
    s0_batch = random_complex(rng, (NHARM,)) if shared_profile else random_complex(rng, (BATCH, NHARM))
    cs_data_batch = np.stack(
        [truncate_like_get_cs(random_complex(rng, (NCHAN, NHARM)), params) for _ in range(BATCH)]
    )
    gain_batch = rng.uniform(0.5, 2.0, size=BATCH)
    probe = 2  # which batch element to perturb

    def merit_only(ht_probe):
        ht = ht_batch.copy()
        ht[probe] = ht_probe
        merit, _grad, _nonzero = pycyc.cyclic_merit_and_grad_batched(
            ht, params, s0_batch, cs_data_batch, gain_batch=gain_batch
        )
        return merit[probe]

    analytic_merit, analytic_grad, _ = pycyc.cyclic_merit_and_grad_batched(
        ht_batch, params, s0_batch, cs_data_batch, gain_batch=gain_batch
    )
    numeric_grad = wirtinger_finite_diff_grad(merit_only, ht_batch[probe])

    np.testing.assert_allclose(analytic_grad[probe], numeric_grad, rtol=1e-5, atol=1e-5)


def test_cyclic_merit_and_grad_batched_rejects_diagnostics():
    params = make_cs(NCHAN, NHARM, NLAG, BW_MHZ, REF_FREQ_HZ)
    ht_batch = np.zeros((BATCH, NLAG), dtype=np.complex128)
    s0_batch = np.zeros((NHARM,), dtype=np.complex128)
    cs_data_batch = np.zeros((BATCH, NCHAN, NHARM), dtype=np.complex128)

    with pytest.raises(NotImplementedError):
        pycyc.cyclic_merit_and_grad_batched(ht_batch, params, s0_batch, cs_data_batch, dump_residual=True)
    with pytest.raises(NotImplementedError):
        pycyc.cyclic_merit_and_grad_batched(
            ht_batch, params, s0_batch, cs_data_batch, on_residual=lambda *a: None
        )
