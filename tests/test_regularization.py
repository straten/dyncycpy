"""
Regression test for a fragility found while verifying
dynamic_frequency_response_refactor's behavior-preservation on real data
(see /home/wvanstra/.claude/plans/lovely-forging-mountain.md, Stage 7):
apply_delay_shrinkage_threshold's noise estimate (delay_noise_power_wavefield)
counted "informative" samples via np.count_nonzero's exact bit-equality test.
An array that is mathematically zero in some bins (permanently causally-
zeroed delay columns, or bins already zeroed by a prior shrinkage pass) can
pick up ~1e-14-relative floating-point noise from an unrelated Fourier
round-trip elsewhere in the pipeline (a domain change that is a mathematical
identity but not bit-exact) without becoming physically nonzero -- the exact
count then miscounts such bins as informative, which was found to move the
estimated noise floor (and hence how much of a real wavefield gets shrunk to
zero) by tens of percent on real P2067 data. Fixed by comparing against a
relative-epsilon floor instead of exact zero.
"""

import numpy as np

from pycyc.regularization import apply_delay_shrinkage_threshold


def _synthetic_wavefield(rng, ndoppler=64, ndelay=32, zero_frac=0.6):
    """A synthetic h(tau;omega)-like array: mostly small values (simulating
    an already-shrinkage-zeroed background) with a sparse set of larger
    "signal" bins survived from a previous shrinkage pass -- exactly the
    kind of array apply_delay_shrinkage_threshold sees on its second
    (updateWavefield) call in the real pipeline."""
    x = 0.05 * (rng.standard_normal((ndoppler, ndelay)) + 1j * rng.standard_normal((ndoppler, ndelay)))
    zero_mask = rng.random((ndoppler, ndelay)) < zero_frac
    x[zero_mask] = 0.0
    signal_mask = (rng.random((ndoppler, ndelay)) < 0.05) & ~zero_mask
    nsignal = np.count_nonzero(signal_mask)
    x[signal_mask] = 10.0 * (rng.standard_normal(nsignal) + 1j * rng.standard_normal(nsignal))
    return x


def test_delay_shrinkage_threshold_robust_to_sub_epsilon_perturbation_of_zeros():
    rng = np.random.default_rng(42)
    x = _synthetic_wavefield(rng)
    zero_mask = x == 0.0
    assert np.count_nonzero(zero_mask) > 0, "test fixture must contain exact zeros"

    # Perturb only the exact zeros, by an amount representative of FFT
    # round-off noise relative to the array's own scale (~1e-14 relative) --
    # everything that was already nonzero is left bit-identical.
    scale = np.max(np.abs(x))
    tiny = 1e-14 * scale
    perturbation = tiny * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    x_perturbed = x.copy()
    x_perturbed[zero_mask] += perturbation[zero_mask]
    assert np.all(x_perturbed[zero_mask] != 0.0), "perturbation must actually break exact-zero equality"

    out = apply_delay_shrinkage_threshold(x.copy(), threshold=1.0, baseline_threshold=2.0)
    out_perturbed = apply_delay_shrinkage_threshold(x_perturbed.copy(), threshold=1.0, baseline_threshold=2.0)

    zero_frac = np.mean(out == 0.0)
    zero_frac_perturbed = np.mean(out_perturbed == 0.0)
    assert abs(zero_frac - zero_frac_perturbed) < 1e-3, (
        f"sub-epsilon perturbation of exact zeros changed the shrinkage zero-fraction "
        f"from {zero_frac} to {zero_frac_perturbed} -- the noise estimate is not robust "
        f"to floating-point-level noise in already-zeroed bins"
    )
    np.testing.assert_allclose(
        np.sum(np.abs(out) ** 2), np.sum(np.abs(out_perturbed) ** 2), rtol=1e-6
    )
