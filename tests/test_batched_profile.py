"""
CuPy-by-Claude Stage 5 (see /home/willem/.claude/plans/stateful-dreaming-wall.md):
CyclicSolver.updateProfile_batched, compared end-to-end against the
existing (threaded/serial) updateProfile on synthetic multi-subint data.
Lower priority per the plan (a bonus once the FISTA gradient path itself
was done), but benefits both --solver fista and --solver outer, and
reuses Stage 1-3's infrastructure directly.

Reuses test_outer_loop's synthetic-solver builder (already comprehensive
enough for the *full* updateProfile, not just the batched math in
isolation): both branches call the real updateProfile() once first with
use_gpu=False to let it establish self.pp_scattered (mirroring
initProfile()'s own first call in real use), matching
updateProfile_batched's documented requirement that
compute_scattered_profile/save_dynamic_spectrum never be True when it's
invoked -- true in real use because cycsolve.py only sets self.use_gpu
after initProfile() completes.

The "matches threaded" comparison forces xp=numpy via monkeypatching
pycyc.solver.get_xp (not pycyc.backend.get_xp -- solver.py's `from
.backend import get_xp` binds its own name, unaffected by patching the
original module attribute after that import), even though self.use_gpu=True
is what exercises the real if self.use_gpu: dispatch inside
updateProfile() -- this keeps the test both a real integration check of
that dispatch wiring *and* independent of whether cupy happens to be
installed on the machine running the suite, matching Stage 1/2's
xp=numpy-guaranteed tight-tolerance style. GPU/complex64 precision is
covered separately in test_batched_gpu.py (importorskip-gated, loose
tolerance), the same split Stage 3 established for compute_gradient_batched.
"""

import numpy as np
import pytest

import pycyc

from .test_outer_loop import _build_synthetic_cyclic_solver


def _prime_and_compare(rng_seed, nsubint, model_gain_variations, chunk_size=None, monkeypatch=None):
    rng = np.random.default_rng(rng_seed)
    CS_threaded = _build_synthetic_cyclic_solver(rng, nsubint=nsubint, inject_jitter=False)
    rng = np.random.default_rng(rng_seed)
    CS_batched = _build_synthetic_cyclic_solver(rng, nsubint=nsubint, inject_jitter=False)

    for CS in (CS_threaded, CS_batched):
        CS.model_gain_variations = model_gain_variations
        if model_gain_variations:
            CS.optimal_gains = np.ones(nsubint)
        # first call (use_gpu=False for both) establishes self.pp_scattered,
        # mirroring initProfile()'s own one-time first call in real use
        CS.updateProfile()

    CS_batched.use_gpu = True
    if chunk_size is not None:
        CS_batched.gpu_chunk_size = chunk_size
    if monkeypatch is not None:
        monkeypatch.setattr(pycyc.solver, "get_xp", lambda use_gpu: np)
        # gpu_dtype's downcast is keyed on self.use_gpu, not on which
        # backend get_xp actually returns -- disable it too, so forcing
        # numpy here gives a genuinely lossless (not just GPU-avoiding)
        # comparison against the threaded/complex128 path
        CS_batched.gpu_dtype = None

    CS_threaded.updateProfile()
    CS_batched.updateProfile()
    return CS_threaded, CS_batched


@pytest.mark.parametrize("model_gain_variations", [False, True])
def test_update_profile_batched_matches_threaded(model_gain_variations, monkeypatch):
    CS_threaded, CS_batched = _prime_and_compare(
        101, nsubint=6, model_gain_variations=model_gain_variations, monkeypatch=monkeypatch
    )

    np.testing.assert_allclose(CS_batched.pp_intrinsic, CS_threaded.pp_intrinsic, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(CS_batched.ph_numer, CS_threaded.ph_numer, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(CS_batched.ph_denom, CS_threaded.ph_denom, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(CS_batched.intrinsic_profiles, CS_threaded.intrinsic_profiles, rtol=1e-8, atol=1e-8)
    if model_gain_variations:
        np.testing.assert_allclose(CS_batched.optimal_gains, CS_threaded.optimal_gains, rtol=1e-8, atol=1e-8)


def test_update_profile_batched_chunking_matches_single_batch(monkeypatch):
    CS_threaded, CS_whole = _prime_and_compare(
        103, nsubint=9, model_gain_variations=False, monkeypatch=monkeypatch
    )
    _, CS_chunked = _prime_and_compare(
        103, nsubint=9, model_gain_variations=False, chunk_size=4, monkeypatch=monkeypatch
    )

    np.testing.assert_allclose(CS_chunked.pp_intrinsic, CS_whole.pp_intrinsic, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(CS_chunked.intrinsic_profiles, CS_whole.intrinsic_profiles, rtol=1e-8, atol=1e-8)


def test_update_profile_batched_rejects_scattered_profile():
    """The very first updateProfile() call (pp_scattered still None) sets
    compute_scattered_profile=True -- updateProfile_batched must refuse
    that case rather than silently skip/mishandle it, since real usage
    (cycsolve.py) relies on use_gpu only being set after that first call
    already happened through the ordinary path."""
    rng = np.random.default_rng(107)
    CS = _build_synthetic_cyclic_solver(rng, nsubint=4, inject_jitter=False)
    CS.use_gpu = True
    assert CS.pp_scattered is None
    with pytest.raises(NotImplementedError):
        CS.updateProfile()


def test_update_profile_batched_rejects_non_include_Nyquist():
    rng = np.random.default_rng(109)
    CS = _build_synthetic_cyclic_solver(rng, nsubint=4, inject_jitter=False)
    CS.updateProfile()  # prime pp_scattered with the ordinary path first
    CS.use_gpu = True
    CS.include_Nyquist = False
    with pytest.raises(NotImplementedError):
        CS.updateProfile()
