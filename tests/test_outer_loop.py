"""
Stage 4 of the jitter/degeneracy plan (see
/home/willem/.claude/plans/stateful-dreaming-wall.md): outer-loop
orchestration tying Stages 1-3 together (CyclicSolver.outer_loop /
._refresh_jitter_model).

End-to-end integration smoke test on a small synthetic multi-subint
dataset -- a different style from the rest of this suite, appropriate for
orchestration code with too much shared mutable state between steps to
usefully finite-difference or Monte-Carlo test in isolation. Builds a
minimal CyclicSolver by hand (bypassing __init__'s PSRFITS loading, the
same approach used for CyclicSolver-level checks in the earlier refactor's
test suite), with a known wavefield and a known rigid-shift-plus-low-rank
jitter profile injected into synthetic cyclic spectra, and confirms the
whole pipeline (per-subint wavefield fit -> profile pooling -> epsilon/gain
re-fit -> PCA refresh -> jitter-aware profile fed back into the next pass)
runs to completion, drives the merit down, and finds/stabilizes on the
injected jitter rank.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

import pycyc


def _build_synthetic_cyclic_solver(rng, nsubint=8, inject_jitter=True):
    nchan, nphase = 16, 10
    nharm = nphase // 2 + 1  # 6, matches include_Nyquist=True below
    nlag = nchan

    CS = object.__new__(pycyc.CyclicSolver)
    CS.bw = 1.0
    CS.ref_freq = 1e5
    CS.pad_cyclic_spectra = False
    CS.include_Nyquist = True
    CS.maxharm = None
    CS.exclude_DC = 0
    CS.nlag = nlag
    CS.nchan = nchan
    CS.nharm = nharm
    CS.nphase = nphase
    CS.nspec = nsubint
    CS.nsubint = nsubint
    CS.npol = 1
    CS.shear_phasors = pycyc.create_shear_phasors(nchan, nharm, CS.bw, CS.ref_freq)
    CS.dump_residual = False
    CS.dump_gradient = False
    CS.iprint = 0
    CS.rindex = 0
    CS.filename = "synthetic.ar"
    CS.save_cyclic_spectra = True
    CS.save_dynamic_spectrum = False
    CS.model_gain_variations = False
    CS.normalize_cyclic_spectra = False
    CS.nthread = 1
    CS.hf_prev = None
    CS.nopt = 0
    CS.delay_taper = None
    CS.maxinitharm = None
    CS.optimized_filters = np.zeros((nsubint, nchan), dtype=np.complex128)
    CS.intrinsic_profiles = np.zeros((nsubint, 1, nphase))
    CS.dynamic_spectrum = np.zeros((nsubint, nchan))
    CS.model_jitter = True
    CS.jitter_warmup_passes = 1
    CS.jitter_max_rank = None
    CS.jitter_profiles = None
    CS.jitter_rank = 0
    CS._jitter_basis = None
    CS.jitter_principal_angle = None
    CS.intrinsic_ph_sum = None
    CS.intrinsic_ph_sumsq = None
    CS.compute_scattered_profile = False
    CS.pp_scattered = None
    CS.scattered_profiles = np.zeros((nsubint, nphase))
    CS.ph_numer = np.zeros((1, nsubint, nharm), dtype=np.complex128)
    CS.ph_denom = np.zeros((1, nsubint, nharm), dtype=np.complex128)
    CS.intrinsic_ph = np.zeros((1, nharm), dtype=np.complex128)
    CS.cs_norm = np.zeros((nsubint, 1))
    CS.align_frequency_responses = False
    CS.reduce_temporal_phase_noise = False
    CS.minimize_spectral_entropy = False
    CS.minimize_spectral_entropy_delay = False
    CS.enforce_orthogonal_real_imag = False
    CS.enforce_real_at_origin = False
    CS.conserve_wavefield_energy = False

    ht_true = rng.standard_normal(nlag) + 1j * rng.standard_normal(nlag)
    ht_true[0] += 3.0  # dominant lag, easy for loop() to identify via phase_gradient
    hf_true = pycyc.time2freq(ht_true)
    s0_true = rng.standard_normal(nharm) + 1j * rng.standard_normal(nharm)

    true_eps = rng.uniform(-0.3, 0.3, nsubint)
    alpha = np.arange(nharm)
    s_t_true = s0_true[None, :] * np.exp(1j * np.outer(true_eps, alpha))
    if inject_jitter:
        jitter_basis_true = (rng.standard_normal((2, nharm)) + 1j * rng.standard_normal((2, nharm))) * 0.3
        jitter_weights_true = (rng.standard_normal((nsubint, 2)) + 1j * rng.standard_normal((nsubint, 2))) * 0.5
        s_t_true = s_t_true + jitter_weights_true @ jitter_basis_true

    params = CS._model_params()
    cyclic_spectra = np.zeros((nsubint, 1, nchan, nharm), dtype=np.complex128)
    for isub in range(nsubint):
        cs_model, _hp, _hm = pycyc.make_model_cs(params, hf_true, s_t_true[isub])
        noise = 0.02 * (rng.standard_normal((nchan, nharm)) + 1j * rng.standard_normal((nchan, nharm)))
        cyclic_spectra[isub, 0] = cs_model + noise
    CS.cyclic_spectra = cyclic_spectra

    # rough (perturbed, not exact) initial wavefield guess
    ht_guess = ht_true + 0.5 * (rng.standard_normal(nlag) + 1j * rng.standard_normal(nlag))
    CS.h_time_delay = np.tile(ht_guess, (nsubint, 1))
    CS.h_doppler_delay = pycyc.time2freq(CS.h_time_delay, axis=0)

    ph0 = np.zeros(nharm, dtype=np.complex128)
    ph0[1] = 1.0  # normalize_profile divides by ph[1]; must be nonzero
    CS.pp_intrinsic = CS.harm2phase(ph0)

    return CS


def test_outer_loop_runs_end_to_end_and_lowers_merit():
    rng = np.random.default_rng(7)
    CS = _build_synthetic_cyclic_solver(rng, inject_jitter=True)

    pass_merits = []
    loop_kwargs = dict(make_plots=False, maxfun=20, iprint=-1, use_minphase=False)

    # run one pass at a time so we can track merit per pass without
    # depending on solver.py's internal logging
    for pass_idx in range(4):
        total_merit = 0.0
        for isub in range(CS.nspec):
            CS.loop(isub=isub, **loop_kwargs)
            if CS.objval:
                total_merit += CS.objval[-1]
        CS.updateProfile()
        if CS.model_jitter and pass_idx >= CS.jitter_warmup_passes:
            CS._refresh_jitter_model()
        pass_merits.append(total_merit)

    assert pass_merits[-1] < pass_merits[0] * 0.5  # substantial improvement, not just noise
    assert CS.jitter_rank > 0  # found the injected jitter
    assert CS.jitter_profiles is not None
    assert CS.jitter_profiles.shape == (CS.nspec, CS.nharm)


def test_outer_loop_method_matches_manual_pass_loop():
    """CyclicSolver.outer_loop itself (not just the manual per-pass loop
    used above to track merit) runs to completion and reaches the same
    kind of state -- rank found, profiles populated -- confirming the
    public entry point is wired correctly, not just its pieces."""
    rng = np.random.default_rng(11)
    CS = _build_synthetic_cyclic_solver(rng, inject_jitter=True)

    CS.outer_loop(n_passes=3, loop_kwargs=dict(make_plots=False, maxfun=20, iprint=-1, use_minphase=False))

    assert CS.jitter_rank > 0
    assert CS.jitter_profiles is not None


def test_outer_loop_basis_stabilizes_across_passes():
    """With model_jitter on for several passes past warmup, the recovered
    jitter rank and basis should stabilize (principal angle between
    successive passes' bases shrinking towards ~0), rather than continuing
    to change indefinitely -- the convergence signature the plan's Stage 4
    diagnostics were designed to surface."""
    rng = np.random.default_rng(7)
    CS = _build_synthetic_cyclic_solver(rng, inject_jitter=True)
    CS.jitter_warmup_passes = 1

    loop_kwargs = dict(make_plots=False, maxfun=20, iprint=-1, use_minphase=False)
    angles = []
    for pass_idx in range(4):
        for isub in range(CS.nspec):
            CS.loop(isub=isub, **loop_kwargs)
        CS.updateProfile()
        if pass_idx >= CS.jitter_warmup_passes:
            basis_before = CS._jitter_basis
            CS._refresh_jitter_model()
            if basis_before is not None:
                angle = pycyc.subspace_principal_angle(basis_before, CS._jitter_basis)
                angles.append(angle)

    assert len(angles) >= 1
    assert angles[-1] < 1e-3  # basis has essentially stopped changing by the last pass


def test_outer_loop_without_jitter_model_leaves_jitter_rank_zero():
    """With model_jitter off, outer_loop must reduce to today's ordinary
    fixed-profile behavior -- no jitter refresh should ever run."""
    rng = np.random.default_rng(13)
    CS = _build_synthetic_cyclic_solver(rng, inject_jitter=False)
    CS.model_jitter = False

    CS.outer_loop(n_passes=2, loop_kwargs=dict(make_plots=False, maxfun=10, iprint=-1, use_minphase=False))

    assert CS.jitter_rank == 0
    assert CS.jitter_profiles is None


def test_outer_loop_stops_early_once_converged():
    """Given a generous n_passes cap, outer_loop should stop itself once
    merit (and, once active, the jitter basis) has stabilized for
    `patience` consecutive passes, rather than always running to
    n_passes -- the stopping condition outer_loop previously lacked."""
    rng = np.random.default_rng(7)
    CS = _build_synthetic_cyclic_solver(rng, inject_jitter=True)
    CS.jitter_warmup_passes = 1

    call_count = {"loop": 0}
    real_loop = CS.loop

    def counting_loop(*args, **kwargs):
        call_count["loop"] += 1
        return real_loop(*args, **kwargs)

    CS.loop = counting_loop

    CS.outer_loop(
        n_passes=20,
        loop_kwargs=dict(make_plots=False, maxfun=20, iprint=-1, use_minphase=False),
    )

    passes_run = call_count["loop"] / CS.nspec
    assert passes_run < 20  # stopped before exhausting the hard cap
    assert CS.jitter_rank > 0


def test_outer_loop_patience_disables_early_stopping():
    """patience<=0 must recover the old always-run-n_passes behavior
    exactly, for callers (e.g. scripts driving outer_loop with their own
    fixed iteration budget) that don't want early stopping."""
    rng = np.random.default_rng(13)
    CS = _build_synthetic_cyclic_solver(rng, inject_jitter=False)
    CS.model_jitter = False

    call_count = {"loop": 0}
    real_loop = CS.loop

    def counting_loop(*args, **kwargs):
        call_count["loop"] += 1
        return real_loop(*args, **kwargs)

    CS.loop = counting_loop

    n_passes = 5
    CS.outer_loop(
        n_passes=n_passes,
        patience=0,
        loop_kwargs=dict(make_plots=False, maxfun=10, iprint=-1, use_minphase=False),
    )

    assert call_count["loop"] == n_passes * CS.nspec
