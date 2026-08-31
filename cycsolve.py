#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import math
import pickle
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import fista
import pycyc
from plotting import (
    plot_Doppler_vs_delay,
    plot_intrinsic_vs_observed,
    plot_jitter_eigenvectors,
    plot_power_vs_delay,
)

mpl.rcParams["image.aspect"] = "auto"

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# do arg parsing here
p = argparse.ArgumentParser()
p.add_argument(
    "--init",
    type=str,
    help="file containing the initial wavefield and intrinsic profile",
)

p.add_argument(
    "--init-profile",
    type=str,
    help="file containing the initial intrinsic profile",
)

p.add_argument(
    "--zap",
    type=float,
    default=0.05556,
    help="fraction of band edges to zap",
)

p.add_argument(
    "--iter",
    type=int,
    default=1000,
    help="maximum number of iterations (fista solver) or outer-loop passes (outer solver)",
)

p.add_argument(
    "--alpha",
    type=float,
    default=1e-6,
    help="initial step size, alpha = 1 / Lipschitz constant (fista solver only)",
)

p.add_argument(
    "--solver",
    type=str,
    choices=["fista", "outer"],
    default="fista",
    help=(
        "main optimization loop to use: 'fista' (this script's original, "
        "experimental FISTA-based loop over the shared wavefield) or "
        "'outer' (pycyc.CyclicSolver.outer_loop, alternating per-subint "
        "L-BFGS-B wavefield fits with profile/jitter-model updates)"
    ),
)

p.add_argument(
    "--gpu",
    action="store_true",
    help=(
        "run the fista solver's gradient and profile computations on the "
        "GPU via CuPy (pycyc.CyclicSolver.compute_gradient_batched / "
        ".updateProfile_batched, CS.use_gpu=True) instead of the CPU "
        "ThreadPoolExecutor path. Requires cupy (pip install cupy-cudaXXx "
        "matching your CUDA version). Ignored for --solver outer, which "
        "has no GPU path -- see the CuPy port plan "
        "(/home/willem/.claude/plans/stateful-dreaming-wall.md) for why "
        "the classical per-subint L-BFGS-B loop isn't a GPU target."
    ),
)

p.add_argument(
    "--gpu-chunk-size",
    type=int,
    default=None,
    help=(
        "subints processed per batched GPU call (CS.gpu_chunk_size); "
        "default (unset) processes all subints in one call. Only matters "
        "with --gpu; see the CuPy port plan's memory-budget notes."
    ),
)

p.add_argument(
    "--debug",
    action="store_true",
    help=(
        "set the pycyc.* loggers (pycyc.solver, pycyc.jitter, ...) to DEBUG "
        "-- raises the 'pycyc' logger specifically, not the root logger, so "
        "this doesn't also turn on matplotlib/other libraries' own debug "
        "spam. fista.py's separate 'H-FISTA' logger is untouched for now."
    ),
)

args, files = p.parse_known_args()

if args.debug:
    logging.getLogger("pycyc").setLevel(logging.DEBUG)
else:
    logging.getLogger("pycyc").setLevel(logging.INFO)

init = args.init
init_profile = args.init_profile

max_iterations = args.iter

# should probably estimate this as described in Oslowski & Walker (2023)
alpha_init = args.alpha

CS = pycyc.CyclicSolver(zap_edges=args.zap)

# use the minimum of the last N estimates of alpha = 1 / Lipschitz
alpha_history = 2

# solve sub-integrations in parallel using nthread threads
CS.nthread = 8

use_gpu = args.gpu and args.solver != "outer"
if args.gpu and args.solver == "outer":
    print("cycsolve: --gpu has no effect with --solver outer (CPU-only); ignoring")
if args.gpu_chunk_size is not None:
    CS.gpu_chunk_size = args.gpu_chunk_size
# CS.use_gpu itself is set later, after initProfile()/initWavefield() --
# updateProfile_batched (see pycyc/solver.py) doesn't support the
# one-time compute_scattered_profile/save_dynamic_spectrum work
# initProfile()'s own first updateProfile() call does, so use_gpu must
# stay off through setup and only turn on for the main loop.

# CS.enforce_real_at_origin = True

# compute and save cyclic spectra when loading periodic spectra
CS.save_cyclic_spectra = True

# use a single integrated profile as the reference profile for each sub-integration
CS.use_integrated_profile = True

# maintain constant total energy
CS.conserve_wavefield_energy = True

# set cyclic spectra to zero where shifted content is out of band
CS.pad_cyclic_spectra = False

# set h(tau,omega) to zero for tau < 0, for the whole run (permanently
# enforced, not just an early-iteration warmup) -- letting this expire
# after a fixed number of iterations, as this used to do, was found to
# let white noise accumulate unconstrained in the (physically required to
# be exactly zero) negative-delay region: growing to ~11% of total
# wavefield power / ~8x the reference noise floor by iteration 150 of a
# 150-iteration fista run with no L1 penalty (_lambda=None below) to
# otherwise suppress it. See the delay/omega noise-growth investigation.
CS.enforce_causality = True

# Remove two real per-subint degenerate directions from the gradient every
# iteration -- an arbitrary overall phase and an arbitrary rigid delay
# shift of each subint's own impulse response, both exact null directions
# of the per-subint merit (see pycyc.regularization.subtract_degenerate_dof/
# subtract_degenerate_delay_and_phase, and fista.take_fista_step's own
# comment on this same invariance). Previously left off (the CyclicSolver
# default): FISTA's momentum term amplifies whatever difference exists
# between consecutive iterates regardless of whether the gradient actually
# opposes it, so with nothing projecting these 2*nspec directions out, any
# drift along them (initialization asymmetry, interaction with the other
# regularizers) compounds unchecked. Diagnosed from h(tau=0,omega) growing
# a large, non-physical near-DC spike and surrounding Gibbs ringing across
# iterations on real P2067_full data -- a coherent per-subint phase/delay
# drift is exactly the kind of low-order perturbation that concentrates
# near zero Doppler when Fourier-transformed along the subint axis.
CS.subtract_degenerate_projections = True

warmup_passes = 1
CS.model_jitter = True

# seems to have little impact and too great a computational cost
CS.minimize_spectral_entropy = False

# maximum Doppler shift cut-off (fraction of Doppler shifts to keep)
# CS.low_pass_filter_Doppler = 0.5

# align the phase and delay of time-adjacent frequency responses computed from the wavefield
# CS.align_frequency_responses = True

# use a delay-dependent threshold to perform shrinkage -- targets the
# positive-delay white-noise band (elevated, delay-varying noise floor
# spread flat across all Doppler shifts) that a single global threshold
# can't adapt to. See the delay/omega noise-growth investigation.
CS.delay_noise_shrinkage_threshold = 1.0
CS.delay_noise_selection_threshold = 2.0

# CS.noise_shrinkage_threshold = 1.0

# include a separate gain variation term for each sub-integration
# CS.model_gain_variations = True

# when updating the profile, minimize phase differences between h(tau,t) and h(tau,t+1)
# CS.reduce_temporal_phase_noise = True

# reduce temporal phase noise by minimizing the spectral entropy
# CS.minimize_spectral_entropy = True

# Number of iterations between profile updates (fista solver only -- the
# outer solver always re-estimates the profile once per pass)
update_profile = True
update_profile_period = 2
update_profile_every_iteration_until = 5
update_profile_after = 0
plot_all = True

# Early-stopping: treat the fit as converged once demerit's (get_reduced_chisq's)
# relative change from the previous accepted step is smaller than the natural
# sampling-fluctuation scale of a chi-squared statistic with CS.get_dof()
# degrees of freedom, sqrt(2/dof) -- same statistical criterion as
# pycyc.CyclicSolver.outer_loop's merit_n_sigma (see its docstring), applied
# here to consecutive FISTA iterations instead of consecutive outer_loop
# passes. merit_n_sigma scales that noise floor (1.0 = one sigma); require
# merit_patience consecutive non-reset steps below it, since a single
# iteration's step size can be small without the fit having truly converged.
# Set merit_patience=0 to disable and always run exactly max_iterations.
#
# merit_patience=5 (tried first) was fooled by an ordinary FISTA momentum
# ripple on real P2067_full data: an overshoot past the best-so-far point
# at iteration 51, followed by 5 small-relative-change recovery steps
# (52-56) that never actually got back below the pre-overshoot merit,
# triggered a false "converged" at iteration 57 (see the two adaptive-
# restart commits this reverts -- neither restart variant fixed the
# ripple itself without hurting overall convergence, so raising patience
# to outlast a single ripple is the more direct fix). 15 comfortably
# exceeds that ripple's ~6-7 iteration duration; revisit if a longer-lived
# ripple is ever seen to fool this too.
merit_n_sigma = 1.0
merit_patience = 15

# CS.doppler_window = ('kaiser', 8.0)

# maximum Doppler shift cut-off (fraction of Doppler shifts to keep)
# CS.low_pass_filter_Doppler = 0.5

# CS.temporal_taper_alpha = 0.25
# CS.spectral_taper_alpha = 0.25

# CS.first_wavefield_delay = 0
CS.first_wavefield_from_best_harmonic = 10

# break the exact symmetry of the bare-delta initial wavefield guess with
# a small amount of complex Gaussian noise, to test whether FISTA is
# trapped near a shallow local minimum by the naive starting point. rms
# calibrated to roughly match the real reference-noise-floor amplitude
# observed in a converged run on this dataset (sqrt(2.62e-2) ~= 0.162,
# from cycsolve_fista_iter150's final wavefield) -- a physically-motivated
# "just above the eventual noise floor" perturbation scale, not a guess.
# Tested alone (50-iteration comparison run): no measurable improvement
# over the bare delta -- final demerit and wavefield concentration were
# statistically indistinguishable from an unperturbed control.
# CS.initial_guess_noise_perturbation_rms = 0.162

# CS.noise_threshold = 1.0
# CS.noise_smoothing_duty_cycle = 0.05

if init is not None:
    print(f"cycsolve: loading initial wavefield and intrinsic profile from {init}")
    CS.load_initial_guess(init)

print(f"cycsolve: loading {len(files)} files")
for file in files:
    CS.load(file)

print(f"cycsolve: {CS.nsubint} spectra loaded")

CS.initProfile()

if init_profile is not None:
    print(f"cycsolve: loading initial intrinsic profile from {init_profile}")
    CS.load_initial_profile(init_profile)
    CS.conserve_wavefield_energy = False
    CS.enforce_causality = 0
    update_profile = False

initial_profile_from_first_subintegration = False

if initial_profile_from_first_subintegration:
    CS.loop(isub=0, make_plots=False, ipol=0, tolfact=10, iprint=0)
    CS.pp_intrinsic = CS.intrinsic_profiles[0, 0]

plt.plot(CS.pp_intrinsic)
plt.savefig("cycsolve_init_profile.png")
plt.close()
with open("cycsolve_init_profile.pkl", "wb") as fh:
    pickle.dump(CS.pp_intrinsic, fh)

plt.plot(CS.cs_norm)
plt.savefig("cycsolve_cs_norm.png")
plt.close()
with open("cycsolve_cs_norm.pkl", "wb") as fh:
    pickle.dump(CS.cs_norm, fh)

pp_scattered = np.copy(CS.pp_scattered)

CS.initWavefield()

if use_gpu:
    print("cycsolve: running the fista gradient/profile computation on the GPU (CuPy)")
    CS.use_gpu = True

if args.solver == "fista":
    y_n = np.copy(CS.h_doppler_delay)
    x_n = np.copy(CS.h_doppler_delay)
    t_n = 1

    demerits = np.array([])
    alphas = np.array([])

    alpha = alpha_init

    best_merit = CS.get_reduced_chisq()
    best_x = np.copy(x_n)
    L_max = 1.0 / alpha

    print(f"starting merit={best_merit}")

    step_factor = 1.0
    acceleration = 1.2
    bad_step = 0

    prev_merit = best_merit
    merit_stable_count = 0

    # Start timer
    prev_time = start_time = time.time()
    min_step_factor = 0.5

    for i in range(max_iterations + 1):
        CS.nopt += 1

        if update_profile and (
            i < update_profile_every_iteration_until
            or ((update_profile_after == 0 or i > update_profile_after) and i % update_profile_period == 0)
        ):
            print("cycsolve: update profile")
            # CyclicSolver.evaluate() leaves CS.h_doppler_delay holding
            # whatever it last set it to -- the previous iteration's
            # x_np1 (== x_n), not y_n (this iteration's momentum-
            # extrapolated starting point, tracked separately as a local
            # variable here). Sync it from y_n first, so updateProfile()
            # (which derives self.h_time_delay from self.h_doppler_delay
            # as its own first step, and may further adjust either place
            # -- power normalization always, spectral-entropy
            # minimization if CS.minimize_spectral_entropy) actually
            # gauge-fixes the point FISTA is about to take a gradient
            # step from, not a stale copy of x_n. Cheap: an array copy,
            # not a gradient recompute.
            CS.h_doppler_delay = np.copy(y_n)
            CS.updateProfile()
            # Resync y_n so whatever updateProfile() just did (power
            # normalization, and spectral-entropy minimization when
            # enabled) actually reaches the FISTA iterate sequence,
            # instead of being silently discarded by the next
            # evaluate() call's overwrite of CS.h_doppler_delay.
            y_n = np.copy(CS.h_doppler_delay)

            if CS.model_jitter and i >= warmup_passes:
                CS._refresh_jitter_model()

        x_n, y_n, L, t_n, demerits = fista.take_fista_step(
            iter=i,
            func=CS,
            backtrack=False,
            alpha=alpha,
            eta=5,
            y_n=y_n,
            _lambda=None,
            delay_for_inf=-int(CS.nchan / 2),
            zero_penalty_coords=np.array([]),
            fix_phase_value=None,
            fix_phase_coords=None,
            fix_support=np.array([]),
            t_n=t_n,
            x_n=x_n,
            demerits=demerits,
            eps=None,
        )

        assert math.isfinite(CS.get_reduced_chisq())

        # CS.enforce_causality is now permanently True (see comment at its
        # assignment above) rather than an expiring countdown, so there is
        # nothing left to do here each iteration.

        if i == 0 or L > L_max:
            L_max = L

        reduced_chisq = CS.get_reduced_chisq()

        if reduced_chisq < best_merit:
            best_merit = reduced_chisq
            best_x = np.copy(x_n)
        else:
            print(f"\n** greater than best={best_merit}")

        if reduced_chisq > prev_merit:
            print("**** bad step")

        really_bad = not math.isfinite(reduced_chisq) or reduced_chisq > 2.0 * prev_merit

        # Statistically-scaled early stopping: see merit_n_sigma/merit_patience's
        # definition above. A really_bad step is a reset, not evidence either
        # way about convergence, so it doesn't count toward or break the streak.
        if not really_bad:
            dof = CS.get_dof()
            merit_rel_change = abs(reduced_chisq - prev_merit) / abs(prev_merit) if prev_merit != 0 else math.inf
            merit_tol = merit_n_sigma * math.sqrt(2.0 / dof) if dof > 0 else 0.0
            merit_stable_count = merit_stable_count + 1 if merit_rel_change < merit_tol else 0

        if really_bad:
            print("**** really bad step - RESET")
            t_n = 1
            CS.h_doppler_delay[:] = y_n[:] = x_n[:] = best_x[:]
        else:
            alphas = np.append(alphas, 1.0 / L)
            prev_merit = reduced_chisq

        if merit_patience > 0 and merit_stable_count >= merit_patience:
            print(
                f"cycsolve: merit converged (relative change below the {merit_n_sigma}-sigma "
                f"noise floor for {merit_stable_count} consecutive iterations) after iteration {i}"
            )
            break

        if alphas.size == 0:
            alpha = 1.0 / L  # this should happen only if the first step is bad
            print(f"alpha init={alpha_init} led to very bad first step.  next alpha={alpha}")
        elif alpha_history == 0 or alphas.size < alpha_history:
            alpha = np.min(alphas)
        else:
            alpha = np.min(alphas[-alpha_history:])

        if really_bad:
            alpha *= 0.2
            print(f"reducing alpha to {alpha}")
            alphas = np.append(alphas, alpha)

        print(f"{i:03d} demerit={reduced_chisq} alpha={alpha} last={1.0/L} min={1.0/L_max} t_n={t_n}", flush=True)
        end_time = time.time()

        iter_time = end_time - prev_time
        elapsed_time = end_time - start_time

        prev_time = end_time
        print(f"Elapsed time: {elapsed_time/60} min   Iteration time: {iter_time/60} min\n", flush=True)

        if plot_all or i < 10 or i % 10 == 0:
            base = "cycsolve_" + f"{i:03d}"
            try:
                plot_Doppler_vs_delay(x_n, CS.mean_time_offset, CS.bw, base + "_wavefield.png")
            except Exception:
                print("##################################### wavefield plot failed")
            with open(base + "_wavefield.pkl", "wb") as fh:
                pickle.dump(x_n, fh)

            if CS.model_gain_variations:
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(CS.optimal_gains)
                    fig.savefig(base + "_optimal_gains.png")
                    plt.close()
                except Exception:
                    print("##################################### optimal gains plot failed")
                with open(base + "_optimal_gains.pkl", "wb") as fh:
                    pickle.dump(CS.optimal_gains, fh)

            try:
                plot_power_vs_delay(x_n, CS.bw, base + "_impulse_response.png")
            except Exception:
                print("##################################### impulse response plot failed")

            plot_intrinsic_vs_observed(CS, pp_scattered, base + "_compare.png")
            plt.close()

            if CS.model_jitter and i >= warmup_passes:
                # eigenvectors: every basis vector from the jitter PCA fit
                # (not just the jitter_rank retained ones); rank: how many
                # of those are actually retained, against a Marchenko-
                # Pastur threshold whose noise_variance_per_harmonic input
                # has been empirically recalibrated against a "non-pulsar"
                # high-harmonic noise-only band (noise_h_start/noise_kappa
                # -- see pycyc.jitter.fit_jitter_basis_calibrated); weights:
                # the per-subint principal-component coefficients for the
                # retained rank-N subspace, one row per subint.
                with open(base + "_jitter.pkl", "wb") as fh:
                    pickle.dump(
                        {
                            "eigenvectors": CS.jitter_full_basis,
                            "rank": CS.jitter_rank,
                            "weights": CS.jitter_weights,
                            "eigenvalues": CS.jitter_eigenvalues,
                            "threshold": CS.jitter_threshold,
                            "noise_h_start": CS.jitter_noise_h_start,
                            "noise_kappa": CS.jitter_noise_kappa,
                        },
                        fh,
                    )

                # the retained (rank-N) eigenvectors are complex, in the
                # harmonic domain (like intrinsic_ph vs. pp_intrinsic) --
                # inverse-FFT each back to the real pulse-phase domain the
                # same way CyclicSolver.harm2phase does for pp_intrinsic,
                # for a qualitative look at their shape.
                try:
                    retained = CS.jitter_full_basis[: CS.jitter_rank]
                    eigenvector_profiles = np.array([CS.harm2phase(vec) for vec in retained])
                    plot_jitter_eigenvectors(eigenvector_profiles, base + "_jitter_eigenvectors.png")
                except Exception:
                    print("##################################### jitter eigenvector plot failed")

    # Restore the best-ever iterate before saving: whatever CS.h_doppler_delay
    # holds when the loop exits (whether by early-stopping above or simply
    # reaching max_iterations) is just wherever the last accepted FISTA step
    # happened to land, which is not necessarily best_x -- a step can be
    # worse than the best seen so far (as tracked by best_merit/best_x)
    # without being "really bad" enough to trigger the in-loop reset. The
    # per-iteration diagnostic plots above intentionally show the actual
    # trajectory (x_n, warts and all); only the final saved solution needs
    # this correction.
    CS.h_doppler_delay = np.copy(best_x)
    CS.h_time_delay = pycyc.freq2time(CS.h_doppler_delay, axis=0)

else:  # args.solver == "outer"
    # CyclicSolver.outer_loop alternates a classical per-subint L-BFGS-B
    # wavefield fit (CyclicSolver.loop) with profile/jitter-model updates,
    # instead of the FISTA branch's joint gradient step over the shared
    # Doppler-delay wavefield above. It has its own merit- and
    # jitter-basis-stability stopping condition (see its docstring), so
    # --iter here is an upper bound on passes, not necessarily the number
    # actually run.
    CS.outer_loop(
        n_passes=max_iterations,
        warmup_passes=warmup_passes,
        loop_kwargs=dict(iprint=-1),
    )

    # outer_loop updates self.h_time_delay per subint directly; refresh the
    # Doppler-delay wavefield from it for plotting (unload_solution below
    # only needs h_time_delay, so this is for diagnostics only).
    CS.h_doppler_delay = pycyc.time2freq(CS.h_time_delay, axis=0)

    try:
        plot_Doppler_vs_delay(CS.h_doppler_delay, CS.mean_time_offset, CS.bw, "cycsolve_outer_wavefield.png")
    except Exception:
        print("##################################### wavefield plot failed")
    with open("cycsolve_outer_wavefield.pkl", "wb") as fh:
        pickle.dump(CS.h_doppler_delay, fh)

    if CS.model_gain_variations:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(CS.optimal_gains)
            fig.savefig("cycsolve_outer_optimal_gains.png")
            plt.close()
        except Exception:
            print("##################################### optimal gains plot failed")
        with open("cycsolve_outer_optimal_gains.pkl", "wb") as fh:
            pickle.dump(CS.optimal_gains, fh)

    try:
        plot_power_vs_delay(CS.h_doppler_delay, CS.bw, "cycsolve_outer_impulse_response.png")
    except Exception:
        print("##################################### impulse response plot failed")

    plot_intrinsic_vs_observed(CS, pp_scattered, "cycsolve_outer_compare.png")
    plt.close()

CS.unload_solution("cycsolve_best.fits")
