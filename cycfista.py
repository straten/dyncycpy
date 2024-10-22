#!/usr/bin/env python
# coding: utf-8

import pickle
import sys
import time
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftshift

import fista
import pycyc
from plotting import plot_intrinsic_vs_observed

mpl.rcParams["image.aspect"] = "auto"

CS = pycyc.CyclicSolver(zap_edges=0.05556)

# use the minimum of the last N estimates of alpha = 1 / Lipschitz
alpha_history = 10

# solve sub-integrations in parallel using nthread threads
CS.nthread = 8

# compute and save cyclic spectra when loading periodic spectra
CS.save_cyclic_spectra = True

# use a single integrated profile as the reference profile for each sub-integration
CS.use_integrated_profile = True

# include a separate gain variation term for each sub-integration
CS.model_gain_variations = True

# set h(tau,omega) to zero for tau < 0 for the first N iterations
CS.enforce_causality = 8

# when updating the profile, minimize phase differences between h(tau,t) and h(tau,t+1)
CS.reduce_temporal_phase_noise = True

# Number of iterations between profile updates
update_profile_period = 10
update_profile_every_iteration_until = 15

# CS.noise_shrinkage_threshold = 1.0
# CS.doppler_window = ('kaiser', 8.0)

# CS.delay_noise_shrinkage_threshold = 1.0
# CS.delay_noise_selection_threshold = 2.0

# CS.temporal_taper_alpha = 0.25
# CS.spectral_taper_alpha = 0.25

# CS.first_wavefield_delay = 0
# CS.first_wavefield_from_best_harmonic = 10

# CS.noise_threshold = 1.0
# CS.noise_smoothing_duty_cycle = 0.05

# do arg parsing here
p = argparse.ArgumentParser()
p.add_argument(
    "--init",
    type=str,
    help="file containing the initial wavefield and intrinsic profile",
)

args, files = p.parse_known_args()
init = args.init

if init is not None:
    print(f"cycfista: loading initial wavefield and intrinsic profile from {init}")
    CS.load_initial_guess(init)

print(f"cycfista: loading {len(files)} files")
for file in files:
    CS.load(file)

print(f"cycfista: {CS.nsubint} spectra loaded")

CS.initProfile()

plt.plot(CS.pp_intrinsic)
plt.savefig("cycfista_init_profile.png")
plt.close()
with open("cycfista_init_profile.pkl", "wb") as fh:
    pickle.dump(CS.pp_intrinsic, fh)

plt.plot(CS.cs_norm)
plt.savefig("cycfista_cs_norm.png")
plt.close()
with open("cycfista_cs_norm.pkl", "wb") as fh:
    pickle.dump(CS.cs_norm, fh)

pp_scattered = np.copy(CS.pp_scattered)

CS.initWavefield()

y_n = np.copy(CS.h_doppler_delay)
x_n = np.copy(CS.h_doppler_delay)
t_n = 1

demerits = np.array([])
alphas = np.array([])

alpha = 1e-6

best_merit = CS.get_reduced_chisq()
best_x = np.copy(x_n)
L_max = 1.0 / alpha

print(f"starting merit={best_merit}")

step_factor = 1.0
acceleration = 1.2
bad_step = 0

prev_merit = best_merit

# Start timer
start_time = time.time()
min_step_factor = 0.5

for i in range(1000):
    CS.nopt += 1

    if i < update_profile_every_iteration_until or (i + 1) % update_profile_period == 0:
        print("cycfista: update profile")
        CS.updateProfile()

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

    if CS.enforce_causality:
        CS.enforce_causality -= 1
        print(f"enforcing causality for {CS.enforce_causality} more iterations")

    if i == 0 or L > L_max:
        L_max = L

    if CS.get_reduced_chisq() < best_merit:
        best_merit = CS.get_reduced_chisq()
        best_x = np.copy(x_n)
    else:
        print(f"\n** greater than best={best_merit}")

    if CS.get_reduced_chisq() > prev_merit:
        print("**** bad step")

    alphas = np.append(alphas, 1.0 / L)

    if alpha_history == 0 or alphas.size < alpha_history:
        alpha = np.min(alphas)
    else:
        alpha = np.min(alphas[-alpha_history:])

    prev_merit = CS.get_reduced_chisq()

    print(f"\n{i:03d} demerit={CS.get_reduced_chisq()} alpha={alpha} last={1.0/L} min={1.0/L_max} t_n={t_n}")
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time/60} min", flush=True)

    if i < 10 or i % 10 == 0:
        base = "cycfista_" + f"{i:03d}"
        plotthis = np.log10(np.abs(fftshift(x_n)) + 1e-2)
        try:
            fig, ax = plt.subplots(figsize=(8, 9))
            img = ax.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r", vmin=-1)
            fig.colorbar(img)
            fig.savefig(base + "_wavefield.png")
            plt.close()
        except Exception:
            print("##################################### wavefield plot failed")
        with open(base + "_wavefield.pkl", "wb") as fh:
            pickle.dump(x_n, fh)

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
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(np.log10(np.sum(np.abs(x_n) ** 2, axis=0)))
            fig.savefig(base + "_impulse_response.png")
            plt.close()
        except Exception:
            print("##################################### impulse response plot failed")

        plot_intrinsic_vs_observed(CS, pp_scattered, base + "_compare.png")
        plt.close()
