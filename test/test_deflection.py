#!/usr/bin/env python
# coding: utf-8

import argparse
import math
import pickle
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftshift

import fista
import pycyc
from plotting import plot_intrinsic_vs_observed

mpl.rcParams["image.aspect"] = "auto"

# do arg parsing here
p = argparse.ArgumentParser()
p.add_argument(
    "--init",
    type=str,
    help="file containing the initial wavefield and intrinsic profile",
)
p.add_argument(
    "--amp",
    type=float,
    default=1e-6,
    help="relative amplitude of deflection",
)

args, files = p.parse_known_args()
init = args.init
relative_deflection = args.amp

CS = pycyc.CyclicSolver()

# solve sub-integrations in parallel using nthread threads
CS.nthread = 8

# compute and save cyclic spectra when loading periodic spectra
CS.save_cyclic_spectra = True

# use a single integrated profile as the reference profile for each sub-integration
CS.use_integrated_profile = True

# maintain constant total power in the wavefield
# CS.conserve_wavefield_energy = True

# reduce temporal phase noise by minimizing the spectral entropy
# CS.minimize_spectral_entropy = True

if init is not None:
    print(f"test_deflection: loading initial wavefield and intrinsic profile from {init}")
    CS.load_initial_guess(init)

print(f"test_deflection: loading {len(files)} files")
for file in files:
    CS.load(file)

print(f"test_deflection: {CS.nsubint} spectra loaded")

CS.initProfile()

plt.plot(CS.pp_intrinsic)
plt.savefig("test_deflection_init_profile.png")
plt.close()
with open("test_deflection_init_profile.pkl", "wb") as fh:
    pickle.dump(CS.pp_intrinsic, fh)

plt.plot(CS.cs_norm)
plt.savefig("test_deflection_cs_norm.png")
plt.close()
with open("test_deflection_cs_norm.pkl", "wb") as fh:
    pickle.dump(CS.cs_norm, fh)

pp_scattered = np.copy(CS.pp_scattered)

CS.initWavefield()

initial_doppler_delay = np.copy(CS.h_doppler_delay)

for i in range(4):

    ph = 0.5 * i * np.pi

    delta = relative_deflection * initial_doppler_delay
    offset_doppler_delay = initial_doppler_delay + delta

    y_val, gradient = CS.evaluate(offset_doppler_delay)

    # Note that numpy vdot takes the complex conjugate of the first argument
    z = np.vdot(gradient, delta)

    print(f"test_deflection: gradient phase difference={np.angle(z)}")
    z /= np.abs(z)

    re_delta = np.real(delta)
    im_delta = np.imag(delta)
    print(f"test_deflection: power in delta={np.vdot(delta, delta)}")
    print(f"test_deflection: power in Re[delta]={np.vdot(re_delta,re_delta)}")
    print(f"test_deflection: power in Im[delta]={np.vdot(im_delta,im_delta)}")

    re_gradient = np.real(gradient)
    im_gradient = np.imag(gradient)
    print(f"test_deflection: power in gradient={np.vdot(gradient, gradient)}")
    print(f"test_deflection: power in Re[gradient]={np.vdot(re_gradient,re_gradient)}")
    print(f"test_deflection: power in Im[gradient]={np.vdot(im_gradient,im_gradient)}")

    base = "test_deflection_" + f"{i}"
    plotthis = np.log10(np.abs(fftshift(gradient)) + 1e-2)
    try:
        fig, ax = plt.subplots(figsize=(8, 9))
        img = ax.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r", vmin=-1)
        fig.colorbar(img)
        fig.savefig(base + "_wavefield.png")
        plt.close()
    except Exception:
        print("##################################### wavefield plot failed")
    with open(base + "_wavefield.pkl", "wb") as fh:
        pickle.dump(gradient, fh)

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
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.log10(np.sum(np.abs(gradient) ** 2, axis=0) + 1e-16))
        fig.savefig(base + "_impulse_response.png")
        plt.close()
    except Exception:
        print("##################################### impulse response plot failed")

    plot_intrinsic_vs_observed(CS, pp_scattered, base + "_compare.png")
    plt.close()
