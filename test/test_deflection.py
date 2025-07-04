#!/usr/bin/env python
# coding: utf-8

import pycyc

import argparse
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftshift


mpl.rcParams["image.aspect"] = "auto"

def analysis (base, gradient, delta):

    re_gradient = np.real(gradient)
    im_gradient = np.imag(gradient)
    print(f"test_deflection: power in gradient={np.real(np.vdot(gradient, gradient))}")
    print(f"test_deflection: power in Re[gradient]={np.vdot(re_gradient,re_gradient)}")
    print(f"test_deflection: power in Im[gradient]={np.vdot(im_gradient,im_gradient)}")

    plotthis = np.log10(np.abs(fftshift(gradient)) + 1e-2)
    try:
        fig, ax = plt.subplots(figsize=(8, 9))
        img = ax.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r")
        fig.colorbar(img)
        fig.savefig(base + "_gradient.png")
        plt.close()
    except Exception:
        print("##################################### gradient plot failed")
    with open(base + "_gradient.pkl", "wb") as fh:
        pickle.dump(gradient, fh)

    if delta is not None:
        re_delta = np.real(delta)
        im_delta = np.imag(delta)
        print(f"test_deflection: power in delta={np.real(np.vdot(delta, delta))}")
        print(f"test_deflection: power in Re[delta]={np.vdot(re_delta,re_delta)}")
        print(f"test_deflection: power in Im[delta]={np.vdot(im_delta,im_delta)}")        
        plotthis = np.log10(np.abs(fftshift(delta)) + 1e-2)
        try:
            fig, ax = plt.subplots(figsize=(8, 9))
            img = ax.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r")
            fig.colorbar(img)
            fig.savefig(base + "_deflection.png")
            plt.close()
        except Exception:
            print("##################################### deflection plot failed")
        with open(base + "_deflection.pkl", "wb") as fh:
            pickle.dump(delta, fh)

        # Note that numpy vdot takes the complex conjugate of the first argument
        z = np.vdot(gradient, delta)

        print(f"test_deflection: gradient phase difference={np.angle(z)}")
        z /= np.abs(z)

# do arg parsing here
parser = argparse.ArgumentParser()
parser.add_argument(
    "--init",
    type=str,
    help="file containing the initial wavefield and intrinsic profile",
)
parser.add_argument(
    "--amp",
    type=float,
    default=1e-6,
    help="relative amplitude of deflection",
)

parser.add_argument('--dc', dest='dc', action='store_true', help="deflect the DC bin")
parser.add_argument('--no-dc', dest='dc', action='store_false', help="do not deflect the DC bin")
parser.set_defaults(dc=True)

args, files = parser.parse_known_args()
init = args.init
relative_deflection = args.amp
deflect_dc = args.dc

if deflect_dc:
    print("\ntest_deflection: deflecting DC")
else:
    print("\ntest_deflection: not deflecting DC")

CS = pycyc.CyclicSolver()

# solve sub-integrations in parallel using nthread threads
CS.nthread = 8

# compute and save cyclic spectra when loading periodic spectra
CS.save_cyclic_spectra = True

# normalize each cyclic spectrum
CS.normalize_cyclic_spectra = True

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

print(f"\ntest_deflection: no deflection baseline")
y_val, gradient = CS.evaluate(initial_doppler_delay)
analysis("no_deflection", gradient, None)

for i in range(4):

    ph = 0.5 * i * np.pi

    print(f"\ntest_deflection: deflecting with phase shifted by={ph} radians")

    delta_factor = relative_deflection * np.exp(1.j * ph)
    delta = delta_factor * initial_doppler_delay
    
    if not deflect_dc:
        delta[0,0] = 0.0

    offset_doppler_delay = initial_doppler_delay + delta

    y_val, gradient = CS.evaluate(offset_doppler_delay)

    base = "test_deflection_" + f"{i}"
    analysis (base,gradient,delta)
