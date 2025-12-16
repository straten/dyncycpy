#!/usr/bin/env python
# coding: utf-8

# This notebook reproduces the results of Walker, Demorest & van Straten (2013)

import numpy as np
import pycyc
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from plotting import plot_intrinsic_vs_observed
import copy
import pickle
import glob

from scipy.fft import rfft, fft, fftshift, ifft, fftn, ifftn

# reload a module to incorporate code changes
import importlib
import sys
import pycyc

CS = pycyc.CyclicSolver(zap_edges = 0.05556)

# compute and save cyclic spectra when loading periodic spectra
CS.save_cyclic_spectra = True

# solve sub-integrations in parallel using nthread threads
CS.nthread = 8

files = glob.glob("P2067/*.pb2")
files = np.sort(files)

print(f"loading {len(files)} files")
for file in files:
    CS.load(file)

print(f"computing initial profile from {CS.nsubint} sub-integrations")

CS.initProfile()
plt.plot(CS.pp_intrinsic)
plt.savefig('initProfile.png')
plt.clf()

filters = {}
intrinsic_profiles = {}

# For first pass, process at most the first 80 sub-integrations
init_nspec = np.minimum(80, CS.nspec)

for isub in range(0, init_nspec):
    CS.loop(isub=isub, make_plots=False, ipol=0, tolfact=10)
filters[0] = copy.deepcopy(CS.optimized_filters)
intrinsic_profiles[0] = copy.deepcopy(CS.intrinsic_profiles)

with open("filters_0.pkl", "wb") as fh:
    pickle.dump(filters[0], fh)

with open("profiles_0.pkl", "wb") as fh:
    pickle.dump(intrinsic_profiles[0], fh)

# three more passes through first init_nspec subints:
for ipass in range(1, 4):
    CS.pp_intrinsic = np.zeros((CS.nphase))
    for isub in range(0, init_nspec):
        CS.loop(isub=isub, make_plots=False, ipol=0, tolfact=10, hf_prev=np.copy(filters[ipass-1][isub]))
    
    filters[ipass] = copy.deepcopy(CS.optimized_filters)
    intrinsic_profiles[ipass] = copy.deepcopy(CS.intrinsic_profiles)

with open(f"filters_{ipass}.pkl", "wb") as fh:
    pickle.dump(filters[ipass], fh)

with open(f"profiles_{ipass}.pkl", "wb") as fh:
    pickle.dump(intrinsic_profiles[ipass], fh)

filters_full = None
intrinsic_profiles_full = None

# Now pass through all the data with intrinsic profile so far (output cleared)

CS.pp_intrinsic = np.zeros((CS.nphase))
for isub in range(0, CS.nspec):
    CS.loop(isub=isub, make_plots=False, ipol=0, tolfact=10)

CS.unload_solution("l_bfgs_b_best.fits")

filters_full = copy.deepcopy(CS.optimized_filters)
intrinsic_profiles_full = copy.deepcopy(CS.intrinsic_profiles)

with open(f"filters_full.pkl", "wb") as fh:
    pickle.dump(filters_full, fh)

with open(f"profiles_full.pkl", "wb") as fh:
    pickle.dump(intrinsic_profiles_full, fh)

# Reproduce Figure 2 of WDvS13
plot_intrinsic_vs_observed(CS, CS.pp_scattered, savefig='intrinsic_vs_observed.png')
plt.clf()

# Reproduce the bottom panel of Figure 7 of WDvS13
avg=np.sum(abs(ifft(filters_full,axis=1)),axis=0)
plt.plot(np.log(avg))
plt.savefig('impulse.png')
plt.clf()

# Reproduce Figure 8 of WDvS13
subimp = ifft(filters_full, axis=1)
# Perform a forward FFT along the time (sub-integration) to doppler shift axis
wavefield = fft(subimp, axis=0)
plotthis = np.log10(np.abs(fftshift(wavefield)))
plt.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r", vmin=-2)
plt.colorbar()
plt.savefig('wavefield.png')
plt.clf()

