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
from scipy.fft import rfft, fft, fftshift, ifft, fftn, ifftn

# reload a module to incorporate code changes
import importlib
import sys
import pycyc

CS = pycyc.CyclicSolver("P2067/chan07/53873.27864.07.15s.pb2", zap_edges = 0.05556, pscrunch=True)
CS.data.shape, CS.nspec

CS.load("P2067/chan07/53873.31676.07.15s.pb2")
CS.data.shape, CS.nspec

CS.initProfile()
plt.plot(CS.pp_int)
plt.savefig('initProfile.png')
plt.clf()

pp_scattered = np.copy(CS.pp_ref)

filters = {}
intrinsic_profiles = {}

# For first pass, either loop:

for isub in range(0, 80):
    CS.loop(isub=isub, make_plots=False, ipol=0, tolfact=10)
filters[0] = copy.deepcopy(CS.optimized_filters)
intrinsic_profiles[0] = copy.deepcopy(CS.intrinsic_profiles)

with open("filters_0.pkl", "wb") as fh:
    pickle.dump(filters[0], fh)

with open("profiles_0.pkl", "wb") as fh:
    pickle.dump(intrinsic_profiles[0], fh)

# three more passes through 20 minutes (80 15 sec subints):
for ipass in range(1, 4):
    CS.pp_ref = np.copy(CS.pp_int)
    CS.pp_int = np.zeros((CS.nphase))
    for isub in range(0, 80):
        CS.loop(isub=isub, make_plots=False, ipol=0, tolfact=10, hf_prev=np.copy(filters[ipass-1][isub]))
    
    filters[ipass] = copy.deepcopy(CS.optimized_filters)
    intrinsic_profiles[ipass] = copy.deepcopy(CS.intrinsic_profiles)

with open(f"filters_{ipass}.pkl", "wb") as fh:
    pickle.dump(filters[ipass], fh)

with open(f"profiles_{ipass}.pkl", "wb") as fh:
    pickle.dump(intrinsic_profiles[ipass], fh)

# Now pass through all the data with intrinsic profile so far (output cleared)

CS.pp_ref = np.copy(CS.pp_int)
CS.pp_int = np.zeros((CS.nphase))
for isub in range(0, CS.data.shape[0]):
    CS.loop(isub=isub, make_plots=False, ipol=0, tolfact=10)

filters_full = {}
intrinsic_profiles_full = {}

ipass = 0
filters_full[ipass] = copy.deepcopy(CS.optimized_filters)
intrinsic_profiles_full[ipass] = copy.deepcopy(CS.intrinsic_profiles)

with open(f"filters_full_{ipass}.pkl", "wb") as fh:
    pickle.dump(filters_full[ipass], fh)

with open(f"profiles_full_{ipass}.pkl", "wb") as fh:
    pickle.dump(intrinsic_profiles_full[ipass], fh)

# Reproduce Figure 2 of WDvS13
plot_intrinsic_vs_observed(CS, np.average(CS.data, axis=(0, 1, 2)), savefig='intrinsic_vs_observed.png')
plt.clf()

# Reproduce the bottom panel of Figure 7 of WDvS13
avg=np.sum(abs(ifft(filters_full[0],axis=1)),axis=0)
plt.plot(np.log(avg))
plt.savefig('impulse.png')
plt.clf()

# Reproduce Figure 8 of WDvS13, using the wavefield derived from only the second file (subint 236 onward)
subfilts=filters_full[0][236:]
subimp = ifft(subfilts, axis=1)
# Perform a forward FFT along the time (sub-integration) to doppler shift axis
wavefield = fft(subimp, axis=0)
plotthis = np.log10(np.abs(fftshift(wavefield)))
plt.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r", vmin=-2)
plt.colorbar()
plt.savefig('wavefield.png')
plt.clf()

