#!/usr/bin/env python
# coding: utf-8

# This notebook runs an analysis similar to that of Walker, Demorest & van Straten (2013)

import pickle
import time
import sys

import numpy as np
import pycyc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from plotting import plot_intrinsic_vs_observed
import copy
from scipy.fft import rfft, fft, fftshift, ifft, fftn, ifftn

# reload a module to incorporate code changes
import pycyc

CS = pycyc.CyclicSolver(zap_edges=0.05556)

CS.nthread = 8
CS.save_cyclic_spectra = True

inputArgs = sys.argv
print(f"cyclbfgsb: loading {len(inputArgs)-1} files")
for file in inputArgs[1:]:
    CS.load(file)

print(f"cyclbfgsb: {CS.nsubint} spectra loaded")

CS.initProfile()

plt.plot(CS.pp_intrinsic)
plt.savefig("cyclbfgsb_init_profile.png")
plt.close()
with open("cyclbfgsb_init_profile.pkl", "wb") as fh:
    pickle.dump(CS.pp_intrinsic, fh)

plt.plot(CS.cs_norm)
plt.savefig("cyclbfgsb_cs_norm.png")
plt.close()
with open("cyclbfgsb_cs_norm.pkl", "wb") as fh:
    pickle.dump(CS.cs_norm, fh)

pp_scattered = np.copy(CS.pp_scattered)

filters = {}
intrinsic_profiles = {}

# four passes through first 80 files (20 minutes assuming 15 sec subints):
hf_prev=None
nsub = 80
for ipass in range(4):
    for isub in range(0, nsub):
        if ipass > 0:
            hf_prev=np.copy(filters[ipass-1][isub])

        print(f'cyclbfgsb: pass={ipass} sub-integration={isub}', flush=True)
        CS.loop(isub=isub, make_plots=False, ipol=0, tolfact=10, hf_prev=hf_prev)
    
    print(f'cyclbfgsb: pass {ipass} finished', flush=True)

    filters[ipass] = copy.deepcopy(CS.optimized_filters)
    intrinsic_profiles[ipass] = copy.deepcopy(CS.intrinsic_profiles)

    CS.pp_intrinsic /= nsub

    with open(f"filters_{ipass}.pkl", "wb") as fh:
        pickle.dump(filters[ipass], fh)

    with open(f"profiles_{ipass}.pkl", "wb") as fh:
        pickle.dump(intrinsic_profiles[ipass], fh)

# Now pass through all the data with intrinsic profile so far (output cleared)

# CS.pp_intrinsic = np.zeros((CS.nphase))
for isub in range(0, CS.nsubint):
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
plot_intrinsic_vs_observed(CS, pp_scattered, savefig='intrinsic_vs_observed.png')
plt.clf()

# Reproduce the bottom panel of Figure 7 of WDvS13
avg=np.sum(abs(ifft(filters_full[0],axis=1)),axis=0)
plt.plot(np.log(avg))
plt.savefig('impulse.png')
plt.clf()

# Reproduce Figure 8 of WDvS13
subfilts=filters_full[0]
subimp = ifft(subfilts, axis=1)
# Perform a forward FFT along the time (sub-integration) to doppler shift axis
wavefield = fft(subimp, axis=0)
plotthis = np.log10(np.abs(fftshift(wavefield)))
plt.imshow(plotthis.T, aspect="auto", origin="lower", cmap="cubehelix_r", vmin=-2)
plt.colorbar()
plt.savefig('wavefield.png')
plt.clf()

