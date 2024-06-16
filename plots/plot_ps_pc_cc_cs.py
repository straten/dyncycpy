#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, irfft, rfft
import scipy.stats

import numpy as np
import psrchive
import sys

def ps2cs(ps, workers=2, axis=1):
    return rfft(ps, axis=axis, workers=workers) / np.sqrt(2*ps.shape[axis])

def ps2pc(ps, workers=2, axis=0):
    return rfft(ps, axis=axis, workers=workers) / np.sqrt(2*ps.shape[axis])

def ps2pc2(ps, workers=2, axis=0):
    nchan = ps.shape[0]
    nbin = ps.shape[1]
    pc = np.zeros((nchan//2+1,nbin), dtype="csingle")
    for ibin in range(nbin):
        pc[:,ibin] = rfft(ps[:,ibin], axis=axis, workers=workers)
    return pc / np.sqrt(2*ps.shape[axis])

def cs2cc(cs, workers=2, axis=0):
    return fft(cs, axis=axis, workers=workers) / np.sqrt(cs.shape[axis])

def cc2cs(cc, workers=2, axis=0):
    return ifft(cc, axis=axis, workers=workers) * np.sqrt(cc.shape[axis])

def cc2pc(cc, workers=2, axis=1):
    return ifft(cc, axis=axis, workers=workers) * np.sqrt(cc.shape[axis])

def pc2cc(pc,workers=2, axis=1):
    return fft(pc, axis=axis, workers=workers) /  np.sqrt(ps.shape[axis])

def power(x):
    return np.sum(abs(x)**2)/x.size

inputArgs = sys.argv
filename = inputArgs[1]
ar = psrchive.Archive_load(filename)
# ar.remove_baseline()
bw = ar.get_bandwidth()
cf = ar.get_centre_frequency()

data=ar.get_data()
ps = data[0,0,:,:]
pc = ps2pc2(ps)
cc = pc2cc(pc)
cs = cc2cs(cc)

nchan=ps.shape[0]
halfchan=nchan//2
nbin=ps.shape[1]

print(f"nbin={nbin} nchan={nchan} halfchan={halfchan}")

ps_power=power(ps)

print(f'ps shape={ps.shape} power={power(ps)/ps_power}')
print(f'cs shape={cs.shape} power={power(cs)/ps_power}')
print(f'pc shape={pc.shape} power={power(pc)/ps_power}')
print(f'cc shape={cc.shape} power={power(cc)/ps_power}')

SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 32

plt.rc('figure', figsize=[20,15])

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axs = plt.subplots(2,2)

fmin=cf-bw/2
fmax=cf+bw/2
delay_mus = nchan / (bw*2)

min_delay = 10
min_delay_mus = min_delay / bw

cmap="viridis"

toplot=ps
tostat=ps[(nchan//4):(3*nchan//4),:]
med_plot=np.median(tostat,axis=None)
med_plot,count=scipy.stats.mode(tostat,axis=None,keepdims=False)
max_plot=np.max(tostat,axis=None)
range_plot = max_plot - med_plot
minplot=med_plot
maxplot=med_plot + 0.7*range_plot
axs[0,0].imshow(toplot, vmin=minplot, vmax=maxplot, aspect="auto", origin="lower", cmap="plasma", extent=[0, 1, fmin, fmax])
axs[0,0].set(ylabel="Frequency (MHz)", xlabel="Phase (turns)")

toplot=np.log10(abs(cs[:,1:50]))
mplot,count=scipy.stats.mode(toplot,axis=None,keepdims=False)
axs[0,1].imshow(toplot, vmin=mplot, aspect="auto", origin="lower", cmap=cmap, extent=[1, 50, fmin, fmax])
axs[0,1].set(ylabel="Frequency (MHz)", xlabel="Spin Harmonic")

toplot=np.log10(abs(pc[min_delay:,:]))
mplot,count=scipy.stats.mode(toplot,axis=None,keepdims=False)
max_plot=np.max(toplot,axis=None)
range_plot = max_plot - mplot
maxplot=mplot + 0.1*range_plot
axs[1,0].imshow(toplot, vmin=mplot, vmax=maxplot, aspect="auto", origin="lower", cmap=cmap, extent=[0, 1, min_delay_mus, delay_mus],)
axs[1,0].set(ylabel="Delay ($\mu$s)", xlabel="Phase (turns)")

toplot=np.log10(abs(cc[min_delay:,1:50]))
mplot,count=scipy.stats.mode(toplot,axis=None,keepdims=False)
axs[1,1].imshow(toplot, vmin=mplot, aspect="auto", origin="lower", cmap=cmap, extent=[1, 50, min_delay_mus, delay_mus])
axs[1,1].set(ylabel="Delay ($\mu$s)", xlabel="Spin Harmonic")

plt.savefig('ps_pc_cc_cs.png')
plt.close()

