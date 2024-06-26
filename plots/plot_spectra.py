#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, irfft, rfft
import scipy.stats

import numpy as np
import psrchive

def ps2cs(ps, workers=2, axis=1):
    return rfft(ps, axis=axis, workers=workers) / np.sqrt(2*(ps.shape[axis]))

def ps2pc(ps, workers=2, axis=0):
    return rfft(ps, axis=axis, workers=workers) / np.sqrt(2*(ps.shape[axis]))

def cs2cc(cs, workers=2, axis=0):
    return fft(cs, axis=axis, workers=workers) / np.sqrt(cs.shape[axis])

def cc2cs(cc, workers=2, axis=0):
    return ifft(cc, axis=axis, workers=workers) * np.sqrt(cc.shape[axis])

def cc2pc(cc, workers=2, axis=1):
    return ifft(cc, axis=axis, workers=workers) * np.sqrt(cc.shape[axis])

def pc2cc(pc,workers=2, axis=1):
    return fft(pc, axis=axis, workers=workers) /  np.sqrt(pc.shape[axis])

def power(x):
    return np.sum(abs(x)**2)/x.size


def plot_four(ps,bw,cf,clockwise):

    if clockwise:
        cs = ps2cs(ps)
        cc = cs2cc(cs)
        pc = cc2pc(cc)
    else:
        pc = ps2pc(ps)
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

    plt.rc('figure', figsize=[16,12])

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

    positive_delays = True

    if positive_delays:
        min_delay = 0
        max_delay = nchan//8
        min_delay_mus = min_delay / bw
        max_delay_mus = max_delay / bw
    else:
        min_delay = nchan//2 - nchan//16
        max_delay = nchan//2 + nchan//16
        pc = fftshift(pc,axes=0)
        cc = fftshift(cc,axes=0)
        max_delay_mus = (max_delay - nchan//2) / bw
        min_delay_mus = (min_delay - nchan//2) / bw

    cmap="bwr"

    print(f'min_delay={min_delay} max_delay={max_delay}')

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

    scale=0.25

    toplot=np.real(cs[:,1:50])
    minplot=np.min(toplot,axis=None)
    maxplot=np.abs(minplot)
    axs[0,1].imshow(toplot, vmin=scale*minplot, vmax=scale*maxplot, aspect="auto", origin="lower", cmap=cmap, extent=[1, 50, fmin, fmax])
    axs[0,1].set(ylabel="Frequency (MHz)", xlabel="Spin Harmonic")

    scale=0.05

    toplot=np.real(pc[min_delay:max_delay,:])
    minplot=np.min(toplot,axis=None)
    maxplot=np.abs(minplot)
    axs[1,0].imshow(toplot, vmin=scale*minplot, vmax=scale*maxplot, aspect="auto", origin="lower", cmap=cmap, extent=[0, 1, min_delay_mus, max_delay_mus],)
    axs[1,0].set(ylabel="Delay ($\mu$s)", xlabel="Phase (turns)")

    scale=0.05

    toplot=np.real(cc[min_delay:max_delay,1:50])
    minplot=np.min(toplot,axis=None)
    maxplot=np.abs(minplot)
    mplot,count=scipy.stats.mode(toplot,axis=None,keepdims=False)
    axs[1,1].imshow(toplot, vmin=scale*minplot, vmax=scale*maxplot, aspect="auto", origin="lower", cmap=cmap, extent=[1, 50, min_delay_mus, max_delay_mus])
    axs[1,1].set(ylabel="Delay ($\mu$s)", xlabel="Spin Harmonic")

    if clockwise:
        plt.savefig('ps_cs_cc_pc.png')
    else:
        plt.savefig('ps_pc_cc_cs.png')

    plt.close()

def plot_spectra(filename) -> None:
    ar = psrchive.Archive_load(filename)
    # ar.remove_baseline()
    bw = ar.get_bandwidth()
    cf = ar.get_centre_frequency()

    data=ar.get_data()
    slice = data[0,0,:,:]

    nchan = slice.shape[0]
    nbin=slice.shape[1]

    zap_edges = 0.05556
    if zap_edges > 0:
        zap_chan = int(zap_edges * nchan)
        slice = slice[zap_chan:-zap_chan,:]
        bw *= slice.shape[0] / nchan
        nchan = slice.shape[0]

    spectral_window = None
    # spectral_window = ('kaiser', 8.6)
    # spectral_window = ('tukey', 0.05556)

    if spectral_window is not None:
        spectral_taper = scipy.signal.get_window(spectral_window, nchan)
        for ichan in range(nchan):
            slice[ichan,:] *= spectral_taper[ichan]

    ps = np.zeros((nchan,nbin),dtype=np.float64)
    ps[:,:] = slice[:,:]

    plot_four(ps,bw,cf,clockwise=True)
    plot_four(ps,bw,cf,clockwise=False)

def main() -> None:
    """Plot the cyclic spectrum in four different ways."""
    import argparse

    # do arg parsing here
    p = argparse.ArgumentParser()
    p.add_argument(
        "filename",
        type=str,
        help="the file to process",
    )

    args = vars(p.parse_args())
    plot_spectra(**args)


if __name__ == "__main__":
  main()
