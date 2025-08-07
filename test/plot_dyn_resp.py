#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from scipy.fft import ifft2, ifft
import scipy.stats

import numpy as np
import psrchive

def plot_response(dynamic_response,bw,cf):

    delay_Doppler = ifft2(dynamic_response)
    # delay_Doppler = ifft(dynamic_response, axis=1)

    ntime=dynamic_response.shape[0]
    nchan=dynamic_response.shape[1]

    print(f"ntime={ntime} nchan={nchan}")

    SMALL_SIZE = 16
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 32

    plt.rc('figure', figsize=[24,12])

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(1,2)

    fmin=cf-bw/2
    fmax=cf+bw/2

    min_delay_mus = 0
    max_delay_mus = nchan / bw

    print(f'min_delay={min_delay_mus} max_delay={max_delay_mus}')

    tres = 15 # seconds
    tmax = ntime * tres
    max_omega = 1/tres

    toplot=np.real(dynamic_response)
    pmax = np.max(abs(toplot))
    pextent = [fmin, fmax, 0, tmax]
    img = axs[0].imshow(toplot, aspect="auto", origin="lower", cmap="bwr", interpolation=None, extent=pextent, vmin=-pmax, vmax=pmax)
    axs[0].set(xlabel="Frequency (MHz)", ylabel="Time (s)", title="Real")
    fig.colorbar(img)

    toplot=np.imag(dynamic_response)
    pmax = np.max(abs(toplot))
    pextent = [fmin, fmax, 0, tmax]
    img = axs[1].imshow(toplot, aspect="auto", origin="lower", cmap="bwr", interpolation=None, extent=pextent, vmin=-pmax, vmax=pmax)
    axs[1].set(xlabel="Frequency (MHz)", ylabel="Time (s)", title="Imaginary")
    fig.colorbar(img)

    filename = "dyn_resp.png"

    plt.savefig(filename)
    plt.close()

    fig, axs = plt.subplots(1,1)

    toplot=np.log10(np.abs(delay_Doppler))
    max_plot=np.max(toplot,axis=None)
    min_plot=max_plot - 5
    axs.imshow(toplot, vmin=min_plot, vmax=max_plot, aspect="auto", origin="lower", cmap="binary", interpolation='none', extent=[min_delay_mus, max_delay_mus, 0, max_omega])
    axs.set(xlabel="Delay ($\mu$s)", ylabel="Diff. Doppler (Hz)")

    filename = "wavefield.png"

    plt.savefig(filename)
    plt.close()

def plot_spectra(filename) -> None:
    ar = psrchive.Archive_load(filename)

    bw = ar.get_bandwidth()
    cf = ar.get_centre_frequency()

    ext = ar.get_dynamic_response()
    data=ext.get_data()
    nchan = ext.get_nchan()
    ntime = ext.get_ntime()
    data = np.reshape(data, (ntime, nchan))

    plot_response(data,bw,cf)

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
